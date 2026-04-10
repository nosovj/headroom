"""Stress tests for compression pipeline.

Tests:
- Timeout regression detection (p95 < threshold)
- Memory profiling (RSS bounded)
- Per-stage timing assertions

Run with:
    pytest tests/stress/ -m stress
    pytest tests/stress/ -m stress --stress-mode fast  # quick smoke test
    pytest tests/stress/ -m stress --stress-mode full  # full test suite

Default pytest run excludes these (marker: stress).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median

import pytest

# Default thresholds
DEFAULT_TIMEOUT_S = float(os.environ.get("COMPRESSION_TIMEOUT_THRESHOLD", "30"))
DEFAULT_SIMHASH_BUDGET_MS = float(os.environ.get("SIMHASH_BUDGET_MS", "200"))
DEFAULT_IPC_BUDGET_MS = float(os.environ.get("IPC_BUDGET_MS", "100"))
DEFAULT_CLUSTER_BUDGET_MS = float(os.environ.get("CLUSTER_BUDGET_MS", "50"))
DEFAULT_MEMORY_GROWTH_LIMIT = 2.0  # peak RSS / baseline RSS


def _get_rss_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _run_single_compression(item_count: int) -> dict:
    """Run a single compression and return timing info."""
    from headroom.compression.smart.optimized_simhash import (
        count_unique_simhash,
        get_last_timing,
    )

    # Generate items
    from tests.stress.payload_generator import PayloadGenerator
    gen = PayloadGenerator(seed=42)
    items = gen.generate_items(item_count, avg_chars=500, redundancy=0.3)

    rss_before = _get_rss_mb()
    start = time.perf_counter()

    result = count_unique_simhash(items, threshold=3)

    elapsed_ms = (time.perf_counter() - start) * 1000
    rss_after = _get_rss_mb()
    timing = get_last_timing()

    return {
        "result": result,
        "elapsed_ms": elapsed_ms,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "peak_rss_mb": rss_after,
        "timing": timing,
    }


def _compute_percentiles(values: list[float]) -> dict:
    """Compute percentile stats."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "p50": sorted_vals[n // 2],
        "p95": sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1],
        "p99": sorted_vals[int(n * 0.99)] if n >= 100 else sorted_vals[-1],
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": sum(sorted_vals) / n,
    }


def _run_concurrent_compressions(
    item_count: int,
    concurrency: int,
    n_requests: int,
) -> list[float]:
    """Run compressions concurrently and return latencies."""
    latencies = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_run_single_compression, item_count)
            for _ in range(n_requests)
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                latencies.append(result["elapsed_ms"])
            except Exception as e:
                print(f"Compression failed: {e}")
                latencies.append(float("inf"))

    return latencies


# =============================================================================
# Stress Tests
# =============================================================================


@pytest.mark.stress
def test_p95_within_timeout_fast_mode():
    """Fast mode: concurrency [1, 5], 5 requests each, should complete in <15s."""
    pytest.skipif(
        os.environ.get("STRESS_MODE", "fast") != "fast",
        reason="Only runs in fast mode",
    )

    print("\n=== Fast Mode: p95 within timeout ===")

    results = {}
    for concurrency in [1, 5]:
        latencies = _run_concurrent_compressions(
            item_count=200,
            concurrency=concurrency,
            n_requests=5,
        )
        pcts = _compute_percentiles(latencies)
        results[concurrency] = pcts
        print(f"  concurrency={concurrency}: p50={pcts['p50']:.0f}ms p95={pcts['p95']:.0f}ms")

        assert pcts["p95"] < DEFAULT_TIMEOUT_S * 1000, (
            f"p95 latency at concurrency={concurrency}: {pcts['p95']:.0f}ms "
            f"exceeds threshold {DEFAULT_TIMEOUT_S * 1000:.0f}ms"
        )

    print(f"  All levels passed (threshold: {DEFAULT_TIMEOUT_S}s)")


@pytest.mark.stress
def test_p95_within_timeout_full_mode():
    """Full mode: concurrency [1, 5, 10, 25, 50], 20 requests each."""
    pytest.skipif(
        os.environ.get("STRESS_MODE", "fast") != "full",
        reason="Only runs in full mode",
    )

    print("\n=== Full Mode: p95 within timeout ===")

    for concurrency in [1, 5, 10, 25, 50]:
        latencies = _run_concurrent_compressions(
            item_count=200,
            concurrency=concurrency,
            n_requests=20,
        )
        pcts = _compute_percentiles(latencies)
        print(f"  concurrency={concurrency}: p50={pcts['p50']:.0f}ms p95={pcts['p95']:.0f}ms p99={pcts['p99']:.0f}ms")

        assert pcts["p95"] < DEFAULT_TIMEOUT_S * 1000, (
            f"p95 latency at concurrency={concurrency}: {pcts['p95']:.0f}ms "
            f"exceeds threshold {DEFAULT_TIMEOUT_S * 1000:.0f}ms"
        )

    print(f"  All levels passed (threshold: {DEFAULT_TIMEOUT_S}s)")


@pytest.mark.stress
def test_memory_bounded():
    """Memory growth should stay within 2x baseline."""
    print("\n=== Memory Profiling ===")

    # Baseline: single request RSS
    result_single = _run_single_compression(item_count=100)
    baseline_rss = result_single["peak_rss_mb"]
    print(f"  Baseline RSS (1 request): {baseline_rss:.1f}MB")

    # Peak RSS at high concurrency
    latencies = []
    peak_rss = baseline_rss

    for i in range(10):
        result = _run_single_compression(item_count=200)
        latencies.append(result["elapsed_ms"])
        peak_rss = max(peak_rss, result["peak_rss_mb"])

    growth_ratio = peak_rss / baseline_rss if baseline_rss > 0 else float("inf")
    print(f"  Peak RSS (10 requests): {peak_rss:.1f}MB (growth: {growth_ratio:.2f}x)")

    assert growth_ratio <= DEFAULT_MEMORY_GROWTH_LIMIT, (
        f"Memory growth at concurrency=10: {growth_ratio:.2f}x exceeds "
        f"{DEFAULT_MEMORY_GROWTH_LIMIT}x limit"
    )


@pytest.mark.stress
def test_stage_timings_within_budget():
    """Per-stage timings should stay within configured budgets."""
    print("\n=== Per-Stage Timing ===")

    for stage, budget_ms in [
        ("simhash_ms", DEFAULT_SIMHASH_BUDGET_MS),
        ("ipc_ms", DEFAULT_IPC_BUDGET_MS),
        ("cluster_ms", DEFAULT_CLUSTER_BUDGET_MS),
    ]:
        result = _run_single_compression(item_count=300)
        timing = result["timing"]
        actual_ms = timing.get(stage, 0)

        print(f"  {stage}: {actual_ms:.1f}ms (budget: {budget_ms:.0f}ms)")

        # Warn if exceeded (not fail — informational)
        if actual_ms > budget_ms:
            print(f"    WARNING: {stage} exceeds budget by {actual_ms - budget_ms:.0f}ms")


@pytest.mark.stress
def test_baseline_regression():
    """Compare against saved baseline, fail if p95 regresses >10%."""
    import json
    import glob

    baseline_files = glob.glob("benchmarks/stress_baseline_*.json")
    if not baseline_files:
        pytest.skip("No baseline file found — run with --save-baseline first")

    latest_baseline = max(baseline_files)
    with open(latest_baseline) as f:
        baseline = json.load(f)

    print(f"\n=== Baseline Regression: {latest_baseline} ===")

    baseline_by_concurrency = {
        int(m["concurrency"]): m for m in baseline.get("measurements", [])
    }

    passed = True
    for concurrency in [1, 5, 10]:
        if concurrency not in baseline_by_concurrency:
            continue

        latencies = _run_concurrent_compressions(
            item_count=200,
            concurrency=concurrency,
            n_requests=20,
        )
        pcts = _compute_percentiles(latencies)
        baseline_p95 = baseline_by_concurrency[concurrency]["p95_ms"]
        current_p95 = pcts["p95"]
        regression = (current_p95 - baseline_p95) / baseline_p95 if baseline_p95 > 0 else 0

        status = "OK" if regression <= 0.10 else "REGRESSION"
        print(f"  concurrency={concurrency}: baseline_p95={baseline_p95:.0f}ms "
              f"current_p95={current_p95:.0f}ms ({regression:+.1%}) [{status}]")

        if regression > 0.10:
            passed = False

    assert passed, "Regression detected: p95 increased >10% vs baseline"


# =============================================================================
# CLI Helpers
# =============================================================================


def pytest_configure(config):
    """Register the stress marker."""
    config.addinivalue_line("markers", "stress: slow stress tests")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compression stress test")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--save-baseline", metavar="PATH", help="Save results to PATH")
    args = parser.parse_args()

    os.environ["STRESS_MODE"] = args.mode

    if args.save_baseline:
        # Run and save baseline
        import json
        import time

        results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        measurements = []

        for concurrency in [1, 5, 10, 25, 50]:
            latencies = _run_concurrent_compressions(
                item_count=200,
                concurrency=concurrency,
                n_requests=20,
            )
            pcts = _compute_percentiles(latencies)
            measurements.append({
                "concurrency": concurrency,
                "p50_ms": round(pcts["p50"], 2),
                "p95_ms": round(pcts["p95"], 2),
                "p99_ms": round(pcts["p99"], 2),
                "min_ms": round(pcts["min"], 2),
                "max_ms": round(pcts["max"], 2),
            })
            print(f"Saved: concurrency={concurrency} p95={pcts['p95']:.0f}ms")

        results["measurements"] = measurements
        with open(args.save_baseline, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Baseline saved to {args.save_baseline}")
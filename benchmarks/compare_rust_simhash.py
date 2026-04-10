#!/usr/bin/env python3
"""Benchmark Rust vs Cython vs Python simhash implementations.

Run with:
    python3 benchmarks/compare_rust_simhash.py
"""

import argparse
import json
import sys
import time
from statistics import median

# Add parent directory to path
sys.path.insert(0, ".")

from headroom.compression.smart.optimized_simhash import (
    compute_simhash as python_compute_simhash,
    compute_simhash_batch as python_compute_simhash_batch,
    count_unique_simhash as python_count_unique_simhash,
    _get_cython_simhash,
    _get_rust_simhash,
)


def generate_texts(count, avg_chars=500, seed=42):
    import random

    random.seed(seed)
    words = [
        "hello", "world", "testing", "compression", "simhash",
        "optimization", "performance", "benchmark", "latency", "throughput",
    ]
    texts = []
    for i in range(count):
        length = random.randint(int(avg_chars * 0.5), int(avg_chars * 1.5))
        words_in_text = length // 6
        text = " ".join(random.choices(words, k=words_in_text)) + f" item_{i}"
        texts.append(text)
    return texts


ITEM_COUNTS = [50, 200, 500, 1000]
REPETITIONS = 3


def benchmark_single():
    """Benchmark single-item simhash across implementations."""
    cython_fn = _get_cython_simhash()
    rust_fn = _get_rust_simhash()

    texts = generate_texts(100, avg_chars=500, seed=42)
    baseline_text = texts[0]

    results = {}

    # Python
    times = []
    for _ in range(REPETITIONS):
        start = time.perf_counter()
        for _ in range(1000):
            python_compute_simhash(baseline_text)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    results["python"] = median(times) / 1000  # ms per item

    # Cython
    if cython_fn:
        times = []
        for _ in range(REPETITIONS):
            start = time.perf_counter()
            for _ in range(1000):
                cython_fn(baseline_text)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        results["cython"] = median(times) / 1000
    else:
        results["cython"] = None

    # Rust
    if rust_fn:
        times = []
        for _ in range(REPETITIONS):
            start = time.perf_counter()
            for _ in range(1000):
                rust_fn[0](baseline_text)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        results["rust"] = median(times) / 1000
    else:
        results["rust"] = None

    return results


def benchmark_batch(item_count):
    """Benchmark batch simhash across implementations."""
    cython_fn = _get_cython_simhash()
    rust_fn = _get_rust_simhash()

    texts = generate_texts(item_count, avg_chars=500, seed=42)

    results = {}

    # Python (force no Rust to test actual fallback)
    import headroom.compression.smart.optimized_simhash as opt
    opt._rust_simhash = False  # Disable Rust to test Python path
    times = []
    for _ in range(REPETITIONS):
        start = time.perf_counter()
        python_compute_simhash_batch(texts, num_workers=4)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    results["python_mp"] = median(times)
    opt._rust_simhash = rust_fn  # Re-enable Rust

    # Rust (bypasses GIL and IPC)
    if rust_fn:
        times = []
        for _ in range(REPETITIONS):
            start = time.perf_counter()
            rust_fn[1](texts)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        results["rust"] = median(times)
    else:
        results["rust"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Rust vs Cython vs Python simhash")
    parser.add_argument("--json-output", help="Write results to JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("Simhash Performance Comparison: Rust vs Cython vs Python")
    print("=" * 70)
    print()

    # Single-item benchmark
    print("Single-item simhash (1000 iterations):")
    print("-" * 40)
    single_results = benchmark_single()

    if single_results["cython"]:
        print(f"  Python:  {single_results['python']:.4f}ms per item")
        print(f"  Cython:  {single_results['cython']:.4f}ms per item")
        if single_results["rust"]:
            speedup = single_results["python"] / single_results["rust"]
            print(f"  Rust:    {single_results['rust']:.4f}ms per item ({speedup:.1f}x faster than Python)")
    elif single_results["rust"]:
        print(f"  Python:  {single_results['python']:.4f}ms per item")
        print(f"  Rust:    {single_results['rust']:.4f}ms per item")
        speedup = single_results["python"] / single_results["rust"]
        print(f"  Rust is {speedup:.1f}x faster than Python")
    print()

    # Batch benchmark
    print(f"Batch simhash ({ITEM_COUNTS} items):")
    print("-" * 40)
    batch_results = {}
    for n in ITEM_COUNTS:
        batch_results[n] = benchmark_batch(n)

    print(f"{'Items':>6} | {'Python+MP':>12} | {'Rust':>12} | {'Speedup':>10}")
    print("-" * 50)
    for n in ITEM_COUNTS:
        python_ms = batch_results[n]["python_mp"]
        rust_ms = batch_results[n]["rust"]
        if rust_ms:
            speedup = python_ms / rust_ms
            print(f"{n:6d} | {python_ms:12.2f} | {rust_ms:12.2f} | {speedup:9.1f}x")
        else:
            print(f"{n:6d} | {python_ms:12.2f} | {'N/A':>12} | {'N/A':>10}")
    print()

    # Count unique benchmark
    print("count_unique_simhash (1000 items, threshold=3):")
    print("-" * 40)
    rust_fn = _get_rust_simhash()
    texts = generate_texts(1000, avg_chars=500, seed=42)

    import headroom.compression.smart.optimized_simhash as opt
    opt._rust_simhash = False  # Disable Rust to test Python path
    times = []
    for _ in range(REPETITIONS):
        start = time.perf_counter()
        python_count_unique_simhash(texts, 3, num_workers=4)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    python_ms = median(times)
    opt._rust_simhash = rust_fn  # Re-enable Rust

    if rust_fn:
        times = []
        for _ in range(REPETITIONS):
            start = time.perf_counter()
            rust_fn[2](texts, 3)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        rust_ms = median(times)
        speedup = python_ms / rust_ms
        print(f"  Python+MP: {python_ms:.2f}ms")
        print(f"  Rust:      {rust_ms:.2f}ms")
        print(f"  Rust is {speedup:.1f}x faster")
    else:
        print(f"  Python+MP: {python_ms:.2f}ms")
        print(f"  Rust: N/A")
    print()

    # Summary
    print("=" * 70)
    rust_available = _get_rust_simhash() is not None
    if rust_available:
        print("✓ Rust extension available - recommended for production")
        print("  - Bypasses GIL entirely (true parallelism)")
        print("  - No IPC overhead (direct function calls)")
        print("  - Rayon for batch parallelization")
    else:
        print("✗ Rust extension not available")
        print("  - Falling back to Python multiprocessing")
        print("  - Install with: pip install headroom-ai[simhash-rust]")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "single_item": single_results,
        "batch": {str(k): v for k, v in batch_results.items()},
        "rust_available": rust_available,
    }

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.json_output}")

    sys.exit(0)


if __name__ == "__main__":
    main()
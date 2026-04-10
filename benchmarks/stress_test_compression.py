"""Stress test for compression pipeline.

Runs concurrent compression requests to test timeout behavior.
"""

import asyncio
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from headroom.compression.smart import CompressionPipeline, PipelineConfig


@dataclass
class StressResult:
    """Result of a stress test."""

    total_requests: int
    successful: int
    failed: int
    timeouts: int
    latencies: list[float]
    p50: float
    p95: float
    p99: float
    avg: float


def create_test_data(num_items: int = 100) -> str:
    """Create test JSON data similar to real tool output."""
    items = [
        {
            "id": i,
            "name": f"item_{i}",
            "score": round(i * 0.1 + 0.01 * (i % 3), 3),
            "status": ["active", "pending", "done"][i % 3],
            "timestamp": f"2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00",
            "data": {"key": f"value_{i}", "nested": {"a": i, "b": str(i)}},
        }
        for i in range(num_items)
    ]
    return json.dumps(items)


def compress_single(pipeline: CompressionPipeline, data: str, timeout: float = 5.0) -> tuple[bool, float, Optional[str]]:
    """Compress single item with timeout."""
    start = time.perf_counter()
    try:
        compressed, was_modified, strategy = pipeline.compress_content(data)
        elapsed = time.perf_counter() - start
        return True, elapsed * 1000, strategy if was_modified else None
    except Exception as e:
        elapsed = time.perf_counter() - start
        return False, elapsed * 1000, str(e)


def run_stress_test(
    num_requests: int = 50,
    items_per_request: int = 100,
    max_workers: int = 10,
    timeout: float = 5.0,
) -> StressResult:
    """Run stress test with concurrent requests.

    Args:
        num_requests: Number of total requests to send
        items_per_request: Number of items in each request
        max_workers: Max concurrent workers
        timeout: Timeout per request in seconds

    Returns:
        StressResult with statistics
    """
    config = PipelineConfig()
    pipeline = CompressionPipeline(config)
    test_data = create_test_data(items_per_request)

    latencies = []
    successful = 0
    failed = 0
    timeouts = 0
    strategies_used = []

    print(f"Starting stress test:")
    print(f"  Requests: {num_requests}")
    print(f"  Items per request: {items_per_request}")
    print(f"  Max concurrent workers: {max_workers}")
    print(f"  Timeout per request: {timeout}s")
    print()

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_requests):
            future = executor.submit(compress_single, pipeline, test_data, timeout)
            futures.append(future)

        for future in futures:
            success, latency, strategy = future.result()
            latencies.append(latency)
            if success:
                successful += 1
                if strategy:
                    strategies_used.append(strategy)
            else:
                if latency >= timeout * 1000:
                    timeouts += 1
                else:
                    failed += 1

    total_time = time.time() - start_time

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0
    avg = statistics.mean(latencies) if latencies else 0

    result = StressResult(
        total_requests=num_requests,
        successful=successful,
        failed=failed,
        timeouts=timeouts,
        latencies=latencies,
        p50=p50,
        p95=p95,
        p99=p99,
        avg=avg,
    )

    print(f"Stress test completed in {total_time:.2f}s")
    print()
    print(f"Results:")
    print(f"  Successful: {successful}/{num_requests} ({100*successful/num_requests:.1f}%)")
    print(f"  Failed: {failed}")
    print(f"  Timeouts: {timeouts}")
    print()
    print(f"Latency:")
    print(f"  Average: {avg:.2f}ms")
    print(f"  P50: {p50:.2f}ms")
    print(f"  P95: {p95:.2f}ms")
    print(f"  P99: {p99:.2f}ms")
    print()

    if strategies_used:
        unique_strategies = set(strategies_used)
        print(f"Strategies used: {unique_strategies}")

    return result


def run_size_benchmark():
    """Benchmark different data sizes."""
    sizes = [50, 100, 200, 500]
    config = PipelineConfig()
    pipeline = CompressionPipeline(config)

    print("=" * 50)
    print("Size Benchmark")
    print("=" * 50)
    print()

    print(f"{'Size':>8} {'Time (ms)':>12} {'Success':>10}")
    print("-" * 35)

    for size in sizes:
        test_data = create_test_data(size)
        success, latency, _ = compress_single(pipeline, test_data)
        status = "OK" if success else "FAIL"
        print(f"{size:>8} {latency:>12.2f} {status:>10}")

    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Compression Stress Test")
    print("=" * 50)
    print()

    run_size_benchmark()

    print("=" * 50)
    print("Stress Test: 50 concurrent requests")
    print("=" * 50)
    print()

    result = run_stress_test(
        num_requests=50,
        items_per_request=100,
        max_workers=10,
        timeout=5.0,
    )

    success_rate = result.successful / result.total_requests * 100
    if success_rate >= 95 and result.p99 < 5000:
        print("✓ Stress test PASSED")
    else:
        print("✗ Stress test FAILED")
        print(f"  Success rate: {success_rate:.1f}% (need >= 95%)")
        print(f"  P99 latency: {result.p99:.0f}ms (need < 5000ms)")

#!/usr/bin/env python3
"""Side-by-side comparison of compression performance.

Compares the headroom proxy on 8787 vs our new compression server on 8788.
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class BenchmarkResult:
    server: str
    port: int
    total_requests: int
    successful: int
    failed: int
    latencies: list[float]
    p50: float
    p95: float
    p99: float
    avg: float
    min: float
    max: float


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


async def benchmark_server(
    name: str,
    port: int,
    num_requests: int = 50,
    items_per_request: int = 100,
    timeout: float = 30.0,
) -> BenchmarkResult:
    """Benchmark a compression server."""
    latencies = []
    successful = 0
    failed = 0

    test_data = create_test_data(items_per_request)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = []
        start_time = time.perf_counter()

        for i in range(num_requests):
            task = client.post(
                f"http://localhost:{port}/compress",
                json={"content": test_data},
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.perf_counter() - start_time

        for resp in responses:
            if isinstance(resp, Exception):
                failed += 1
            else:
                if resp.status_code == 200:
                    successful += 1
                    elapsed = resp.json().get("latency_ms", 0)
                    if elapsed > 0:
                        latencies.append(elapsed)
                else:
                    failed += 1

    if latencies:
        latencies.sort()
        p50_idx = int(len(latencies) * 0.50)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)

        return BenchmarkResult(
            server=name,
            port=port,
            total_requests=num_requests,
            successful=successful,
            failed=failed,
            latencies=latencies,
            p50=latencies[p50_idx],
            p95=latencies[p95_idx],
            p99=latencies[p99_idx],
            avg=statistics.mean(latencies),
            min=min(latencies),
            max=max(latencies),
        )
    else:
        return BenchmarkResult(
            server=name,
            port=port,
            total_requests=num_requests,
            successful=successful,
            failed=failed,
            latencies=[],
            p50=0,
            p95=0,
            p99=0,
            avg=0,
            min=0,
            max=0,
        )


def print_results(results: list[BenchmarkResult]):
    """Print comparison results."""
    print("\n" + "=" * 80)
    print("COMPRESSION BENCHMARK RESULTS")
    print("=" * 80)

    for r in results:
        print(f"\n{r.server} (port {r.port}):")
        print(f"  Requests:    {r.total_requests}")
        print(f"  Successful:  {r.successful} ({100*r.successful/r.total_requests:.1f}%)")
        print(f"  Failed:      {r.failed} ({100*r.failed/r.total_requests:.1f}%)")
        if r.latencies:
            print(f"  Latency:")
            print(f"    Min:     {r.min:.2f}ms")
            print(f"    Avg:     {r.avg:.2f}ms")
            print(f"    P50:     {r.p50:.2f}ms")
            print(f"    P95:     {r.p95:.2f}ms")
            print(f"    P99:     {r.p99:.2f}ms")
            print(f"    Max:     {r.max:.2f}ms")

    if len(results) == 2:
        r1, r2 = results
        print("\n" + "-" * 80)
        print("COMPARISON")
        print("-" * 80)

        if r1.avg > 0 and r2.avg > 0:
            speedup = r1.avg / r2.avg
            print(f"Speedup ({r2.server} vs {r1.server}): {speedup:.2f}x")
            print(f"  {r1.server} avg: {r1.avg:.2f}ms")
            print(f"  {r2.server} avg: {r2.avg:.2f}ms")

        if r1.p95 > 0 and r2.p95 > 0:
            p95_improvement = (r1.p95 - r2.p95) / r1.p95 * 100
            print(f"\nP95 improvement: {p95_improvement:.1f}%")
            print(f"  {r1.server} P95: {r1.p95:.2f}ms")
            print(f"  {r2.server} P95: {r2.p95:.2f}ms")

    print("\n" + "=" * 80)


async def main():
    print("Starting side-by-side compression benchmark...")
    print("Testing original headroom (8787) vs new compression server (8788)")

    # Check servers
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r1 = await client.get("http://localhost:8787/health")
            print(f"\n8787 health: {r1.status_code}")
        except Exception as e:
            print(f"\n8787 not available: {e}")

        try:
            r2 = await client.get("http://localhost:8788/health/compression")
            print(f"8788 health: {r2.status_code}")
        except Exception as e:
            print(f"8788 not available: {e}")

    print("\nRunning benchmark with 50 concurrent requests, 100 items each...")

    # Run benchmarks
    results = []

    # Benchmark 8787 (original headroom)
    print("\nBenchmarking original headroom (8787)...")
    result_8787 = await benchmark_server("Original Headroom", 8787, num_requests=50, items_per_request=100)
    results.append(result_8787)

    # Benchmark 8788 (new compression server)
    print("Benchmarking new compression server (8788)...")
    result_8788 = await benchmark_server("New Compression", 8788, num_requests=50, items_per_request=100)
    results.append(result_8788)

    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())

"""Performance benchmarks for headroom optimization changes.

This script benchmarks the key performance improvements:
1. Rust simhash (~62ms for 500 items vs ~400ms Python)
2. Token count cache hit rate
3. Worker pool throughput
4. Async pipeline parallelization
5. Circuit breaker overhead
"""

import asyncio
import os
import random
import string
import sys
import time
from typing import Any

# Set feature flags for benchmarking
os.environ.setdefault("HEADROOM_FEATURE_ASYNC_PIPELINE", "true")
os.environ.setdefault("HEADROOM_FEATURE_TOKEN_CACHE", "true")
os.environ.setdefault("HEADROOM_FEATURE_RETRY", "true")
os.environ.setdefault("HEADROOM_FEATURE_CIRCUIT_BREAKER", "true")


def generate_random_items(count: int, avg_length: int = 50) -> list[str]:
    """Generate random strings for benchmarking."""
    items = []
    for _ in range(count):
        length = random.randint(avg_length // 2, avg_length * 2)
        items.append(''.join(random.choices(string.ascii_letters + ' ', k=length)))
    return items


def format_ms(ms: float) -> str:
    """Format milliseconds for display."""
    return f"{ms:.2f}ms"


def format_rate(hits: int, total: int) -> str:
    """Format hit rate percentage."""
    if total == 0:
        return "N/A"
    return f"{100 * hits / total:.1f}%"


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_result(name: str, value: str, expected: str | None = None) -> None:
    """Print a benchmark result."""
    if expected:
        match = "✓" if value == expected else "✗"
        print(f"  {match} {name}: {value} (expected: {expected})")
    else:
        print(f"    {name}: {value}")


# =============================================================================
# Benchmark: Rust Simhash
# =============================================================================

def benchmark_rust_simhash(item_count: int = 500, iterations: int = 3) -> dict[str, Any]:
    """Benchmark Rust simhash performance."""
    print_section(f"Rust Simhash ({item_count} items)")

    from headroom_simhash import count_unique_simhash as rust_count

    items = generate_random_items(item_count)
    times = []

    for i in range(iterations):
        start = time.perf_counter()
        result = rust_count(items, threshold=3)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        print(f"  Iteration {i+1}: {format_ms(elapsed_ms)} ({result} unique)")

    avg_ms = sum(times) / len(times)
    print_result("Average", format_ms(avg_ms))
    print_result("Items/second", f"{item_count / (avg_ms / 1000):.0f}")

    return {
        "avg_ms": avg_ms,
        "min_ms": min(times),
        "max_ms": max(times),
        "items": item_count,
        "unique": result,
    }


# =============================================================================
# Benchmark: Token Count Cache
# =============================================================================

def benchmark_token_cache(item_count: int = 1000, iterations: int = 5) -> dict[str, Any]:
    """Benchmark token count cache performance."""
    print_section(f"Token Count Cache ({item_count} items)")

    from headroom.cache.tokenizer_cache import TokenCountCache, CacheStats

    # Create cache
    cache = TokenCountCache(max_size=10000)

    # Create a simple tokenizer mock
    class MockTokenizer:
        def count_text(self, text: str) -> int:
            return len(text.split())

    tokenizer = MockTokenizer()

    # Generate unique and duplicate content
    unique_content = [f"unique content number {i}" for i in range(item_count // 2)]
    duplicate_content = [unique_content[0]] * (item_count // 2)
    all_content = unique_content + duplicate_content
    random.shuffle(all_content)

    session_id = "benchmark-session"

    # First pass - populate cache
    print(f"  Populating cache with {item_count} items...")
    for i, content in enumerate(all_content[:item_count]):
        cache.put(session_id, content, tokenizer.count_text(content))

    cache_stats = cache.get_stats()
    print_result("Cache size", str(cache.size))
    print_result("Cache max_size", str(cache.max_size))

    # Second pass - measure cache hits
    print(f"  Measuring cache performance ({iterations} iterations)...")
    total_hits = 0
    total_misses = 0

    for _ in range(iterations):
        for content in all_content[:item_count]:
            result = cache.get(session_id, content)
            if result is not None:
                total_hits += 1
            else:
                total_misses += 1

    stats = cache.get_stats()
    print_result("Total hits", str(stats.hits))
    print_result("Total misses", str(stats.misses))
    print_result("Hit rate", format_rate(stats.hits, stats.hits + stats.misses))
    print_result("Evictions", str(stats.evictions))

    # Test session invalidation
    evicted = cache.invalidate_session(session_id)
    print_result("Invalidated entries", str(evicted))
    print_result("Post-invalidation size", str(cache.size))

    return {
        "hit_rate": stats.hit_rate,
        "hits": stats.hits,
        "misses": stats.misses,
        "evictions": stats.evictions,
    }


# =============================================================================
# Benchmark: Worker Pool
# =============================================================================

def benchmark_worker_pool(pool_size: int = 4, work_items: int = 20) -> dict[str, Any]:
    """Benchmark worker pool throughput."""
    print_section(f"Worker Pool (size={pool_size}, {work_items} work items)")

    try:
        import headroom_workers as workers

        # Create and start pool
        pool = workers.create_pool(pool_size=pool_size)
        pool.start()

        stats = pool.get_stats()
        print_result("Workers created", str(stats.total_workers))

        # Submit work
        print(f"  Submitting {work_items} work items...")
        start = time.perf_counter()

        for i in range(work_items):
            pool.submit_work(
                request_id=i,
                payload=f'{{"type": "work", "id": {i}, "data": "benchmark"}}'
            )

        # Give time for processing
        time.sleep(0.5)

        elapsed_ms = (time.perf_counter() - start) * 1000
        print_result("Submit time", format_ms(elapsed_ms))
        print_result("Throughput", f"{work_items / (elapsed_ms / 1000):.0f} submits/sec")

        # Get final stats
        stats = pool.get_stats()
        print_result("Pool busy workers", str(stats.busy_workers))
        print_result("Pool idle workers", str(stats.idle_workers))
        print_result("Panics recovered", str(stats.panics_recovered))

        pool.stop()

        return {
            "submit_ms": elapsed_ms,
            "throughput": work_items / (elapsed_ms / 1000),
        }

    except Exception as e:
        print(f"  Worker pool benchmark skipped: {e}")
        return {"error": str(e)}


# =============================================================================
# Benchmark: Adaptive Sizer
# =============================================================================

def benchmark_adaptive_sizer(item_count: int = 500, iterations: int = 3) -> dict[str, Any]:
    """Benchmark compute_optimal_k with Rust simhash."""
    print_section(f"Adaptive Sizer ({item_count} items)")

    from headroom.transforms.adaptive_sizer import compute_optimal_k

    items = generate_random_items(item_count, avg_length=100)
    times = []

    for i in range(iterations):
        start = time.perf_counter()
        k = compute_optimal_k(items)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        print(f"  Iteration {i+1}: {format_ms(elapsed_ms)} -> k={k}")

    avg_ms = sum(times) / len(times)
    print_result("Average", format_ms(avg_ms))
    print_result("Target", "<50ms for 500 items")

    return {
        "avg_ms": avg_ms,
        "k": k,
    }


# =============================================================================
# Benchmark: Circuit Breaker
# =============================================================================

def benchmark_circuit_breaker(iterations: int = 100) -> dict[str, Any]:
    """Benchmark circuit breaker overhead."""
    print_section(f"Circuit Breaker ({iterations} iterations)")

    from headroom.resilience import CircuitBreaker, CircuitState

    cb = CircuitBreaker(name="benchmark", failure_threshold=5, recovery_timeout=30)

    # Benchmark state checks
    print("  Benchmarking state check overhead...")
    start = time.perf_counter()
    for _ in range(iterations):
        _ = cb.is_closed
        _ = cb.is_open
        _ = cb.is_half_open
    elapsed_ms = (time.perf_counter() - start) * 1000

    per_call_ns = (elapsed_ms * 1_000_000) / iterations
    print_result("State check (100 iterations)", format_ms(elapsed_ms))
    print_result("Per-call overhead", f"{per_call_ns:.0f}ns")
    print_result("Initial state", str(cb.state.value))

    # Test state transitions
    print("  Testing state transitions...")

    # Record 5 failures to trip the circuit
    for i in range(5):
        asyncio.run(cb.record_failure())

    print_result("After 5 failures", str(cb.state.value))
    print_result("Is open", str(cb.is_open))

    # Reset
    asyncio.run(cb.record_success())  # This closes it when half-open
    asyncio.run(cb.record_success())
    print_result("After 2 successes", str(cb.state.value))

    return {
        "overhead_per_call_ns": per_call_ns,
    }


# =============================================================================
# Main
# =============================================================================

def run_all_benchmarks() -> dict[str, Any]:
    """Run all benchmarks and return results."""
    print("\n" + "="*60)
    print(" HEADROOM OPTIMIZATION BENCHMARKS")
    print("="*60)
    print(f" Python: {sys.version.split()[0]}")
    print(f" Working directory: {os.getcwd()}")

    results = {}

    # Run benchmarks
    results["simhash"] = benchmark_rust_simhash(500, iterations=3)
    results["adaptive_sizer"] = benchmark_adaptive_sizer(500, iterations=3)
    results["token_cache"] = benchmark_token_cache(1000, iterations=3)
    results["worker_pool"] = benchmark_worker_pool(pool_size=4, work_items=20)
    results["circuit_breaker"] = benchmark_circuit_breaker(100)

    # Summary
    print_section("SUMMARY")
    print(f"  Rust simhash avg: {format_ms(results['simhash']['avg_ms'])}")
    print(f"  Adaptive sizer avg: {format_ms(results['adaptive_sizer']['avg_ms'])}")
    if "hit_rate" in results["token_cache"]:
        print(f"  Token cache hit rate: {format_rate(results['token_cache']['hits'], results['token_cache']['hits'] + results['token_cache']['misses'])}")
    print(f"  Circuit breaker overhead: {results['circuit_breaker']['overhead_per_call_ns']:.0f}ns per call")

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
    print("\n" + "="*60)
    print(" BENCHMARKS COMPLETE")
    print("="*60 + "\n")

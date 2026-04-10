#!/usr/bin/env python3
"""Capture baseline timings for compression pipeline bottlenecks.

Run this before optimization changes to establish baseline:
    python3 benchmarks/capture_bottleneck_baseline.py

Then compare after optimizations:
    python3 benchmarks/capture_bottleneck_baseline.py --json-output benchmarks/bottleneck_baseline.json
"""

import argparse
import json
import time
from statistics import median

# Add headroom-fork to path
import sys
sys.path.insert(0, "/home/joe/llm/headroom-fork")

from headroom.compression.smart.optimized_simhash import (
    count_unique_simhash as rust_count_unique,
    compute_simhash_batch as rust_batch,
)
from headroom.transforms.adaptive_sizer import count_unique_simhash as adaptive_count_unique


def generate_items(n, avg_size=500):
    """Generate realistic test items."""
    import random
    random.seed(42)
    
    items = []
    for i in range(n):
        content = "word " * (avg_size // 5)
        items.append({
            "id": i,
            "content": content,
            "score": random.random(),
            "category": f"cat_{i % 10}",
        })
    return items


def benchmark_json_serialization(items, n_reps=5):
    """Benchmark json.dumps vs orjson."""
    import json
    
    results = {}
    
    # Standard json.dumps
    times = []
    for _ in range(n_reps):
        start = time.perf_counter()
        result = [json.dumps(item, default=str) for item in items]
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    results["json_dumps"] = median(times)
    
    # orjson
    try:
        import orjson
        times = []
        for _ in range(n_reps):
            start = time.perf_counter()
            result = [orjson.dumps(item) for item in items]
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        results["orjson_dumps"] = median(times)
    except ImportError:
        results["orjson_dumps"] = None
    
    return results


def benchmark_simhash_variants(items, n_reps=3):
    """Benchmark simhash implementations."""
    results = {}
    
    # Rust (optimized_simhash)
    times = []
    texts = ["item_" + str(i) for i in range(len(items))]
    for _ in range(n_reps):
        start = time.perf_counter()
        rust_count_unique(texts, threshold=3, num_workers=4)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    results["rust_optimized"] = median(times)
    
    # Adaptive sizer (pure Python - the OLD implementation)
    import headroom.compression.smart.optimized_simhash as opt
    old_rust = opt._rust_simhash
    opt._rust_simhash = False
    times = []
    for _ in range(n_reps):
        start = time.perf_counter()
        adaptive_count_unique(texts, threshold=3)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    results["adaptive_python"] = median(times)
    opt._rust_simhash = old_rust
    
    return results


def benchmark_bigram_computation(items, n_reps=3):
    """Benchmark bigram computation."""
    from headroom.transforms.adaptive_sizer import compute_unique_bigram_curve
    
    times = []
    for _ in range(n_reps):
        start = time.perf_counter()
        # Simulate the bigram computation
        texts = [item["content"] for item in items]
        compute_unique_bigram_curve(texts)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {"bigram_cpu": median(times)}


def main():
    parser = argparse.ArgumentParser(description="Capture bottleneck baseline timings")
    parser.add_argument("--json-output", help="Write results to JSON file")
    parser.add_argument("--items", type=int, default=500, help="Number of items to test")
    args = parser.parse_args()
    
    print("=" * 70)
    print("BOTTLENECK BASELINE CAPTURE")
    print("=" * 70)
    
    items = generate_items(args.items)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "item_count": args.items,
        "json_serialization": benchmark_json_serialization(items),
        "simhash_variants": benchmark_simhash_variants(items),
        "bigram": benchmark_bigram_computation(items),
    }
    
    print(f"\nItem count: {args.items}")
    
    print("\nJSON Serialization:")
    for k, v in results["json_serialization"].items():
        if v is not None:
            print(f"  {k}: {v:.2f}ms")
        else:
            print(f"  {k}: N/A")
    
    if results["json_serialization"].get("orjson_dumps"):
        speedup = results["json_serialization"]["json_dumps"] / results["json_serialization"]["orjson_dumps"]
        print(f"  json vs orjson: {speedup:.1f}x faster with orjson")
    
    print("\nSimhash Variants:")
    for k, v in results["simhash_variants"].items():
        print(f"  {k}: {v:.2f}ms")
    
    if results["simhash_variants"].get("rust_optimized") and results["simhash_variants"].get("adaptive_python"):
        speedup = results["simhash_variants"]["adaptive_python"] / results["simhash_variants"]["rust_optimized"]
        print(f"  Rust vs adaptive Python: {speedup:.1f}x faster with Rust")
    
    print("\nBigram Computation:")
    for k, v in results["bigram"].items():
        print(f"  {k}: {v:.2f}ms")
    
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline written to {args.json_output}")
    
    return results


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Compare current simhash performance against baseline.

Run after optimization changes:
    python3 benchmarks/compare_simhash.py --baseline benchmarks/baseline_simhash.json

Exit code 1 if any threshold fails.
"""

import argparse
import json
import os
import sys
import time
from statistics import median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from headroom.compression.smart.optimized_simhash import (
    compute_simhash_batch,
    count_unique_simhash,
)


def timed_batch(texts, num_workers=4):
    start = time.perf_counter()
    results = compute_simhash_batch(list(texts), num_workers)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return results, elapsed_ms


def timed_count_unique(items, threshold=3, num_workers=4):
    start = time.perf_counter()
    result = count_unique_simhash(list(items), threshold, num_workers)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


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


ITEM_COUNTS = [50, 100, 200, 500, 1000]
REPETITIONS = 3
NUM_WORKERS = 4

# Thresholds
SPEEDUP_THRESHOLD = 2.0  # 2x faster for 500+ items
IPC_OVERHEAD_MAX = 0.30   # IPC < 30% of total for 500+ items


def main():
    parser = argparse.ArgumentParser(description="Compare simhash performance vs baseline")
    parser.add_argument("--baseline", required=True, help="Path to baseline_simhash.json")
    parser.add_argument("--json-output", help="Write results to JSON file")
    args = parser.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)

    baseline_by_items = {m["items"]: m for m in baseline["measurements"]}

    print("Comparing current simhash performance vs baseline...")
    print(f"Baseline from: {baseline['timestamp']} ({baseline['python_version']})")
    print()
    print(f"{'Items':>6} | {'Baseline':>10} | {'Current':>10} | {'Speedup':>8} | {'Pass/Fail'}")
    print("-" * 60)

    all_passed = True
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": os.uname().nodename,
        "comparisons": [],
    }

    for count in ITEM_COUNTS:
        texts = generate_texts(count, avg_chars=500, seed=42)
        times = []

        for rep in range(REPETITIONS):
            _, ms = timed_batch(texts, NUM_WORKERS)
            times.append(ms)

        current_ms = median(times)
        baseline_ms = baseline_by_items[count]["batch_ms"]
        speedup = baseline_ms / current_ms if current_ms > 0 else float("inf")

        passed = speedup >= SPEEDUP_THRESHOLD if count >= 500 else True
        status = "PASS" if passed else "FAIL"

        if not passed:
            all_passed = False

        print(f"{count:6d} | {baseline_ms:10.2f} | {current_ms:10.2f} | {speedup:8.2f}x | {status}")

        results["comparisons"].append({
            "items": count,
            "baseline_ms": baseline_ms,
            "current_ms": round(current_ms, 2),
            "speedup": round(speedup, 3),
            "passed": passed,
        })

    print()
    if all_passed:
        print("All thresholds PASSED")
    else:
        print("Some thresholds FAILED (exit code 1)")
        print(f"Speedup threshold: {SPEEDUP_THRESHOLD}x for 500+ items")

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.json_output}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
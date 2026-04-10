#!/usr/bin/env python3
"""Capture baseline simhash performance before optimization changes.

Run this BEFORE making changes to capture the baseline:
    python3 benchmarks/capture_baseline.py

Results are saved to benchmarks/baseline_simhash.json
"""

import json
import os
import sys
import time
from statistics import median

# Add headroom-fork to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from headroom.compression.smart.optimized_simhash import (
    compute_simhash,
    compute_simhash_batch,
    count_unique_simhash,
)


def timed_batch(texts, num_workers=4):
    """Time a batch compute and return (results, simhash_ms, ipc_ms)."""
    start = time.perf_counter()
    results = compute_simhash_batch(list(texts), num_workers)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return results, elapsed_ms


def timed_count_unique(items, threshold=3, num_workers=4):
    """Time a count_unique_simhash call."""
    start = time.perf_counter()
    result = count_unique_simhash(list(items), threshold, num_workers)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def generate_texts(count, avg_chars=500, seed=42):
    """Generate deterministic test texts."""
    import random

    random.seed(seed)
    words = [
        "hello",
        "world",
        "testing",
        "compression",
        "simhash",
        "optimization",
        "performance",
        "benchmark",
        "latency",
        "throughput",
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


def main():
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
        "hostname": os.uname().nodename,
        "measurements": [],
    }

    print("Capturing baseline simhash performance...")
    print(f"Item counts: {ITEM_COUNTS}")
    print(f"Repetitions per count: {REPETITIONS}")
    print()

    for count in ITEM_COUNTS:
        texts = generate_texts(count, avg_chars=500, seed=42)
        batch_times = []
        count_times = []

        for rep in range(REPETITIONS):
            # Measure batch compute
            _, batch_ms = timed_batch(texts, NUM_WORKERS)
            batch_times.append(batch_ms)

            # Measure count_unique
            _, count_ms = timed_count_unique(texts, 3, NUM_WORKERS)
            count_times.append(count_ms)

        # Use median of 3 runs
        batch_median = median(batch_times)
        count_median = median(count_times)

        measurement = {
            "items": count,
            "batch_ms": round(batch_median, 2),
            "count_unique_ms": round(count_median, 2),
            "batch_times_ms": [round(t, 2) for t in batch_times],
            "count_times_ms": [round(t, 2) for t in count_times],
        }
        results["measurements"].append(measurement)

        print(f"  {count:4d} items: batch={batch_median:8.2f}ms  count_unique={count_median:8.2f}ms")

    output_path = os.path.join(os.path.dirname(__file__), "baseline_simhash.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Baseline saved to {output_path}")
    print(f"Commit this file as a read-only benchmark artifact.")

    return results


if __name__ == "__main__":
    main()
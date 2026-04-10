#!/usr/bin/env python3.11
"""Benchmark the compression pipeline to verify Rust simhash is working."""

import sys
sys.path.insert(0, '/home/joe/llm/headroom-fork')

import time

print("=" * 60)
print("HEADROOM SIMHASH BENCHMARK")
print("=" * 60)

# Apply the patch as the unified proxy does
from headroom.compression.smart.optimized_simhash import count_unique_simhash as rust_count_unique
import headroom.transforms.adaptive_sizer as ad_mod
ad_mod.count_unique_simhash = rust_count_unique

print("\n1. Verify Rust extension is loaded:")
from headroom.compression.smart.optimized_simhash import _get_rust_simhash
rust = _get_rust_simhash()
print(f"   Rust available: {rust is not False}")
if rust:
    print(f"   count_unique fn: {rust[2]}")

print(f"\n2. Verify patch applied:")
print(f"   ad_mod.count_unique_simhash is rust: {ad_mod.count_unique_simhash is rust_count_unique}")

print(f"\n3. Benchmark count_unique_simhash directly:")
from headroom_simhash import count_unique_simhash as direct_rust
items = ['item ' + str(i) + ' word ' * 30 for i in range(500)]

times = []
for i in range(5):
    start = time.perf_counter()
    result = direct_rust(items, 3)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.1f}ms (result={result})")

print(f"\n   Direct Rust avg: {sum(times)/len(times):.1f}ms")

print(f"\n4. Benchmark compute_optimal_k (full pipeline):")
from headroom.transforms.adaptive_sizer import compute_optimal_k

times = []
for i in range(5):
    start = time.perf_counter()
    k = compute_optimal_k(items, bias=1.0, min_k=3, max_k=None)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.1f}ms (k={k})")

print(f"\n   compute_optimal_k avg: {sum(times)/len(times):.1f}ms")

print(f"\n5. Compare with baseline (before cutover):")
print(f"   unified-proxy (before): 256.5ms avg")
print(f"   optimized-fork (before): 2.6ms avg")
current_avg = sum(times)/len(times)
if current_avg < 50:
    print(f"   Current: {current_avg:.1f}ms - OPTIMIZED ✓")
elif current_avg < 200:
    print(f"   Current: {current_avg:.1f}ms - PARTIAL ✓")
else:
    print(f"   Current: {current_avg:.1f}ms - NEEDS WORK")

print("\n" + "=" * 60)
"""Integration test for compression pipeline.

Tests the full compression pipeline end-to-end.
"""

import json
import time
from headroom.compression.smart import CompressionPipeline, PipelineConfig


def create_test_data(num_items: int = 100) -> str:
    """Create test JSON data."""
    items = [
        {"id": i, "name": f"item_{i}", "score": round(i * 0.1, 2), "active": i % 2 == 0}
        for i in range(num_items)
    ]
    return json.dumps(items)


def test_pipeline_single():
    """Test pipeline with single compression."""
    config = PipelineConfig()
    pipeline = CompressionPipeline(config)
    
    test_data = create_test_data(100)
    
    start = time.perf_counter()
    compressed, was_modified, strategy = pipeline.compress_content(test_data)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"Single compression test:")
    print(f"  Original size: {len(test_data)} chars")
    print(f"  Compressed size: {len(compressed)} chars")
    print(f"  Was modified: {was_modified}")
    print(f"  Strategy: {strategy}")
    print(f"  Time: {elapsed:.2f}ms")
    print()
    
    return elapsed


def test_pipeline_repeated(num_iterations: int = 10):
    """Test pipeline with repeated compressions."""
    config = PipelineConfig()
    pipeline = CompressionPipeline(config)
    
    test_data = create_test_data(100)
    
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        compressed, was_modified, strategy = pipeline.compress_content(test_data)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Repeated compression test ({num_iterations} iterations):")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    print()
    
    return avg_time


def test_various_sizes():
    """Test pipeline with various data sizes."""
    config = PipelineConfig()
    pipeline = CompressionPipeline(config)
    
    sizes = [10, 50, 100, 500, 1000]
    
    print("Size variation test:")
    print(f"{'Size':>8} {'Time (ms)':>12} {'Compressed':>12} {'Ratio':>8}")
    print("-" * 45)
    
    for size in sizes:
        test_data = create_test_data(size)
        original_len = len(test_data)
        
        start = time.perf_counter()
        compressed, was_modified, strategy = pipeline.compress_content(test_data)
        elapsed = (time.perf_counter() - start) * 1000
        
        ratio = len(compressed) / original_len if original_len > 0 else 1.0
        print(f"{size:>8} {elapsed:>12.2f} {len(compressed):>12} {ratio:>8.2f}")
    
    print()


def test_invalid_json():
    """Test pipeline with invalid JSON."""
    config = PipelineConfig()
    pipeline = CompressionPipeline(config)
    
    result, was_modified, strategy = pipeline.compress_content("not valid json")
    
    print("Invalid JSON test:")
    print(f"  Input returned unchanged: {result == 'not valid json'}")
    print(f"  Strategy: {strategy}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Compression Pipeline Integration Test")
    print("=" * 50)
    print()
    
    test_invalid_json()
    test_various_sizes()
    test_pipeline_single()
    test_pipeline_repeated(10)
    
    print("Done!")

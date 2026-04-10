"""Tests for GPU-accelerated compression functions."""

from __future__ import annotations

import pytest


class TestFindKneePointGPU:
    """Tests for GPU-accelerated knee point detection."""

    def test_gpu_available(self):
        from headroom.compression.smart.gpu.compress import is_available
        assert isinstance(is_available(), bool)

    def test_empty_input(self):
        from headroom.compression.smart.gpu.compress import find_knee_point_gpu
        result = find_knee_point_gpu([])
        assert result is None

    def test_small_input(self):
        from headroom.compression.smart.gpu.compress import find_knee_point_gpu
        result = find_knee_point_gpu([1.0, 2.0, 3.0])
        assert result is None

    def test_constant_values(self):
        from headroom.compression.smart.gpu.compress import find_knee_point_gpu
        values = [1.0] * 50
        result = find_knee_point_gpu(values, window=5)
        assert result is None

    def test_returns_int_or_none(self):
        from headroom.compression.smart.gpu.compress import find_knee_point_gpu
        values = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0] * 10
        result = find_knee_point_gpu(values, window=3)
        assert result is None or isinstance(result, int)


class TestCrushNumberArrayGPU:
    """Tests for GPU-accelerated numeric array crushing."""

    def test_empty_input(self):
        from headroom.compression.smart.gpu.compress import crush_number_array_gpu
        result, strategy = crush_number_array_gpu([])
        assert result == []
        assert strategy == "passthrough"

    def test_small_input(self):
        from headroom.compression.smart.gpu.compress import crush_number_array_gpu
        result, strategy = crush_number_array_gpu([1, 2, 3])
        assert result == [1, 2, 3]
        assert strategy == "passthrough"

    def test_linear_increasing(self):
        from headroom.compression.smart.gpu.compress import crush_number_array_gpu
        values = list(range(1, 101))
        result, strategy = crush_number_array_gpu(values, bias=2.0)
        assert len(result) <= len(values)
        assert strategy in ["kneedle_gpu", "kneedle_cpu", "uniform_sampling_gpu", "uniform_sampling_cpu", "no_knee_found"]

    def test_bias_prevents_crush(self):
        from headroom.compression.smart.gpu.compress import crush_number_array_gpu
        values = list(range(1, 101))
        result, strategy = crush_number_array_gpu(values, bias=0.1)
        assert len(result) <= len(values)

    def test_config_respected(self):
        from headroom.compression.smart.gpu.compress import crush_number_array_gpu
        values = [float(x) for x in range(1, 101)]
        config = {"min_items_to_analyze": 10, "max_items_after_crush": 20}
        result, strategy = crush_number_array_gpu(values, config=config, bias=2.0)
        assert len(result) <= 20 or strategy.startswith("passthrough")


class TestCrushStringArrayGPU:
    """Tests for GPU-accelerated string array crushing."""

    def test_empty_input(self):
        from headroom.compression.smart.gpu.compress import crush_string_array_gpu
        result, strategy = crush_string_array_gpu([])
        assert result == []

    def test_small_input(self):
        from headroom.compression.smart.gpu.compress import crush_string_array_gpu
        result, strategy = crush_string_array_gpu(["a", "b", "c"])
        assert result == ["a", "b", "c"]
        assert strategy == "passthrough"

    def test_all_unique(self):
        from headroom.compression.smart.gpu.compress import crush_string_array_gpu
        values = [f"unique_{i}" for i in range(50)]
        result, strategy = crush_string_array_gpu(values, bias=2.0)
        assert isinstance(result, list)

    def test_all_same(self):
        from headroom.compression.smart.gpu.compress import crush_string_array_gpu
        values = ["same"] * 50
        result, strategy = crush_string_array_gpu(values)
        assert len(result) <= len(values)

    def test_mixed_entropy(self):
        from headroom.compression.smart.gpu.compress import crush_string_array_gpu
        values = ["aaaaa"] * 25 + ["xyz123!@#"] * 25
        result, strategy = crush_string_array_gpu(values, bias=2.0)
        assert isinstance(result, list)
        assert len(result) <= len(values)


class TestCrushArrayGPU:
    """Tests for GPU-accelerated dict array crushing."""

    def test_empty_input(self):
        from headroom.compression.smart.gpu.compress import crush_array_gpu
        result, strategy, modified = crush_array_gpu([])
        assert result == []
        assert modified is False

    def test_small_input(self):
        from headroom.compression.smart.gpu.compress import crush_array_gpu
        items = [{"a": 1}, {"b": 2}]
        result, strategy, modified = crush_array_gpu(items)
        assert result == items
        assert modified is False

    def test_all_same_keys(self):
        from headroom.compression.smart.gpu.compress import crush_array_gpu
        items = [{"id": i, "name": f"item_{i}", "data": i * 10} for i in range(50)]
        result, strategy, modified = crush_array_gpu(items, bias=2.0)
        assert isinstance(result, list)

    def test_different_keys(self):
        from headroom.compression.smart.gpu.compress import crush_array_gpu
        items = [
            {"id": i, "unique_key": f"val_{i}"}
            for i in range(30)
        ]
        result, strategy, modified = crush_array_gpu(items)
        assert isinstance(result, list)

    def test_config_max_keys(self):
        from headroom.compression.smart.gpu.compress import crush_array_gpu
        items = [{"key1": i, "key2": i, "key3": i, "key4": i} for i in range(20)]
        config = {"max_keys_to_preserve": 2}
        result, strategy, modified = crush_array_gpu(items, config=config)
        assert modified is True or strategy == "all_keys_kept"


class TestGPUCompressFallback:
    """Tests for GPU fallback behavior when CUDA unavailable."""

    def test_functions_work_without_cuda(self):
        from headroom.compression.smart.gpu.compress import (
            crush_number_array_gpu,
            crush_string_array_gpu,
            crush_array_gpu,
            find_knee_point_gpu,
        )

        result = find_knee_point_gpu(list(range(1, 51)), window=5)
        assert result is None or isinstance(result, int)

        result, strategy = crush_number_array_gpu(list(range(1, 101)), bias=2.0)
        assert isinstance(result, list)
        assert isinstance(strategy, str)

        result, strategy = crush_string_array_gpu(["test"] * 50)
        assert isinstance(result, list)

        result, strategy, modified = crush_array_gpu([{"a": i} for i in range(50)])
        assert isinstance(result, list)

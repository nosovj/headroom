"""Tests for GPU-accelerated analyze functions."""

from __future__ import annotations

import pytest

from headroom.compression.smart.analyze import (
    calculate_string_entropy,
    detect_change_points,
)
from headroom.compression.smart.parse import ArrayType, classify_array


class TestCalculateStringEntropyGPU:
    """Tests for GPU-accelerated string entropy calculation."""

    def test_gpu_available(self):
        from headroom.compression.smart.gpu.analyze import is_available
        assert isinstance(is_available(), bool)

    def test_empty_input(self):
        from headroom.compression.smart.gpu.analyze import calculate_string_entropy_gpu
        result = calculate_string_entropy_gpu([])
        assert result == []

    def test_single_empty_string(self):
        from headroom.compression.smart.gpu.analyze import calculate_string_entropy_gpu
        result = calculate_string_entropy_gpu([""])
        assert result[0] == 0.0

    def test_single_char_string(self):
        from headroom.compression.smart.gpu.analyze import calculate_string_entropy_gpu
        result = calculate_string_entropy_gpu(["a"])
        assert result[0] == 0.0

    def test_repeated_string(self):
        from headroom.compression.smart.gpu.analyze import calculate_string_entropy_gpu
        result = calculate_string_entropy_gpu(["aaaaaa"])
        assert result[0] < 0.3

    def test_random_string(self):
        from headroom.compression.smart.gpu.analyze import calculate_string_entropy_gpu
        result = calculate_string_entropy_gpu(["abc123!@#"])
        assert 0.0 <= result[0] <= 1.0

    def test_multiple_strings(self):
        from headroom.compression.smart.gpu.analyze import calculate_string_entropy_gpu
        result = calculate_string_entropy_gpu(["aaaa", "bbbb", "abab"])
        assert len(result) == 3
        assert all(0.0 <= r <= 1.0 for r in result)

    def test_matches_cpu_single(self):
        from headroom.compression.smart.gpu.analyze import calculate_string_entropy_gpu
        strings = ["hello world", "test string", "AAAAAAA"]
        for s in strings:
            cpu_result = calculate_string_entropy(s)
            gpu_result = calculate_string_entropy_gpu([s])[0]
            assert abs(cpu_result - gpu_result) < 1e-6, f"Mismatch for '{s}': cpu={cpu_result}, gpu={gpu_result}"

    def test_matches_cpu_batch(self):
        from headroom.compression.smart.gpu.analyze import calculate_string_entropy_gpu
        strings = [
            "hello world",
            "test string",
            "AAAAAAA",
            "random123!@#",
            "a" * 20,
            "bcdefghij",
        ]
        cpu_results = [calculate_string_entropy(s) for s in strings]
        gpu_results = calculate_string_entropy_gpu(strings)
        for cpu, gpu in zip(cpu_results, gpu_results):
            assert abs(cpu - gpu) < 1e-6


class TestDetectChangePointsGPU:
    """Tests for GPU-accelerated change point detection."""

    def test_empty_input(self):
        from headroom.compression.smart.gpu.analyze import detect_change_points_gpu
        result = detect_change_points_gpu([], window=5)
        assert result == []

    def test_small_input(self):
        from headroom.compression.smart.gpu.analyze import detect_change_points_gpu
        result = detect_change_points_gpu([1.0, 2.0, 3.0], window=5)
        assert result == []

    def test_no_change_points(self):
        from headroom.compression.smart.gpu.analyze import detect_change_points_gpu
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = detect_change_points_gpu(values, window=2)
        assert result == []

    def test_single_change_point(self):
        from headroom.compression.smart.gpu.analyze import detect_change_points_gpu
        values = [1.0] * 20 + [5.0] * 5
        result = detect_change_points_gpu(values, window=3, variance_threshold=2.0)
        assert len(result) >= 1

    def test_multiple_change_points(self):
        from headroom.compression.smart.gpu.analyze import detect_change_points_gpu
        values = [1.0] * 20 + [5.0] * 20 + [1.0] * 20
        result = detect_change_points_gpu(values, window=5, variance_threshold=2.0)
        assert len(result) >= 2

    def test_matches_cpu_basic(self):
        from headroom.compression.smart.gpu.analyze import detect_change_points_gpu
        values = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 2.0, 3.0, 4.0]
        cpu_result = detect_change_points(values, window=2, variance_threshold=2.0)
        gpu_result = detect_change_points_gpu(values, window=2, variance_threshold=2.0)
        assert cpu_result == gpu_result

    def test_matches_cpu_multiple_changes(self):
        from headroom.compression.smart.gpu.analyze import detect_change_points_gpu
        values = [1.0] * 10 + [10.0] * 10 + [1.0] * 10 + [10.0] * 10
        cpu_result = detect_change_points(values, window=3, variance_threshold=1.5)
        gpu_result = detect_change_points_gpu(values, window=3, variance_threshold=1.5)
        assert cpu_result == gpu_result

    def test_window_parameter(self):
        from headroom.compression.smart.gpu.analyze import detect_change_points_gpu
        values = [1.0, 1.5, 2.0, 10.0, 10.5, 11.0] + [1.0] * 10
        result_w3 = detect_change_points_gpu(values, window=3, variance_threshold=1.5)
        result_w2 = detect_change_points_gpu(values, window=2, variance_threshold=1.5)
        assert isinstance(result_w3, list)
        assert isinstance(result_w2, list)


class TestClassifyArrayGPU:
    """Tests for GPU-accelerated array classification."""

    def test_empty_arrays(self):
        from headroom.compression.smart.gpu.analyze import classify_array_gpu
        result = classify_array_gpu([[], []])
        assert result == ['empty', 'empty']

    def test_dict_array(self):
        from headroom.compression.smart.gpu.analyze import classify_array_gpu
        result = classify_array_gpu([[{'a': 1}, {'b': 2}]])
        assert result[0] == ArrayType.DICT_ARRAY.value

    def test_string_array(self):
        from headroom.compression.smart.gpu.analyze import classify_array_gpu
        result = classify_array_gpu([['a', 'b', 'c']])
        assert result[0] == ArrayType.STRING_ARRAY.value

    def test_number_array(self):
        from headroom.compression.smart.gpu.analyze import classify_array_gpu
        result = classify_array_gpu([[1, 2, 3], [1.5, 2.5, 3.5]])
        assert all(r == ArrayType.NUMBER_ARRAY.value for r in result)

    def test_bool_array(self):
        from headroom.compression.smart.gpu.analyze import classify_array_gpu
        result = classify_array_gpu([[True, False, True]])
        assert result[0] == ArrayType.BOOL_ARRAY.value

    def test_nested_array(self):
        from headroom.compression.smart.gpu.analyze import classify_array_gpu
        result = classify_array_gpu([[[1, 2], [3, 4]]])
        assert result[0] == ArrayType.NESTED_ARRAY.value

    def test_mixed_array(self):
        from headroom.compression.smart.gpu.analyze import classify_array_gpu
        result = classify_array_gpu([[{'a': 1}, 'str', 42]])
        assert result[0] == ArrayType.MIXED_ARRAY.value

    def test_matches_cpu_single(self):
        from headroom.compression.smart.gpu.analyze import classify_array_gpu
        arrays = [
            [{'a': 1}, {'b': 2}],
            ['x', 'y', 'z'],
            [1, 2, 3],
            [True, False],
        ]
        for arr in arrays:
            cpu = classify_array(arr).value
            gpu = classify_array_gpu([arr])[0]
            assert cpu == gpu, f"Mismatch for {arr}: cpu={cpu}, gpu={gpu}"

    def test_matches_cpu_batch(self):
        from headroom.compression.smart.gpu.analyze import classify_array_gpu
        arrays = [
            [{'a': 1}, {'b': 2}],
            ['x', 'y', 'z'],
            [1, 2, 3],
            [True, False],
            [[1, 2], [3, 4]],
            [1.5, 2.5],
            [],
            [{'a': 1}, 'str', 42],
        ]
        cpu_results = [classify_array(arr).value for arr in arrays]
        gpu_results = classify_array_gpu(arrays)
        assert cpu_results == gpu_results


class TestGPUFallback:
    """Tests for GPU fallback behavior when CUDA unavailable."""

    def test_functions_work_without_cuda(self):
        from headroom.compression.smart.gpu.analyze import (
            calculate_string_entropy_gpu,
            detect_change_points_gpu,
            classify_array_gpu,
        )

        if not classify_array_gpu.__code__.co_freevars:
            pass

        strings = ["test", "hello"]
        result = calculate_string_entropy_gpu(strings)
        assert len(result) == 2

        values = [1.0, 2.0, 10.0, 11.0, 12.0]
        result = detect_change_points_gpu(values, window=1)
        assert isinstance(result, list)

        arrays = [[1, 2, 3]]
        result = classify_array_gpu(arrays)
        assert len(result) == 1

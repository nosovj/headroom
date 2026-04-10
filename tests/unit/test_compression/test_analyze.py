"""Unit tests for compression/smart/analyze module."""

import math

import pytest

from headroom.compression.smart.analyze import (
    FieldStats,
    analyze_field,
    calculate_string_entropy,
    detect_change_points,
    detect_sequential_pattern,
    is_uuid_format,
)


class TestCalculateStringEntropy:
    """Tests for calculate_string_entropy function."""

    def test_empty_string(self):
        """Empty string returns 0.0."""
        assert calculate_string_entropy("") == 0.0

    def test_single_char(self):
        """Single char returns 0.0."""
        assert calculate_string_entropy("a") == 0.0

    def test_repetitive_string(self):
        """Repetitive string returns low entropy."""
        result = calculate_string_entropy("aaaaaaaaaa")
        assert result < 0.3

    def test_random_string(self):
        """Random-looking string returns high entropy."""
        result = calculate_string_entropy("abc123xyz!@#")
        assert result > 0.7

    def test_entropy_bounds(self):
        """Entropy is always in [0.0, 1.0]."""
        test_strings = ["", "a", "aa", "abc", "aaaa", "ababab", "abcdef", "0123456789"]
        for s in test_strings:
            result = calculate_string_entropy(s)
            assert 0.0 <= result <= 1.0, f"Failed for {s}: {result}"

    def test_deterministic(self):
        """Same input always returns same output."""
        s = "hello world test string"
        result1 = calculate_string_entropy(s)
        result2 = calculate_string_entropy(s)
        assert result1 == result2

    def test_all_same_char(self):
        """All same character has zero entropy."""
        result = calculate_string_entropy("aaaaaaaaaa")
        assert result == 0.0


class TestIsUuidFormat:
    """Tests for is_uuid_format function."""

    def test_valid_uuid(self):
        """Valid UUID returns True."""
        assert is_uuid_format("123e4567-e89b-12d3-a456-426614174000") is True

    def test_invalid_uuid_wrong_length(self):
        """Wrong length returns False."""
        assert is_uuid_format("123e4567-e89b-12d3-a456") is False

    def test_invalid_uuid_bad_chars(self):
        """Non-hex chars return False."""
        assert is_uuid_format("123e4567-e89b-12d3-a456-42661417400g") is False

    def test_invalid_uuid_wrong_separators(self):
        """Wrong separator pattern returns False."""
        assert is_uuid_format("123e4567e89b12d3a456426614174000") is False

    def test_non_string(self):
        """Non-string returns False."""
        assert is_uuid_format(123) is False
        assert is_uuid_format(None) is False

    def test_empty_string(self):
        """Empty string returns False."""
        assert is_uuid_format("") is False


class TestDetectSequentialPattern:
    """Tests for detect_sequential_pattern function."""

    def test_sequential_ids(self):
        """Sequential IDs like [1,2,3,4,5] return True."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert detect_sequential_pattern(values) is True

    def test_sequential_with_gaps(self):
        """IDs with small gaps still detected as sequential."""
        values = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
        assert detect_sequential_pattern(values) is True

    def test_non_sequential_random(self):
        """Random values return False."""
        values = [5, 2, 8, 1, 9, 3, 7, 4, 6, 10]
        assert detect_sequential_pattern(values) is False

    def test_too_few_values(self):
        """Less than 5 values returns False."""
        values = [1, 2, 3]
        assert detect_sequential_pattern(values) is False

    def test_descending_not_sequential(self):
        """Descending order is not sequential (IDs are ascending)."""
        values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        assert detect_sequential_pattern(values) is False

    def test_string_numbers(self):
        """String numbers that are sequential return True."""
        values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        assert detect_sequential_pattern(values) is True

    def test_deterministic(self):
        """Same input always returns same output."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result1 = detect_sequential_pattern(values)
        result2 = detect_sequential_pattern(values)
        assert result1 == result2


class TestDetectChangePoints:
    """Tests for detect_change_points function."""

    def test_no_change(self):
        """Constant values return empty list."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = detect_change_points(values)
        assert result == []

    def test_single_change(self):
        """Values with one change point return that index."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        result = detect_change_points(values, variance_threshold=0.5)
        assert len(result) >= 1
        assert any(4 <= idx <= 6 for idx in result)

    def test_multiple_changes(self):
        """Values with multiple changes return at least one change index."""
        values = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0]
        result = detect_change_points(values, variance_threshold=0.5)
        assert len(result) >= 1

    def test_too_few_values(self):
        """Less than 2*window values returns empty."""
        values = [1.0, 2.0, 3.0]
        result = detect_change_points(values, window=5)
        assert result == []

    def test_deterministic(self):
        """Same input always returns same output."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        result1 = detect_change_points(values)
        result2 = detect_change_points(values)
        assert result1 == result2


class TestAnalyzeField:
    """Tests for analyze_field function."""

    def test_numeric_field(self):
        """Numeric field returns correct stats."""
        items = [{"id": 1, "score": 0.5}, {"id": 2, "score": 0.6}, {"id": 3, "score": 0.7}]
        stats = analyze_field("score", items)
        assert stats.name == "score"
        assert stats.field_type == "numeric"
        assert stats.count == 3
        assert stats.unique_count == 3
        assert stats.unique_ratio == 1.0
        assert stats.min_val == 0.5
        assert stats.max_val == 0.7
        assert stats.mean_val is not None

    def test_string_field(self):
        """String field returns correct stats."""
        items = [
            {"name": "alice"},
            {"name": "bob"},
            {"name": "charlie"},
        ]
        stats = analyze_field("name", items)
        assert stats.name == "name"
        assert stats.field_type == "string"
        assert stats.unique_count == 3
        assert stats.avg_length is not None
        assert stats.avg_length > 0

    def test_constant_field(self):
        """Constant field is detected."""
        items = [{"type": "error"}, {"type": "error"}, {"type": "error"}]
        stats = analyze_field("type", items)
        assert stats.is_constant is True
        assert stats.constant_value == "error"
        assert stats.unique_count == 1

    def test_missing_values(self):
        """Items with missing values handled correctly."""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        stats = analyze_field("id", items)
        assert stats.count == 3
        assert stats.unique_count == 3

    def test_null_field(self):
        """Field with only null values returns null type."""
        items = [{"a": None}, {"a": None}]
        stats = analyze_field("a", items)
        assert stats.field_type == "null"

    def test_deterministic(self):
        """Same input always returns same output."""
        items = [
            {"id": 1, "score": 0.5},
            {"id": 2, "score": 0.6},
            {"id": 3, "score": 0.7},
        ]
        result1 = analyze_field("score", items)
        result2 = analyze_field("score", items)
        assert result1.unique_count == result2.unique_count
        assert result1.min_val == result2.min_val
        assert result1.max_val == result2.max_val

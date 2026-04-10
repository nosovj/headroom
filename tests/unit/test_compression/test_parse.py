"""Unit tests for compression/smart/parse module."""

import pytest

from headroom.compression.smart.parse import (
    ArrayType,
    classify_array,
    is_json_string,
    parse_json_content,
    safe_json_loads,
)


class TestClassifyArray:
    """Tests for classify_array function."""

    def test_empty_list(self):
        """Empty list returns EMPTY type."""
        assert classify_array([]) == ArrayType.EMPTY

    def test_dict_array(self):
        """List of dicts returns DICT_ARRAY."""
        items = [{"a": 1}, {"b": 2}, {"c": 3}]
        assert classify_array(items) == ArrayType.DICT_ARRAY

    def test_string_array(self):
        """List of strings returns STRING_ARRAY."""
        items = ["a", "b", "c"]
        assert classify_array(items) == ArrayType.STRING_ARRAY

    def test_number_array(self):
        """List of numbers returns NUMBER_ARRAY."""
        items = [1, 2.5, 3, 4.0]
        assert classify_array(items) == ArrayType.NUMBER_ARRAY

    def test_bool_array(self):
        """List of booleans returns BOOL_ARRAY."""
        items = [True, False, True]
        assert classify_array(items) == ArrayType.BOOL_ARRAY

    def test_nested_array(self):
        """List of lists returns NESTED_ARRAY."""
        items = [[1, 2], [3, 4], [5, 6]]
        assert classify_array(items) == ArrayType.NESTED_ARRAY

    def test_mixed_array(self):
        """Mixed types returns MIXED_ARRAY."""
        items = [{"a": 1}, "string", 42]
        assert classify_array(items) == ArrayType.MIXED_ARRAY

    def test_mixed_with_bool_and_int(self):
        """Bool and int mixed returns MIXED since not all are bool."""
        items = [True, False, 1, 2]
        assert classify_array(items) == ArrayType.MIXED_ARRAY

    def test_single_dict(self):
        """Single dict returns DICT_ARRAY."""
        items = [{"a": 1}]
        assert classify_array(items) == ArrayType.DICT_ARRAY

    def test_single_string(self):
        """Single string returns STRING_ARRAY."""
        items = ["hello"]
        assert classify_array(items) == ArrayType.STRING_ARRAY

    def test_single_number(self):
        """Single number returns NUMBER_ARRAY."""
        items = [42]
        assert classify_array(items) == ArrayType.NUMBER_ARRAY

    def test_single_bool(self):
        """Single boolean returns BOOL_ARRAY."""
        items = [True]
        assert classify_array(items) == ArrayType.BOOL_ARRAY

    def test_single_nested(self):
        """Single nested list returns NESTED_ARRAY."""
        items = [[1, 2, 3]]
        assert classify_array(items) == ArrayType.NESTED_ARRAY

    def test_int_and_float_returns_number(self):
        """Mixed int and float returns NUMBER_ARRAY."""
        items = [1, 2.0, 3, 4.5]
        assert classify_array(items) == ArrayType.NUMBER_ARRAY

    def test_float_only(self):
        """Float-only list returns NUMBER_ARRAY."""
        items = [1.1, 2.2, 3.3]
        assert classify_array(items) == ArrayType.NUMBER_ARRAY


class TestSafeJsonLoads:
    """Tests for safe_json_loads function."""

    def test_valid_json_object(self):
        """Valid JSON object returns parsed result and True."""
        result, success = safe_json_loads('{"a": 1, "b": 2}')
        assert success is True
        assert result == {"a": 1, "b": 2}

    def test_valid_json_array(self):
        """Valid JSON array returns parsed result and True."""
        result, success = safe_json_loads('[1, 2, 3]')
        assert success is True
        assert result == [1, 2, 3]

    def test_valid_json_string(self):
        """Valid JSON string returns parsed result and True."""
        result, success = safe_json_loads('"hello world"')
        assert success is True
        assert result == "hello world"

    def test_valid_json_number(self):
        """Valid JSON number returns parsed result and True."""
        result, success = safe_json_loads("42")
        assert success is True
        assert result == 42

    def test_invalid_json(self):
        """Invalid JSON returns None and False."""
        result, success = safe_json_loads("not valid json")
        assert success is False
        assert result is None

    def test_malformed_json(self):
        """Malformed JSON returns None and False."""
        result, success = safe_json_loads('{"unclosed": ')
        assert success is False
        assert result is None

    def test_empty_string(self):
        """Empty string returns None and False."""
        result, success = safe_json_loads("")
        assert success is False
        assert result is None

    def test_whitespace_only(self):
        """Whitespace-only returns None and False."""
        result, success = safe_json_loads("   \n\t  ")
        assert success is False
        assert result is None


class TestParseJsonContent:
    """Tests for parse_json_content function."""

    def test_valid_json_array(self):
        """Valid JSON array returns parsed result and array type."""
        result, success, arr_type = parse_json_content('[1, 2, 3]')
        assert success is True
        assert result == [1, 2, 3]
        assert arr_type == ArrayType.NUMBER_ARRAY

    def test_valid_json_object(self):
        """Valid JSON object returns parsed result with None array type."""
        result, success, arr_type = parse_json_content('{"key": "value"}')
        assert success is True
        assert result == {"key": "value"}
        assert arr_type is None

    def test_invalid_json(self):
        """Invalid JSON returns None for all values."""
        result, success, arr_type = parse_json_content("not valid")
        assert success is False
        assert result is None
        assert arr_type is None

    def test_complex_array(self):
        """Complex array with dicts returns DICT_ARRAY."""
        content = '[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]'
        result, success, arr_type = parse_json_content(content)
        assert success is True
        assert arr_type == ArrayType.DICT_ARRAY


class TestIsJsonString:
    """Tests for is_json_string function."""

    def test_valid_object(self):
        """Valid object returns True."""
        assert is_json_string('{"key": "value"}') is True

    def test_valid_array(self):
        """Valid array returns True."""
        assert is_json_string('[1, 2, 3]') is True

    def test_invalid_string(self):
        """Invalid string returns False."""
        assert is_json_string("not json") is False

    def test_empty_string(self):
        """Empty string returns False."""
        assert is_json_string("") is False

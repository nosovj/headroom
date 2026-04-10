"""Parse layer for compression pipeline.

Handles JSON parsing and content type detection with clean interfaces
and no side effects.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


class ArrayType(Enum):
    """Classification of JSON array types."""

    DICT_ARRAY = "dict_array"  # [{"a":1}, {"b":2}, ...]
    STRING_ARRAY = "string_array"  # ["a", "b", "c", ...]
    NUMBER_ARRAY = "number_array"  # [1, 2.5, 3, ...]
    BOOL_ARRAY = "bool_array"  # [true, false, ...]
    NESTED_ARRAY = "nested_array"  # [[...], [...], ...]
    MIXED_ARRAY = "mixed_array"  # [{"a":1}, "str", 42, ...]
    EMPTY = "empty"


def classify_array(items: list) -> ArrayType:
    """Classify a JSON array by its element types.

    Uses set-of-types check on ALL elements (not sampling) to guarantee
    correct classification. Fast because type() is O(1).

    Args:
        items: List of items to classify.

    Returns:
        ArrayType classification of the array.
    """
    if not items:
        return ArrayType.EMPTY
    types = set()
    has_bool = False
    for item in items:
        if isinstance(item, bool):
            has_bool = True
        types.add(type(item))
    if has_bool and types <= {bool, int}:
        if all(isinstance(i, bool) for i in items):
            return ArrayType.BOOL_ARRAY
    if types == {dict}:
        return ArrayType.DICT_ARRAY
    if types == {str}:
        return ArrayType.STRING_ARRAY
    if types <= {int, float} and not has_bool:
        return ArrayType.NUMBER_ARRAY
    if types == {list}:
        return ArrayType.NESTED_ARRAY
    return ArrayType.MIXED_ARRAY


def safe_json_loads(text: str) -> tuple[Any | None, bool]:
    """Safely parse JSON, returning (result, success).

    Args:
        text: JSON string to parse.

    Returns:
        Tuple of (parsed_result or None, success_bool).
    """
    try:
        return json.loads(text), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def parse_json_content(content: str) -> tuple[Any | None, bool, ArrayType | None]:
    """Parse JSON content and classify the array type.

    Args:
        content: JSON string to parse.

    Returns:
        Tuple of (parsed_object or None, parse_success_bool, array_type or None).
        If parse fails, array_type will be None.
    """
    parsed, success = safe_json_loads(content)
    if not success:
        return None, False, None

    if isinstance(parsed, list):
        arr_type = classify_array(parsed)
        return parsed, True, arr_type

    return parsed, True, None


def is_json_string(content: str) -> bool:
    """Check if content is valid JSON without parsing fully.

    Args:
        content: String to check.

    Returns:
        True if content is valid JSON.
    """
    _, success = safe_json_loads(content)
    return success

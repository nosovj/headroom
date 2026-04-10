"""Serialize layer for compression pipeline.

JSON serialization with orjson integration and caching.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any


_orjson_available = False
try:
    import orjson

    _orjson_available = True
except ImportError:
    pass


def json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize object to JSON string.

    Uses orjson if available for better performance.

    Args:
        obj: Object to serialize.
        **kwargs: Additional arguments:
            - sort_keys: If True, sort dictionary keys (default: False)
            - indent: If given, pretty-print with that indent level
            - default: Function to convert non-serializable objects (fallback only)

    Returns:
        JSON string representation.
    """
    if _orjson_available:
        option = orjson.OPT_NON_STR_KEYS
        if kwargs.get("sort_keys"):
            option |= orjson.OPT_SORT_KEYS
        indent = kwargs.get("indent")
        if indent is not None:
            option |= orjson.OPT_INDENT_2
        try:
            return orjson.dumps(obj, option=option).decode()
        except TypeError:
            # orjson can't serialize - fall back to json.dumps with default=str
            return json.dumps(obj, default=str, **kwargs)
    else:
        # Fallback to json.dumps with all kwargs passed through
        kwargs.setdefault("default", str)
        return json.dumps(obj, **kwargs)


def json_loads(text: str) -> Any:
    """Parse JSON string.

    Uses orjson if available for better performance.

    Args:
        text: JSON string to parse.

    Returns:
        Parsed Python object.
    """
    if _orjson_available:
        return orjson.loads(text)
    else:
        return json.loads(text)


@lru_cache(maxsize=1000)
def cached_dumps(obj: tuple, indent: bool = False) -> str:
    """Cached JSON dumps for frequently serialized objects.

    Uses tuple as cache key since dict is not hashable.
    Cache size limited to 1000 entries.

    Args:
        obj: Object to serialize (passed as tuple for hashing).
        indent: Whether to indent output.

    Returns:
        JSON string representation.
    """
    parsed = json.loads(obj) if isinstance(obj, str) else obj
    if indent:
        return json.dumps(parsed, indent=2)
    return json.dumps(parsed)


def orjson_available() -> bool:
    """Check if orjson is available.

    Returns:
        True if orjson is installed and usable.
    """
    return _orjson_available


def serialize_compressed(
    items: list,
    indent: bool = False,
    use_cache: bool = True,
) -> str:
    """Serialize compressed items to JSON.

    Args:
        items: List of items to serialize.
        indent: Whether to format with indentation.
        use_cache: Whether to use serialization cache.

    Returns:
        JSON string of compressed items.
    """
    if use_cache:
        result = cached_dumps(tuple(items), indent=indent)
    else:
        result = json_dumps(items, indent=2 if indent else None)
    return result

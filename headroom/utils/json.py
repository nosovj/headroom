"""Fast JSON serialization using orjson with fallback to standard library.

This module provides a single function that uses orjson when available
for optimal performance, with graceful fallback to json.dumps when orjson
is not installed.

Usage:
    from headroom.utils.json import json_dumps
    result = json_dumps(obj)
"""

from __future__ import annotations

import json
from typing import Any

_orjson_available = False
try:
    import orjson

    _orjson_available = True
except ImportError:
    orjson = None


def json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize object to JSON string using fastest available serializer.

    Uses orjson if available for ~3x performance improvement over json.dumps.
    Falls back to standard library json.dumps if orjson is not installed.

    Args:
        obj: Object to serialize.
        **kwargs: Additional arguments passed to json.dumps when orjson unavailable.
                  For orjson, only 'indent' is supported (creates OPT_INDENT_2).

    Returns:
        JSON string representation of obj.
    """
    if _orjson_available:
        option = orjson.OPT_NON_STR_KEYS
        indent = kwargs.get("indent")
        if indent is not None:
            option |= orjson.OPT_INDENT_2
        return orjson.dumps(obj, option=option).decode()
    return json.dumps(obj, **kwargs)


def json_loads(text: str) -> Any:
    """Parse JSON string using fastest available parser.

    Args:
        text: JSON string to parse.

    Returns:
        Parsed Python object.
    """
    if _orjson_available:
        return orjson.loads(text)
    return json.loads(text)


def orjson_available() -> bool:
    """Check if orjson is available.

    Returns:
        True if orjson is installed and usable.
    """
    return _orjson_available
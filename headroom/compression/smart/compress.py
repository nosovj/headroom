"""Compress layer for compression pipeline.

Wrapper around SmartCrusher with clean interface for the compression pipeline.
This module provides the crushing algorithms that reduce array size while
preserving important items.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

_orjson_available = False
try:
    import orjson

    _orjson_available = True
except ImportError:
    orjson = None

from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig


def _orjson_dumps(obj, **kwargs):
    """Fast JSON dumps using orjson, fallback to json.dumps."""
    if _orjson_available:
        option = orjson.OPT_NON_STR_KEYS
        indent = kwargs.get("indent")
        if indent is not None:
            option |= orjson.OPT_INDENT_2
        return orjson.dumps(obj, option=option).decode()
    return json.dumps(obj, **kwargs)

logger = logging.getLogger(__name__)


@dataclass
class CompressConfig:
    """Configuration for compression."""

    enabled: bool = True
    min_items_to_analyze: int = 5
    min_tokens_to_crush: int = 200
    variance_threshold: float = 2.0
    uniqueness_threshold: float = 0.1
    similarity_threshold: float = 0.8
    max_items_after_crush: int = 15
    preserve_change_points: bool = True


@dataclass
class CompressResult:
    """Result from compression."""

    items: list
    strategy: str
    was_modified: bool
    items_before: int
    items_after: int


def create_smart_crusher(config: CompressConfig | None = None) -> SmartCrusher:
    """Create a SmartCrusher instance with the given config.

    Args:
        config: Compression configuration.

    Returns:
        Configured SmartCrusher instance.
    """
    if config is None:
        config = CompressConfig()

    sc_config = SmartCrusherConfig(
        enabled=config.enabled,
        min_items_to_analyze=config.min_items_to_analyze,
        min_tokens_to_crush=config.min_tokens_to_crush,
        variance_threshold=config.variance_threshold,
        uniqueness_threshold=config.uniqueness_threshold,
        similarity_threshold=config.similarity_threshold,
        max_items_after_crush=config.max_items_after_crush,
        preserve_change_points=config.preserve_change_points,
    )

    return SmartCrusher(config=sc_config)


def crush_array(
    items: list[dict],
    config: CompressConfig | None = None,
    query_context: str = "",
    tool_name: str | None = None,
    bias: float = 1.0,
) -> CompressResult:
    """Crush an array of dicts using SmartCrusher.

    Args:
        items: List of dict items to compress.
        config: Compression configuration.
        query_context: Context for relevance scoring.
        tool_name: Name of tool that produced output.
        bias: Compression bias (>1 = keep more, <1 = keep fewer).

    Returns:
        CompressResult with crushed items and metadata.
    """
    crusher = create_smart_crusher(config)

    original_count = len(items)

    crushed, was_modified, analysis_info = crusher._smart_crush_content(
        json.dumps(items, default=str),
        query_context=query_context,
        tool_name=tool_name,
        bias=bias,
    )

    if was_modified:
        result_items = json.loads(crushed)
    else:
        result_items = items

    return CompressResult(
        items=result_items,
        strategy=analysis_info or "passthrough",
        was_modified=was_modified,
        items_before=original_count,
        items_after=len(result_items),
    )


def crush_string_array(
    items: list[str],
    config: CompressConfig | None = None,
    bias: float = 1.0,
) -> CompressResult:
    """Crush an array of strings.

    Args:
        items: List of strings to compress.
        config: Compression configuration.
        bias: Compression bias.

    Returns:
        CompressResult with crushed items.
    """
    crusher = create_smart_crusher(config)

    original_count = len(items)

    result_items, strategy = crusher._crush_string_array(items, bias=bias)

    return CompressResult(
        items=result_items,
        strategy=strategy,
        was_modified=len(result_items) != original_count,
        items_before=original_count,
        items_after=len(result_items),
    )


def crush_number_array(
    items: list[int | float],
    config: CompressConfig | None = None,
    bias: float = 1.0,
) -> CompressResult:
    """Crush an array of numbers.

    Args:
        items: List of numbers to compress.
        config: Compression configuration.
        bias: Compression bias.

    Returns:
        CompressResult with crushed items.
    """
    crusher = create_smart_crusher(config)

    original_count = len(items)

    result_items, strategy = crusher._crush_number_array(items, bias=bias)

    return CompressResult(
        items=result_items,
        strategy=strategy,
        was_modified=len(result_items) != original_count,
        items_before=original_count,
        items_after=len(result_items),
    )


def crush_object(
    obj: dict,
    config: CompressConfig | None = None,
    bias: float = 1.0,
) -> tuple[dict, str, bool]:
    """Crush an object with many keys.

    Args:
        obj: Object to compress.
        config: Compression configuration.
        bias: Compression bias.

    Returns:
        Tuple of (crushed_obj, strategy, was_modified).
    """
    crusher = create_smart_crusher(config)

    result, strategy = crusher._crush_object(obj, bias=bias)

    return result, strategy, True

"""Analyze layer for compression pipeline.

Statistical analysis, classification, and pattern detection with no side effects.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldStats:
    """Statistics for a single field across array items."""

    name: str
    field_type: str  # "numeric", "string", "boolean", "object", "array", "null"
    count: int
    unique_count: int
    unique_ratio: float
    is_constant: bool
    constant_value: Any = None

    # Numeric-specific stats
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    variance: float | None = None
    change_points: list[int] = field(default_factory=list)

    # String-specific stats
    avg_length: float | None = None
    top_values: list[tuple[str, int]] = field(default_factory=list)


def calculate_string_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string, normalized to [0, 1].

    High entropy (>0.7) suggests random/ID-like content.
    Low entropy (<0.3) suggests repetitive/predictable content.

    Args:
        s: String to analyze.

    Returns:
        Entropy value between 0.0 and 1.0.
    """
    if not s or len(s) < 2:
        return 0.0

    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1

    entropy = 0.0
    length = len(s)
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(min(len(freq), length))
    if max_entropy > 0:
        return entropy / max_entropy
    return 0.0


def is_uuid_format(value: Any) -> bool:
    """Check if a string looks like a UUID (structural pattern).

    Args:
        value: String to check.

    Returns:
        True if value appears to be a UUID format.
    """
    if not isinstance(value, str) or len(value) != 36:
        return False
    parts = value.split("-")
    if len(parts) != 5:
        return False
    expected_lens = [8, 4, 4, 4, 12]
    for part, expected_len in zip(parts, expected_lens):
        if len(part) != expected_len:
            return False
        if not all(c in "0123456789abcdefABCDEF" for c in part):
            return False
    return True


def detect_sequential_pattern(values: list[Any], check_order: bool = True) -> bool:
    """Detect if numeric values form a sequential pattern (like IDs: 1,2,3,...).

    Args:
        values: List of values to check.
        check_order: If True, also check if values are in ascending order.

    Returns:
        True if values appear to be auto-incrementing or sequential.
    """
    if len(values) < 5:
        return False

    nums = []
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nums.append(v)
        elif isinstance(v, str):
            try:
                nums.append(int(v))
            except ValueError:
                pass

    if len(nums) < 5:
        return False

    if len(nums) < 2:
        return False

    sorted_nums = sorted(nums)
    diffs = [sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]

    if not diffs:
        return False

    avg_diff = sum(diffs) / len(diffs)
    if 0.5 <= avg_diff <= 2.0:
        consistent_count = sum(1 for d in diffs if 0.5 <= d <= 2.0)
        is_sequential = consistent_count / len(diffs) > 0.8

        if check_order and is_sequential:
            ascending_count = sum(1 for i in range(len(nums) - 1) if nums[i] <= nums[i + 1])
            is_ascending = ascending_count / (len(nums) - 1) > 0.7
            return is_ascending

        return is_sequential

    return False


def detect_change_points(
    values: list[float],
    window: int = 5,
    variance_threshold: float = 2.0,
) -> list[int]:
    """Detect indices where values change significantly.

    Args:
        values: Numeric values to analyze.
        window: Sliding window size for comparison.
        variance_threshold: Number of standard deviations to flag as change.

    Returns:
        List of indices where significant changes occur.
    """
    if len(values) < window * 2:
        return []

    change_points = []

    overall_std = statistics.stdev(values) if len(values) > 1 else 0
    if overall_std == 0:
        return []

    threshold = variance_threshold * overall_std

    for i in range(window, len(values) - window + 1):
        before_mean = statistics.mean(values[i - window : i])
        after_mean = statistics.mean(values[i : i + window])

        if abs(after_mean - before_mean) > threshold:
            change_points.append(i)

    if change_points:
        deduped = [change_points[0]]
        for cp in change_points[1:]:
            if cp - deduped[-1] > window:
                deduped.append(cp)
        return deduped

    return []


def analyze_field(key: str, items: list[dict]) -> FieldStats:
    """Analyze a single field across all items.

    Args:
        key: Field name to analyze.
        items: List of dict items containing the field.

    Returns:
        FieldStats with computed statistics.
    """
    values = [item.get(key) for item in items if isinstance(item, dict)]
    non_null_values = [v for v in values if v is not None]

    if not non_null_values:
        return FieldStats(
            name=key,
            field_type="null",
            count=len(values),
            unique_count=0,
            unique_ratio=0.0,
            is_constant=True,
            constant_value=None,
        )

    first_val = non_null_values[0]
    if isinstance(first_val, bool):
        field_type = "boolean"
    elif isinstance(first_val, (int, float)):
        field_type = "numeric"
    elif isinstance(first_val, str):
        field_type = "string"
    elif isinstance(first_val, dict):
        field_type = "object"
    elif isinstance(first_val, list):
        field_type = "array"
    else:
        field_type = "unknown"

    str_values = [str(v) for v in values]
    unique_values = set(str_values)
    unique_count = len(unique_values)
    unique_ratio = unique_count / len(values) if values else 0

    is_constant = unique_count == 1
    constant_value = non_null_values[0] if is_constant else None

    stats = FieldStats(
        name=key,
        field_type=field_type,
        count=len(values),
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        is_constant=is_constant,
        constant_value=constant_value,
    )

    if field_type == "numeric":
        nums = [v for v in non_null_values if isinstance(v, (int, float)) and math.isfinite(v)]
        if nums:
            try:
                stats.min_val = min(nums)
                stats.max_val = max(nums)
                stats.mean_val = statistics.mean(nums)
                stats.variance = statistics.variance(nums) if len(nums) > 1 else 0
                stats.change_points = detect_change_points(nums)
            except (OverflowError, ValueError):
                stats.min_val = None
                stats.max_val = None
                stats.mean_val = None
                stats.variance = 0
                stats.change_points = []

    elif field_type == "string":
        strs = [v for v in non_null_values if isinstance(v, str)]
        if strs:
            stats.avg_length = statistics.mean(len(s) for s in strs)
            stats.top_values = Counter(strs).most_common(5)

    return stats

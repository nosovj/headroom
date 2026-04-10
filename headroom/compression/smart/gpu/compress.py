"""GPU-accelerated compression functions for the pipeline.

Uses PyTorch for tensor-based Kneedle algorithm and statistical computations.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


def find_knee_point_gpu(
    values: list[float],
    window: int = 5,
) -> int | None:
    """Find knee point in numeric values using GPU acceleration.

    The knee point is where the rate of change transitions from steep to shallow.

    Args:
        values: Numeric values to analyze.
        window: Smoothing window size.

    Returns:
        Index of knee point, or None if not found.
    """
    if len(values) < window * 2 + 1:
        return None

    if torch is None:
        raise ImportError("PyTorch required for GPU operations")

    if not torch.cuda.is_available():
        return _find_knee_point_cpu(values, window)

    tensor = torch.tensor(values, dtype=torch.float32, device='cuda')

    cumsum = torch.cumsum(tensor, dim=0)
    n = len(values)

    x = torch.arange(1, n + 1, dtype=torch.float32, device='cuda')
    y = cumsum

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()

    if denominator == 0:
        return None

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    fitted = slope * x + intercept
    residuals = torch.abs(y - fitted)
    max_residual_idx = int(residuals.argmax().item())

    if max_residual_idx < window or max_residual_idx > n - window:
        return None

    return max_residual_idx


def _find_knee_point_cpu(values: list[float], window: int = 5) -> int | None:
    """CPU fallback for knee point detection."""
    if len(values) < window * 2 + 1:
        return None

    cumsum = []
    total = 0
    for v in values:
        total += v
        cumsum.append(total)

    n = len(values)
    x = list(range(1, n + 1))

    x_mean = sum(x) / n
    y_mean = sum(cumsum) / n

    numerator = sum((x[i] - x_mean) * (cumsum[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return None

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    max_residual = -1
    max_idx = None

    for i in range(n):
        fitted = slope * x[i] + intercept
        residual = abs(cumsum[i] - fitted)
        if residual > max_residual:
            max_residual = residual
            max_idx = i

    if max_idx is not None and (max_idx < window or max_idx > n - window):
        return None

    return max_idx


def crush_number_array_gpu(
    items: list[int | float],
    config: dict | None = None,
    bias: float = 1.0,
) -> tuple[list[int | float], str]:
    """Crush numeric array using GPU acceleration.

    Args:
        items: List of numbers to compress.
        config: Compression configuration dict.
        bias: Compression bias (>1 = keep more).

    Returns:
        Tuple of (crushed_items, strategy).
    """
    if config is None:
        config = {}

    min_items = config.get("min_items_to_analyze", 5)
    max_after = config.get("max_items_after_crush", 15)

    if len(items) < min_items:
        return items, "passthrough"

    if torch is None or not torch.cuda.is_available():
        return _crush_number_array_cpu(items, config, bias)

    tensor = torch.tensor(items, dtype=torch.float32, device='cuda')

    variance = tensor.var().item()
    if variance < 0.01:
        return items, "low_variance_passthrough"

    knee_idx = find_knee_point_gpu(items, window=5)

    target_size = max(min_items, int(len(items) * (1.0 / bias)))
    target_size = min(target_size, max_after)

    if target_size >= len(items):
        return items, "bias_prevents_crush"

    if knee_idx is not None and knee_idx > min_items and knee_idx < len(items) - min_items:
        step = len(items) / target_size
        indices = [int(i * step) for i in range(target_size)]
        result = [items[i] for i in indices]
        return result, "kneedle_gpu"
    else:
        step = len(items) / target_size
        indices = [int(i * step) for i in range(target_size)]
        result = [items[i] for i in indices]
        return result, "uniform_sampling_gpu"


def _crush_number_array_cpu(
    items: list[int | float],
    config: dict,
    bias: float = 1.0,
) -> tuple[list[int | float], str]:
    """CPU fallback for numeric array crushing."""
    if len(items) < config.get("min_items_to_analyze", 5):
        return items, "passthrough"

    variance = __import__('statistics').variance(items) if len(items) > 1 else 0
    if variance < 0.01:
        return items, "low_variance_passthrough"

    knee = _find_knee_point_cpu(items, window=5)

    max_after = config.get("max_items_after_crush", 15)
    target_size = max(5, int(len(items) * (1.0 / bias)))
    target_size = min(target_size, max_after)

    if target_size >= len(items):
        return items, "bias_prevents_crush"

    if knee is not None and knee > 5 and knee < len(items) - 5:
        step = len(items) / target_size
        indices = [int(i * step) for i in range(target_size)]
        return [items[i] for i in indices], "kneedle_cpu"
    else:
        step = len(items) / target_size
        indices = [int(i * step) for i in range(target_size)]
        return [items[i] for i in indices], "uniform_sampling_cpu"


def crush_string_array_gpu(
    items: list[str],
    config: dict | None = None,
    bias: float = 1.0,
) -> tuple[list[str], str]:
    """Crush string array using GPU acceleration.

    Args:
        items: List of strings to compress.
        config: Compression configuration dict.
        bias: Compression bias.

    Returns:
        Tuple of (crushed_items, strategy).
    """
    if config is None:
        config = {}

    min_items = config.get("min_items_to_analyze", 5)

    if len(items) < min_items:
        return items, "passthrough"

    from ..analyze import calculate_string_entropy

    entropies = [calculate_string_entropy(s) for s in items]

    if torch is not None and torch.cuda.is_available():
        entropies_tensor = torch.tensor(entropies, device='cuda')
        threshold = entropies_tensor.mean().item() + entropies_tensor.std().item()
    else:
        import statistics
        threshold = statistics.mean(entropies) + statistics.stdev(entropies)

    unique_threshold = config.get("uniqueness_threshold", 0.1)
    unique_ratio = len(set(items)) / len(items) if items else 0

    if unique_ratio < unique_threshold:
        seen = []
        for item in items:
            if item not in seen:
                seen.append(item)
        return seen[:max(1, int(len(items) * unique_ratio * 2))], "dedup_gpu"

    high_entropy = [i for i, e in enumerate(entropies) if e > threshold]
    low_entropy = [i for i, e in enumerate(entropies) if e <= threshold]

    max_after = config.get("max_items_after_crush", 15)
    target_size = max(min_items, int(len(items) * (1.0 / bias)))
    target_size = min(target_size, max_after)

    if target_size >= len(items):
        return items, "bias_prevents_crush"

    result = []
    if low_entropy:
        step = len(low_entropy) / max(1, target_size // 2)
        for i in range(target_size // 2):
            idx = min(int(i * step), len(low_entropy) - 1)
            result.append(items[low_entropy[idx]])

    remaining = target_size - len(result)
    if remaining > 0 and high_entropy:
        step = len(high_entropy) / remaining
        for i in range(remaining):
            idx = min(int(i * step), len(high_entropy) - 1)
            result.append(items[high_entropy[idx]])

    return result, "entropy_based_gpu"


def crush_array_gpu(
    items: list[dict],
    config: dict | None = None,
    query_context: str = "",
    tool_name: str | None = None,
    bias: float = 1.0,
) -> tuple[list[dict], str, bool]:
    """Crush array of dicts using GPU acceleration.

    Args:
        items: List of dict items to compress.
        config: Compression configuration dict.
        query_context: Context for relevance scoring.
        tool_name: Name of tool that produced output.
        bias: Compression bias.

    Returns:
        Tuple of (crushed_items, strategy, was_modified).
    """
    if config is None:
        config = {}

    min_items = config.get("min_items_to_analyze", 5)

    if len(items) < min_items:
        return items, "too_few_items", False

    all_keys = set()
    for item in items:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    if not all_keys:
        return items, "no_dict_items", False

    key_scores = {}
    for key in all_keys:
        values = [item.get(key) for item in items if isinstance(item, dict) and key in item]
        unique_ratio = len(set(str(v) for v in values)) / len(values) if values else 0

        numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
        if torch is not None and torch.cuda.is_available():
            if len(numeric_values) > 1:
                variance = torch.tensor(numeric_values, device='cuda').var().item()
            else:
                variance = 0.0
        else:
            import statistics
            variance = statistics.variance(numeric_values) if len(numeric_values) > 1 else 0.0

        key_scores[key] = unique_ratio + (1.0 / (variance + 1.0))

    sorted_keys = sorted(key_scores.keys(), key=lambda k: key_scores[k], reverse=True)

    max_keys = config.get("max_keys_to_preserve", 10)
    keys_to_keep = sorted_keys[:max_keys]

    if len(all_keys) <= len(keys_to_keep):
        return items, "all_keys_kept", False

    result = []
    for item in items:
        if isinstance(item, dict):
            filtered = {k: item.get(k) for k in keys_to_keep if k in item}
            result.append(filtered)
        else:
            result.append(item)

    return result, "key_filtering_gpu", True


def is_available() -> bool:
    """Check if GPU is available for compression.

    Returns:
        True if CUDA GPU is available.
    """
    if torch is None:
        return False
    return torch.cuda.is_available()

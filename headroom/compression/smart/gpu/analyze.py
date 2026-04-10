"""GPU-accelerated analyze functions for compression pipeline.

Uses PyTorch for vectorized operations on statistical analysis.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


def calculate_string_entropy_gpu(strings: list[str]) -> list[float]:
    """Calculate Shannon entropy for multiple strings using GPU.

    Fully vectorized implementation using scatter_add for GPU speedup.

    Args:
        strings: List of strings to analyze.

    Returns:
        List of entropy values between 0.0 and 1.0.
    """
    if not strings:
        return []

    if torch is None:
        raise ImportError("PyTorch required for GPU operations")

    if not torch.cuda.is_available():
        from ..analyze import calculate_string_entropy
        return [calculate_string_entropy(s) for s in strings]

    n = len(strings)
    if n == 1:
        from ..analyze import calculate_string_entropy
        return [calculate_string_entropy(strings[0])]

    lengths_list = [len(s) for s in strings]
    lengths = torch.tensor(lengths_list, dtype=torch.float32, device='cuda')
    if (lengths < 2).all():
        return [0.0] * n

    max_len = max(lengths_list)
    vocab = sorted(set(c for s in strings for c in s))
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)

    indices_matrix = torch.zeros((n, max_len), dtype=torch.long, device='cuda')
    for i, s in enumerate(strings):
        indices_matrix[i, :len(s)] = torch.tensor([char_to_idx[c] for c in s], dtype=torch.long, device='cuda')

    lengths_long = torch.tensor(lengths_list, dtype=torch.long, device='cuda')
    mask = torch.arange(max_len, device='cuda').unsqueeze(0) < lengths_long.unsqueeze(1)
    masked = indices_matrix * mask.long()

    char_counts = torch.zeros((n, vocab_size), dtype=torch.long, device='cuda')
    char_counts.scatter_add_(1, masked, torch.ones((n, max_len), dtype=torch.long, device='cuda'))

    counts_float = char_counts.float()
    probs = counts_float / lengths.unsqueeze(1)
    probs = probs.clamp(min=1e-10)

    entropy = -(probs * torch.log2(probs)).sum(dim=1)

    unique_chars = (char_counts > 0).sum(dim=1).float()
    max_entropy = torch.log2(torch.minimum(unique_chars, lengths).clamp(min=1))

    normalized = entropy / max_entropy.clamp(min=1e-10)

    return normalized.cpu().tolist()


def detect_change_points_gpu(
    values: list[float],
    window: int = 5,
    variance_threshold: float = 2.0,
) -> list[int]:
    """Detect change points in numeric values using GPU acceleration.

    Args:
        values: Numeric values to analyze.
        window: Sliding window size for comparison.
        variance_threshold: Number of standard deviations to flag as change.

    Returns:
        List of indices where significant changes occur.
    """
    if len(values) < window * 2:
        return []

    if torch is None:
        raise ImportError("PyTorch required for GPU operations")

    if not torch.cuda.is_available():
        from ..analyze import detect_change_points
        return detect_change_points(values, window, variance_threshold)

    tensor = torch.tensor(values, dtype=torch.float32, device='cuda')

    overall_std = tensor.std().item()
    if overall_std == 0:
        return []

    threshold = variance_threshold * overall_std

    n = len(values)
    padded = torch.cat([torch.zeros(window, device='cuda'), tensor, torch.zeros(window, device='cuda')])

    before_means = torch.zeros(n, device='cuda')
    after_means = torch.zeros(n, device='cuda')

    for i in range(n):
        before_means[i] = padded[i:i + window].mean()
        after_means[i] = padded[i + window:i + window * 2].mean()

    diffs = torch.abs(after_means - before_means)
    change_mask = diffs > threshold

    change_indices = change_mask.nonzero().squeeze(-1).cpu().tolist()

    if not change_indices:
        return []

    deduped = [change_indices[0]]
    for idx in change_indices[1:]:
        if idx - deduped[-1] > window:
            deduped.append(idx)

    return deduped


def classify_array_gpu(arrays: list[list]) -> list[str]:
    """Classify multiple arrays by type using GPU batch processing.

    Args:
        arrays: List of arrays to classify.

    Returns:
        List of ArrayType string values.
    """
    if not arrays:
        return []

    if torch is None:
        raise ImportError("PyTorch required for GPU operations")

    if not torch.cuda.is_available():
        from ..parse import classify_array, ArrayType
        return [classify_array(arr).value for arr in arrays]

    n = len(arrays)
    result = [None] * n

    type_counts = {
        'dict': torch.zeros(n, device='cuda'),
        'str': torch.zeros(n, device='cuda'),
        'int': torch.zeros(n, device='cuda'),
        'float': torch.zeros(n, device='cuda'),
        'bool': torch.zeros(n, device='cuda'),
        'list': torch.zeros(n, device='cuda'),
        'none': torch.zeros(n, device='cuda'),
    }

    lengths = torch.zeros(n, device='cuda', dtype=torch.long)

    for i, arr in enumerate(arrays):
        lengths[i] = len(arr)
        for item in arr:
            if item is None:
                type_counts['none'][i] += 1
            elif isinstance(item, bool):
                type_counts['bool'][i] += 1
            elif isinstance(item, dict):
                type_counts['dict'][i] += 1
            elif isinstance(item, str):
                type_counts['str'][i] += 1
            elif isinstance(item, int):
                type_counts['int'][i] += 1
            elif isinstance(item, float):
                type_counts['float'][i] += 1
            elif isinstance(item, list):
                type_counts['list'][i] += 1

    total = lengths.float().clamp(min=1)

    for i in range(n):
        if lengths[i] == 0:
            result[i] = 'empty'
            continue

        dc = type_counts['dict'][i].item()
        sc = type_counts['str'][i].item()
        ic = type_counts['int'][i].item()
        fc = type_counts['float'][i].item()
        bc = type_counts['bool'][i].item()
        lc = type_counts['list'][i].item()

        if dc == total[i]:
            result[i] = 'dict_array'
        elif sc == total[i]:
            result[i] = 'string_array'
        elif lc == total[i]:
            result[i] = 'nested_array'
        elif bc > 0 and (bc + ic) == total[i]:
            if bc == total[i]:
                result[i] = 'bool_array'
            else:
                result[i] = 'mixed_array'
        elif (ic + fc) == total[i] and bc == 0:
            result[i] = 'number_array'
        else:
            result[i] = 'mixed_array'

    return result


def is_available() -> bool:
    """Check if GPU is available for compression.

    Returns:
        True if CUDA GPU is available.
    """
    if torch is None:
        return False
    return torch.cuda.is_available()

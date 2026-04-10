"""GPU-accelerated operations for adaptive sizing.

These functions accelerate the compute_optimal_k pipeline using GPU
parallelization for tokenization, bigram extraction, and clustering.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


def compute_unique_bigram_curve_gpu(items: Sequence[str]) -> list[int]:
    """Build cumulative unique bigram coverage curve using GPU.

    For each item (in order), extracts word-level bigrams, adds them to a
    running set, and records the total unique count.

    Args:
        items: Sequence of string items in importance order.

    Returns:
        List where curve[k] = number of unique bigrams after seeing items[0:k+1].
    """
    if not items:
        return []

    if torch is None or not torch.cuda.is_available():
        return _compute_bigram_curve_cpu(items)

    n = len(items)

    words_list: list[list[str]] = []
    for item in items:
        words_list.append(item.lower().split())

    max_words = max(len(w) for w in words_list) if words_list else 0
    if max_words == 0:
        return [0] * n

    padded: list[list[str]] = []
    for words in words_list:
        padded.append(words + [""] * (max_words - len(words)))

    word_tensor = torch.tensor(
        [[hash(w) % 1000000 for w in words] for words in padded],
        dtype=torch.int64,
        device="cuda",
    )

    bigram_matrix = torch.zeros((n, max_words - 1 if max_words > 1 else 1), dtype=torch.int64, device="cuda")

    if max_words > 1:
        bigram_matrix = word_tensor[:, :-1] * 1000000 + word_tensor[:, 1:]

    seen_bigrams: set[int] = set()
    curve: list[int] = []

    for i in range(n):
        row = bigram_matrix[i]
        if max_words > 1:
            for j in range(max_words - 1):
                bigram_val = int(row[j].item())
                if bigram_val != 0:
                    seen_bigrams.add(bigram_val)
        curve.append(len(seen_bigrams))

    return curve


def _compute_bigram_curve_cpu(items: Sequence[str]) -> list[int]:
    """Build cumulative unique bigram coverage curve (CPU version)."""
    seen_bigrams: set[tuple[str, str]] = set()
    curve: list[int] = []

    for item in items:
        words = item.lower().split()
        if len(words) < 2:
            seen_bigrams.add((words[0] if words else "", ""))
        else:
            for j in range(len(words) - 1):
                seen_bigrams.add((words[j], words[j + 1]))
        curve.append(len(seen_bigrams))

    return curve


def cluster_by_hamming_gpu(
    fingerprints: list[int],
    threshold: int = 3,
) -> list[list[int]]:
    """Cluster fingerprints by Hamming distance using GPU.

    Uses vectorized bitwise operations to compute Hamming distances
    between all fingerprints simultaneously.

    Args:
        fingerprints: List of 64-bit fingerprint integers.
        threshold: Max Hamming distance to consider items as duplicates.

    Returns:
        List of clusters, each containing indices of items in that cluster.
    """
    if not fingerprints:
        return []

    n = len(fingerprints)
    if n == 1:
        return [[0]]

    if torch is None or not torch.cuda.is_available():
        return _cluster_by_hamming_cpu(fingerprints, threshold)

    fp_tensor = torch.tensor(fingerprints, dtype=torch.uint64, device="cuda")

    clusters: list[list[int]] = []
    cluster_reps: list[int] = []

    for i in range(n):
        fp = fp_tensor[i]

        if not cluster_reps:
            cluster_reps.append(int(fp.item()))
            clusters.append([i])
            continue

        matched = False
        for j, rep in enumerate(cluster_reps):
            dist = bin(int(fp.item()) ^ rep).count("1")
            if dist <= threshold:
                clusters[j].append(i)
                matched = True
                break

        if not matched:
            cluster_reps.append(int(fp.item()))
            clusters.append([i])

    return clusters


def _cluster_by_hamming_cpu(
    fingerprints: list[int],
    threshold: int = 3,
) -> list[list[int]]:
    """Cluster fingerprints by Hamming distance (CPU version)."""
    if not fingerprints:
        return []

    n = len(fingerprints)
    if n == 1:
        return [[0]]

    clusters: list[list[int]] = []
    cluster_reps: list[int] = []

    for i in range(n):
        fp = fingerprints[i]
        matched = False

        for j, rep in enumerate(cluster_reps):
            dist = bin(fp ^ rep).count("1")
            if dist <= threshold:
                clusters[j].append(i)
                matched = True
                break

        if not matched:
            cluster_reps.append(fp)
            clusters.append([i])

    return clusters


def cluster_by_hamming_batch_gpu(
    fingerprints: list[int],
    threshold: int = 3,
) -> list[list[int]]:
    """Batch GPU clustering - computes all pairwise distances at once.

    More efficient for small n but allocates O(n^2) memory.

    Args:
        fingerprints: List of 64-bit fingerprint integers.
        threshold: Max Hamming distance to consider items as duplicates.

    Returns:
        List of clusters, each containing indices.
    """
    if not fingerprints:
        return []

    n = len(fingerprints)
    if n == 1:
        return [[0]]

    if torch is None or not torch.cuda.is_available():
        return _cluster_by_hamming_cpu(fingerprints, threshold)

    if n > 10000:
        return cluster_by_hamming_gpu(fingerprints, threshold)

    fp_tensor = torch.tensor(fingerprints, dtype=torch.uint64, device="cuda")

    fp_a = fp_tensor.unsqueeze(1)
    fp_b = fp_tensor.unsqueeze(0)

    xor_matrix = fp_a ^ fp_b

    hamming_matrix = torch.zeros((n, n), dtype=torch.int32, device="cuda")
    for bit in range(64):
        hamming_matrix += ((xor_matrix >> bit) & 1).int()

    assigned = torch.zeros(n, dtype=torch.bool, device="cuda")
    clusters: list[list[int]] = []

    for i in range(n):
        if assigned[i]:
            continue

        cluster_mask = (hamming_matrix[i] <= threshold) & (~assigned)
        cluster_indices = torch.where(cluster_mask)[0].cpu().tolist()

        for idx in cluster_indices:
            assigned[idx] = True

        clusters.append(cluster_indices)

    return clusters


def detect_structural_outliers_gpu(
    items: list[dict],
    outlier_ratio: float = 0.1,
) -> list[int]:
    """Detect structurally outlier items using GPU acceleration.

    Items are outliers if they have fields that appear in < outlier_ratio
    fraction of items.

    Args:
        items: List of dictionary items.
        outlier_ratio: Fraction threshold for outlier detection.

    Returns:
        List of indices of outlier items.
    """
    if not items or len(items) < 3:
        return []

    if torch is None or not torch.cuda.is_available():
        return _detect_structural_outliers_cpu(items, outlier_ratio)

    n = len(items)

    all_keys = set()
    for item in items:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    if not all_keys:
        return []

    key_list = list(all_keys)
    key_to_idx = {k: i for i, k in enumerate(key_list)}
    num_keys = len(key_list)

    field_matrix = torch.zeros((n, num_keys), dtype=torch.int8, device="cuda")

    for i, item in enumerate(items):
        if isinstance(item, dict):
            for key, value in item.items():
                if key in key_to_idx:
                    field_matrix[i, key_to_idx[key]] = 1

    field_counts = field_matrix.sum(dim=0)

    key_outlier_mask = field_counts < (n * outlier_ratio)

    item_outlier_scores = (field_matrix[:, key_outlier_mask] > 0).sum(dim=1).float()

    threshold = key_outlier_mask.sum().float().clamp(min=1)

    outlier_mask = item_outlier_scores > threshold

    outlier_indices = torch.where(outlier_mask)[0].cpu().tolist()

    return outlier_indices


def _detect_structural_outliers_cpu(
    items: list[dict],
    outlier_ratio: float = 0.1,
) -> list[int]:
    """Detect structurally outlier items (CPU version)."""
    if not items or len(items) < 3:
        return []

    all_keys: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    if not all_keys:
        return []

    key_counts: dict[str, int] = {k: 0 for k in all_keys}

    for item in items:
        if isinstance(item, dict):
            for key in all_keys:
                if key in item:
                    key_counts[key] += 1

    outlier_keys = {k for k, count in key_counts.items() if count < len(items) * outlier_ratio}

    outlier_indices: list[int] = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            outlier_count = sum(1 for k in outlier_keys if k in item)
            if outlier_count > len(outlier_keys) * 0.5:
                outlier_indices.append(i)

    return outlier_indices


def compute_statistics_gpu(values: list[float]) -> dict[str, float]:
    """Compute multiple statistics for a list of values using GPU.

    Args:
        values: List of numeric values.

    Returns:
        Dictionary with mean, variance, stdev, min, max, sum.
    """
    if not values:
        return {"mean": 0.0, "variance": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}

    if torch is None or not torch.cuda.is_available():
        return _compute_statistics_cpu(values)

    tensor = torch.tensor(values, dtype=torch.float32, device="cuda")

    return {
        "mean": float(tensor.mean().item()),
        "variance": float(tensor.var().item()),
        "stdev": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "sum": float(tensor.sum().item()),
    }


def _compute_statistics_cpu(values: list[float]) -> dict[str, float]:
    """Compute statistics (CPU version)."""
    if not values:
        return {"mean": 0.0, "variance": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}

    import statistics

    return {
        "mean": statistics.mean(values),
        "variance": statistics.variance(values) if len(values) > 1 else 0.0,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def is_available() -> bool:
    """Check if GPU acceleration is available.

    Returns:
        True if CUDA GPU is available.
    """
    if torch is None:
        return False
    return torch.cuda.is_available()

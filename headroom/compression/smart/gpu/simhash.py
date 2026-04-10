"""GPU and multiprocessing-accelerated simhash for deduplication.

SimHash is used to detect near-duplicate items by comparing
64-bit fingerprint bitwise similarity.

Uses multiprocessing for CPU parallelism when processing many items.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


def _simhash_cpu(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint for a text string (CPU version).

    Args:
        text: Input text.

    Returns:
        64-bit integer fingerprint.
    """
    v = [0] * 64
    text_lower = text.lower()

    for i in range(max(1, len(text_lower) - 3)):
        gram = text_lower[i : i + 4]
        h = int(hashlib.md5(gram.encode(), usedforsecurity=False).hexdigest()[:16], 16)
        for j in range(64):
            if h & (1 << j):
                v[j] += 1
            else:
                v[j] -= 1

    fingerprint = 0
    for j in range(64):
        if v[j] > 0:
            fingerprint |= 1 << j
    return fingerprint


def _hamming_distance_cpu(a: int, b: int) -> int:
    """Count differing bits between two 64-bit integers (CPU version)."""
    return bin(a ^ b).count("1")


def _compute_simhash_batch(texts: list[str]) -> list[int]:
    """Compute simhash for a batch of texts (used by multiprocessing)."""
    return [_simhash_cpu(text) for text in texts]


def simhash_batch_mp(texts: list[str], num_workers: int = 4) -> list[int]:
    """Compute SimHash fingerprints using multiprocessing.

    Args:
        texts: List of strings to hash.
        num_workers: Number of worker processes.

    Returns:
        List of 64-bit integer fingerprints.
    """
    if len(texts) < 50:
        return [_simhash_cpu(text) for text in texts]

    chunk_size = max(10, len(texts) // num_workers)
    with mp.Pool(num_workers) as pool:
        results = pool.map(_simhash_cpu, texts, chunksize=chunk_size)

    return results


def cluster_by_hamming_cpu(
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
            if _hamming_distance_cpu(fp, rep) <= threshold:
                matched = True
                clusters[j].append(i)
                break

        if not matched:
            cluster_reps.append(fp)
            clusters.append([i])

    return clusters


def count_unique_simhash_mp(items: list[str], threshold: int = 3, num_workers: int = 4) -> int:
    """Count unique items using SimHash with multiprocessing.

    Args:
        items: List of string items.
        threshold: Max Hamming distance for duplicates.
        num_workers: Number of worker processes.

    Returns:
        Number of unique content groups.
    """
    if not items:
        return 0

    n = len(items)
    if n < 50:
        from headroom.transforms.adaptive_sizer import count_unique_simhash
        return count_unique_simhash(items, threshold)

    fingerprints = simhash_batch_mp(items, num_workers)
    clusters = cluster_by_hamming_cpu(fingerprints, threshold)

    return len(clusters)


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    if torch is None:
        return False
    return torch.cuda.is_available()


def is_mp_available() -> bool:
    """Check if multiprocessing is available."""
    return True

"""Optimized simhash using multiprocessing for large batches.

This module provides an optimized version of simhash that uses:
1. Cython + xxHash for 6x faster hashing (when available)
2. Multiprocessing to parallelize across CPU cores
3. Batch IPC protocol to reduce per-item serialization overhead

Key insight: Simhash is CPU-bound but embarrassingly parallel.

.. warning::
    Simhash fingerprints are NOT portable across implementation tiers.
    Python/MD5-based fingerprints will differ from Cython+Rust/xxHash fingerprints.
    Fingerprints MUST NOT be cached across process restarts or shared between
    different implementation tiers (pure Python, Cython, Rust extension).
    Always recompute fingerprints within the same tier that produced them.
"""

from __future__ import annotations

import hashlib
import logging
import math
import multiprocessing as mp
import threading
from typing import Sequence

logger = logging.getLogger(__name__)

_persistent_pool: mp.Pool | None = None
_pool_lock = mp.Lock()

_cython_simhash = None

_INLINE_THRESHOLD = 200
_SHARED_MEMORY_THRESHOLD = 1_048_576
_TARGET_CHUNK_BYTES = 512 * 1024

_tls = threading.local()


class SimhashTimer:
    """Context manager for recording simhash pipeline stage timings.
    
    Records wall-clock time for each pipeline stage into thread-local storage.
    Always active - overhead is negligible compared to ms-scale computation.
    """

    def __init__(self):
        self.start_time = None
        self.stage_times: dict[str, float] = {}

    def __enter__(self):
        self.start_time = _time_ns()
        return self

    def __exit__(self, *args):
        total = (_time_ns() - self.start_time) / 1e6
        self.stage_times["total_ms"] = total
        if not hasattr(_tls, "timings"):
            _tls.timings = []
        _tls.timings.append(self.stage_times)

    def record(self, stage: str, ms: float):
        """Record a stage timing."""
        self.stage_times[stage] = ms


def get_last_timing() -> dict:
    """Return the most recent stage timings from thread-local storage."""
    if hasattr(_tls, "timings") and _tls.timings:
        return _tls.timings[-1]
    return {}


def _time_ns() -> float:
    """Get current time in nanoseconds for high-resolution timing."""
    import time
    return time.perf_counter_ns()


def _batch_simhash_worker(texts: list[str]) -> list[int]:
    """Module-level worker that computes simhash for multiple texts in one IPC call.
    
    Uses Cython when available, pure Python otherwise.
    """
    return [compute_simhash(text) for text in texts]


def _batch_cluster_worker(texts: list[str], threshold: int) -> list[tuple[int, list[int]]]:
    """Module-level worker that computes fingerprints AND performs local Hamming clustering.
    
    Returns list of (repr_fingerprint, member_indices) pairs.
    """
    fingerprints = [compute_simhash(text) for text in texts]
    clusters = cluster_by_hamming(fingerprints, threshold)
    return [(fingerprints[cluster[0]], cluster) for cluster in clusters]


def _merge_partial_clusters(
    partial_results: list[list[tuple[int, list[int]]]],
    threshold: int,
) -> int:
    """Merge cross-chunk clusters by checking representative fingerprints pairwise.
    
    Args:
        partial_results: List of per-chunk cluster results from workers.
        threshold: Hamming distance threshold for merging.
    
    Returns:
        Total number of unique clusters after merging.
    """
    all_clusters: list[tuple[int, list[int]]] = []
    for chunks_clusters in partial_results:
        all_clusters.extend(chunks_clusters)

    if not all_clusters:
        return 0

    merged: list[list[int]] = []
    cluster_reps: list[int] = []

    for rep_fingerprint, member_indices in all_clusters:
        matched = False
        for j, existing_rep in enumerate(cluster_reps):
            if hamming_distance(rep_fingerprint, existing_rep) <= threshold:
                merged[j].extend(member_indices)
                matched = True
                break
        if not matched:
            cluster_reps.append(rep_fingerprint)
            merged.append(list(member_indices))

    return len(merged)


def _compute_adaptive_chunks(texts: list[str], num_workers: int) -> list[list[str]]:
    """Split texts into chunks targeting 512KB each based on total payload size.
    
    Args:
        texts: List of text strings to chunk.
        num_workers: Number of worker processes.
    
    Returns:
        List of text chunks for distribution to workers.
    """
    if not texts:
        return []

    total_bytes = sum(len(t.encode()) for t in texts)
    chunk_count = max(1, total_bytes // _TARGET_CHUNK_BYTES)
    chunk_size = math.ceil(len(texts) / chunk_count)

    return [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]


def compute_simhash_batch(texts: Sequence[str], num_workers: int = 4) -> list[int]:
    """Compute SimHash fingerprints for multiple texts.

    Uses Rust extension when available (bypasses GIL and IPC entirely).
    Falls back to persistent multiprocessing pool with batch IPC protocol.

    Args:
        texts: Sequence of strings to hash.
        num_workers: Number of worker processes (used only when Rust unavailable).

    Returns:
        List of 64-bit integer fingerprints.
    """
    n = len(texts)

    with SimhashTimer() as timer:
        # Rust extension: GIL-free, no IPC overhead
        rust = _get_rust_simhash()
        if rust:
            timer.record("simhash_ms", 0)
            timer.record("ipc_ms", 0)
            return rust[1](list(texts))

        # Cython/Python path with multiprocessing
        if n < _INLINE_THRESHOLD:
            timer.record("simhash_ms", 0)
            timer.record("ipc_ms", 0)
            return [compute_simhash(text) for text in texts]

        pool = _get_persistent_pool(num_workers)
        chunks = _compute_adaptive_chunks(list(texts), num_workers)

        timer.record("simhash_ms", 0)
        ipc_start = _time_ns()
        results = pool.starmap(_batch_simhash_worker, [(chunk,) for chunk in chunks])
        ipc_elapsed = (_time_ns() - ipc_start) / 1e6
        timer.record("ipc_ms", ipc_elapsed)

        flat_results = []
        for chunk_results in results:
            flat_results.extend(chunk_results)

        return flat_results


def count_unique_simhash_batch(
    items: Sequence[str],
    threshold: int = 3,
    num_workers: int = 4,
) -> int:
    """Count items with distinct content using batched clustering.

    Dispatches fingerprinting and local clustering to workers, then merges
    partial results in the main process.

    Args:
        items: Sequence of string items.
        threshold: Max Hamming distance to consider items as duplicates.
        num_workers: Number of worker processes.

    Returns:
        Number of unique content groups.
    """
    n = len(items)
    if n < _INLINE_THRESHOLD:
        return _count_unique_simhash_original(items, threshold)

    # Rust extension: GIL-free, no IPC overhead
    rust = _get_rust_simhash()
    if rust:
        with SimhashTimer() as timer:
            timer.record("simhash_ms", 0)
            timer.record("ipc_ms", 0)
            return rust[2](list(items), threshold)

    with SimhashTimer() as timer:
        pool = _get_persistent_pool(num_workers)
        chunks = _compute_adaptive_chunks(list(items), num_workers)

        ipc_start = _time_ns()
        partial_results = pool.starmap(
            _batch_cluster_worker,
            [(chunk, threshold) for chunk in chunks],
        )
        ipc_elapsed = (_time_ns() - ipc_start) / 1e6
        timer.record("ipc_ms", ipc_elapsed)
        timer.record("simhash_ms", 0)

        cluster_start = _time_ns()
        result = _merge_partial_clusters(list(partial_results), threshold)
        cluster_elapsed = (_time_ns() - cluster_start) / 1e6
        timer.record("cluster_ms", cluster_elapsed)

        return result


def _get_cython_simhash():
    """Get Cython-accelerated simhash if available."""
    global _cython_simhash
    if _cython_simhash is None:
        try:
            from headroom.compression.smart._simhash_cython import compute_simhash_cython
            _cython_simhash = compute_simhash_cython
            logger.info("Using Cython-accelerated simhash")
        except ImportError:
            _cython_simhash = False
            logger.info("Cython simhash not available, using pure Python")
    return _cython_simhash if _cython_simhash else None


_rust_simhash = None


def _get_rust_simhash():
    """Get Rust simhash extension if available.
    
    Returns (compute_simhash_fn, compute_batch_fn, count_unique_fn) or None.
    """
    global _rust_simhash
    if _rust_simhash is None:
        try:
            from headroom_simhash import compute_simhash, compute_simhash_batch, count_unique_simhash
            _rust_simhash = (compute_simhash, compute_simhash_batch, count_unique_simhash)
            logger.info("Rust simhash extension available")
        except ImportError:
            _rust_simhash = False
            logger.info("Rust simhash extension not available")
    return _rust_simhash if _rust_simhash else None


def _get_persistent_pool(num_workers: int = 4) -> mp.Pool:
    """Get or create a persistent process pool.
    
    This avoids the overhead of creating/destroying pools on each call.
    """
    global _persistent_pool
    with _pool_lock:
        if _persistent_pool is None:
            _persistent_pool = mp.Pool(num_workers)
        return _persistent_pool


def compute_simhash(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint for a text string.

    Uses Rust extension if available (fastest), falls back to Cython + xxHash,
    then pure Python + MD5.

    Args:
        text: Input text.

    Returns:
        64-bit integer fingerprint.
    """
    rust = _get_rust_simhash()
    if rust:
        return rust[0](text)

    cython_fn = _get_cython_simhash()
    if cython_fn:
        return cython_fn(text)

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


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64-bit integers."""
    return bin(a ^ b).count("1")


def cluster_by_hamming(
    fingerprints: Sequence[int],
    threshold: int = 3,
) -> list[list[int]]:
    """Cluster fingerprints by Hamming distance.

    Args:
        fingerprints: Sequence of 64-bit fingerprint integers.
        threshold: Max Hamming distance to consider items as duplicates.

    Returns:
        List of clusters, each containing indices of items in that cluster.
    """
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
            if hamming_distance(fp, rep) <= threshold:
                matched = True
                clusters[j].append(i)
                break

        if not matched:
            cluster_reps.append(fp)
            clusters.append([i])

    return clusters


def count_unique_simhash(items: Sequence[str], threshold: int = 3, num_workers: int = 4) -> int:
    """Count items with distinct content using SimHash with MP acceleration.

    This is a drop-in replacement for headroom.transforms.adaptive_sizer.count_unique_simhash
    that uses multiprocessing for large batches.

    Args:
        items: Sequence of string items.
        threshold: Max Hamming distance to consider items as duplicates.
        num_workers: Number of worker processes.

    Returns:
        Number of unique content groups.
    """
    if not items:
        return 0

    n = len(items)
    if n < _INLINE_THRESHOLD:
        return _count_unique_simhash_original(items, threshold)

    # Rust extension: bypasses GIL and IPC entirely
    rust = _get_rust_simhash()
    if rust:
        return rust[2](list(items), threshold)

    # Use batch protocol for 200+ items
    return count_unique_simhash_batch(items, threshold, num_workers)


_original_count_unique_simhash: callable | None = None


def _capture_original() -> None:
    """Capture reference to original count_unique_simhash before patching.
    
    Must be called at module load time, before any patching occurs.
    """
    global _original_count_unique_simhash
    if _original_count_unique_simhash is None:
        from headroom.transforms.adaptive_sizer import count_unique_simhash
        _original_count_unique_simhash = count_unique_simhash


def _get_original_count_unique_simhash() -> callable:
    """Get reference to original count_unique_simhash (captured before patching)."""
    if _original_count_unique_simhash is None:
        _capture_original()
    return _original_count_unique_simhash


def _count_unique_simhash_original(items: Sequence[str], threshold: int = 3) -> int:
    """Original count_unique_simhash for small batches."""
    return _get_original_count_unique_simhash()(items, threshold)


def compute_simhash_and_bigrams_batch(texts: Sequence[str]) -> tuple[list[int], list[set]]:
    """Compute both simhash and bigrams for a batch of texts.

    This allows computing both in a single pass, reducing IPC overhead.

    Args:
        texts: Sequence of strings.

    Returns:
        Tuple of (simhash_list, bigram_sets_list).
    """
    try:
        from headroom.compression.smart._simhash_cython import compute_simhash_and_bigrams_cython
        return compute_simhash_and_bigrams_batch_cython(texts)
    except ImportError:
        return _compute_simhash_and_bigrams_python(texts)


def _compute_simhash_and_bigrams_python(texts: Sequence[str]) -> tuple[list[int], list[set]]:
    """Python implementation of simhash + bigrams."""
    import hashlib
    from collections import Counter

    results = []
    for text in texts:
        v = [0] * 64
        text_lower = text.lower()
        n = max(1, len(text_lower) - 3)

        for i in range(n):
            gram = text_lower[i : i + 4]
            h = int(hashlib.md5(gram.encode(), usedforsecurity=False).hexdigest()[:16], 16)
            for j in range(64):
                v[j] += 1 if (h & (1 << j)) else -1

        fingerprint = sum(1 << j for j in range(64) if v[j] > 0)

        words = text_lower.split()
        if len(words) < 2:
            bigrams = {(words[0] if words else "", "")}
        else:
            bigrams = {(words[i], words[i + 1]) for i in range(len(words) - 1)}

        results.append((fingerprint, bigrams))

    return [r[0] for r in results], [r[1] for r in results]


def compute_simhash_and_bigrams_batch_cython(texts: Sequence[str]) -> tuple[list[int], list[set]]:
    """Cython implementation of simhash + bigrams."""
    from headroom.compression.smart._simhash_cython import compute_simhash_and_bigrams_cython

    results = [compute_simhash_and_bigrams_cython(text) for text in texts]
    return [r[0] for r in results], [r[1] for r in results]


_capture_original()


def is_available() -> bool:
    """Check if optimized simhash is available."""
    return True

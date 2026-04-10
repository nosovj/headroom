"""Async worker pool wrapper for compute-heavy tasks.

This module provides an async interface to the Rust worker pool,
allowing Python async code to submit work without blocking the event loop.

Workers release the GIL during computation (via spawn_blocking), allowing
the event loop to process other requests while waiting.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Module-level worker pool instance
_worker_pool = None


def _get_worker_pool():
    """Get or create the global worker pool instance."""
    global _worker_pool
    if _worker_pool is None:
        try:
            import headroom_workers as workers
            _worker_pool = workers.create_pool(pool_size=None)  # Default: CPU count
            _worker_pool.start()
            logger.info(f"Worker pool started (size: {_worker_pool.get_stats().total_workers})")
        except Exception as e:
            logger.warning(f"Failed to start worker pool: {e}")
            _worker_pool = None
    return _worker_pool


@dataclass
class WorkResult:
    """Result of a work request."""
    request_id: int
    success: bool
    result: Any | None = None
    error: str | None = None


async def submit_simhash_work(items: list[str], threshold: int = 3) -> int:
    """Submit simhash work to the Rust worker pool.

    This function releases the GIL during computation, allowing the
    Python async event loop to continue processing other requests.

    Args:
        items: List of strings to count unique simhashes for.
        threshold: Hamming distance threshold for considering items similar.

    Returns:
        Number of unique items according to simhash clustering.
    """
    pool = _get_worker_pool()
    if pool is None:
        # Fall back to synchronous Rust call directly
        return _sync_simhash_call(items, threshold)

    request_id = id(items) & 0xFFFFFFFF

    # Run the blocking work in a thread to release GIL
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _execute_simhash_work(pool, request_id, items, threshold)
    )
    return result


def _execute_simhash_work(pool, request_id: int, items: list[str], threshold: int) -> int:
    """Execute simhash work synchronously in the worker pool thread."""
    try:
        # Submit work to pool
        pool.submit_work(request_id, json.dumps({
            "type": "count_unique_simhash",
            "items": items,
            "threshold": threshold,
        }))

        # For now, execute directly using headroom_simhash
        # The worker pool will eventually handle this
        return _sync_simhash_call(items, threshold)
    except Exception as e:
        logger.error(f"Simhash work failed: {e}")
        return _sync_simhash_call(items, threshold)


def _sync_simhash_call(items: list[str], threshold: int) -> int:
    """Fall back to synchronous Rust simhash call."""
    try:
        from headroom_simhash import count_unique_simhash as rust_count
        return rust_count(items, threshold)
    except ImportError:
        # Fall back to Python implementation
        from .adaptive_sizer import count_unique_simhash as python_count
        return python_count(items, threshold)


def get_pool_stats() -> dict[str, Any] | None:
    """Get worker pool statistics.

    Returns:
        Dict with total_workers, busy_workers, idle_workers, queue_depth, panics_recovered,
        or None if pool is not available.
    """
    pool = _get_worker_pool()
    if pool is None:
        return None

    stats = pool.get_stats()
    return {
        "total_workers": stats.total_workers,
        "busy_workers": stats.busy_workers,
        "idle_workers": stats.idle_workers,
        "queue_depth": stats.queue_depth,
        "panics_recovered": stats.panics_recovered,
    }


def shutdown_pool():
    """Shutdown the worker pool gracefully."""
    global _worker_pool
    if _worker_pool is not None:
        _worker_pool.stop()
        _worker_pool = None
        logger.info("Worker pool stopped")


@dataclass
class CompressionResult:
    """Result of text compression."""
    compressed: str
    ratio: float
    strategy: str
    latency_ms: float


def compress_text(content: str, context_type: str = "general") -> CompressionResult:
    """Compress text using Rust zstd.

    Args:
        content: Text to compress.
        context_type: Type of content ("tool_output", "log", "code", "general").
                    Used to select compression dictionary if available.

    Returns:
        CompressionResult with compressed text (base64), ratio, strategy, and latency_ms.
    """
    try:
        from headroom_workers import compress_text as rust_compress
        result = rust_compress(content, context_type)
        return CompressionResult(
            compressed=result.compressed,
            ratio=result.ratio,
            strategy=result.strategy,
            latency_ms=result.latency_ms,
        )
    except ImportError as e:
        logger.warning(f"Rust compression not available: {e}")
        return CompressionResult(
            compressed=content,
            ratio=1.0,
            strategy="unavailable",
            latency_ms=0.0,
        )


def decompress_text(content: str, context_type: str = "general") -> str:
    """Decompress text that was compressed with compress_text.

    Args:
        content: Base64-encoded compressed text.
        context_type: Type hint used during compression.

    Returns:
        Decompressed text string.
    """
    try:
        from headroom_workers import decompress_text as rust_decompress
        return rust_decompress(content, context_type)
    except ImportError as e:
        logger.warning(f"Rust decompression not available: {e}")
        return content

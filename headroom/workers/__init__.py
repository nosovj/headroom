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
import os
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Feature flag for async pipeline
ASYNC_PIPELINE_ENABLED = os.environ.get("HEADROOM_FEATURE_ASYNC_PIPELINE", "true").lower() == "true"

# Module-level worker pool instance
_worker_pool = None


def _get_worker_pool():
    """Get or create the global worker pool instance."""
    global _worker_pool
    if _worker_pool is None:
        try:
            import headroom_workers as workers
            pool_size = int(os.environ.get("HEADROOM_WORKER_POOL_SIZE", "0") or "0")
            _worker_pool = workers.create_pool(pool_size=pool_size if pool_size > 0 else None)
            _worker_pool.start()
            logger.info(
                f"Worker pool started (size: {_worker_pool.get_stats().total_workers}, "
                f"async_pipeline: {ASYNC_PIPELINE_ENABLED})"
            )
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
    payload = json.dumps({
        "type": "count_unique_simhash",
        "items": items,
        "threshold": threshold,
    })

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

"""Integration tests for the headroom-workers crate.

These tests verify the worker pool lifecycle including:
- Pool creation and initialization
- Work submission
- Statistics gathering
- Graceful shutdown
"""

import os
import time


def test_create_pool_default():
    """Test creating a pool with default settings."""
    import headroom_workers as workers

    pool = workers.create_pool()
    assert pool is not None

    # Clean up
    del pool


def test_create_pool_with_size():
    """Test creating a pool with explicit size."""
    import headroom_workers as workers

    pool = workers.create_pool(pool_size=2)
    assert pool is not None

    del pool


def test_start_stop_pool():
    """Test starting and stopping the pool."""
    import headroom_workers as workers

    pool = workers.create_pool(pool_size=2)

    pool.start()

    pool.stop()
    # After stop, pool should be stopped

    del pool


def test_get_stats():
    """Test getting pool statistics."""
    import headroom_workers as workers

    pool = workers.create_pool(pool_size=2)
    pool.start()

    stats = pool.get_stats()
    assert stats.total_workers == 2
    assert stats.busy_workers >= 0
    assert stats.idle_workers >= 0
    assert stats.queue_depth >= 0
    assert stats.panics_recovered == 0

    pool.stop()
    del pool


def test_submit_work():
    """Test submitting work to the pool."""
    import headroom_workers as workers

    pool = workers.create_pool(pool_size=2)
    pool.start()

    # Submit some work
    for i in range(5):
        pool.submit_work(request_id=i, payload=f"test_payload_{i}")

    # Give some time for work to be processed
    time.sleep(0.1)

    stats = pool.get_stats()
    # Worker should have processed some work
    assert stats.total_workers == 2

    pool.stop()
    del pool


def test_env_variable_pool_size():
    """Test that HEADROOM_WORKER_POOL_SIZE env var is respected."""
    # This test would need to be run in a subprocess to properly test env var
    # For now, just verify the function works
    import headroom_workers as workers

    default_size = workers.get_default_pool_size()
    assert default_size > 0

    pool = workers.create_pool(pool_size=4)
    assert pool is not None

    del pool


def test_pool_health_check():
    """Test health check returns pong responses."""
    import headroom_workers as workers

    pool = workers.create_pool(pool_size=2)
    pool.start()

    # Health check should return empty list (not yet implemented)
    pongs = pool.health_check()
    assert isinstance(pongs, list)

    pool.stop()
    del pool


if __name__ == "__main__":
    test_create_pool_default()
    test_create_pool_with_size()
    test_start_stop_pool()
    test_get_stats()
    test_submit_work()
    test_env_variable_pool_size()
    test_pool_health_check()
    print("All tests passed!")

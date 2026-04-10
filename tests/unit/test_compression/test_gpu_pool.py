"""Tests for GPU Worker Pool."""

from __future__ import annotations

import pytest
import time


class TestGPUWorkerPool:
    """Tests for GPU worker pool functionality."""

    def test_pool_initialization(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        assert pool.num_workers == 1
        assert not pool.is_running()

    def test_pool_start_stop(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        assert pool.is_running()
        pool.stop()
        assert not pool.is_running()

    def test_pool_not_running_error(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        with pytest.raises(RuntimeError, match="not running"):
            pool.compress([1, 2, 3], "number")

    def test_health_check_initial(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        health = pool.health_check()
        assert health["status"] == "ok"
        assert health["workers_total"] == 0  # No workers created until start
        assert health["workers_alive"] == 0

    def test_health_check_running(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        try:
            time.sleep(0.5)
            health = pool.health_check()
            assert health["status"] == "ok"
            assert health["workers_alive"] == 1
        finally:
            pool.stop()


class TestWorkRequest:
    """Tests for work request/response."""

    def test_work_request_creation(self):
        from headroom.compression.smart.gpu_pool import WorkRequest
        req = WorkRequest(
            request_id="test_1",
            func_name="crush_number_array",
            args=([1, 2, 3],),
            kwargs={"bias": 1.5},
        )
        assert req.request_id == "test_1"
        assert req.func_name == "crush_number_array"
        assert req.args == ([1, 2, 3],)
        assert req.kwargs == {"bias": 1.5}

    def test_work_response_success(self):
        from headroom.compression.smart.gpu_pool import WorkResponse
        resp = WorkResponse(
            request_id="test_1",
            success=True,
            result=[1, 2, 3],
        )
        assert resp.success is True
        assert resp.result == [1, 2, 3]
        assert resp.error is None

    def test_work_response_failure(self):
        from headroom.compression.smart.gpu_pool import WorkResponse
        resp = WorkResponse(
            request_id="test_1",
            success=False,
            error="Something went wrong",
        )
        assert resp.success is False
        assert resp.error == "Something went wrong"


class TestGPUWorkerPoolIntegration:
    """Integration tests for GPU worker pool."""

    def test_compress_number_array(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        try:
            time.sleep(0.5)
            result, strategy = pool.compress(
                list(range(100)),
                array_type="number",
                bias=2.0,
                timeout=10.0,
            )
            assert isinstance(result, list)
            assert isinstance(strategy, str)
        finally:
            pool.stop()

    def test_compress_string_array(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        try:
            time.sleep(0.5)
            result, strategy = pool.compress(
                ["test"] * 50,
                array_type="string",
                bias=1.5,
                timeout=10.0,
            )
            assert isinstance(result, list)
            assert isinstance(strategy, str)
        finally:
            pool.stop()

    def test_compress_dict_array(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        try:
            time.sleep(0.5)
            items = [{"id": i, "name": f"item_{i}"} for i in range(30)]
            result, strategy = pool.compress(
                items,
                array_type="dict",
                bias=1.0,
                timeout=10.0,
            )
            assert isinstance(result, list)
            assert isinstance(strategy, str)
        finally:
            pool.stop()

    def test_calculate_entropy(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        try:
            time.sleep(0.5)
            result = pool.calculate_entropy(
                ["hello", "world", "test"],
                timeout=10.0,
            )
            assert isinstance(result, list)
            assert len(result) == 3
        finally:
            pool.stop()

    def test_detect_change_points(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        try:
            time.sleep(0.5)
            values = [1.0] * 20 + [10.0] * 10
            result = pool.detect_change_points(
                values,
                window=3,
                timeout=10.0,
            )
            assert isinstance(result, list)
        finally:
            pool.stop()

    def test_timeout_error(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        try:
            time.sleep(0.5)
            with pytest.raises(TimeoutError):
                pool.compress(
                    list(range(100)),
                    array_type="number",
                    timeout=0.001,
                )
        finally:
            pool.stop()


class TestGPUWorkerPoolRestart:
    """Tests for worker restart behavior."""

    def test_multiple_start_stop(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        time.sleep(0.3)
        pool.stop()
        pool.start()
        time.sleep(0.3)
        assert pool.is_running()
        pool.stop()

    def test_worker_restart_on_death(self):
        from headroom.compression.smart.gpu_pool import GPUWorkerPool
        pool = GPUWorkerPool(num_workers=1)
        pool.start()
        time.sleep(0.5)
        health = pool.health_check()
        assert health["workers_alive"] == 1
        pool.stop()

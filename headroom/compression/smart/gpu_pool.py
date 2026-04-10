"""GPU Worker Pool for compression.

Provides persistent worker processes with preloaded GPU models for crash recovery.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import signal
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkRequest:
    """Request to worker process."""
    request_id: str
    func_name: str
    args: tuple
    kwargs: dict


@dataclass
class WorkResponse:
    """Response from worker process."""
    request_id: str
    success: bool
    result: Any = None
    error: str | None = None


class GPUWorkerProcess(mp.Process):
    """Worker process with GPU acceleration."""

    def __init__(self, work_queue: mp.Queue, result_queue: mp.Queue, worker_id: int):
        super().__init__()
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self._gpu_available = False

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._initialize_worker()
        self._process_loop()

    def _initialize_worker(self) -> None:
        """Initialize GPU and load models."""
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
            if self._gpu_available:
                torch.cuda.set_device(self._worker_id % torch.cuda.device_count())
                logger.info(f"Worker {self.worker_id}: GPU initialized")
            else:
                logger.info(f"Worker {self.worker_id}: Running in CPU mode")
        except Exception as e:
            logger.warning(f"Worker {self.worker_id}: GPU init failed: {e}")
            self._gpu_available = False

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info(f"Worker {self.worker_id}: Received signal {signum}, shutting down")
        self._cleanup()
        raise SystemExit(0)

    def _cleanup(self) -> None:
        """Clean up GPU resources."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _process_loop(self) -> None:
        """Main work processing loop."""
        while True:
            try:
                request = self.work_queue.get(timeout=1.0)
                if request is None:
                    break
                response = self._process_request(request)
                self.result_queue.put(response)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error: {e}")

    def _process_request(self, request: WorkRequest) -> WorkResponse:
        """Process a single work request."""
        try:
            result = self._dispatch(request)
            return WorkResponse(
                request_id=request.request_id,
                success=True,
                result=result,
            )
        except Exception as e:
            return WorkResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
            )

    def _dispatch(self, request: WorkRequest) -> Any:
        """Dispatch request to appropriate handler."""
        if request.func_name == "crush_number_array":
            return self._crush_number_array(*request.args, **request.kwargs)
        elif request.func_name == "crush_string_array":
            return self._crush_string_array(*request.args, **request.kwargs)
        elif request.func_name == "crush_array":
            return self._crush_array(*request.args, **request.kwargs)
        elif request.func_name == "calculate_entropy":
            return self._calculate_entropy(*request.args, **request.kwargs)
        elif request.func_name == "detect_change_points":
            return self._detect_change_points(*request.args, **request.kwargs)
        else:
            raise ValueError(f"Unknown function: {request.func_name}")

    def _crush_number_array(self, items: list, config: dict | None = None, bias: float = 1.0):
        from headroom.compression.smart.gpu.compress import crush_number_array_gpu
        return crush_number_array_gpu(items, config, bias)

    def _crush_string_array(self, items: list, config: dict | None = None, bias: float = 1.0):
        from headroom.compression.smart.gpu.compress import crush_string_array_gpu
        return crush_string_array_gpu(items, config, bias)

    def _crush_array(self, items: list, config: dict | None = None, **kwargs):
        from headroom.compression.smart.gpu.compress import crush_array_gpu
        return crush_array_gpu(items, config, **kwargs)

    def _calculate_entropy(self, strings: list):
        from headroom.compression.smart.gpu.analyze import calculate_string_entropy_gpu
        return calculate_string_entropy_gpu(strings)

    def _detect_change_points(self, values: list, **kwargs):
        from headroom.compression.smart.gpu.analyze import detect_change_points_gpu
        return detect_change_points_gpu(values, **kwargs)


class GPUWorkerPool:
    """Pool of GPU workers for compression tasks."""

    def __init__(
        self,
        num_workers: int = 2,
        max_retries: int = 3,
        health_check_interval: float = 30.0,
    ):
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        self._work_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None
        self._workers: list[GPUWorkerProcess] = []
        self._running = False
        self._next_request_id = 0
        self._pending_requests: dict[str, tuple[WorkRequest, float]] = {}
        self._last_health_check = 0.0

    def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return

        self._work_queue = mp.Queue()
        self._result_queue = mp.Queue()

        for i in range(self.num_workers):
            worker = GPUWorkerProcess(
                work_queue=self._work_queue,
                result_queue=self._result_queue,
                worker_id=i,
            )
            worker.start()
            self._workers.append(worker)

        self._running = True
        logger.info(f"GPU worker pool started with {self.num_workers} workers")

    def stop(self) -> None:
        """Stop the worker pool gracefully."""
        if not self._running:
            return

        for _ in range(self.num_workers):
            self._work_queue.put(None)

        for worker in self._workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning(f"Worker {worker.worker_id} did not exit gracefully, terminating")
                worker.terminate()

        self._workers.clear()
        self._running = False
        logger.info("GPU worker pool stopped")

    def compress(
        self,
        items: list,
        array_type: str,
        config: dict | None = None,
        bias: float = 1.0,
        timeout: float = 30.0,
    ) -> tuple[Any, str]:
        """Submit compression task to pool.

        Args:
            items: Items to compress.
            array_type: Type of array ("number", "string", "dict").
            config: Compression config.
            bias: Compression bias.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (result, strategy).

        Raises:
            TimeoutError: If task doesn't complete within timeout.
            RuntimeError: If all retries fail.
        """
        func_map = {
            "number": "crush_number_array",
            "string": "crush_string_array",
            "dict": "crush_array",
        }

        func_name = func_map.get(array_type, "crush_array")

        request_id = self._submit_request(func_name, (items,), {"config": config, "bias": bias})

        result = self._wait_for_result(request_id, timeout)

        if array_type == "dict" and len(result) == 3:
            return result[0], result[1]
        return result

    def calculate_entropy(self, strings: list, timeout: float = 10.0) -> list[float]:
        """Calculate string entropies using GPU.

        Args:
            strings: List of strings.
            timeout: Timeout in seconds.

        Returns:
            List of entropy values.
        """
        request_id = self._submit_request("calculate_entropy", (strings,), {})
        return self._wait_for_result(request_id, timeout)

    def detect_change_points(
        self,
        values: list,
        timeout: float = 10.0,
        **kwargs,
    ) -> list[int]:
        """Detect change points using GPU.

        Args:
            values: Numeric values.
            timeout: Timeout in seconds.
            **kwargs: Additional arguments.

        Returns:
            List of change point indices.
        """
        request_id = self._submit_request("detect_change_points", (values,), kwargs)
        return self._wait_for_result(request_id, timeout)

    def _submit_request(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Submit a work request to the queue."""
        if not self._running:
            raise RuntimeError("Worker pool not running")

        self._next_request_id += 1
        request_id = f"req_{self._next_request_id}"

        request = WorkRequest(
            request_id=request_id,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
        )

        self._work_queue.put(request)
        self._pending_requests[request_id] = (request, time.time())

        return request_id

    def _wait_for_result(self, request_id: str, timeout: float) -> tuple[Any, str]:
        """Wait for a result with timeout and retry on failure."""
        start_time = time.time()
        retries = 0

        while retries < self.max_retries:
            try:
                elapsed = time.time() - start_time
                remaining_timeout = timeout - elapsed
                if remaining_timeout <= 0:
                    raise TimeoutError(f"Request {request_id} timed out")

                response = self._result_queue.get(timeout=remaining_timeout)

                if response.request_id != request_id:
                    self._result_queue.put(response)
                    continue

                del self._pending_requests[request_id]

                if response.success:
                    return response.result
                else:
                    raise RuntimeError(f"Request failed: {response.error}")

            except queue.Empty:
                retries += 1
                if retries >= self.max_retries:
                    raise TimeoutError(f"Request {request_id} timed out after {self.max_retries} retries")
                logger.warning(f"Request {request_id} timed out, retry {retries}/{self.max_retries}")

        raise RuntimeError(f"Request {request_id} failed after {self.max_retries} retries")

    def health_check(self) -> dict[str, Any]:
        """Check health of worker pool.

        Returns:
            Dict with health status.
        """
        now = time.time()
        if now - self._last_health_check < self.health_check_interval:
            return {"status": "ok", "workers": len(self._workers)}

        alive_workers = sum(1 for w in self._workers if w.is_alive())

        if alive_workers < len(self._workers):
            logger.warning(f"Only {alive_workers}/{len(self._workers)} workers alive")
            self._restart_dead_workers()

        self._last_health_check = now

        return {
            "status": "ok" if alive_workers == len(self._workers) else "degraded",
            "workers_total": len(self._workers),
            "workers_alive": alive_workers,
        }

    def _restart_dead_workers(self) -> None:
        """Restart any dead workers."""
        for i, worker in enumerate(self._workers):
            if not worker.is_alive():
                logger.info(f"Restarting worker {i}")
                new_worker = GPUWorkerProcess(
                    work_queue=self._work_queue,
                    result_queue=self._result_queue,
                    worker_id=i,
                )
                new_worker.start()
                self._workers[i] = new_worker

    def is_running(self) -> bool:
        """Check if pool is running."""
        return self._running

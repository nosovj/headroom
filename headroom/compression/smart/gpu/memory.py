"""GPU memory management with OOM detection for compression.

Provides GPU memory reservation and OOM error propagation (NO fallback).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


class GPUOutOfMemoryError(Exception):
    """Raised when GPU memory is exhausted (NO fallback)."""
    pass


class CompressionMemoryManager:
    """Manages GPU memory for compression with strict OOM handling.

    Reserves GPU memory to avoid competing with Ollama.
    Default: Reserve 12.5% (4GB on 32GB GPU) for compression.
    """

    DEFAULT_MEMORY_FRACTION = 0.125

    def __init__(
        self,
        memory_fraction: float = DEFAULT_MEMORY_FRACTION,
        oom_on_failure: bool = True,
    ):
        """Initialize compression memory manager.

        Args:
            memory_fraction: Fraction of GPU memory to reserve.
                              Default 0.125 = 12.5% (4GB on 32GB GPU).
            oom_on_failure: If True, raise GPUOutOfMemoryError on OOM.
                           If False, log warning and continue (for testing only).
        """
        self.memory_fraction = memory_fraction
        self.oom_on_failure = oom_on_failure
        self._reserved = False
        self._device_count = 0

    def reserve(self) -> bool:
        """Reserve GPU memory for compression.

        Returns:
            True if memory was reserved successfully.

        Raises:
            GPUOutOfMemoryError: If memory reservation fails and oom_on_failure is True.
        """
        if torch is None:
            logger.info("PyTorch not available, skipping GPU memory reservation")
            return True

        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping GPU memory reservation")
            return True

        try:
            self._device_count = torch.cuda.device_count()
            total = torch.cuda.get_device_properties(0).total_memory
            reserved_bytes = int(total * self.memory_fraction)
            reserved_gb = reserved_bytes / 1024**3
            total_gb = total / 1024**3

            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)

            self._reserved = True
            logger.info(
                f"GPU memory reserved for compression: {reserved_gb:.2f}GB "
                f"({self.memory_fraction * 100:.1f}% of {total_gb:.2f}GB)"
            )
            return True

        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"GPU OOM during memory reservation: {e}"
            logger.error(error_msg)
            if self.oom_on_failure:
                raise GPUOutOfMemoryError(error_msg) from e
            return False

        except Exception as e:
            error_msg = f"Failed to reserve GPU memory: {e}"
            logger.error(error_msg)
            if self.oom_on_failure:
                raise RuntimeError(error_msg) from e
            return False

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage.

        Returns:
            Dict with memory info, or {"available": False} if not available.
        """
        if torch is None or not torch.cuda.is_available():
            return {"available": False}

        try:
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = total - reserved

            return {
                "available": True,
                "total_gb": total,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "free_gb": free,
                "device_count": self._device_count,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {"available": False, "error": str(e)}

    def check_available(self, required_gb: float) -> bool:
        """Check if required memory is available.

        Args:
            required_gb: Required memory in GB.

        Returns:
            True if required memory is available.
        """
        info = self.get_memory_info()
        if not info.get("available"):
            return False
        return info.get("free_gb", 0) >= required_gb

    def allocate_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Allocate a tensor with OOM handling.

        Args:
            shape: Tensor shape.
            dtype: Tensor data type.

        Returns:
            Allocated tensor.

        Raises:
            GPUOutOfMemoryError: If allocation fails.
        """
        if not self._reserved:
            self.reserve()

        try:
            tensor = torch.empty(shape, dtype=dtype, device='cuda')
            return tensor

        except torch.cuda.OutOfMemoryError as e:
            error_msg = (
                f"GPU OOM: Cannot allocate tensor of shape {shape}, dtype {dtype}. "
                f"Consider reducing batch size or model size."
            )
            logger.error(error_msg)
            raise GPUOutOfMemoryError(error_msg) from e

    def reset(self) -> None:
        """Reset GPU memory state."""
        if torch is None:
            return

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._reserved = False
            logger.info("GPU memory state reset")
        except Exception as e:
            logger.warning(f"Failed to reset GPU memory: {e}")

    def is_reserved(self) -> bool:
        """Check if memory has been reserved."""
        return self._reserved


_global_memory_manager: CompressionMemoryManager | None = None


def get_memory_manager() -> CompressionMemoryManager:
    """Get global compression memory manager.

    Returns:
        Global CompressionMemoryManager instance.
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = CompressionMemoryManager()
    return _global_memory_manager


def reserve_compression_memory(
    fraction: float = CompressionMemoryManager.DEFAULT_MEMORY_FRACTION,
    oom_on_failure: bool = True,
) -> bool:
    """Reserve GPU memory using global manager.

    Args:
        fraction: Memory fraction to reserve.
        oom_on_failure: If True, raise on OOM.

    Returns:
        True if successful.

    Raises:
        GPUOutOfMemoryError: If reservation fails and oom_on_failure is True.
    """
    manager = get_memory_manager()
    manager.memory_fraction = fraction
    manager.oom_on_failure = oom_on_failure
    return manager.reserve()


class MemoryGuardedOperation:
    """Context manager for memory-guarded GPU operations.

    Ensures OOM errors propagate correctly (NO fallback).
    """

    def __init__(self, manager: CompressionMemoryManager | None = None):
        self.manager = manager or get_memory_manager()
        self._original_fraction = None

    def __enter__(self) -> "MemoryGuardedOperation":
        if not self.manager.is_reserved():
            self.manager.reserve()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is torch.cuda.OutOfMemoryError:
            error_msg = f"GPU OOM in guarded operation: {exc_val}"
            logger.error(error_msg)
            raise GPUOutOfMemoryError(error_msg) from exc_val
        return False

    def execute(self, func, *args, **kwargs):
        """Execute function with memory guarding.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of function execution.

        Raises:
            GPUOutOfMemoryError: If GPU OOM occurs.
        """
        with self:
            return func(*args, **kwargs)

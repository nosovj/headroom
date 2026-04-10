"""GPU memory management for compression.

Manages GPU memory allocation to avoid competing with Ollama.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MAX_MEMORY_FRACTION = 0.125


class GPUMemoryManager:
    """Manages GPU memory for compression.

    Reserves memory for compression to avoid competing with Ollama.
    Default: Reserve 12.5% (4GB on 32GB GPU) for compression.
    """

    def __init__(self, memory_fraction: float = MAX_MEMORY_FRACTION):
        """Initialize GPU memory manager.

        Args:
            memory_fraction: Fraction of GPU memory to reserve for compression.
                              Default 0.125 = 12.5% (4GB on 32GB GPU).
        """
        self.memory_fraction = memory_fraction
        self._initialized = False

    def reserve(self) -> bool:
        """Reserve GPU memory for compression.

        Returns:
            True if memory was reserved successfully or not needed.
            False if memory reservation failed.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("CUDA not available, skipping GPU memory reservation")
                return True

            total = torch.cuda.get_device_properties(0).total_memory
            reserved = int(total * self.memory_fraction)

            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)

            logger.info(
                f"GPU memory reserved for compression: {reserved / 1024**3:.2f}GB "
                f"({self.memory_fraction * 100:.1f}% of {total / 1024**3:.2f}GB)"
            )
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to reserve GPU memory: {e}")
            return False

    def get_memory_info(self) -> dict[str, float]:
        """Get current GPU memory usage.

        Returns:
            Dict with total, allocated, and free memory in GB.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}

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
            }
        except Exception:
            return {"available": False}

    def reset(self) -> None:
        """Reset GPU memory state."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._initialized = False
        except Exception as e:
            logger.warning(f"Failed to reset GPU memory: {e}")


_global_memory_manager: GPUMemoryManager | None = None


def get_memory_manager() -> GPUMemoryManager:
    """Get global GPU memory manager.

    Returns:
        Global GPUMemoryManager instance.
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = GPUMemoryManager()
    return _global_memory_manager


def reserve_gpu_memory(fraction: float = MAX_MEMORY_FRACTION) -> bool:
    """Reserve GPU memory using global manager.

    Args:
        fraction: Memory fraction to reserve.

    Returns:
        True if successful.
    """
    manager = get_memory_manager()
    manager.memory_fraction = fraction
    return manager.reserve()

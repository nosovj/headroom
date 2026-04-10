"""Model warmup for GPU compression.

Preloads GPU models at startup to avoid cold-start latency.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WarmupConfig:
    """Configuration for warmup."""

    timeout: float = 60.0
    device: str = "cuda"


class CompressionWarmupper:
    """Preloads GPU models at startup.

    Usage:
        warmupper = CompressionWarmupper()
        await warmupper.warmup()
        # Now accept requests
    """

    def __init__(self, config: WarmupConfig | None = None):
        """Initialize warmup handler.

        Args:
            config: Warmup configuration.
        """
        self.config = config or WarmupConfig()
        self._models_loaded = False
        self._loading = False

    async def warmup(self) -> None:
        """Load all GPU models.

        Raises:
            asyncio.TimeoutError: If warmup exceeds timeout.
        """
        if self._models_loaded:
            logger.info("Models already loaded, skipping warmup")
            return

        if self._loading:
            logger.info("Warmup already in progress")
            return

        self._loading = True
        logger.info("Starting GPU model warmup...")

        try:
            await asyncio.wait_for(
                self._load_models(),
                timeout=self.config.timeout
            )
            self._models_loaded = True
            logger.info("GPU model warmup complete")
        except asyncio.TimeoutError:
            logger.error(f"Warmup timed out after {self.config.timeout}s")
            raise
        finally:
            self._loading = False

    async def _load_models(self) -> None:
        """Load all required GPU models.

        This loads:
        - SmartCrusher (CPU-based, but initializes caches)
        - KompressCompressor models if GPU available
        """
        from headroom.compression.smart import create_smart_crusher
        from headroom.compression.smart.optimized_simhash import count_unique_simhash as mp_count_unique_simhash

        logger.info("Patching count_unique_simhash with MP-accelerated version...")
        import headroom.transforms.adaptive_sizer as adaptive_sizer
        adaptive_sizer.count_unique_simhash = mp_count_unique_simhash
        logger.info("Patch applied, initializing SmartCrusher...")
        create_smart_crusher()
        logger.info("SmartCrusher initialized")

        if self.config.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    logger.info("GPU available, preparing GPU context...")
                    await self._warmup_gpu()
                else:
                    logger.warning("CUDA not available, running in CPU mode")
            except ImportError:
                logger.warning("PyTorch not available, running in CPU mode")

    async def _warmup_gpu(self) -> None:
        """Warmup GPU-specific models."""
        try:
            import torch

            with torch.device("cuda"):
                dummy = torch.zeros(10, device="cuda")
                del dummy
                torch.cuda.synchronize()
            logger.info("GPU context warmup complete")
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")

    def health_check(self) -> bool:
        """Check if models are loaded and ready.

        Returns:
            True if models are loaded, False otherwise.
        """
        return self._models_loaded

    def reset(self) -> None:
        """Reset warmup state.

        Useful for testing or forced restart.
        """
        self._models_loaded = False
        self._loading = False

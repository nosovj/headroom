"""Unit tests for compression/smart/warmup module."""

import asyncio

import pytest

from headroom.compression.smart.warmup import (
    CompressionWarmupper,
    WarmupConfig,
)


class TestCompressionWarmupper:
    """Tests for CompressionWarmupper."""

    def test_initial_state(self):
        """Warmupper starts in not-loaded state."""
        warmup = CompressionWarmupper()
        assert warmup.health_check() is False
        assert warmup._models_loaded is False

    def test_warmup_config(self):
        """WarmupConfig defaults are correct."""
        config = WarmupConfig()
        assert config.timeout == 60.0
        assert config.device == "cuda"

    def test_reset(self):
        """Reset clears warmup state."""
        warmup = CompressionWarmupper()
        warmup._models_loaded = True
        warmup.reset()
        assert warmup.health_check() is False

    def test_double_warmup_skip(self):
        """Second warmup is skipped if already loaded."""
        warmup = CompressionWarmupper()
        warmup._models_loaded = True
        warmup._loading = False
        warmup.warmup()
        assert warmup._models_loaded is True

    def test_async_warmup(self):
        """Warmup completes without error."""
        warmup = CompressionWarmupper()
        asyncio.run(warmup.warmup())
        assert warmup.health_check() is True

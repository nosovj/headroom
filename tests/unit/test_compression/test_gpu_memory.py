"""Tests for GPU memory management."""

from __future__ import annotations

import pytest


class TestCompressionMemoryManager:
    """Tests for CompressionMemoryManager."""

    def test_initialization(self):
        from headroom.compression.smart.gpu.memory import CompressionMemoryManager
        manager = CompressionMemoryManager()
        assert manager.memory_fraction == 0.125
        assert not manager.is_reserved()

    def test_custom_fraction(self):
        from headroom.compression.smart.gpu.memory import CompressionMemoryManager
        manager = CompressionMemoryManager(memory_fraction=0.25)
        assert manager.memory_fraction == 0.25

    def test_oom_on_failure_default(self):
        from headroom.compression.smart.gpu.memory import CompressionMemoryManager
        manager = CompressionMemoryManager()
        assert manager.oom_on_failure is True

    def test_oom_on_failure_disabled(self):
        from headroom.compression.smart.gpu.memory import CompressionMemoryManager
        manager = CompressionMemoryManager(oom_on_failure=False)
        assert manager.oom_on_failure is False


class TestGPUOutOfMemoryError:
    """Tests for GPUOutOfMemoryError."""

    def test_exception_exists(self):
        from headroom.compression.smart.gpu.memory import GPUOutOfMemoryError
        assert issubclass(GPUOutOfMemoryError, Exception)

    def test_exception_message(self):
        from headroom.compression.smart.gpu.memory import GPUOutOfMemoryError
        error = GPUOutOfMemoryError("Test OOM error")
        assert str(error) == "Test OOM error"

    def test_exception_chaining(self):
        from headroom.compression.smart.gpu.memory import GPUOutOfMemoryError
        original = ValueError("Original")
        try:
            raise GPUOutOfMemoryError("OOM") from original
        except GPUOutOfMemoryError as e:
            assert e.__cause__ is original


class TestGetMemoryInfo:
    """Tests for memory info retrieval."""

    def test_memory_info_when_not_available(self):
        from headroom.compression.smart.gpu.memory import CompressionMemoryManager
        manager = CompressionMemoryManager()
        info = manager.get_memory_info()
        assert isinstance(info, dict)


class TestCheckAvailable:
    """Tests for memory availability check."""

    def test_check_available_returns_bool(self):
        from headroom.compression.smart.gpu.memory import CompressionMemoryManager
        manager = CompressionMemoryManager()
        result = manager.check_available(1.0)
        assert isinstance(result, bool)


class TestMemoryGuardedOperation:
    """Tests for MemoryGuardedOperation context manager."""

    def test_context_manager_init(self):
        from headroom.compression.smart.gpu.memory import MemoryGuardedOperation
        op = MemoryGuardedOperation()
        assert op.manager is not None

    def test_context_manager_with_manager(self):
        from headroom.compression.smart.gpu.memory import (
            CompressionMemoryManager,
            MemoryGuardedOperation,
        )
        manager = CompressionMemoryManager()
        op = MemoryGuardedOperation(manager)
        assert op.manager is manager

    def test_execute_returns_result(self):
        from headroom.compression.smart.gpu.memory import MemoryGuardedOperation
        op = MemoryGuardedOperation()
        result = op.execute(lambda x: x * 2, 5)
        assert result == 10

    def test_execute_with_args(self):
        from headroom.compression.smart.gpu.memory import MemoryGuardedOperation
        op = MemoryGuardedOperation()
        result = op.execute(lambda x, y: x + y, 3, 4)
        assert result == 7


class TestGlobalManager:
    """Tests for global memory manager."""

    def test_get_memory_manager(self):
        from headroom.compression.smart.gpu.memory import get_memory_manager
        manager = get_memory_manager()
        assert manager is not None
        assert isinstance(manager, type(get_memory_manager()))

    def test_same_manager_returned(self):
        from headroom.compression.smart.gpu.memory import get_memory_manager
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()
        assert manager1 is manager2


class TestReserveCompressionMemory:
    """Tests for reserve_compression_memory function."""

    def test_function_exists(self):
        from headroom.compression.smart.gpu.memory import reserve_compression_memory
        assert callable(reserve_compression_memory)

    def test_default_fraction(self):
        from headroom.compression.smart.gpu.memory import (
            CompressionMemoryManager,
            reserve_compression_memory,
        )
        manager = reserve_compression_memory()
        assert manager is True
        assert manager is not False


class TestReset:
    """Tests for memory reset."""

    def test_reset(self):
        from headroom.compression.smart.gpu.memory import CompressionMemoryManager
        manager = CompressionMemoryManager()
        manager._reserved = True
        manager.reset()
        assert not manager.is_reserved()


class TestDefaultConstants:
    """Tests for default constants."""

    def test_default_memory_fraction(self):
        from headroom.compression.smart.gpu.memory import CompressionMemoryManager
        assert CompressionMemoryManager.DEFAULT_MEMORY_FRACTION == 0.125

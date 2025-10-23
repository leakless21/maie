"""
Comprehensive test suite for modern AsyncProcessor interfaces.

This test suite validates the new AsyncProcessor base class and related utilities
that replace the deprecated asyncio.get_event_loop() pattern.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.processors.async_base import (
    AsyncProcessor,
    run_in_executor,
    safe_async_execute,
)


class TestAsyncProcessorModern:
    """Test suite for modern async processor interfaces."""

    def create_async_processor(self, execute_result=None, should_fail=False):
        """Create a concrete async processor for testing."""

        class TestAsyncProcessor(AsyncProcessor):
            def __init__(self, execute_result=None, should_fail=False):
                self.execute_result = execute_result or "test_result"
                self.should_fail = should_fail
                self.unload_called = False
                self.setup_called = False

            async def execute(self, *args, **kwargs):
                if self.should_fail:
                    raise ValueError("Test execution failure")
                return self.execute_result

            async def unload(self):
                self.unload_called = True

            async def _setup_resources(self):
                self.setup_called = True

        return TestAsyncProcessor(
            execute_result=execute_result, should_fail=should_fail
        )

    @pytest.mark.asyncio
    async def test_async_execute_with_running_loop(self):
        """Test run_in_executor utility uses get_running_loop correctly."""

        # Use a sync function to exercise the loop->thread bridge
        def sync_fn(x):
            return f"modern_{x}"

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = AsyncMock()
            mock_run_in_executor = AsyncMock()
            mock_run_in_executor.return_value = "modern_test"
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            # call the utility that must call get_running_loop internally
            from src.processors.async_base import run_in_executor

            result = await run_in_executor(sync_fn, "test_data")

            mock_get_loop.assert_called_once()
            mock_run_in_executor.assert_called_once_with(None, sync_fn, "test_data")
            assert result == "modern_test"

    @pytest.mark.asyncio
    async def test_context_manager_resource_management(self):
        """Test proper resource cleanup in context manager."""
        processor = self.create_async_processor()

        async with processor.use() as p:
            assert isinstance(p, AsyncProcessor)

        assert processor.unload_called

    def test_sync_compatibility_bridge(self):
        """Test synchronous compatibility method."""
        processor = self.create_async_processor("sync_test")

        result = processor.execute_sync("test_data")
        assert result == "sync_test"

    @pytest.mark.asyncio
    async def test_safe_async_execute_success(self):
        """Test safe async execution on success."""
        processor = self.create_async_processor("safe_test")

        result, error = await safe_async_execute(processor, "test_data")

        assert result == "safe_test"
        assert error is None

    @pytest.mark.asyncio
    async def test_safe_async_execute_failure(self):
        """Test safe async execution on failure."""
        processor = self.create_async_processor(should_fail=True)

        result, error = await safe_async_execute(processor, "test_data")

        assert result is None
        assert error is not None
        assert error["type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_run_in_executor_uses_running_loop(self):
        """Test run_in_executor uses get_running_loop."""

        def sync_func(x):
            return f"processed_{x}"

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = AsyncMock()
            mock_run_in_executor = AsyncMock()
            mock_run_in_executor.return_value = "processed_test"
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            result = await run_in_executor(sync_func, "test")

            mock_get_loop.assert_called_once()
            mock_run_in_executor.assert_called_once_with(None, sync_func, "test")
            assert result == "processed_test"

    @pytest.mark.asyncio
    async def test_processor_session_context_manager(self):
        """Test processor_session context manager."""
        from src.processors.async_base import processor_session

        processor = self.create_async_processor("session_test")

        async with processor_session(processor) as p:
            assert p.setup_called
            result = await p.execute("test")

        assert result == "session_test"
        assert processor.unload_called

    @pytest.mark.asyncio
    async def test_processor_session_error_handling(self):
        """Test processor_session handles errors properly."""
        from src.processors.async_base import processor_session

        processor = self.create_async_processor(should_fail=True)

        with pytest.raises(ValueError, match="Test execution failure"):
            async with processor_session(processor) as p:
                await p.execute("test")

        # Should still call cleanup
        assert processor.unload_called

    def test_no_running_loop_error(self):
        """Test proper error when no loop is running."""
        self.create_async_processor()

        with pytest.raises(RuntimeError):
            # Should fail if called outside async context
            asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_async_processor_abstract_methods(self):
        """Test that AsyncProcessor enforces abstract methods."""
        # This should fail to instantiate because execute and unload are abstract
        with pytest.raises(TypeError):
            AsyncProcessor()

    @pytest.mark.asyncio
    async def test_reduce_memory_default_implementation(self):
        """Test that reduce_memory has a default no-op implementation."""
        processor = self.create_async_processor()

        # Should not raise any exceptions
        await processor.reduce_memory()

    @pytest.mark.asyncio
    async def test_cleanup_on_error_default_implementation(self):
        """Test cleanup_on_error default implementation."""
        processor = self.create_async_processor()

        # Should not raise any exceptions
        await processor.cleanup_on_error(None)
        await processor.cleanup_on_error(ValueError("test"))

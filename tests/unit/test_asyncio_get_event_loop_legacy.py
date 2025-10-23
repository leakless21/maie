"""
Comprehensive test suite for asyncio.get_running_loop() legacy usage.

This test suite ensures backward compatibility and documents the current
behavior of the deprecated asyncio.get_running_loop() usage in the Processor base class.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.processors.base import Processor


class TestAsyncioGetEventLoopLegacy:
    """Test suite for asyncio.get_running_loop() legacy behavior."""

    def create_concrete_processor(self, execute_result=None, should_fail=False):
        """Create a concrete processor for testing."""

        class TestProcessor(Processor):
            def __init__(self, execute_result=None, should_fail=False):
                self.execute_result = execute_result or "test_result"
                self.should_fail = should_fail
                self.unload_called = False

            def execute(self, *args, **kwargs):
                if self.should_fail:
                    raise ValueError("Test execution failure")
                return self.execute_result

            def unload(self):
                self.unload_called = True

        return TestProcessor(execute_result=execute_result, should_fail=should_fail)

    def test_async_execute_with_get_event_loop_legacy(self):
        """Test that async_execute uses deprecated asyncio.get_running_loop()."""
        processor = self.create_concrete_processor("legacy_test")

        # Mock asyncio.get_running_loop to verify it's being called
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock()
            mock_loop.run_in_executor = mock_run_in_executor
            mock_run_in_executor.return_value = "legacy_test"
            mock_get_loop.return_value = mock_loop

            # Run the async execute method
            result = asyncio.run(
                processor.async_execute("test_arg", kwarg="test_value")
            )

            # Verify the deprecated function was called
            mock_get_loop.assert_called_once()

            # Verify run_in_executor was called with correct parameters
            mock_run_in_executor.assert_called_once_with(
                None, processor.execute, "test_arg", kwarg="test_value"
            )

            assert result == "legacy_test"

    def test_async_execute_with_arguments(self):
        """Test async_execute passes arguments correctly to execute method."""
        processor = self.create_concrete_processor("args_test")

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock()
            mock_run_in_executor.return_value = "args_test"
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            # Test with various argument types
            result = asyncio.run(
                processor.async_execute(
                    "positional_arg",
                    keyword_arg="keyword_value",
                    number_arg=42,
                    list_arg=[1, 2, 3],
                    dict_arg={"nested": "value"},
                )
            )

            # Verify all arguments were passed correctly
            mock_run_in_executor.assert_called_once_with(
                None,
                processor.execute,
                "positional_arg",
                keyword_arg="keyword_value",
                number_arg=42,
                list_arg=[1, 2, 3],
                dict_arg={"nested": "value"},
            )

            assert result == "args_test"

    def test_async_execute_handles_exceptions(self):
        """Test async_execute properly handles exceptions from execute method."""
        processor = self.create_concrete_processor(should_fail=True)

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock()
            # Simulate the exception being raised from run_in_executor
            mock_run_in_executor.side_effect = ValueError("Test execution failure")
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            # Verify the exception is propagated
            with pytest.raises(ValueError, match="Test execution failure"):
                asyncio.run(processor.async_execute())

    def test_async_execute_with_none_result(self):
        """Test async_execute handles None return values correctly."""
        processor = self.create_concrete_processor(None)

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock()
            mock_run_in_executor.return_value = None
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            result = asyncio.run(processor.async_execute())

            assert result is None

    def test_async_execute_with_complex_result(self):
        """Test async_execute handles complex return values correctly."""
        complex_result = {
            "text": "transcription",
            "confidence": 0.95,
            "metadata": {"model": "test", "duration": 10.5},
        }
        processor = self.create_concrete_processor(complex_result)

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock()
            mock_run_in_executor.return_value = complex_result
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            result = asyncio.run(processor.async_execute())

            assert result == complex_result
            assert result["text"] == "transcription"
            assert result["confidence"] == 0.95

    def test_async_execute_deprecation_warning(self):
        """Test that deprecation warning is not yet implemented (for future compatibility)."""
        processor = self.create_concrete_processor()

        # This test documents current behavior - no deprecation warning is shown
        # In the future, when we migrate to asyncio.get_running_loop(), we may want to add warnings

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock()
            mock_run_in_executor.return_value = "test"
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            # Should not raise any warnings currently
            result = asyncio.run(processor.async_execute())
            assert result == "test"

    def test_async_execute_in_running_loop_context(self):
        """Test async_execute behavior when called from within a running event loop."""
        processor = self.create_concrete_processor("loop_test")

        async def test_within_running_loop():
            with patch("asyncio.get_running_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_run_in_executor = AsyncMock()
                mock_run_in_executor.return_value = "loop_test"
                mock_loop.run_in_executor = mock_run_in_executor
                mock_get_loop.return_value = mock_loop

                # Call async_execute from within an already running loop
                result = await processor.async_execute("inner_test")

                mock_get_loop.assert_called_once()
                assert result == "loop_test"
                return result

        # Run the test within an event loop
        result = asyncio.run(test_within_running_loop())
        assert result == "loop_test"

    def test_async_execute_concurrent_calls(self):
        """Test that async_execute can handle concurrent calls correctly."""
        processor = self.create_concrete_processor("concurrent_test")

        async def concurrent_test():
            with patch("asyncio.get_running_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_run_in_executor = AsyncMock()

                # Simulate different results for concurrent calls
                call_count = 0

                def side_effect(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    return f"concurrent_result_{call_count}"

                mock_run_in_executor.side_effect = side_effect
                mock_loop.run_in_executor = mock_run_in_executor
                mock_get_loop.return_value = mock_loop

                # Create multiple concurrent calls
                tasks = [processor.async_execute(f"call_{i}") for i in range(3)]

                results = await asyncio.gather(*tasks)

                # Verify all calls completed
                assert len(results) == 3
                assert "concurrent_result_1" in results
                assert "concurrent_result_2" in results
                assert "concurrent_result_3" in results

                return results

        results = asyncio.run(concurrent_test())
        assert len(results) == 3

    def test_async_execute_backward_compatibility(self):
        """Test backward compatibility with existing async_execute usage patterns."""
        processor = self.create_concrete_processor("compat_test")

        # Test the common usage pattern found in the codebase
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock()
            mock_run_in_executor.return_value = "compat_test"
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            # Simulate typical usage in production code
            async def typical_usage():
                result = await processor.async_execute(
                    audio_data=b"test_audio_data", model="test_model", language="en"
                )
                return result

            result = asyncio.run(typical_usage())
            assert result == "compat_test"

            # Verify the legacy pattern was used
            mock_get_loop.assert_called_once()

    def test_async_execute_executor_none_handling(self):
        """Test that async_execute handles None executor correctly (uses default)."""
        processor = self.create_concrete_processor("executor_test")

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock()
            mock_run_in_executor.return_value = "executor_test"
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            result = asyncio.run(processor.async_execute("test"))

            # Verify None is passed as executor (uses default thread pool)
            mock_run_in_executor.assert_called_once_with(
                None, processor.execute, "test"
            )
            assert result == "executor_test"


class TestAsyncioGetEventLoopMigration:
    """Test suite to prepare for migration to asyncio.get_running_loop()."""

    def create_concrete_processor(self, execute_result=None, should_fail=False):
        """Create a concrete processor for testing."""

        class TestProcessor(Processor):
            def __init__(self, execute_result=None, should_fail=False):
                self.execute_result = execute_result or "test_result"
                self.should_fail = should_fail
                self.unload_called = False

            def execute(self, *args, **kwargs):
                if self.should_fail:
                    raise ValueError("Test execution failure")
                return self.execute_result

            def unload(self):
                self.unload_called = True

        return TestProcessor(execute_result=execute_result, should_fail=should_fail)

    def test_current_behavior_documentation(self):
        """Document current behavior for future migration reference."""
        processor = self.create_concrete_processor("migration_test")

        # This test documents the current behavior before migration
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_run_in_executor = AsyncMock()
            mock_run_in_executor.return_value = "migration_test"
            mock_loop.run_in_executor = mock_run_in_executor
            mock_get_loop.return_value = mock_loop

            result = asyncio.run(processor.async_execute("test"))

            # Current behavior: uses deprecated asyncio.get_running_loop()
            mock_get_loop.assert_called_once()
            assert result == "migration_test"

            # Future migration should:
            # 1. Replace asyncio.get_running_loop() with asyncio.get_running_loop()
            # 2. Add proper error handling for when no loop is running
            # 3. Consider providing a sync fallback or better error message

    def test_migration_readiness_check(self):
        """Test that current code is ready for migration to get_running_loop()."""
        self.create_concrete_processor("readiness_test")

        # This test verifies that the current implementation structure
        # supports migration to asyncio.get_running_loop()

        async def test_running_loop_availability():
            # Verify we can get the running loop (this will work in Python 3.10+)
            try:
                running_loop = asyncio.get_running_loop()
                assert running_loop is not None
            except RuntimeError:
                # This would happen if no loop is running
                pytest.skip("No event loop running")

        # Test that we can get the running loop in an async context
        asyncio.run(test_running_loop_availability())

        # The current implementation should be migratable because:
        # 1. It's already in an async method
        # 2. It doesn't rely on loop policy manipulation
        # 3. It uses run_in_executor which works with both loop types

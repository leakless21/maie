"""
Tests for modern error handling patterns including context managers and native error leveraging.

This test suite verifies that the new error handling system provides:
- Proper exception chaining with native error preservation
- Context manager-based resource cleanup
- Semantic error wrapping with debugging context
"""

import pytest
from unittest.mock import MagicMock, patch

from src.api.errors import (
    ModelLoadError,
    NetworkError,
    FileSystemError,
    ProcessingError,
)
from src.core.error_handler import (
    ProcessingErrorHandler,
    GpuModelResource,
    RedisConnectionResource,
    leverage_native_error,
)


class TestLeverageNativeError:
    """Test native error leveraging patterns."""

    def test_leverage_torch_native_error(self):
        """Test that PyTorch native errors are properly leveraged."""
        native_error = RuntimeError("CUDA out of memory")

        with pytest.raises(ModelLoadError) as exc_info:
            leverage_native_error(
                native_error,
                "GPU memory insufficient for model loading",
                ModelLoadError,
                model_path="test_model.pt",
            )

        # Verify semantic error was raised
        assert exc_info.value.error_code == "MODEL_LOAD_ERROR"
        assert "GPU memory insufficient" in str(exc_info.value)

        # Verify native error is chained
        assert exc_info.value.__cause__ is native_error
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "CUDA out of memory" in str(exc_info.value.__cause__)

        # Verify context is preserved
        assert exc_info.value.details["model_path"] == "test_model.pt"
        assert exc_info.value.details["native_error_type"] == "RuntimeError"

    def test_leverage_redis_native_error(self):
        """Test that Redis native errors are properly leveraged."""
        native_error = ConnectionError("Connection refused")

        with pytest.raises(NetworkError) as exc_info:
            leverage_native_error(
                native_error,
                "Redis server unavailable",
                NetworkError,
                host="localhost",
                port=6379,
            )

        # Verify semantic error was raised
        assert exc_info.value.error_code == "NETWORK_ERROR"
        assert "Redis server unavailable" in str(exc_info.value)

        # Verify native error is chained
        assert exc_info.value.__cause__ is native_error
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    def test_leverage_filesystem_native_error(self):
        """Test that filesystem native errors are properly leveraged."""
        native_error = FileNotFoundError("No such file or directory")

        with pytest.raises(FileSystemError) as exc_info:
            leverage_native_error(
                native_error,
                "Audio file not found",
                FileSystemError,
                file_path="/path/to/audio.wav",
            )

        # Verify semantic error was raised
        assert exc_info.value.error_code == "FILE_SYSTEM_ERROR"
        assert "Audio file not found" in str(exc_info.value)

        # Verify native error is chained
        assert exc_info.value.__cause__ is native_error
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)


class TestProcessingErrorHandler:
    """Test context manager-based error handling."""

    def test_successful_operation_cleanup(self):
        """Test that resources are cleaned up on successful operations."""
        mock_redis = MagicMock()
        mock_resource = MagicMock()

        with ProcessingErrorHandler(
            redis_client=mock_redis, task_key="task:123", stage="test_stage"
        ) as handler:
            handler.register_resource(mock_resource)

        # Verify cleanup was called
        mock_resource.cleanup.assert_called_once()

    def test_error_operation_cleanup(self):
        """Test that resources are cleaned up even when errors occur."""
        mock_redis = MagicMock()
        mock_resource = MagicMock()

        with pytest.raises(ValueError):
            with ProcessingErrorHandler(
                redis_client=mock_redis, task_key="task:123", stage="test_stage"
            ) as handler:
                handler.register_resource(mock_resource)
                raise ValueError("Test error")

        # Verify cleanup was called despite error
        mock_resource.cleanup.assert_called_once()

    def test_redis_status_update_on_error(self):
        """Test that Redis status is updated when errors occur."""
        mock_redis = MagicMock()

        with pytest.raises(ValueError):
            with ProcessingErrorHandler(
                redis_client=mock_redis, task_key="task:123", stage="test_stage"
            ):
                raise ValueError("Test error")

        # Verify Redis was updated with FAILED status
        mock_redis.hset.assert_called()

    def test_redis_status_update_on_success(self):
        """Test that Redis status is updated on success."""
        mock_redis = MagicMock()

        with ProcessingErrorHandler(
            redis_client=mock_redis, task_key="task:123", stage="test_stage"
        ):
            pass  # Successful operation

        # Verify Redis was updated with COMPLETED status
        mock_redis.hset.assert_called()


class TestResourceCleanup:
    """Test resource cleanup implementations."""

    def test_gpu_model_resource_cleanup(self):
        """Test GPU model resource cleanup."""
        mock_model = MagicMock()
        resource = GpuModelResource(mock_model, "test_model")

        with patch("src.core.error_handler.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = MagicMock()

            resource.cleanup()

            mock_torch.cuda.empty_cache.assert_called_once()

    def test_redis_connection_resource_cleanup(self):
        """Test Redis connection resource cleanup."""
        mock_redis = MagicMock()
        resource = RedisConnectionResource(mock_redis)

        resource.cleanup()

        mock_redis.close.assert_called_once()

    def test_context_manager_protocol(self):
        """Test that resources support context manager protocol."""
        mock_model = MagicMock()
        resource = GpuModelResource(mock_model, "test_model")

        with resource as r:
            assert r is resource

        # Cleanup should be called automatically
        mock_model.cleanup.assert_called_once()


class TestErrorChaining:
    """Test exception chaining for better debugging."""

    def test_exception_chaining_preserves_context(self):
        """Test that exception chaining preserves original error context."""
        try:
            try:
                raise OSError("Permission denied")
            except OSError as e:
                leverage_native_error(
                    e, "File access failed", FileSystemError, path="/test/file"
                )
        except FileSystemError as final_error:
            # Verify the chain
            assert isinstance(final_error.__cause__, OSError)
            assert str(final_error.__cause__) == "Permission denied"
            assert "File access failed" in str(final_error)

    def test_multiple_level_chaining(self):
        """Test multiple levels of exception chaining."""
        try:
            try:
                raise ConnectionError("Network timeout")
            except ConnectionError as e:
                leverage_native_error(
                    e, "Service unavailable", NetworkError, service="api"
                )
        except NetworkError as e:
            try:
                leverage_native_error(
                    e, "Processing failed", ProcessingError, operation="test"
                )
            except ProcessingError as final_error:
                # Verify the chain: ProcessingError -> NetworkError -> ConnectionError
                assert isinstance(final_error.__cause__, NetworkError)
                assert isinstance(final_error.__cause__.__cause__, ConnectionError)
                assert "Network timeout" in str(final_error.__cause__.__cause__)


class TestErrorContextPreservation:
    """Test that error context is properly preserved and enhanced."""

    def test_error_details_enhancement(self):
        """Test that error details are properly enhanced."""
        native_error = ValueError("Invalid input")

        with pytest.raises(ProcessingError) as exc_info:
            leverage_native_error(
                native_error,
                "Input validation failed",
                ProcessingError,
                input_field="email",
                input_value="invalid-email",
            )

        details = exc_info.value.details
        assert details["native_error_type"] == "ValueError"
        assert details["native_error_message"] == "Invalid input"
        assert details["input_field"] == "email"
        assert details["input_value"] == "invalid-email"

    def test_structured_logging_context(self):
        """Test that structured logging includes proper context."""
        mock_redis = MagicMock()

        with patch("src.core.error_handler.logger") as mock_logger:
            with pytest.raises(ValueError):
                with ProcessingErrorHandler(
                    redis_client=mock_redis, task_key="task:123", stage="test_stage"
                ):
                    raise ValueError("Test error")

            # Verify structured error logging
            mock_logger.error.assert_called()
            call_args = mock_logger.error.call_args
            assert "stage" in call_args[1]
            assert "task_key" in call_args[1]
            assert "error_type" in call_args[1]
            assert call_args[1]["stage"] == "test_stage"
            assert call_args[1]["task_key"] == "task:123"

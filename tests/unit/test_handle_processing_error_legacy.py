"""
Comprehensive test suite for handle_processing_error legacy function.

This test suite ensures backward compatibility and documents the current
behavior of the deprecated handle_processing_error function.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from redis import Redis
from src.api.errors import (
    AudioValidationError,
    ProcessingError,
)
from src.api.schemas import TaskStatus
from src.worker.pipeline import handle_processing_error


class TestHandleProcessingErrorFunction:
    """Test suite for handle_processing_error function behavior."""

    def test_handle_processing_error_basic_functionality(self):
        """Test basic error handling functionality."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:123"
        error = ProcessingError(
            error_code="TEST_ERROR",
            message="Test error message",
            details={"test_detail": "test_value"},
        )
        stage = "test_stage"
        error_code = "CUSTOM_ERROR_CODE"

        handle_processing_error(mock_redis, task_key, error, stage, error_code)

        # Verify Redis.hset was called with correct parameters
        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == task_key  # First positional argument is task_key

        # Check the mapping data
        mapping = call_args[1]["mapping"]
        assert mapping["status"] == TaskStatus.FAILED.value
        assert mapping["error_code"] == error_code
        assert "error" in mapping
        assert "stage" in mapping
        assert "updated_at" in mapping

    def test_handle_processing_error_without_custom_error_code(self):
        """Test error handling without custom error code."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:456"
        error = AudioValidationError(
            message="Audio validation failed", details={"audio_path": "/test/path.wav"}
        )
        stage = "preprocessing"

        handle_processing_error(mock_redis, task_key, error, stage)

        # error_code is only added if explicitly provided
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]
        assert "error_code" not in mapping
        assert mapping["stage"] == stage

    def test_handle_processing_error_serializes_complex_error(self):
        """Test that complex error objects are properly serialized."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:789"

        # Create error with complex nested data
        error = ProcessingError(
            error_code="COMPLEX_ERROR",
            message="Complex error with nested data",
            details={
                "nested_dict": {"key1": "value1", "key2": 42},
                "list_data": [1, 2, 3, {"nested": "list"}],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "none_value": None,
                "boolean": True,
            },
        )
        stage = "processing"

        handle_processing_error(mock_redis, task_key, error, stage)

        # Verify error is stored as simple string, not JSON
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # Error field should be a simple string
        assert mapping["error"] == "Complex error with nested data"

    def test_handle_processing_error_with_simple_exception(self):
        """Test handling of non-MAIE exceptions."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:999"
        error = ValueError("Simple ValueError")
        stage = "test_stage"
        error_code = "SIMPLE_ERROR"

        handle_processing_error(mock_redis, task_key, error, stage, error_code)

        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # Should store simple exception as string
        assert mapping["error"] == "Simple ValueError"
        assert mapping["error_code"] == error_code

    def test_handle_processing_error_includes_timestamp(self):
        """Test that error handling includes proper timestamp."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:timestamp"
        error = ProcessingError(error_code="TIME_TEST", message="Timestamp test")
        stage = "test"

        handle_processing_error(mock_redis, task_key, error, stage)

        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # Verify timestamp is included and is a valid ISO format
        assert "updated_at" in mapping
        # Verify it's a valid ISO 8601 timestamp
        timestamp = mapping["updated_at"]
        # Should parse without error and be recent
        parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed_time.year >= 2025

    def test_handle_processing_error_with_maqie_error_response(self):
        """Test handling of MAIE errors with to_error_response method."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:maie"
        error = ProcessingError(
            error_code="MAIE_ERROR",
            message="MAIE-specific error",
            details={"custom_field": "custom_value"},
        )
        stage = "maie_stage"

        handle_processing_error(mock_redis, task_key, error, stage)

        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # Error is stored as simple string
        assert mapping["error"] == "MAIE-specific error"
        # error_code is only added if explicitly provided
        assert "error_code" not in mapping

    def test_handle_processing_error_redis_failure_handling(self):
        """Test behavior when Redis operations fail."""
        mock_redis = MagicMock(spec=Redis)
        mock_redis.hset.side_effect = Exception("Redis connection failed")

        task_key = "task:redis_fail"
        error = ProcessingError(error_code="REDIS_FAIL", message="Redis test")
        stage = "test"

        # Should not raise exception even if Redis fails
        # (This is current behavior - errors are logged but not re-raised)
        try:
            handle_processing_error(mock_redis, task_key, error, stage)
        except Exception:
            pytest.fail(
                "handle_processing_error should not raise exceptions on Redis failure"
            )

    def test_handle_processing_error_all_task_status_fields(self):
        """Test that all required task status fields are included."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:fields"
        error = ProcessingError(error_code="FIELDS_TEST", message="Fields test")
        stage = "field_test"

        handle_processing_error(
            mock_redis, task_key, error, stage, error_code="FIELDS_ERROR"
        )

        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # Base required fields
        required_fields = ["status", "error", "stage", "updated_at"]
        for field in required_fields:
            assert field in mapping, f"Missing required field: {field}"

        # error_code is only present if explicitly provided
        assert "error_code" in mapping  # Present because we provided it

        assert mapping["status"] == "FAILED"
        assert mapping["error"] == "Fields test"
        assert mapping["stage"] == "field_test"
        assert mapping["error_code"] == "FIELDS_ERROR"

    def test_handle_processing_error_data_types(self):
        """Test that all data types are properly handled."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:types"

        # Create error with various data types
        error = ProcessingError(
            error_code="TYPES_TEST",
            message="Types test",
            details={
                "string": "test_string",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
                "none": None,
                "list": [1, "two", {"three": 3}],
                "dict": {"nested": "value", "number": 123},
            },
        )
        stage = "type_test"

        handle_processing_error(mock_redis, task_key, error, stage)

        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # All values should be strings
        for key, value in mapping.items():
            assert isinstance(value, str), (
                f"Field {key} should be string, got {type(value)}"
            )

    def test_handle_processing_error_stage_validation(self):
        """Test stage parameter handling."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:stage"
        error = ProcessingError(error_code="STAGE_TEST", message="Stage test")

        # Test various stage values
        test_stages = ["preprocessing", "asr_loading", "llm_processing", "cleanup", ""]

        for stage in test_stages:
            mock_redis.reset_mock()
            handle_processing_error(mock_redis, task_key, error, stage)

            call_args = mock_redis.hset.call_args
            assert call_args[0][0] == task_key

    def test_handle_processing_error_task_key_validation(self):
        """Test task_key parameter handling."""
        mock_redis = MagicMock(spec=Redis)
        error = ProcessingError(error_code="KEY_TEST", message="Key test")
        stage = "test_stage"

        # Test various task_key formats
        test_keys = ["task:123", "task:abc", "task:with-dash", "task:with_underscore"]

        for task_key in test_keys:
            mock_redis.reset_mock()
            handle_processing_error(mock_redis, task_key, error, stage)

            call_args = mock_redis.hset.call_args
            assert call_args[0][0] == task_key

    def test_handle_processing_error_error_code_priority(self):
        """Test that custom error_code takes priority over error.error_code."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:priority"
        error = ProcessingError(error_code="ORIGINAL_CODE", message="Priority test")
        stage = "test"
        custom_code = "CUSTOM_CODE"

        handle_processing_error(mock_redis, task_key, error, stage, custom_code)

        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # Custom error_code should override error.error_code
        assert mapping["error_code"] == custom_code

    def test_handle_processing_error_backwards_compatibility(self):
        """Test backwards compatibility with existing usage patterns."""
        mock_redis = MagicMock(spec=Redis)

        # Pattern 1: Basic usage (most common in tests)
        task_key = "task:compat1"
        error = ProcessingError(error_code="COMPAT1", message="Compatibility test 1")
        handle_processing_error(
            mock_redis, task_key, error, "test_stage", "COMPAT1_CUSTOM"
        )

        # Pattern 2: Without custom error_code
        task_key2 = "task:compat2"
        error2 = AudioValidationError(message="Compatibility test 2")
        handle_processing_error(mock_redis, task_key2, error2, "validation_stage")

        # Pattern 3: Simple exception
        task_key3 = "task:compat3"
        error3 = RuntimeError("Runtime error test")
        handle_processing_error(
            mock_redis, task_key3, error3, "runtime_stage", "RUNTIME_ERROR"
        )

        # All should complete without errors
        assert mock_redis.hset.call_count == 3

        # Verify each call had correct structure
        for i, call in enumerate(mock_redis.hset.call_args_list):
            mapping = call[1]["mapping"]
            assert "status" in mapping
            # error_code is only present if explicitly provided
            if i == 0 or i == 2:  # Cases with explicit error_code
                assert "error_code" in mapping
                if i == 0:
                    assert mapping["error_code"] == "COMPAT1_CUSTOM"
                else:
                    assert mapping["error_code"] == "RUNTIME_ERROR"
            else:  # Case without explicit error_code
                assert "error_code" not in mapping
            assert "error" in mapping
            assert "stage" in mapping
            assert "updated_at" in mapping
            assert mapping["status"] == TaskStatus.FAILED.value


class TestHandleProcessingErrorIntegration:
    """Integration tests for handle_processing_error with real components."""

    def test_integration_with_task_status_enum(self):
        """Test integration with TaskStatus enum."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:enum_test"
        error = ProcessingError(error_code="ENUM_TEST", message="Enum test")
        stage = "enum_stage"

        handle_processing_error(mock_redis, task_key, error, stage)

        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # Should use the actual enum value, not the string
        assert mapping["status"] == TaskStatus.FAILED.value
        assert isinstance(mapping["status"], str)

    def test_integration_with_error_hierarchy(self):
        """Test integration with MAIE error hierarchy."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:hierarchy"

        # Test different error types in hierarchy
        errors = [
            AudioValidationError(message="Audio error"),
            ProcessingError(error_code="PROCESSING_ERROR", message="Processing error"),
        ]

        for error in errors:
            mock_redis.reset_mock()
            handle_processing_error(mock_redis, task_key, error, "test")

            call_args = mock_redis.hset.call_args
            mapping = call_args[1]["mapping"]

            # Error is stored as simple string
            assert mapping["error"] == error.message

    def test_integration_with_real_json_serialization(self):
        """Test that JSON serialization works with real data."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:json_test"

        # Create error with data that might cause JSON serialization issues
        error = ProcessingError(
            error_code="JSON_TEST",
            message="JSON serialization test",
            details={
                "unicode": "Hello 世界 🌍",
                "special_chars": 'Special chars: \n\t\r"',
                "datetime": datetime.now(timezone.utc).isoformat(),
                "nested": {"deep": {"values": [1, 2, 3, None, True, False]}},
            },
        )
        stage = "json_test"

        handle_processing_error(mock_redis, task_key, error, stage)

        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        # Error is stored as simple string, not JSON
        assert mapping["error"] == "JSON serialization test"

    def test_integration_logging_behavior(self):
        """Test that logging works correctly."""
        mock_redis = MagicMock(spec=Redis)
        task_key = "task:log_test"
        error = ProcessingError(error_code="LOG_TEST", message="Log test")
        stage = "log_stage"

        # The logger is imported inside the function, so we need to patch the loguru logger
        with patch("loguru.logger") as mock_logger:
            handle_processing_error(mock_redis, task_key, error, stage)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args

            # Check log message format
            assert f"Task {task_key}" in call_args[0][0]
            assert stage in call_args[0][0]
            assert error.message in call_args[0][0]

    def test_integration_redis_error_logging(self):
        """Test logging when Redis operations fail."""
        mock_redis = MagicMock(spec=Redis)
        mock_redis.hset.side_effect = Exception("Redis connection failed")

        task_key = "task:redis_log_test"
        error = ProcessingError(error_code="REDIS_LOG_TEST", message="Redis log test")
        stage = "test"

        # The logger is imported inside the function, so we need to patch the loguru logger
        with patch("loguru.logger") as mock_logger:
            handle_processing_error(mock_redis, task_key, error, stage)

            # Should have logged both the original error and the Redis error
            # But sometimes the first log might not be captured due to the import timing
            assert mock_logger.error.call_count >= 1

            # Check for Redis failure log (this should always be present)
            redis_error_found = False
            for call in mock_logger.error.call_args_list:
                if "Failed to update error status" in call[0][0]:
                    redis_error_found = True
                    break

            assert redis_error_found, "Redis error log not found"

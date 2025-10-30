"""Tests for the logging_utils module."""

import json
from jsonschema.exceptions import ValidationError
from src.utils.logging_utils import (
    log_validation_error,
    log_json_parse_error,
    create_error_summary,
    bind_request_context,
    log_performance_metrics,
    log_api_request,
    log_processing_step,
    log_error_with_context,
    format_log_message,
    log_config_change,
    log_security_event,
)


class TestLogValidationError:
    """Tests for log_validation_error function."""

    def test_log_validation_error(self, caplog):
        """Test log_validation_error function."""
        # This function just logs, so we're testing that it doesn't raise an error
        from jsonschema import validate

        schema = {"type": "string"}
        try:
            validate(instance=123, schema=schema)  # This will raise ValidationError
        except ValidationError as e:
            log_validation_error(e, {"template_id": "test"})
            # Just make sure it runs without error


class TestLogJsonParseError:
    """Tests for log_json_parse_error function."""

    def test_log_json_parse_error(self):
        """Test log_json_parse_error function."""
        try:
            json.loads("invalid json")  # This will raise JSONDecodeError
        except json.JSONDecodeError as e:
            log_json_parse_error(e, {"context": "test"})
            # Just make sure it runs without error


class TestCreateErrorSummary:
    """Tests for create_error_summary function."""

    def test_create_error_summary_basic(self):
        """Test create_error_summary with basic error."""
        error = ValueError("Test error")
        summary = create_error_summary(error, {"field": "name"})
        assert summary["error_type"] == "ValueError"
        assert summary["error_message"] == "Test error"
        assert summary["error_context"]["field"] == "name"
        assert "traceback" in summary

    def test_create_error_summary_validation_error(self):
        """Test create_error_summary with ValidationError."""
        try:
            from jsonschema import validate

            validate(instance=123, schema={"type": "string"})
        except ValidationError as e:
            summary = create_error_summary(e, {"context": "validation"})
            assert summary["error_type"] == "ValidationError"
            assert "error_path" in summary


class TestBindRequestContext:
    """Tests for bind_request_context function."""

    def test_bind_request_context(self):
        """Test bind_request_context function."""
        # This function is a placeholder for structured logging context binding
        bind_request_context("req-123", user_id="user-456")
        # Just make sure it runs without error


class TestLogPerformanceMetrics:
    """Tests for log_performance_metrics function."""

    def test_log_performance_metrics(self):
        """Test log_performance_metrics function."""
        log_performance_metrics("test_operation", 0.1, items_processed=100)
        # Just make sure it runs without error


class TestLogApiRequest:
    """Tests for log_api_request function."""

    def test_log_api_request(self):
        """Test log_api_request function."""
        log_api_request("req-123", "GET", "/test", 200, 0.05, user_id="user-456")
        # Just make sure it runs without error


class TestLogProcessingStep:
    """Tests for log_processing_step function."""

    def test_log_processing_step(self):
        """Test log_processing_step function."""
        log_processing_step("validation", "success", {"step": "test"}, duration=0.01)
        # Just make sure it runs without error


class TestLogErrorWithContext:
    """Tests for log_error_with_context function."""

    def test_log_error_with_context(self):
        """Test log_error_with_context function."""
        error = RuntimeError("Test runtime error")
        log_error_with_context(
            print, error, {"component": "test"}, extra_info={"details": "more info"}
        )
        # Just make sure it runs without error


class TestFormatLogMessage:
    """Tests for format_log_message function."""

    def test_format_log_message_basic(self):
        """Test format_log_message with basic message."""
        result = format_log_message("Test message")
        assert result == "Test message"

    def test_format_log_message_with_data(self):
        """Test format_log_message with additional data."""
        result = format_log_message("Test message", key="value", count=42)
        assert "Test message" in result
        assert "key" in result
        assert "value" in result


class TestLogConfigChange:
    """Tests for log_config_change function."""

    def test_log_config_change(self):
        """Test log_config_change function."""
        log_config_change("max_workers", 4, 8, user="admin")
        # Just make sure it runs without error


class TestLogSecurityEvent:
    """Tests for log_security_event function."""

    def test_log_security_event(self):
        """Test log_security_event function."""
        log_security_event(
            "login_attempt", "high", {"user_id": "123"}, ip_address="192.168.1.1"
        )
        # Just make sure it runs without error

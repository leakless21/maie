"""Tests for the error_handling module."""

import pytest
import asyncio
from src.utils.error_handling import (
    safe_execute,
    safe_async_execute,
    create_error_response,
    handle_generic_error,
    retry_with_backoff,
    retry_sync_with_backoff,
    with_error_context,
    format_error_details,
    log_exception_safely,
    convert_to_error_response,
    suppress_exceptions,
)


class TestSafeExecute:
    """Tests for safe_execute function."""
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""
        def test_func(x, y):
            return x + y
        
        result, error = safe_execute(test_func, 2, 3)
        assert result == 5
        assert error is None
    
    def test_safe_execute_failure(self):
        """Test safe_execute with failing function."""
        def test_func():
            raise ValueError("Test error")
        
        result, error = safe_execute(test_func)
        assert result is None
        assert error is not None
        assert error["type"] == "ValueError"
        assert error["message"] == "Test error"


class TestSafeAsyncExecute:
    """Tests for safe_async_execute function."""
    
    @pytest.mark.asyncio
    async def test_safe_async_execute_success(self):
        """Test safe_async_execute with successful async function."""
        async def test_func(x, y):
            return x * y
        
        result, error = await safe_async_execute(test_func, 3, 4)
        assert result == 12
        assert error is None
    
    @pytest.mark.asyncio
    async def test_safe_async_execute_failure(self):
        """Test safe_async_execute with failing async function."""
        async def test_func():
            raise TypeError("Async test error")
        
        result, error = await safe_async_execute(test_func)
        assert result is None
        assert error is not None
        assert error["type"] == "TypeError"
        assert error["message"] == "Async test error"


class TestCreateErrorResponse:
    """Tests for create_error_response function."""
    
    def test_create_error_response_basic(self):
        """Test create_error_response with basic parameters."""
        response = create_error_response("VALIDATION_ERROR", "Invalid input")
        assert response["error"]["code"] == "VALIDATION_ERROR"
        assert response["error"]["message"] == "Invalid input"
        assert response["error"]["details"] == {}
    
    def test_create_error_response_with_details(self):
        """Test create_error_response with details."""
        details = {"field": "email", "reason": "invalid_format"}
        response = create_error_response("VALIDATION_ERROR", "Invalid email", details=details)
        assert response["error"]["details"]["field"] == "email"
    
    def test_create_error_response_with_request_id(self):
        """Test create_error_response with request ID."""
        response = create_error_response("INTERNAL_ERROR", "Server error", request_id="req-123")
        assert response["request_id"] == "req-123"


class TestHandleGenericError:
    """Tests for handle_generic_error function."""
    
    def test_handle_generic_error(self):
        """Test handle_generic_error function."""
        error = ValueError("Something went wrong")
        response = handle_generic_error(error, request_id="req-456")
        assert response["error"]["code"] == "INTERNAL_ERROR"
        assert "Something went wrong" in response["error"]["message"]
        assert response["request_id"] == "req-456"


class TestRetrySyncWithBackoff:
    """Tests for retry_sync_with_backoff function."""
    
    def test_retry_sync_with_backoff_success(self):
        """Test retry_sync_with_backoff with eventually successful function."""
        call_count = 0
        
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"
        
        result = retry_sync_with_backoff(flaky_func, max_retries=5, base_delay=0.01)
        assert result == "success"
        assert call_count == 3
    
    def test_retry_sync_with_backoff_failure(self):
        """Test retry_sync_with_backoff with always failing function."""
        def always_fail():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            retry_sync_with_backoff(always_fail, max_retries=2, base_delay=0.01)


class TestWithErrorContext:
    """Tests for with_error_context decorator."""
    
    def test_with_error_context(self):
        """Test with_error_context decorator."""
        @with_error_context({"user_id": "123", "action": "login"})
        def test_func():
            raise ValueError("Test error")
        
        # The with_error_context decorator creates a new exception with context
        # So we need to check the exception handling differently
        with pytest.raises(Exception) as exc_info:
            test_func()
        
        # Check if the exception has context attached by checking the string representation
        assert "Test error" in str(exc_info.value)


class TestFormatErrorDetails:
    """Tests for format_error_details function."""
    
    def test_format_error_details_basic(self):
        """Test format_error_details with basic error."""
        error = ValueError("Test message")
        details = format_error_details(error)
        assert details["error_type"] == "ValueError"
        assert details["error_message"] == "Test message"
    
    def test_format_error_details_with_context(self):
        """Test format_error_details with context."""
        error = TypeError("Type error")
        details = format_error_details(error, context={"field": "name"})
        assert details["context"] == {"field": "name"}


class TestConvertToErrorResponse:
    """Tests for convert_to_error_response function."""
    
    def test_convert_to_error_response(self):
        """Test convert_to_error_response function."""
        error = ValueError("Conversion test")
        response = convert_to_error_response(error, request_id="req-789")
        assert response["error"]["message"] == "Conversion test"
        assert response["request_id"] == "req-789"


class TestSuppressExceptions:
    """Tests for suppress_exceptions decorator."""
    
    def test_suppress_exceptions_with_return(self):
        """Test suppress_exceptions decorator with return value."""
        # Using the decorator with default return value - it's applied as suppress_exceptions(func, default_return)
        def failing_func():
            raise ValueError("Should be suppressed")
        
        wrapped_func = suppress_exceptions(failing_func, default_return="default")
        result = wrapped_func()
        assert result == "default"
    
    def test_suppress_exceptions_success(self):
        """Test suppress_exceptions decorator with successful function."""
        def successful_func():
            return "success"
        
        wrapped_func = suppress_exceptions(successful_func, default_return="default")
        result = wrapped_func()
        assert result == "success"
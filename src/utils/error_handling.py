"""Error handling utilities for the MAIE project.

This module provides consistent error handling patterns with structured
error information and safe execution functions.
"""

import asyncio
import time
import functools
from typing import Any, Callable, Dict, Optional
from .types import ExecutionResult, ErrorContext, SafeCallable, AsyncSafeCallable


def safe_execute(func: SafeCallable, *args, **kwargs) -> ExecutionResult:
    """Execute function safely, returning result and structured error info.

    Args:
        func: Function to execute safely
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (result, error_info) where error_info is None if execution succeeds

    Examples:
        >>> def divide(a, b):
        ...     return a / b
        >>> safe_execute(divide, 10, 2)
        (5.0, None)
        >>> safe_execute(divide, 10, 0)
        (None, {'type': 'ZeroDivisionError', 'message': 'division by zero', 'args': (10, 0), 'kwargs': {}})
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "args": args,
            "kwargs": kwargs,
        }
        return None, error_info


async def safe_async_execute(
    func: AsyncSafeCallable, *args, **kwargs
) -> ExecutionResult:
    """Async version of safe_execute.

    Args:
        func: Async function to execute safely
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (result, error_info) where error_info is None if execution succeeds

    Examples:
        >>> async def async_divide(a, b):
        ...     return a / b
        >>> import asyncio
        >>> asyncio.run(safe_async_execute(async_divide, 10, 2))
        (5.0, None)
    """
    try:
        result = await func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "args": args,
            "kwargs": kwargs,
        }
        return None, error_info


def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create standardized error response format.

    Args:
        error_code: Standardized error code
        message: Human-readable error message
        details: Additional error details (optional)
        request_id: Request identifier for correlation (optional)

    Returns:
        Standardized error response dictionary

    Examples:
        >>> create_error_response("VALIDATION_ERROR", "Invalid input provided")
        {'error': {'code': 'VALIDATION_ERROR', 'message': 'Invalid input provided', 'details': {}}}
        >>> create_error_response("PROCESSING_ERROR", "Processing failed", details={"field": "email"})
        {'error': {'code': 'PROCESSING_ERROR', 'message': 'Processing failed', 'details': {'field': 'email'}}}
    """
    response: Dict[str, Any] = {
        "error": {"code": error_code, "message": message, "details": details or {}}
    }

    if request_id:
        # Add request_id at the top level, not nested inside error
        response["request_id"] = request_id

    return response


def handle_generic_error(
    error: Exception, request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Handle generic exceptions with standardized error response.

    Args:
        error: Exception to handle
        request_id: Request identifier for correlation (optional)

    Returns:
        Standardized error response dictionary
    """
    return create_error_response(
        "INTERNAL_ERROR",
        f"An internal error occurred: {str(error)}",
        details={"error_type": type(error).__name__, "error_message": str(error)},
        request_id=request_id,
    )


async def retry_with_backoff(
    func: AsyncSafeCallable, max_retries: int, base_delay: float, *args, **kwargs
) -> Any:
    """Execute function with exponential backoff retry logic.

    Args:
        func: Async function to execute with retries
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries (will be exponentially increased)
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function if successful

    Raises:
        Exception: If all retry attempts fail, raises the last exception

    Examples:
        >>> async def flaky_function():
        ...     import random
        ...     if random.random() < 0.7:  # 70% chance of failure
        ...         raise Exception("Random failure")
        ...     return "success"
        >>> # This would retry up to 3 times with exponential backoff
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                # Last attempt, raise the exception
                raise last_exception

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2**attempt) + (0.1 * attempt)  # Add small jitter
            await asyncio.sleep(delay)

    # This line should never be reached, but included for type safety
    raise last_exception if last_exception else Exception("Retry failed unexpectedly")


def retry_sync_with_backoff(
    func: SafeCallable, max_retries: int, base_delay: float, *args, **kwargs
) -> Any:
    """Execute function with exponential backoff retry logic (sync version).

    Args:
        func: Function to execute with retries
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries (will be exponentially increased)
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function if successful

    Raises:
        Exception: If all retry attempts fail, raises the last exception
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                # Last attempt, raise the exception
                raise last_exception

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2**attempt) + (0.1 * attempt)  # Add small jitter
            time.sleep(delay)

    # This line should never be reached, but included for type safety
    raise last_exception if last_exception else Exception("Retry failed unexpectedly")


def with_error_context(error_context: ErrorContext):
    """Decorator to add error context to function execution.

    Args:
        error_context: Context information to include in error details

    Returns:
        Decorator function
    """

    def decorator(func: SafeCallable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create a new exception with context attached
                e_attrs = {"original_exception": e, "context": error_context}
                # Create a new exception with the same type but with context
                new_exception = type(e)(str(e))
                for attr_name, attr_value in e_attrs.items():
                    setattr(new_exception, attr_name, attr_value)
                raise new_exception

        return wrapper

    return decorator


def format_error_details(
    error: Exception, context: Optional[ErrorContext] = None
) -> Dict[str, Any]:
    """Format error details with optional context for logging.

    Args:
        error: Exception to format
        context: Optional context information

    Returns:
        Formatted error details dictionary
    """
    details = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_args": error.args if hasattr(error, "args") else [],
    }

    if context:
        details["context"] = context

    # Add context if available on the exception (set by with_error_context decorator)
    if hasattr(error, "context"):
        details["exception_context"] = getattr(error, "context", None)

    return details


def log_exception_safely(
    logger_func: Callable, error: Exception, message: str = "Exception occurred"
):
    """Safely log an exception without raising additional errors.

    Args:
        logger_func: Logging function (e.g., logger.error)
        error: Exception to log
        message: Message to log with the exception
    """
    try:
        logger_func(f"{message}: {type(error).__name__}: {str(error)}")
    except Exception:
        # If logging fails, at least don't break the main flow
        pass


def convert_to_error_response(
    error: Exception,
    error_code: str = "INTERNAL_ERROR",
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert an exception to a standardized error response.

    Args:
        error: Exception to convert
        error_code: Error code to use (default: INTERNAL_ERROR)
        request_id: Request ID for correlation

    Returns:
        Standardized error response
    """
    return create_error_response(
        error_code,
        str(error),
        details={"error_type": type(error).__name__, "error_message": str(error)},
        request_id=request_id,
    )


def suppress_exceptions(func: SafeCallable, default_return: Any = None):
    """Decorator to suppress exceptions and return a default value.

    Args:
        func: Function to wrap
        default_return: Value to return if an exception occurs

    Returns:
        Wrapped function that suppresses exceptions
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return default_return

    return wrapper

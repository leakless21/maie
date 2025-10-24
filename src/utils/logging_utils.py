"""Enhanced logging utilities for the MAIE project.

This module provides enhanced logging utilities with structured data
and error context for better debugging and monitoring.
"""

import json
import traceback
from typing import Any, Dict, Optional
from jsonschema.exceptions import ValidationError
from .types import ErrorContext


def log_validation_error(error: ValidationError, context: ErrorContext) -> None:
    """Log validation errors with structured context.

    Args:
        error: ValidationError instance to log
        context: Context information for the error

    Examples:
        >>> from jsonschema import ValidationError
        >>> try:
        ...     # Some validation that fails
        ...     pass
        ... except ValidationError as e:
        ...     log_validation_error(e, {"template_id": "summary_v1", "user_id": "123"})
    """
    import logging

    error_info = {
        "error_message": str(error.message),
        "error_path": list(error.absolute_path),
        "error_schema_path": list(error.schema_path),
        "context": context,
    }

    logging.error(f"Validation error: {json.dumps(error_info, indent=2)}")


def log_json_parse_error(error: json.JSONDecodeError, context: ErrorContext) -> None:
    """Log JSON parsing errors with detailed context.

    Args:
        error: JSONDecodeError instance to log
        context: Context information for the error
    """
    import logging

    error_info = {
        "error_message": str(error),
        "error_line": error.lineno,
        "error_col": error.colno,
        "error_pos": error.pos,
        "context": context,
    }

    logging.error(f"JSON parse error: {json.dumps(error_info, indent=2)}")


def create_error_summary(error: Exception, context: ErrorContext) -> Dict[str, Any]:
    """Create comprehensive error summary for logging.

    Args:
        error: Exception to summarize
        context: Context information for the error

    Returns:
        Dictionary with comprehensive error summary
    """
    summary = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_context": context,
        "traceback": traceback.format_exc(),
    }

    # Add specific information for different error types
    if isinstance(error, ValidationError):
        summary.update(
            {
                "error_path": list(error.absolute_path),
                "error_schema_path": list(error.schema_path),
            }
        )
    elif isinstance(error, json.JSONDecodeError):
        summary.update(
            {
                "error_line": error.lineno,
                "error_col": error.colno,
                "error_pos": error.pos,
            }
        )

    return summary


def bind_request_context(request_id: str, **kwargs) -> None:
    """Bind request context to logger for correlation.

    Args:
        request_id: Unique request identifier
        **kwargs: Additional context key-value pairs
    """
    # This would typically bind to a structured logger like loguru
    # For now, we'll add this as a placeholder for when we integrate with loguru

    # In a real implementation, this would bind context to the logger
    # For example, with loguru: logger.bind(request_id=request_id, **kwargs)
    pass


def log_performance_metrics(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics with structured data.

    Args:
        operation: Name of the operation being measured
        duration: Duration in seconds
        **kwargs: Additional metrics to log
    """
    import logging

    metrics = {
        "operation": operation,
        "duration_seconds": duration,
        "duration_ms": duration * 1000,
        **kwargs,
    }

    logging.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")


def log_api_request(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration: float,
    user_id: Optional[str] = None,
    **kwargs,
) -> None:
    """Log API request with structured data.

    Args:
        request_id: Unique request identifier
        method: HTTP method (GET, POST, etc.)
        path: Request path
        status_code: HTTP status code
        duration: Request duration in seconds
        user_id: Optional user identifier
        **kwargs: Additional request data
    """
    import logging

    request_data = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration * 100,
        "timestamp": "now",  # In practice, you'd use actual timestamp
    }

    if user_id:
        request_data["user_id"] = user_id

    request_data.update(kwargs)

    logging.info(f"API request: {json.dumps(request_data, indent=2)}")


def log_processing_step(
    step_name: str,
    status: str,
    context: ErrorContext,
    duration: Optional[float] = None,
    **kwargs,
) -> None:
    """Log a processing step with status and context.

    Args:
        step_name: Name of the processing step
        status: Status of the step (success, error, warning, etc.)
        context: Context information for the step
        duration: Optional duration of the step in seconds
        **kwargs: Additional step data
    """
    import logging

    step_data = {"step": step_name, "status": status, "context": context}

    if duration is not None:
        step_data["duration_ms"] = duration * 1000

    step_data.update(kwargs)

    logging.info(f"Processing step: {json.dumps(step_data, indent=2)}")


def log_error_with_context(
    logger_func: Any,
    error: Exception,
    context: ErrorContext,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an error with comprehensive context information.

    Args:
        logger_func: Logging function (e.g., logger.error)
        error: Exception to log
        context: Context information for the error
        extra_info: Optional additional information to include
    """
    error_details = create_error_summary(error, context)

    if extra_info:
        error_details["extra_info"] = extra_info

    logger_func(f"Error with context: {json.dumps(error_details, indent=2)}")


def format_log_message(message: str, **kwargs) -> str:
    """Format a log message with structured data.

    Args:
        message: Main log message
        **kwargs: Additional data to include in the log

    Returns:
        Formatted log message with structured data
    """
    if kwargs:
        data = {k: v for k, v in kwargs.items() if v is not None}
        if data:
            return f"{message} | {json.dumps(data)}"

    return message


def log_config_change(
    config_key: str, old_value: Any, new_value: Any, user: Optional[str] = None
) -> None:
    """Log configuration changes for audit trail.

    Args:
        config_key: Name of the configuration key that changed
        old_value: Previous value of the configuration
        new_value: New value of the configuration
        user: Optional user who made the change
    """
    import logging

    change_data = {
        "config_key": config_key,
        "old_value": old_value,
        "new_value": new_value,
        "timestamp": "now",  # In practice, you'd use actual timestamp
    }

    if user:
        change_data["user"] = user

    logging.info(f"Config change: {json.dumps(change_data, indent=2)}")


def log_security_event(
    event_type: str, severity: str, context: ErrorContext, **kwargs
) -> None:
    """Log security-related events with appropriate context.

    Args:
        event_type: Type of security event
        severity: Severity level (low, medium, high, critical)
        context: Context information for the event
        **kwargs: Additional event data
    """
    import logging

    event_data = {
        "event_type": event_type,
        "severity": severity,
        "context": context,
        "timestamp": "now",  # In practice, you'd use actual timestamp
    }

    event_data.update(kwargs)

    logging.warning(f"Security event: {json.dumps(event_data, indent=2)}")

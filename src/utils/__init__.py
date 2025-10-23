"""MAIE utilities package.

This package provides consolidated utility functions for data validation,
JSON processing, error handling, sanitization, configuration validation,
and logging utilities.
"""

# Import all utility modules
from . import types
from . import validation
from . import json_utils
from . import error_handling
from . import sanitization
from . import config_validation
from . import logging_utils

# Convenience imports for most commonly used functions
from .validation import (
    coerce_optional_int,
    coerce_optional_str,
    validate_range,
    validate_port,
    validate_choice,
    coerce_feature_list,
    blank_to_none,
    validate_positive,
    validate_percentage,
    validate_non_empty,
    validate_list_length,
)
from .json_utils import (
    safe_parse_json,
    validate_json_schema,
    validate_llm_output,
    safe_json_loads,
    json_dumps_safe,
)
from .error_handling import (
    safe_execute,
    safe_async_execute,
    create_error_response,
    retry_with_backoff,
    retry_sync_with_backoff,
)
from .sanitization import (
    sanitize_filename,
    sanitize_metadata,
    sanitize_text,
    validate_file_extension,
    validate_mime_type,
    sanitize_path,
    sanitize_url,
)
from .config_validation import (
    validate_audio_settings,
    validate_llm_settings,
    validate_api_settings,
    validate_cleanup_intervals,
    validate_retention_periods,
    validate_disk_thresholds,
    validate_worker_settings,
    validate_logging_settings,
    validate_all_settings,
)
from .logging_utils import (
    log_validation_error,
    log_json_parse_error,
    create_error_summary,
    bind_request_context,
    log_performance_metrics,
    log_api_request,
    log_processing_step,
)

# Define what gets imported with "from utils import *"
__all__ = [
    # Validation functions
    "coerce_optional_int",
    "coerce_optional_str", 
    "validate_range",
    "validate_port",
    "validate_choice",
    "coerce_feature_list",
    "blank_to_none",
    "validate_positive",
    "validate_percentage",
    "validate_non_empty",
    "validate_list_length",
    # JSON utilities
    "safe_parse_json",
    "validate_json_schema",
    "validate_llm_output",
    "safe_json_loads",
    "json_dumps_safe",
    # Error handling functions
    "safe_execute",
    "safe_async_execute",
    "create_error_response",
    "retry_with_backoff",
    "retry_sync_with_backoff",
    # Sanitization functions
    "sanitize_filename",
    "sanitize_metadata",
    "sanitize_text",
    "validate_file_extension",
    "validate_mime_type",
    "sanitize_path",
    "sanitize_url",
    # Config validation functions
    "validate_audio_settings",
    "validate_llm_settings",
    "validate_api_settings",
    "validate_cleanup_intervals",
    "validate_retention_periods",
    "validate_disk_thresholds",
    "validate_worker_settings",
    "validate_logging_settings",
    "validate_all_settings",
    # Logging utilities
    "log_validation_error",
    "log_json_parse_error",
    "create_error_summary",
    "bind_request_context",
    "log_performance_metrics",
    "log_api_request",
    "log_processing_step",
]
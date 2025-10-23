"""Shared type definitions and constants for the MAIE utilities module.

This module contains common type aliases and constants used across all
utility modules to ensure consistent typing and reduce code duplication.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from pathlib import Path
from jsonschema.exceptions import ValidationError


# Common type aliases
JSONDict = Dict[str, Any]
"""Type alias for JSON-compatible dictionaries with string keys and any values."""

ValidationResult = Tuple[bool, Optional[Dict[str, Any]]]
"""Type alias for validation results: (is_valid, error_details)."""

ExecutionResult = Tuple[Any, Optional[Dict[str, Any]]]
"""Type alias for execution results: (result, error_info)."""

JSONParseResult = Tuple[Optional[JSONDict], Optional[str]]
"""Type alias for JSON parsing results: (parsed_data, error_message)."""

# Error context types
ErrorContext = Dict[str, Any]
"""Type alias for error context dictionaries."""

ValidationContext = Dict[str, Any]
"""Type alias for validation context dictionaries."""

# Callable type aliases
SafeCallable = Callable[..., Any]
"""Type alias for functions that can be called safely with any arguments."""

AsyncSafeCallable = Callable[..., Any]  # Actually should be Awaitable but simplified for now
"""Type alias for async functions that can be called safely with any arguments."""

# Common constants
DEFAULT_PORT_RANGE = (1, 65535)
"""Default valid port range: 1-6535."""

DEFAULT_RETRY_COUNT = 3
"""Default number of retry attempts for operations."""

DEFAULT_TIMEOUT = 30.0
"""Default timeout value in seconds."""

DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
"""Default maximum file size in bytes."""

ALLOWED_FILE_EXTENSIONS = ['.mp3', '.wav', '.flac', '.m4a', '.mp4', '.m4v', '.mov', '.avi', '.mkv']
"""List of allowed file extensions for audio/video uploads."""

ALLOWED_MIME_TYPES = [
    'audio/mpeg', 'audio/wav', 'audio/flac', 'audio/mp4',
    'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'
]
"""List of allowed MIME types for audio/video uploads."""

# Security-related constants
MAX_PATH_LENGTH = 255
"""Maximum allowed path length to prevent path traversal attacks."""

FORBIDDEN_PATH_CHARS = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
"""Characters that are forbidden in file paths for security reasons."""

# Validation constants
DEFAULT_MIN_LENGTH = 1
"""Default minimum length for string validation."""

DEFAULT_MAX_LENGTH = 10000
"""Default maximum length for string validation."""

# Common error codes
ERROR_CODES = {
    'VALIDATION_ERROR': 'VALIDATION_ERROR',
    'PARSING_ERROR': 'PARSING_ERROR',
    'PROCESSING_ERROR': 'PROCESSING_ERROR',
    'CONFIG_ERROR': 'CONFIG_ERROR',
    'SECURITY_ERROR': 'SECURITY_ERROR',
    'TIMEOUT_ERROR': 'TIMEOUT_ERROR',
    'NETWORK_ERROR': 'NETWORK_ERROR',
    'FILE_ERROR': 'FILE_ERROR',
}
"""Dictionary of standard error codes used across the application."""
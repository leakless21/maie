"""
Standardized error response classes and error codes for MAIE API.

This module provides consistent error handling across all API endpoints
and worker processes, following the error taxonomy defined in the
error-handling-taxonomy.md documentation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.utils.error_handling import create_error_response as utils_create_error_response, handle_generic_error as utils_handle_generic_error


class ErrorResponse(BaseModel):
    """Standardized error response structure."""

    code: str = Field(
        ..., description="Standardized error code (e.g., AUDIO_DECODE_ERROR)"
    )
    message: str = Field(..., description="Human-readable error message for users")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional context information"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    request_id: Optional[str] = Field(
        None, description="Unique identifier for request tracking"
    )


class ErrorCodes:
    """Standardized error codes for the MAIE system."""

    # Audio Processing Errors
    AUDIO_DECODE_ERROR = "AUDIO_DECODE_ERROR"
    AUDIO_PREPROCESSING_ERROR = "AUDIO_PREPROCESSING_ERROR"
    AUDIO_VALIDATION_ERROR = "AUDIO_VALIDATION_ERROR"

    # Model Loading Errors
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    MODEL_INITIALIZATION_ERROR = "MODEL_INITIALIZATION_ERROR"
    MODEL_MEMORY_ERROR = "MODEL_MEMORY_ERROR"

    # Processing Errors
    PROCESSING_ERROR = "PROCESSING_ERROR"
    ASR_PROCESSING_ERROR = "ASR_PROCESSING_ERROR"
    LLM_PROCESSING_ERROR = "LLM_PROCESSING_ERROR"

    # Configuration Errors
    CONFIG_VALIDATION_ERROR = "CONFIG_VALIDATION_ERROR"
    CONFIG_LOAD_ERROR = "CONFIG_LOAD_ERROR"

    # API Errors
    API_VALIDATION_ERROR = "API_VALIDATION_ERROR"
    API_AUTHENTICATION_ERROR = "API_AUTHENTICATION_ERROR"
    API_RATE_LIMIT_ERROR = "API_RATE_LIMIT_ERROR"

    # Infrastructure Errors
    REDIS_CONNECTION_ERROR = "REDIS_CONNECTION_ERROR"
    FILE_SYSTEM_ERROR = "FILE_SYSTEM_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"


class MAIEError(Exception):
    """Base exception class for MAIE-specific errors."""

    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.request_id = request_id
        super().__init__(message)

    def to_error_response(self) -> ErrorResponse:
        """Convert exception to standardized error response."""
        return ErrorResponse(
            code=self.error_code,
            message=self.message,
            details=self.details,
            request_id=self.request_id,
        )


class AudioProcessingError(MAIEError):
    """Base class for audio processing errors."""

    pass


class AudioDecodeError(AudioProcessingError):
    """Failed to decode audio file format."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.AUDIO_DECODE_ERROR,
            message=message
            or "Unable to process audio file. Please ensure the file is in a supported format (WAV, MP3, FLAC).",
            details=details,
            request_id=request_id,
        )


class AudioPreprocessingError(AudioProcessingError):
    """Failed during audio preprocessing."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.AUDIO_PREPROCESSING_ERROR,
            message=message
            or "Audio preprocessing failed. Please check the audio file and try again.",
            details=details,
            request_id=request_id,
        )


class AudioValidationError(AudioProcessingError):
    """Audio file failed validation checks."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.AUDIO_VALIDATION_ERROR,
            message=message
            or "Audio file validation failed. Please check file size and duration requirements.",
            details=details,
            request_id=request_id,
        )


class ModelError(MAIEError):
    """Base class for model-related errors."""

    pass


class ModelLoadError(ModelError):
    """Failed to load AI model into memory."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.MODEL_LOAD_ERROR,
            message=message
            or "Model loading failed. Please try again or contact support if the issue persists.",
            details=details,
            request_id=request_id,
        )


class ModelInitializationError(ModelError):
    """Failed to initialize model with configuration."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.MODEL_INITIALIZATION_ERROR,
            message=message
            or "Model initialization failed. Please check configuration settings.",
            details=details,
            request_id=request_id,
        )


class ModelMemoryError(ModelError):
    """Insufficient memory for model operations."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.MODEL_MEMORY_ERROR,
            message=message
            or "Insufficient memory for processing. Please try again later or contact support.",
            details=details,
            request_id=request_id,
        )


class ProcessingError(MAIEError):
    """Base class for processing errors."""

    pass


class ASRProcessingError(ProcessingError):
    """ASR (Automatic Speech Recognition) processing failed."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.ASR_PROCESSING_ERROR,
            message=message
            or "Speech recognition failed. Please ensure clear audio and try again.",
            details=details,
            request_id=request_id,
        )


class LLMProcessingError(ProcessingError):
    """LLM (Large Language Model) processing failed."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.LLM_PROCESSING_ERROR,
            message=message
            or "Text processing failed. Please try again or contact support.",
            details=details,
            request_id=request_id,
        )


class ConfigurationError(MAIEError):
    """Base class for configuration errors."""

    pass


class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
            message=message or "Configuration error. Please check system settings.",
            details=details,
            request_id=request_id,
        )


class ConfigLoadError(ConfigurationError):
    """Failed to load configuration from file or environment."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.CONFIG_LOAD_ERROR,
            message=message
            or "Configuration loading failed. Please check system setup.",
            details=details,
            request_id=request_id,
        )


class APIError(MAIEError):
    """Base class for API errors."""

    pass


class APIValidationError(APIError):
    """Request validation failed."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.API_VALIDATION_ERROR,
            message=message
            or "Invalid request format. Please check your request parameters.",
            details=details,
            request_id=request_id,
        )


class APIAuthenticationError(APIError):
    """Authentication failed."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.API_AUTHENTICATION_ERROR,
            message=message
            or "Authentication failed. Please check your API credentials.",
            details=details,
            request_id=request_id,
        )


class APIRateLimitError(APIError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.API_RATE_LIMIT_ERROR,
            message=message
            or "Rate limit exceeded. Please wait before making another request.",
            details=details,
            request_id=request_id,
        )


class InfrastructureError(MAIEError):
    """Base class for infrastructure errors."""

    pass


class RedisConnectionError(InfrastructureError):
    """Failed to connect to Redis."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.REDIS_CONNECTION_ERROR,
            message=message
            or "Service temporarily unavailable. Please try again later.",
            details=details,
            request_id=request_id,
        )


class FileSystemError(InfrastructureError):
    """File system operation failed."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.FILE_SYSTEM_ERROR,
            message=message
            or "File system error. Please try again or contact support.",
            details=details,
            request_id=request_id,
        )


class NetworkError(InfrastructureError):
    """Network communication failed."""

    def __init__(
        self,
        message: str = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            error_code=ErrorCodes.NETWORK_ERROR,
            message=message
            or "Network error. Please check your connection and try again.",
            details=details,
            request_id=request_id,
        )


# Error handling utilities
def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        code=error_code,
        message=message,
        details=details,
        request_id=request_id,
    )


def handle_maie_error(error: MAIEError) -> ErrorResponse:
    """Convert MAIE error to standardized response."""
    return error.to_error_response()


def handle_generic_error(
    error: Exception, request_id: Optional[str] = None
) -> ErrorResponse:
    """Handle generic exceptions by converting to standardized error response."""
    return ErrorResponse(
        code=ErrorCodes.PROCESSING_ERROR,
        message="An unexpected error occurred. Please try again or contact support.",
        details={"error_type": type(error).__name__} if request_id else None,
        request_id=request_id,
    )


# Export all error classes and utilities
__all__ = [
    "ErrorResponse",
    "ErrorCodes",
    "MAIEError",
    "AudioProcessingError",
    "AudioDecodeError",
    "AudioPreprocessingError",
    "AudioValidationError",
    "ModelError",
    "ModelLoadError",
    "ModelInitializationError",
    "ModelMemoryError",
    "ProcessingError",
    "ASRProcessingError",
    "LLMProcessingError",
    "ConfigurationError",
    "ConfigValidationError",
    "ConfigLoadError",
    "APIError",
    "APIValidationError",
    "APIAuthenticationError",
    "APIRateLimitError",
    "InfrastructureError",
    "RedisConnectionError",
    "FileSystemError",
    "NetworkError",
    "create_error_response",
    "handle_maie_error",
    "handle_generic_error",
]

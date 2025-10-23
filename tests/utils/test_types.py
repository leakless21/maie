"""Tests for the types module."""

import pytest
from src.utils.types import (
    JSONDict,
    ValidationResult,
    ExecutionResult,
    JSONParseResult,
    ErrorContext,
    ValidationContext,
    DEFAULT_PORT_RANGE,
    DEFAULT_RETRY_COUNT,
    DEFAULT_TIMEOUT,
    ALLOWED_FILE_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_PATH_LENGTH,
    FORBIDDEN_PATH_CHARS,
    DEFAULT_MIN_LENGTH,
    DEFAULT_MAX_LENGTH,
    ERROR_CODES
)


def test_json_dict_type():
    """Test that JSONDict can be used as a type hint."""
    data: JSONDict = {"key": "value", "nested": {"subkey": "subvalue"}}
    assert isinstance(data, dict)
    assert data["key"] == "value"


def test_validation_result_type():
    """Test that ValidationResult can be used as a type hint."""
    result: ValidationResult = (True, None)
    assert result[0] is True
    assert result[1] is None


def test_execution_result_type():
    """Test that ExecutionResult can be used as a type hint."""
    result: ExecutionResult = ("success", None)
    assert result[0] == "success"
    assert result[1] is None


def test_json_parse_result_type():
    """Test that JSONParseResult can be used as a type hint."""
    result: JSONParseResult = ({"key": "value"}, None)
    assert result[0] == {"key": "value"}
    assert result[1] is None


def test_error_context_type():
    """Test that ErrorContext can be used as a type hint."""
    context: ErrorContext = {"request_id": "123", "user_id": "456"}
    assert context["request_id"] == "123"


def test_validation_context_type():
    """Test that ValidationContext can be used as a type hint."""
    context: ValidationContext = {"field": "name", "value": "test"}
    assert context["field"] == "name"


def test_default_port_range():
    """Test default port range constant."""
    assert DEFAULT_PORT_RANGE == (1, 65535)


def test_default_retry_count():
    """Test default retry count constant."""
    assert DEFAULT_RETRY_COUNT == 3


def test_default_timeout():
    """Test default timeout constant."""
    assert DEFAULT_TIMEOUT == 30.0


def test_allowed_file_extensions():
    """Test allowed file extensions constant."""
    assert ".mp3" in ALLOWED_FILE_EXTENSIONS
    assert ".wav" in ALLOWED_FILE_EXTENSIONS
    assert ".mp4" in ALLOWED_FILE_EXTENSIONS


def test_allowed_mime_types():
    """Test allowed MIME types constant."""
    assert "audio/mpeg" in ALLOWED_MIME_TYPES
    assert "video/mp4" in ALLOWED_MIME_TYPES


def test_max_path_length():
    """Test max path length constant."""
    assert MAX_PATH_LENGTH == 255


def test_forbidden_path_chars():
    """Test forbidden path characters constant."""
    assert ".." in FORBIDDEN_PATH_CHARS
    assert "/" in FORBIDDEN_PATH_CHARS
    assert "\\" in FORBIDDEN_PATH_CHARS


def test_default_min_max_length():
    """Test default min/max length constants."""
    assert DEFAULT_MIN_LENGTH == 1
    assert DEFAULT_MAX_LENGTH == 10000


def test_error_codes():
    """Test error codes constant."""
    assert "VALIDATION_ERROR" in ERROR_CODES
    assert ERROR_CODES["VALIDATION_ERROR"] == "VALIDATION_ERROR"
    assert "PARSING_ERROR" in ERROR_CODES
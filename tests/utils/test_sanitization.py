"""Tests for the sanitization module."""

import pytest
from src.utils.sanitization import (
    sanitize_filename,
    sanitize_metadata,
    sanitize_text,
    validate_file_extension,
    validate_mime_type,
    normalize_phone_number,
    sanitize_path,
    sanitize_url,
    remove_control_characters,
    validate_and_sanitize_filename,
)


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_sanitize_filename_normal(self):
        """Test sanitize_filename with normal filename."""
        result = sanitize_filename("normal_file.txt")
        assert result == "normal_file.txt"

    def test_sanitize_filename_path_traversal(self):
        """Test sanitize_filename with path traversal."""
        result = sanitize_filename("../../etc/passwd")
        assert result == "etc_passwd"

    def test_sanitize_filename_forbidden_chars(self):
        """Test sanitize_filename with forbidden characters."""
        result = sanitize_filename("file<>.txt")
        assert result == "file__.txt"

    def test_sanitize_filename_empty(self):
        """Test sanitize_filename with empty input."""
        result = sanitize_filename("")
        assert result == "unnamed_file"


class TestSanitizeMetadata:
    """Tests for sanitize_metadata function."""

    def test_sanitize_metadata_dict(self):
        """Test sanitize_metadata with dictionary."""
        data = {"key": "value", "nested": {"subkey": "subvalue"}}
        result = sanitize_metadata(data)
        assert result == data

    def test_sanitize_metadata_with_text(self):
        """Test sanitize_metadata with text values."""
        data = {"script": "<script>alert('xss')</script>", "normal": "value"}
        result = sanitize_metadata(data)
        # The script tags should be removed, so the content will be empty or just the text
        assert result["normal"] == "value"
        # The script content should be removed, so it should be empty
        assert result["script"] == ""

    def test_sanitize_metadata_list(self):
        """Test sanitize_metadata with list."""
        data = ["item1", "<script>xss</script>", "item3"]
        result = sanitize_metadata(data)
        assert len(result) == 3
        assert result[0] == "item1"

    def test_sanitize_metadata_circular_reference(self):
        """Test sanitize_metadata with circular reference."""
        # Create the data structure and then add the circular reference separately
        data = {"key": "value"}
        data["self"] = data  # Create circular reference
        result = sanitize_metadata(data)
        assert result["key"] == "value"
        # The circular reference should be detected and replaced with a string
        assert isinstance(result["self"], str)
        assert "Circular Reference" in result["self"]


class TestSanitizeText:
    """Tests for sanitize_text function."""

    def test_sanitize_text_normal(self):
        """Test sanitize_text with normal text."""
        result = sanitize_text("Hello World")
        assert result == "Hello World"

    def test_sanitize_text_with_script(self):
        """Test sanitize_text with script tags."""
        result = sanitize_text("Hello <script>alert('xss')</script> World")
        assert "<script>" not in result
        # The script content should also be removed, so only "Hello World" remains
        assert result == "Hello  World"

    def test_sanitize_text_with_newlines(self):
        """Test sanitize_text with newlines preservation."""
        result = sanitize_text("Line1\nLine2", preserve_newlines=True)
        assert "Line1\nLine2" in result

    def test_sanitize_text_without_newlines(self):
        """Test sanitize_text without newlines preservation."""
        result = sanitize_text("Line1\nLine2", preserve_newlines=False)
        assert "\n" not in result


class TestValidateFileExtension:
    """Tests for validate_file_extension function."""

    def test_validate_file_extension_allowed(self):
        """Test validate_file_extension with allowed extension."""
        assert validate_file_extension("test.mp3") is True
        assert validate_file_extension("test.wav") is True
        assert validate_file_extension("test.MP3") is True  # Should be case insensitive

    def test_validate_file_extension_disallowed(self):
        """Test validate_file_extension with disallowed extension."""
        assert validate_file_extension("test.exe") is False
        assert validate_file_extension("test.bat") is False


class TestValidateMimeType:
    """Tests for validate_mime_type function."""

    def test_validate_mime_type_allowed(self):
        """Test validate_mime_type with allowed types."""
        assert validate_mime_type("audio/mpeg") is True
        assert validate_mime_type("video/mp4") is True

    def test_validate_mime_type_disallowed(self):
        """Test validate_mime_type with disallowed types."""
        assert validate_mime_type("application/exe") is False
        assert validate_mime_type("text/html") is False


class TestNormalizePhoneNumber:
    """Tests for normalize_phone_number function."""

    def test_normalize_phone_number_basic(self):
        """Test normalize_phone_number with basic formats."""
        assert normalize_phone_number("(555) 123-4567") == "5551234567"
        assert normalize_phone_number("555.123.4567") == "5551234567"
        assert normalize_phone_number("555-123-4567") == "5551234567"

    def test_normalize_phone_number_with_country_code(self):
        """Test normalize_phone_number with country code."""
        assert normalize_phone_number("+1-555-123-4567") == "5551234567"
        assert normalize_phone_number("0015551234567") == "5551234567"


class TestSanitizePath:
    """Tests for sanitize_path function."""

    def test_sanitize_path_normal(self):
        """Test sanitize_path with normal path."""
        result = sanitize_path("/safe/path/file.txt")
        assert result == "safe/path/file.txt"

    def test_sanitize_path_with_traversal(self):
        """Test sanitize_path with path traversal."""
        result = sanitize_path("../../etc/passwd")
        assert result == "etc/passwd"


class TestSanitizeUrl:
    """Tests for sanitize_url function."""

    def test_sanitize_url_safe(self):
        """Test sanitize_url with safe URLs."""
        assert sanitize_url("https://example.com") == "https://example.com"
        assert sanitize_url("http://example.com") == "http://example.com"
        assert sanitize_url("/relative/path") == "/relative/path"

    def test_sanitize_url_dangerous(self):
        """Test sanitize_url with dangerous schemes."""
        assert sanitize_url("javascript:alert('xss')") == ""
        assert sanitize_url("data:text/html,<script>") == ""


class TestRemoveControlCharacters:
    """Tests for remove_control_characters function."""

    def test_remove_control_characters(self):
        """Test remove_control_characters function."""
        text = "Hello\x00World\x01Test"
        result = remove_control_characters(text)
        assert result == "HelloWorldTest"

    def test_remove_control_characters_with_allowed(self):
        """Test remove_control_characters with allowed characters."""
        text = "Line1\nLine2\tTab"
        result = remove_control_characters(text, allowed_chars=["\n", "\t"])
        assert result == "Line1\nLine2\tTab"


class TestValidateAndSanitizeFilename:
    """Tests for validate_and_sanitize_filename function."""

    def test_validate_and_sanitize_filename_valid(self):
        """Test validate_and_sanitize_filename with valid filename."""
        result = validate_and_sanitize_filename("test.mp3")
        assert result == "test.mp3"

    def test_validate_and_sanitize_filename_invalid_extension(self):
        """Test validate_and_sanitize_filename with invalid extension."""
        with pytest.raises(ValueError, match="File extension not allowed"):
            validate_and_sanitize_filename("test.exe")

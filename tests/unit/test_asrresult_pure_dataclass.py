"""
Test suite for ASRResult pure dataclass interface.

This test suite verifies the new pure dataclass ASRResult interface
with attribute-based access and validation.
"""

import pytest
from dataclasses import asdict

from src.processors.base import ASRResult


class TestASRResultPureInterface:
    """Test suite for pure dataclass ASRResult interface."""

    def test_basic_attribute_access(self):
        """Test basic attribute access functionality."""
        result = ASRResult(
            text="Hello world",
            segments=[{"start": 0, "end": 1, "text": "Hello"}],
            language="en",
            confidence=0.95,
            duration=1.5,
        )

        # Test attribute access
        assert result.text == "Hello world"
        assert result.segments == [{"start": 0, "end": 1, "text": "Hello"}]
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.duration == 1.5
        assert result.error is None

    def test_minimal_instance(self):
        """Test ASRResult with only required field."""
        result = ASRResult(text="Minimal")

        # Test basic functionality
        assert result.text == "Minimal"
        assert result.segments is None
        assert result.language is None
        assert result.confidence is None
        assert result.duration is None
        assert result.error is None

    def test_serialization_with_asdict(self):
        """Test serialization using dataclasses.asdict()."""
        result = ASRResult(
            text="Hello world",
            segments=[{"start": 0, "end": 1, "text": "Hello"}],
            language="en",
            confidence=0.95,
            duration=1.5,
        )

        dict_result = asdict(result)
        expected = {
            "text": "Hello world",
            "segments": [{"start": 0, "end": 1, "text": "Hello"}],
            "language": "en",
            "confidence": 0.95,
            "duration": 1.5,
            "error": None,
        }

        assert dict_result == expected

    def test_validation_confidence_range(self):
        """Test confidence validation (0.0-1.0)."""
        # Valid confidence
        result = ASRResult(text="Test", confidence=0.5)
        assert result.confidence == 0.5

        # Invalid confidence - too high
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            ASRResult(text="Test", confidence=1.5)

        # Invalid confidence - negative
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            ASRResult(text="Test", confidence=-0.1)

    def test_validation_duration_non_negative(self):
        """Test duration validation (non-negative)."""
        # Valid duration
        result = ASRResult(text="Test", duration=1.5)
        assert result.duration == 1.5

        # Valid zero duration
        result = ASRResult(text="Test", duration=0.0)
        assert result.duration == 0.0

        # Invalid negative duration
        with pytest.raises(ValueError, match="duration must be non-negative"):
            ASRResult(text="Test", duration=-1.0)

    def test_validation_type_errors(self):
        """Test type validation for all fields."""
        # Valid text
        result = ASRResult(text="Valid text")
        assert result.text == "Valid text"

        # Invalid text type
        with pytest.raises(TypeError, match="text must be str"):
            ASRResult(text=123)

        # Invalid confidence type
        with pytest.raises(TypeError, match="confidence must be numeric"):
            ASRResult(text="Test", confidence="0.5")

        # Invalid duration type
        with pytest.raises(TypeError, match="duration must be numeric"):
            ASRResult(text="Test", duration="1.5")

        # Invalid language type
        with pytest.raises(TypeError, match="language must be str"):
            ASRResult(text="Test", language=123)

        # Invalid segments type
        with pytest.raises(TypeError, match="segments must be list"):
            ASRResult(text="Test", segments="not a list")

        # Invalid error type
        with pytest.raises(TypeError, match="error must be dict"):
            ASRResult(text="Test", error="not a dict")

    def test_optional_fields_none_handling(self):
        """Test None handling for optional fields."""
        result = ASRResult(text="Test")

        # All optional fields should be None by default
        assert result.segments is None
        assert result.language is None
        assert result.confidence is None
        assert result.duration is None
        assert result.error is None

    def test_complex_nested_data(self):
        """Test ASRResult with complex nested data structures."""
        complex_segments = [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "Hello world",
                "confidence": 0.95,
                "metadata": {"speaker": "1", "channel": "left"},
            },
            {
                "start": 1.5,
                "end": 3.0,
                "text": "How are you?",
                "confidence": 0.87,
                "metadata": {"speaker": "2", "channel": "right"},
            },
        ]

        result = ASRResult(
            text="Hello world How are you?",
            segments=complex_segments,
            language="en",
            confidence=0.91,
        )

        # Test that complex data is preserved
        assert result.segments == complex_segments
        assert result.segments[0]["metadata"]["speaker"] == "1"

        # Test serialization preserves complex structure
        dict_result = asdict(result)
        assert dict_result["segments"] == complex_segments

    def test_error_handling_invalid_data(self):
        """Test validation error handling."""

        # Test multiple validation errors (first one wins)
        with pytest.raises(TypeError, match="text must be str"):
            ASRResult(text=None, confidence=1.5, duration=-1.0)

    def test_edge_cases(self):
        """Test edge cases and special values."""
        # Test with empty strings and lists
        result = ASRResult(
            text="", segments=[], language="", confidence=0.0, duration=0.0
        )

        assert result.text == ""
        assert result.segments == []
        assert result.language == ""
        assert result.confidence == 0.0
        assert result.duration == 0.0

    def test_with_error_field(self):
        """Test ASRResult with error information."""
        error_info = {"type": "ProcessingError", "message": "Test error"}
        result = ASRResult(text="", error=error_info)

        # Test attribute access
        assert result.text == ""
        assert result.error == error_info

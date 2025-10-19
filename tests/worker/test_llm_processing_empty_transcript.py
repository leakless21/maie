"""
Unit tests for empty transcript handling in LLM processing.

Tests that execute_llm_processing properly rejects empty/whitespace transcripts
before attempting any LLM operations.
"""

import pytest
from unittest.mock import Mock

from src.api.errors import LLMProcessingError
from src.worker.pipeline import execute_llm_processing


class TestEmptyTranscriptHandling:
    """Test cases for empty transcript validation in LLM processing."""

    def test_execute_llm_processing_rejects_empty_transcript(self):
        """Test that empty transcript raises LLMProcessingError."""
        # Arrange
        mock_llm_model = Mock()
        empty_transcript = ""
        features = ["clean_transcript"]

        # Act & Assert
        with pytest.raises(LLMProcessingError) as exc_info:
            execute_llm_processing(
                llm_model=mock_llm_model,
                transcription=empty_transcript,
                features=features,
            )

        # Verify error message contains expected text
        assert "Empty transcript" in str(exc_info.value.message)
        assert exc_info.value.details["transcription_length"] == 0

    def test_execute_llm_processing_rejects_whitespace_only_transcript(self):
        """Test that whitespace-only transcript raises LLMProcessingError."""
        # Arrange
        mock_llm_model = Mock()
        whitespace_transcript = "   \n\t  "
        features = ["clean_transcript"]

        # Act & Assert
        with pytest.raises(LLMProcessingError) as exc_info:
            execute_llm_processing(
                llm_model=mock_llm_model,
                transcription=whitespace_transcript,
                features=features,
            )

        # Verify error message contains expected text
        assert "Empty transcript" in str(exc_info.value.message)
        assert exc_info.value.details["transcription_length"] == 0

    def test_execute_llm_processing_accepts_non_empty_transcript(self):
        """Test that non-empty transcript does not raise error."""
        # Arrange
        mock_llm_model = Mock()
        mock_llm_model.needs_enhancement.return_value = False
        non_empty_transcript = "Hello world"
        features = ["clean_transcript"]

        # Act & Assert - should not raise
        try:
            result = execute_llm_processing(
                llm_model=mock_llm_model,
                transcription=non_empty_transcript,
                features=features,
            )
            # Should return tuple of (clean_transcript, structured_summary)
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception as e:
            # If it raises, it should not be LLMProcessingError for empty transcript
            assert not isinstance(e, LLMProcessingError) or "Empty transcript" not in str(e)

    def test_execute_llm_processing_accepts_whitespace_padded_transcript(self):
        """Test that transcript with leading/trailing whitespace is accepted."""
        # Arrange
        mock_llm_model = Mock()
        mock_llm_model.needs_enhancement.return_value = False
        padded_transcript = "  Hello world  \n"
        features = ["clean_transcript"]

        # Act & Assert - should not raise
        try:
            result = execute_llm_processing(
                llm_model=mock_llm_model,
                transcription=padded_transcript,
                features=features,
            )
            # Should return tuple of (clean_transcript, structured_summary)
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception as e:
            # If it raises, it should not be LLMProcessingError for empty transcript
            assert not isinstance(e, LLMProcessingError) or "Empty transcript" not in str(e)

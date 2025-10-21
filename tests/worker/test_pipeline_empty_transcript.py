"""
Unit tests for empty transcript handling in the main pipeline.

Tests that process_audio_task properly skips LLM loading and raises
LLMProcessingError when ASR returns an empty transcript.
"""

import pytest
from unittest.mock import Mock, patch

from src.api.errors import LLMProcessingError
from src.worker.pipeline import process_audio_task


class TestPipelineEmptyTranscriptHandling:
    """Test cases for empty transcript validation in the main pipeline."""

    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.execute_asr_transcription")
    @patch("src.processors.audio.AudioPreprocessor")
    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.unload_asr_model")
    @patch("src.worker.pipeline.get_current_job")
    def test_pipeline_skips_llm_load_on_empty_transcript(
        self,
        mock_get_current_job,
        mock_unload_asr_model,
        mock_load_asr_model,
        mock_audio_preprocessor,
        mock_execute_asr_transcription,
        mock_load_llm_model,
    ):
        """Test that pipeline skips LLM loading when ASR returns empty transcript."""
        # Arrange
        mock_job = Mock()
        mock_job.id = "test-job-123"
        mock_get_current_job.return_value = mock_job

        # Mock audio preprocessing to return valid audio info
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.preprocess.return_value = {
            "duration": 3.0,
            "sample_rate": 16000,
            "channels": 1,
            "normalized": False,
        }
        mock_audio_preprocessor.return_value = mock_preprocessor_instance

        # Mock ASR model loading/unloading
        mock_asr_model = Mock()
        mock_load_asr_model.return_value = mock_asr_model

        # Mock ASR transcription to return empty transcript
        # Phase 1: Now returns (ASRResult, rtf, metadata) instead of (text, rtf, confidence, metadata)
        from src.processors.base import ASRResult
        empty_result = ASRResult(text="", confidence=0.0)
        mock_execute_asr_transcription.return_value = (empty_result, 0.0, {})

        task_params = {
            "audio_path": "dummy.wav",
            "features": ["clean_transcript"],
            "asr_backend": "chunkformer",
        }

        # Act - pipeline now returns structured error response instead of raising
        result = process_audio_task(task_params)

        # Assert - verify error response
        assert result["status"] == "error"
        assert result["error"]["code"] == "LLM_PROCESSING_ERROR"
        assert "Empty transcript" in result["error"]["message"]

        # Verify LLM model was never loaded
        mock_load_llm_model.assert_not_called()

        # Verify ASR model was loaded and unloaded
        mock_load_asr_model.assert_called_once()
        mock_unload_asr_model.assert_called_once()

    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.execute_asr_transcription")
    @patch("src.processors.audio.AudioPreprocessor")
    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.unload_asr_model")
    @patch("src.worker.pipeline.get_current_job")
    def test_pipeline_skips_llm_load_on_whitespace_only_transcript(
        self,
        mock_get_current_job,
        mock_unload_asr_model,
        mock_load_asr_model,
        mock_audio_preprocessor,
        mock_execute_asr_transcription,
        mock_load_llm_model,
    ):
        """Test that pipeline skips LLM loading when ASR returns whitespace-only transcript."""
        # Arrange
        mock_job = Mock()
        mock_job.id = "test-job-456"
        mock_get_current_job.return_value = mock_job

        # Mock audio preprocessing to return valid audio info
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.preprocess.return_value = {
            "duration": 3.0,
            "sample_rate": 16000,
            "channels": 1,
            "normalized": False,
        }
        mock_audio_preprocessor.return_value = mock_preprocessor_instance

        # Mock ASR model loading/unloading
        mock_asr_model = Mock()
        mock_load_asr_model.return_value = mock_asr_model

        # Mock ASR transcription to return whitespace-only transcript
        # Phase 1: Now returns (ASRResult, rtf, metadata) instead of (text, rtf, confidence, metadata)
        from src.processors.base import ASRResult
        whitespace_result = ASRResult(text="   \n\t  ", confidence=0.0)
        mock_execute_asr_transcription.return_value = (whitespace_result, 0.0, {})

        task_params = {
            "audio_path": "dummy.wav",
            "features": ["clean_transcript"],
            "asr_backend": "chunkformer",
        }

        # Act - pipeline now returns structured error response instead of raising
        result = process_audio_task(task_params)

        # Assert - verify error response
        assert result["status"] == "error"
        assert result["error"]["code"] == "LLM_PROCESSING_ERROR"
        assert "Empty transcript" in result["error"]["message"]

        # Verify LLM model was never loaded
        mock_load_llm_model.assert_not_called()

    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.execute_asr_transcription")
    @patch("src.processors.audio.AudioPreprocessor")
    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.unload_asr_model")
    @patch("src.worker.pipeline.get_current_job")
    def test_pipeline_proceeds_with_non_empty_transcript(
        self,
        mock_get_current_job,
        mock_unload_asr_model,
        mock_load_asr_model,
        mock_audio_preprocessor,
        mock_execute_asr_transcription,
        mock_load_llm_model,
    ):
        """Test that pipeline proceeds normally with non-empty transcript."""
        # Arrange
        mock_job = Mock()
        mock_job.id = "test-job-789"
        mock_get_current_job.return_value = mock_job

        # Mock audio preprocessing to return valid audio info
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.preprocess.return_value = {
            "duration": 3.0,
            "sample_rate": 16000,
            "channels": 1,
            "normalized": False,
        }
        mock_audio_preprocessor.return_value = mock_preprocessor_instance

        # Mock ASR model loading/unloading
        mock_asr_model = Mock()
        mock_load_asr_model.return_value = mock_asr_model

        # Mock ASR transcription to return valid transcript
        # Phase 1: Now returns (ASRResult, rtf, metadata) instead of (text, rtf, confidence, metadata)
        from src.processors.base import ASRResult
        valid_result = ASRResult(text="Hello world", confidence=0.8)
        mock_execute_asr_transcription.return_value = (valid_result, 0.5, {})

        # Mock LLM model and processing
        mock_llm_model = Mock()
        mock_llm_model.needs_enhancement.return_value = False
        mock_load_llm_model.return_value = mock_llm_model

        task_params = {
            "audio_path": "dummy.wav",
            "features": ["clean_transcript"],
            "asr_backend": "chunkformer",
        }

        # Act
        try:
            result = process_audio_task(task_params)
            # Should complete successfully
            assert result is not None
        except Exception as e:
            # If it raises, it should not be LLMProcessingError for empty transcript
            assert not isinstance(e, LLMProcessingError) or "Empty transcript" not in str(e)

        # Verify LLM model was loaded (since transcript is not empty)
        mock_load_llm_model.assert_called_once()
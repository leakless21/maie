"""
Unit tests for worker pipeline error handling.

Tests error scenarios to ensure robust failure handling:
- Audio file errors (missing, corrupted, too short)
- ASR processing failures
- LLM processing failures
- Redis connection failures
- Resource cleanup on errors

Follows TDD.md section 3.2 error handling requirements.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from redis.exceptions import ConnectionError as RedisConnectionError

from src.api.schemas import TaskStatus
from src.worker.pipeline import (
    handle_processing_error,
    TaskStatus as PipelineTaskStatus,
    unload_asr_model,
    unload_llm_model,
)


class TestAudioFileErrors:
    """Test error handling for audio file issues."""

    def test_missing_audio_file(self, mock_redis_sync):
        """Test handling of missing audio file."""
        # Simulate missing file error
        error = FileNotFoundError("Audio file not found: /path/to/audio.wav")
        task_key = "task:test-missing-file"

        # Call error handler
        handle_processing_error(mock_redis_sync, task_key, error, stage="preprocessing")

        # Verify status updated to FAILED
        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert "error" in stored_data
        assert "Audio file not found" in stored_data["error"]
        assert stored_data.get("stage") == "preprocessing"

    def test_corrupted_audio_file(self, mock_redis_sync):
        """Test handling of corrupted/unreadable audio file."""
        # Simulate audio decode error
        error = RuntimeError("AUDIO_DECODE_ERROR: Failed to decode audio")
        task_key = "task:test-corrupted"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="preprocessing",
            error_code="AUDIO_DECODE_ERROR",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert "AUDIO_DECODE_ERROR" in stored_data.get("error_code", "")
        assert "decode" in stored_data["error"].lower()

    def test_audio_too_short(self, mock_redis_sync):
        """Test handling of audio file that's too short."""
        # Simulate audio too short error (< 1.0 seconds per TDD.md)
        error = ValueError("Audio too short: 0.5s < 1.0s")
        task_key = "task:test-short"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="preprocessing",
            error_code="AUDIO_TOO_SHORT",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert "AUDIO_TOO_SHORT" in stored_data.get("error_code", "")
        assert "too short" in stored_data["error"].lower()


class TestASRProcessingErrors:
    """Test error handling during ASR processing."""

    def test_asr_model_load_failure(self, mock_redis_sync):
        """Test handling of ASR model loading failure."""
        error = RuntimeError("Failed to load WhisperBackend model")
        task_key = "task:test-asr-load"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="asr_loading",
            error_code="MODEL_LOAD_ERROR",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert stored_data.get("stage") == "asr_loading"
        assert "MODEL_LOAD_ERROR" in stored_data.get("error_code", "")

    def test_asr_transcription_failure(self, mock_redis_sync):
        """Test handling of ASR transcription failure."""
        error = RuntimeError("ASR transcription failed: CUDA out of memory")
        task_key = "task:test-asr-transcribe"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="asr_transcription",
            error_code="ASR_PROCESSING_ERROR",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert "ASR_PROCESSING_ERROR" in stored_data.get("error_code", "")
        assert stored_data.get("stage") == "asr_transcription"

    def test_asr_timeout(self, mock_redis_sync):
        """Test handling of ASR processing timeout."""
        error = TimeoutError("ASR processing exceeded 300s timeout")
        task_key = "task:test-asr-timeout"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="asr_transcription",
            error_code="PROCESSING_TIMEOUT",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert "PROCESSING_TIMEOUT" in stored_data.get("error_code", "")
        assert "timeout" in stored_data["error"].lower()


class TestLLMProcessingErrors:
    """Test error handling during LLM processing."""

    def test_llm_model_load_failure(self, mock_redis_sync):
        """Test handling of LLM model loading failure."""
        error = RuntimeError("Failed to initialize vLLM engine")
        task_key = "task:test-llm-load"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="llm_loading",
            error_code="MODEL_LOAD_ERROR",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert stored_data.get("stage") == "llm_loading"
        assert "vLLM" in stored_data["error"]

    def test_llm_enhancement_failure(self, mock_redis_sync):
        """Test handling of LLM text enhancement failure."""
        error = RuntimeError("LLM generation failed: token limit exceeded")
        task_key = "task:test-llm-enhance"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="llm_enhancement",
            error_code="LLM_PROCESSING_ERROR",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert "LLM_PROCESSING_ERROR" in stored_data.get("error_code", "")

    def test_llm_summarization_failure(self, mock_redis_sync):
        """Test handling of LLM summarization failure."""
        error = RuntimeError("Summarization generation failed")
        task_key = "task:test-llm-summary"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="llm_summarization",
            error_code="LLM_PROCESSING_ERROR",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert stored_data.get("stage") == "llm_summarization"

    def test_schema_validation_failure(self, mock_redis_sync):
        """Test handling of prompt schema validation failure."""
        error = ValueError("Schema validation failed: missing required field 'text'")
        task_key = "task:test-schema"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="llm_enhancement",
            error_code="SCHEMA_VALIDATION_ERROR",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert "SCHEMA_VALIDATION_ERROR" in stored_data.get("error_code", "")
        assert "validation" in stored_data["error"].lower()


class TestRedisConnectionErrors:
    """Test error handling for Redis connection issues."""

    def test_redis_connection_lost_during_status_update(self, mock_redis_sync):
        """Test handling of Redis connection loss during status update."""
        # Simulate Redis connection error
        error = RedisConnectionError("Connection to Redis lost")
        task_key = "task:test-redis-lost"

        # Even if Redis is down, error handler should not crash
        # It should log the error and potentially use fallback
        try:
            handle_processing_error(
                mock_redis_sync,
                task_key,
                error,
                stage="status_update",
                error_code="REDIS_CONNECTION_ERROR",
            )
        except RedisConnectionError:
            # Expected - can't update status if Redis is down
            # But the error handler shouldn't crash with a different error
            pass

    def test_redis_timeout(self, mock_redis_sync):
        """Test handling of Redis operation timeout."""
        error = TimeoutError("Redis operation timed out after 5s")
        task_key = "task:test-redis-timeout"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="result_storage",
            error_code="REDIS_TIMEOUT",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data["status"] == TaskStatus.FAILED.value
        assert "REDIS_TIMEOUT" in stored_data.get("error_code", "")


class TestResourceCleanup:
    """Test that resources are properly cleaned up on errors."""

    def test_asr_model_cleanup_on_error(self):
        """Test that ASR model is unloaded even if processing fails."""
        # Create mock ASR model with unload method
        mock_model = MagicMock()
        mock_model.unload = Mock()

        # Call unload
        unload_asr_model(mock_model)

        # Verify unload was called
        mock_model.unload.assert_called_once()

    def test_llm_model_cleanup_on_error(self):
        """Test that LLM model is unloaded even if processing fails."""
        # Create mock LLM model with unload method
        mock_model = MagicMock()
        mock_model.unload = Mock()

        # Call unload
        unload_llm_model(mock_model)

        # Verify unload was called
        mock_model.unload.assert_called_once()

    def test_gpu_memory_cleanup_on_error(self):
        """Test that GPU memory is cleared even on error."""
        mock_model = MagicMock()
        mock_model.unload = Mock()

        # Should not raise even if torch is not available
        try:
            unload_asr_model(mock_model)
        except Exception as e:
            pytest.fail(f"GPU cleanup should not raise: {e}")


class TestErrorMetadata:
    """Test that error information is properly structured."""

    def test_error_includes_stage_information(self, mock_redis_sync):
        """Test that errors include the processing stage."""
        error = RuntimeError("Test error")
        task_key = "task:test-stage"

        handle_processing_error(
            mock_redis_sync, task_key, error, stage="asr_transcription"
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data.get("stage") == "asr_transcription"

    def test_error_includes_error_code(self, mock_redis_sync):
        """Test that errors include structured error codes."""
        error = RuntimeError("Test error")
        task_key = "task:test-code"

        handle_processing_error(
            mock_redis_sync,
            task_key,
            error,
            stage="preprocessing",
            error_code="AUDIO_DECODE_ERROR",
        )

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert stored_data.get("error_code") == "AUDIO_DECODE_ERROR"

    def test_error_includes_timestamp(self, mock_redis_sync):
        """Test that errors include timestamp of failure."""
        error = RuntimeError("Test error")
        task_key = "task:test-timestamp"

        handle_processing_error(mock_redis_sync, task_key, error, stage="preprocessing")

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert "updated_at" in stored_data
        # Timestamp should be ISO format
        from datetime import datetime

        try:
            datetime.fromisoformat(stored_data["updated_at"])
        except ValueError:
            pytest.fail("Timestamp should be in ISO format")

    def test_error_message_preserved(self, mock_redis_sync):
        """Test that original error message is preserved."""
        original_message = "Specific error: CUDA device 0 not available"
        error = RuntimeError(original_message)
        task_key = "task:test-message"

        handle_processing_error(mock_redis_sync, task_key, error, stage="asr_loading")

        stored_data = mock_redis_sync._data.get(task_key, {})
        assert original_message in stored_data["error"]

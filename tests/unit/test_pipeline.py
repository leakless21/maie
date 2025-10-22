"""
Unit tests for src/worker/pipeline.py

Tests the sequential audio processing pipeline including:
- Helper functions (_sanitize_metadata, _update_status, _calculate_edit_rate)
- Model loading/unloading (ASR and LLM)
- Processing functions (ASR transcription, LLM enhancement/summarization)
- Main pipeline function (process_audio_task)
- Error handling and edge cases
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.api.errors import (
    ASRProcessingError,
    LLMProcessingError,
    ModelLoadError,
)
from src.api.schemas import TaskStatus
from src.worker.pipeline import (
    _calculate_edit_rate,
    _sanitize_metadata,
    _update_status,
    calculate_metrics,
    execute_asr_transcription,
    execute_llm_processing,
    get_version_metadata,
    load_asr_model,
    load_llm_model,
    process_audio_task,
    unload_asr_model,
    unload_llm_model,
)

try:
    import fakeredis

    FAKEREDIS_AVAILABLE = True
except ImportError:
    fakeredis = None  # type: ignore
    FAKEREDIS_AVAILABLE = False


class TestSanitizeMetadata:
    """Test _sanitize_metadata helper function."""

    def test_sanitize_primitives(self):
        """Test sanitizing primitive types."""
        assert _sanitize_metadata("string") == "string"
        assert _sanitize_metadata(42) == 42
        assert _sanitize_metadata(3.14) == 3.14
        assert _sanitize_metadata(True) == True
        assert _sanitize_metadata(None) is None

    def test_sanitize_dict(self):
        """Test sanitizing dictionary."""
        input_dict = {"key": "value", "number": 42, "nested": {"inner": "data"}}
        result = _sanitize_metadata(input_dict)
        assert result == {"key": "value", "number": 42, "nested": {"inner": "data"}}

    def test_sanitize_list(self):
        """Test sanitizing list."""
        input_list = ["a", 1, {"key": "value"}]
        result = _sanitize_metadata(input_list)
        assert result == ["a", 1, {"key": "value"}]

    def test_sanitize_pydantic_model(self):
        """Test sanitizing Pydantic model."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=123)
        result = _sanitize_metadata(model)
        assert result == {"name": "test", "value": 123}

    def test_sanitize_complex_object(self):
        """Test sanitizing complex objects falls back to string."""

        class ComplexObject:
            def __init__(self):
                self.data = "complex"

        obj = ComplexObject()
        result = _sanitize_metadata(obj)
        assert isinstance(result, str)
        assert "ComplexObject" in result or "complex" in result


class TestUpdateStatus:
    """Test _update_status helper function."""

    @pytest.fixture
    def redis_client(self):
        """Provide a fakeredis client for testing."""
        if not FAKEREDIS_AVAILABLE:
            pytest.skip("fakeredis not available")
        return fakeredis.FakeRedis()

    def test_update_status_basic(self, redis_client):
        """Test basic status update."""
        _update_status(redis_client, "task:123", TaskStatus.PROCESSING_ASR)

        data = redis_client.hgetall("task:123")
        assert data[b"status"] == b"PROCESSING_ASR"
        assert b"updated_at" in data

    def test_update_status_with_details(self, redis_client):
        """Test status update with additional details."""
        details = {"progress": 50, "message": "Halfway done"}

        _update_status(redis_client, "task:123", TaskStatus.PROCESSING_LLM, details)

        data = redis_client.hgetall("task:123")
        assert data[b"status"] == b"PROCESSING_LLM"
        assert data[b"progress"] == b"50"
        assert data[b"message"] == b"Halfway done"
        assert b"updated_at" in data

    def test_update_status_with_complex_details(self, redis_client):
        """Test status update with complex details that need JSON serialization."""
        details = {"metadata": {"model": "test"}, "segments": [1, 2, 3]}

        _update_status(redis_client, "task:123", TaskStatus.COMPLETE, details)

        data = redis_client.hgetall("task:123")
        assert data[b"status"] == b"COMPLETE"
        assert data[b"metadata"] == json.dumps({"model": "test"}).encode()
        assert data[b"segments"] == json.dumps([1, 2, 3]).encode()


class TestCalculateEditRate:
    """Test _calculate_edit_rate helper function."""

    def test_identical_strings(self):
        """Test edit rate for identical strings."""
        rate = _calculate_edit_rate("hello", "hello")
        assert rate == 0.0

    def test_completely_different_strings(self):
        """Test edit rate for completely different strings."""
        rate = _calculate_edit_rate("abc", "xyz")
        assert rate == 1.0

    def test_empty_strings(self):
        """Test edit rate for empty strings."""
        assert _calculate_edit_rate("", "") == 0.0
        assert _calculate_edit_rate("hello", "") == 1.0
        assert _calculate_edit_rate("", "world") == 1.0

    def test_partial_similarity(self):
        """Test edit rate for partially similar strings."""
        # "kitten" -> "sitting" has edit distance 3, max length 7
        rate = _calculate_edit_rate("kitten", "sitting")
        expected = 3 / 7  # 0.428...
        assert abs(rate - expected) < 0.001

    def test_case_sensitivity(self):
        """Test that function is case sensitive."""
        rate = _calculate_edit_rate("Hello", "hello")
        assert rate > 0.0  # Should have some difference


class TestLoadASRModel:
    """Test load_asr_model function."""

    @patch("src.processors.asr.factory.ASRFactory")
    def test_load_whisper_backend(self, mock_factory):
        """Test loading whisper backend."""
        mock_model = Mock()
        mock_factory.create.return_value = mock_model

        result = load_asr_model("whisper", device="cuda")

        mock_factory.create.assert_called_once_with(
            backend_type="whisper", device="cuda"
        )
        assert result == mock_model

    @patch("src.processors.asr.factory.ASRFactory")
    def test_load_chunkformer_backend(self, mock_factory):
        """Test loading chunkformer backend."""
        mock_model = Mock()
        mock_factory.create.return_value = mock_model

        result = load_asr_model("chunkformer", model_path="/path/to/model")

        mock_factory.create.assert_called_once_with(
            backend_type="chunkformer", model_path="/path/to/model"
        )
        assert result == mock_model

    def test_invalid_backend(self):
        """Test loading invalid backend raises ModelLoadError."""
        with pytest.raises(ModelLoadError) as exc_info:
            load_asr_model("invalid_backend")

        assert "Invalid ASR backend" in str(exc_info.value)
        assert exc_info.value.details["backend"] == "invalid_backend"

    @patch("src.processors.asr.factory.ASRFactory")
    def test_factory_exception(self, mock_factory):
        """Test handling of factory exceptions."""
        mock_factory.create.side_effect = RuntimeError("Factory error")

        with pytest.raises(ModelLoadError) as exc_info:
            load_asr_model("whisper")

        assert "Failed to load ASR model" in str(exc_info.value)


class TestUnloadASRModel:
    """Test unload_asr_model function."""

    def test_unload_with_unload_method(self, mocker):
        """Test unloading model with unload method."""
        mock_model = Mock()
        mock_torch = mocker.patch("src.worker.pipeline.torch")
        mock_torch.cuda.is_available.return_value = True

        unload_asr_model(mock_model)

        mock_model.unload.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_unload_without_unload_method(self, mocker):
        """Test unloading model without unload method."""
        mock_model = Mock(spec=[])  # No unload method
        mock_torch = mocker.patch("src.worker.pipeline.torch")
        mock_torch.cuda.is_available.return_value = True

        unload_asr_model(mock_model)

        # Should not call unload since it doesn't exist
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_unload_without_cuda(self, mocker):
        """Test unloading when CUDA is not available."""
        mock_model = Mock()
        mock_torch = mocker.patch("src.worker.pipeline.torch")
        mock_torch.cuda.is_available.return_value = False

        unload_asr_model(mock_model)

        mock_model.unload.assert_called_once()
        mock_torch.cuda.empty_cache.assert_not_called()

    def test_unload_none_model(self, mocker):
        """Test unloading None model."""
        mock_torch = mocker.patch("src.worker.pipeline.torch")
        mock_torch.cuda.is_available.return_value = True

        unload_asr_model(None)

        mock_torch.cuda.empty_cache.assert_called_once()

    def test_unload_exception_handling(self, mocker):
        """Test exception handling during unload."""
        mock_model = Mock()
        mock_model.unload.side_effect = Exception("Unload error")
        mock_torch = mocker.patch("src.worker.pipeline.torch")
        mock_torch.cuda.is_available.return_value = True

        # Should not raise exception
        unload_asr_model(mock_model)

        # empty_cache is in the same try block as unload, so it won't be called if unload fails
        mock_torch.cuda.empty_cache.assert_not_called()


class TestLoadLLMModel:
    """Test load_llm_model function."""

    @patch("src.processors.llm.LLMProcessor")
    def test_load_llm_model(self, mock_processor_class):
        """Test loading LLM model."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        result = load_llm_model()

        mock_processor_class.assert_called_once()
        mock_processor._load_model.assert_called_once()
        assert result == mock_processor

    @patch("src.processors.llm.LLMProcessor")
    def test_load_llm_model_exception(self, mock_processor_class):
        """Test exception during LLM model loading."""
        mock_processor_class.side_effect = Exception("Load error")

        with pytest.raises(ModelLoadError) as exc_info:
            load_llm_model()

        assert "Failed to load LLM model" in str(exc_info.value)


class TestUnloadLLMModel:
    """Test unload_llm_model function."""

    def test_unload_with_unload_method(self, mocker):
        """Test unloading LLM model with unload method."""
        mock_model = Mock()
        mock_torch = mocker.patch("src.worker.pipeline.torch")
        mock_torch.cuda.is_available.return_value = True

        unload_llm_model(mock_model)

        mock_model.unload.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_unload_without_cuda(self, mocker):
        """Test unloading when CUDA is not available."""
        mock_model = Mock()
        mock_torch = mocker.patch("src.worker.pipeline.torch")
        mock_torch.cuda.is_available.return_value = False

        unload_llm_model(mock_model)

        mock_model.unload.assert_called_once()
        mock_torch.cuda.empty_cache.assert_not_called()

    def test_unload_none_model(self, mocker):
        """Test unloading None LLM model."""
        mock_torch = mocker.patch("src.worker.pipeline.torch")
        mock_torch.cuda.is_available.return_value = True

        unload_llm_model(None)

        mock_torch.cuda.empty_cache.assert_called_once()


class TestExecuteASRTranscription:
    """Test execute_asr_transcription function."""

    def test_successful_transcription(self, mocker):
        """Test successful ASR transcription."""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.text = "Hello world"
        mock_result.confidence = 0.95
        mock_result.model_name = "test-model"
        mock_result.checkpoint_hash = "abc123"
        mock_model.execute.return_value = mock_result

        # Mock time to control processing time
        mock_time = mocker.patch("src.worker.pipeline.time")
        mock_time.time.side_effect = [1000.0, 1001.0]  # 1 second processing time

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            # Phase 1: Now returns (ASRResult, rtf, metadata)
            asr_result, rtf, metadata = execute_asr_transcription(
                mock_model, audio_path, 2.0
            )

            assert asr_result.text == "Hello world"
            assert rtf == 0.5  # 1.0 / 2.0
            assert asr_result.confidence == 0.95
            assert metadata["model_name"] == "test-model"
            assert metadata["checkpoint_hash"] == "abc123"
        finally:
            Path(audio_path).unlink()

    def test_transcription_with_transcript_attr(self, mocker):
        """Test transcription result with transcript attribute."""
        from src.processors.base import ASRResult
        
        mock_model = Mock()
        # Mock an ASRResult with text field (standard interface)
        mock_result = ASRResult(text="Alternative text", confidence=0.88)
        mock_model.execute.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            # Phase 1: Now returns (ASRResult, rtf, metadata)
            asr_result, rtf, metadata = execute_asr_transcription(
                mock_model, audio_path, 1.0
            )

            assert asr_result.text == "Alternative text"
            assert asr_result.confidence == 0.88
        finally:
            Path(audio_path).unlink()

    def test_file_not_found(self, mocker):
        """Test handling of file not found error."""
        mock_model = Mock()

        with pytest.raises(ASRProcessingError) as exc_info:
            execute_asr_transcription(mock_model, "/nonexistent/file.wav", 1.0)

        assert "Audio file not found" in str(exc_info.value)

    def test_transcription_exception(self, mocker):
        """Test handling of transcription exceptions."""
        mock_model = Mock()
        mock_model.execute.side_effect = Exception("Transcription failed")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            with pytest.raises(ASRProcessingError) as exc_info:
                execute_asr_transcription(mock_model, audio_path, 1.0)

            assert "ASR transcription failed" in str(exc_info.value)
        finally:
            Path(audio_path).unlink()


class TestExecuteLLMProcessing:
    """Test execute_llm_processing function."""

    def test_text_enhancement_needed(self, mocker):
        """Test LLM processing with text enhancement needed."""
        mock_llm = Mock()
        mock_llm.needs_enhancement.return_value = True
        mock_llm.enhance_text.return_value = {
            "enhanced_text": "Enhanced: Hello world",
            "enhancement_applied": True,
            "edit_rate": 0.2,
            "edit_distance": 2,
        }
        mock_llm.generate_summary.return_value = {
            "summary": "A greeting",
            "retry_count": 0,
            "model_info": {"name": "test-llm"},
        }

        result_transcript, result_summary = execute_llm_processing(
            mock_llm,
            "Hello world",
            ["clean_transcript", "summary"],
            "meeting_notes_v1",
            "whisper",
        )

        assert result_transcript == "Enhanced: Hello world"
        assert result_summary == "A greeting"
        mock_llm.enhance_text.assert_called_once_with("Hello world")
        mock_llm.generate_summary.assert_called_once()

    def test_text_enhancement_not_needed(self, mocker):
        """Test LLM processing with text enhancement not needed."""
        mock_llm = Mock()
        mock_llm.needs_enhancement.return_value = False
        mock_llm.generate_summary.return_value = {
            "summary": "A greeting",
            "retry_count": 0,
            "model_info": {"name": "test-llm"},
        }

        result_transcript, result_summary = execute_llm_processing(
            mock_llm,
            "Hello world",
            ["clean_transcript", "summary"],
            "meeting_notes_v1",
            "chunkformer",
        )

        assert result_transcript == "Hello world"  # Original text
        assert result_summary == "A greeting"
        mock_llm.enhance_text.assert_not_called()

    def test_enhancement_failure_fallback(self, mocker):
        """Test fallback when text enhancement fails."""
        mock_llm = Mock()
        mock_llm.needs_enhancement.return_value = True
        mock_llm.enhance_text.side_effect = Exception("Enhancement failed")
        mock_llm.generate_summary.return_value = {
            "summary": "A greeting",
            "retry_count": 0,
            "model_info": {"name": "test-llm"},
        }

        result_transcript, result_summary = execute_llm_processing(
            mock_llm,
            "Hello world",
            ["clean_transcript", "summary"],
            "meeting_notes_v1",
            "whisper",
        )

        assert result_transcript == "Hello world"  # Fallback to original
        assert result_summary == "A greeting"

    def test_summary_only(self, mocker):
        """Test LLM processing with summary only."""
        mock_llm = Mock()
        mock_llm.generate_summary.return_value = {
            "summary": "Summary only",
            "retry_count": 1,
            "model_info": {"name": "test-llm"},
        }

        result_transcript, result_summary = execute_llm_processing(
            mock_llm, "Hello world", ["summary"], "meeting_notes_v1", "whisper"
        )

        assert result_transcript == "Hello world"  # Always returns clean_transcript
        assert result_summary == "Summary only"

    def test_missing_template_id_for_summary(self, mocker):
        """Test error when template_id missing for summary feature."""
        mock_llm = Mock()

        with pytest.raises(LLMProcessingError) as exc_info:
            execute_llm_processing(
                mock_llm, "Hello world", ["summary"], None, "whisper"
            )

        assert "template_id required" in str(exc_info.value)

    def test_summary_generation_failure(self, mocker):
        """Test handling of summary generation failure."""
        mock_llm = Mock()
        mock_llm.needs_enhancement.return_value = False
        mock_llm.generate_summary.return_value = {
            "error": "Summary failed",
            "retry_count": 2,
        }

        with pytest.raises(LLMProcessingError) as exc_info:
            execute_llm_processing(
                mock_llm, "Hello world", ["summary"], "meeting_notes_v1", "whisper"
            )

        assert "Summary generation failed" in str(exc_info.value)

    def test_unexpected_exception(self, mocker):
        """Test handling of unexpected exceptions."""
        mock_llm = Mock()
        mock_llm.needs_enhancement.side_effect = Exception("Unexpected error")

        with pytest.raises(LLMProcessingError) as exc_info:
            execute_llm_processing(
                mock_llm, "Hello world", ["clean_transcript"], None, "whisper"
            )

        assert "LLM processing failed" in str(exc_info.value)


class TestGetVersionMetadata:
    """Test get_version_metadata function."""

    def test_with_llm_model(self, mocker):
        """Test version metadata collection with LLM model."""
        mock_llm = Mock()
        mock_llm.get_version_info.return_value = {
            "model_name": "test-llm",
            "version": "1.0.0",
            "checkpoint": "abc123",
        }

        asr_metadata = {"model_name": "whisper-base", "language": "en"}
        result = get_version_metadata(asr_metadata, mock_llm)

        assert result["processing_pipeline"] == "1.0.0"  # From settings mock
        assert result["asr"]["model_name"] == "whisper-base"
        assert result["llm"]["model_name"] == "test-llm"

    def test_without_llm_model(self, mocker):
        """Test version metadata collection without LLM model."""
        asr_metadata = {"model_name": "whisper-base"}
        result = get_version_metadata(asr_metadata, None)

        assert result["llm"]["model_name"] == "not_loaded"
        assert result["llm"]["reason"] == "no_model_provided"

    def test_llm_without_get_version_info(self, mocker):
        """Test version metadata when LLM model lacks get_version_info."""
        mock_llm = Mock(spec=[])  # No get_version_info method

        result = get_version_metadata({}, mock_llm)

        assert result["llm"]["model_name"] == "not_loaded"
        assert result["llm"]["reason"] == "method_missing"

    def test_llm_version_info_exception(self, mocker):
        """Test handling of exceptions in LLM version info collection."""
        mock_llm = Mock()
        mock_llm.get_version_info.side_effect = Exception("Version error")

        result = get_version_metadata({}, mock_llm)

        assert result["llm"]["model_name"] == "unavailable"
        assert "error" in result["llm"]


class TestCalculateMetrics:
    """Test calculate_metrics function."""

    def test_basic_metrics(self, mocker):
        """Test basic metrics calculation."""
        start_time = 1000.0
        mocker.patch("src.worker.pipeline.time.time", return_value=1010.0)

        metrics = calculate_metrics("Hello world", None, start_time, 5.0, 0.8)

        assert metrics["total_processing_time"] == 10.0
        assert metrics["total_rtf"] == 2.0  # 10.0 / 5.0
        assert metrics["asr_rtf"] == 0.8
        assert metrics["transcription_length"] == 11
        assert metrics["audio_duration"] == 5.0

    def test_with_text_enhancement(self, mocker):
        """Test metrics calculation with text enhancement."""
        start_time = 1000.0
        mocker.patch("src.worker.pipeline.time.time", return_value=1010.0)

        metrics = calculate_metrics(
            "Hello world", "Hello, world!", start_time, 5.0, 0.8
        )

        assert "edit_rate_cleaning" in metrics
        assert metrics["edit_rate_cleaning"] > 0.0  # Should have some difference

    def test_zero_duration(self, mocker):
        """Test metrics calculation with zero audio duration."""
        start_time = 1000.0
        mocker.patch("src.worker.pipeline.time.time", return_value=1010.0)

        metrics = calculate_metrics("Hello", None, start_time, 0.0, 0.0)

        assert metrics["total_rtf"] == 0.0
        assert metrics["asr_rtf"] == 0.0


class TestProcessAudioTask:
    """Test process_audio_task function."""

    @pytest.fixture
    def redis_client(self):
        """Provide a fakeredis client for testing."""
        if not FAKEREDIS_AVAILABLE:
            pytest.skip("fakeredis not available")
        return fakeredis.FakeRedis()

    def test_successful_processing(self, mocker, redis_client):
        """Test successful audio processing pipeline."""
        # Mock all dependencies
        with patch("src.processors.audio.AudioPreprocessor") as mock_preprocessor_class:
            mock_preprocessor = Mock()
            mock_preprocessor_class.return_value = mock_preprocessor
            mock_preprocessor.preprocess.return_value = {
                "duration": 5.0,
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "normalized_path": None,
            }

            mock_asr_model = Mock()
            mock_asr_result = Mock()
            mock_asr_result.text = "Hello world"
            mock_asr_result.confidence = 0.95
            mock_asr_result.model_name = "test-asr"
            mock_asr_result.segments = []  # Proper empty list instead of Mock
            mock_asr_model.execute.return_value = mock_asr_result

            mock_llm_model = Mock()
            mock_llm_model.needs_enhancement.return_value = False
            mock_llm_model.get_version_info.return_value = {"model_name": "test-llm"}
            mock_llm_model.generate_summary.return_value = {
                "summary": "A greeting",
                "retry_count": 0,
                "model_info": {"name": "test-llm"},
            }

            # Mock the loading functions
            mocker.patch(
                "src.worker.pipeline.load_asr_model", return_value=mock_asr_model
            )
            mocker.patch(
                "src.worker.pipeline.load_llm_model", return_value=mock_llm_model
            )
            mocker.patch("src.worker.pipeline.unload_asr_model")
            mocker.patch("src.worker.pipeline.unload_llm_model")

            # Mock RQ job
            mock_job = Mock()
            mock_job.id = "test-job-123"
            mocker.patch("src.worker.pipeline.get_current_job", return_value=mock_job)

            # Mock Redis connection to use our fakeredis client
            mocker.patch("src.worker.pipeline.Redis", return_value=redis_client)

            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            try:
                task_params = {
                    "audio_path": audio_path,
                    "asr_backend": "whisper",
                    "features": ["clean_transcript", "summary"],
                    "template_id": "meeting_notes_v1",
                }

                result = process_audio_task(task_params)

                # Verify result structure
                assert "versions" in result
                assert "metrics" in result
                assert "results" in result
                assert result["results"]["transcript"] == "Hello world"
                assert result["results"]["summary"] == "A greeting"

                # Verify Redis updates were made
                task_data = redis_client.hgetall("task:test-job-123")
                assert task_data[b"status"] == b"COMPLETE"
                assert b"updated_at" in task_data

            finally:
                Path(audio_path).unlink()

    def test_audio_validation_error(self, mocker, redis_client):
        """Test handling of audio validation errors."""
        mock_job = Mock()
        mock_job.id = "test-job-123"
        mocker.patch("src.worker.pipeline.get_current_job", return_value=mock_job)

        # Mock Redis connection to use our fakeredis client
        mocker.patch("src.worker.pipeline.Redis", return_value=redis_client)

        task_params = {
            "audio_path": None,  # Invalid audio path
            "asr_backend": "whisper",
            "features": ["clean_transcript"],
        }

        result = process_audio_task(task_params)

        assert result["status"] == "error"
        assert "error" in result
        assert result["error"]["code"] == "AUDIO_VALIDATION_ERROR"

        # Verify error status was stored in Redis
        task_data = redis_client.hgetall("task:test-job-123")
        assert task_data[b"status"] == b"FAILED"
        assert b"error_code" in task_data

    def test_preprocessing_error(self, mocker, redis_client):
        """Test handling of audio preprocessing errors."""
        with patch("src.processors.audio.AudioPreprocessor") as mock_preprocessor_class:
            mock_preprocessor = Mock()
            mock_preprocessor_class.return_value = mock_preprocessor
            mock_preprocessor.preprocess.side_effect = Exception("Preprocessing failed")

            mock_job = Mock()
            mock_job.id = "test-job-123"
            mocker.patch("src.worker.pipeline.get_current_job", return_value=mock_job)

            # Mock Redis connection to use our fakeredis client
            mocker.patch("src.worker.pipeline.Redis", return_value=redis_client)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            try:
                task_params = {
                    "audio_path": audio_path,
                    "asr_backend": "whisper",
                    "features": ["clean_transcript"],
                }

                result = process_audio_task(task_params)

                assert result["status"] == "error"
                assert "error" in result
                assert result["error"]["code"] == "AUDIO_PREPROCESSING_ERROR"

                # Verify error status was stored in Redis
                task_data = redis_client.hgetall("task:test-job-123")
                assert task_data[b"status"] == b"FAILED"

            finally:
                Path(audio_path).unlink()

    def test_asr_processing_error(self, mocker, redis_client):
        """Test handling of ASR processing errors."""
        # Mock successful preprocessing
        with patch("src.processors.audio.AudioPreprocessor") as mock_preprocessor_class:
            mock_preprocessor = Mock()
            mock_preprocessor_class.return_value = mock_preprocessor
            mock_preprocessor.preprocess.return_value = {
                "duration": 5.0,
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "normalized_path": None,
            }

            # Mock ASR loading success but execution failure
            mock_asr_model = Mock()
            mock_asr_model.execute.side_effect = Exception("ASR failed")
            mocker.patch(
                "src.worker.pipeline.load_asr_model", return_value=mock_asr_model
            )
            mocker.patch("src.worker.pipeline.unload_asr_model")

            mock_job = Mock()
            mock_job.id = "test-job-123"
            mocker.patch("src.worker.pipeline.get_current_job", return_value=mock_job)

            # Mock Redis connection to use our fakeredis client
            mocker.patch("src.worker.pipeline.Redis", return_value=redis_client)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            try:
                task_params = {
                    "audio_path": audio_path,
                    "asr_backend": "whisper",
                    "features": ["clean_transcript"],
                }

                result = process_audio_task(task_params)

                assert result["status"] == "error"
                assert "error" in result
                assert result["error"]["code"] == "ASR_PROCESSING_ERROR"

                # Verify error status was stored in Redis
                task_data = redis_client.hgetall("task:test-job-123")
                assert task_data[b"status"] == b"FAILED"

            finally:
                Path(audio_path).unlink()

    def test_unexpected_error(self, mocker, redis_client):
        """Test handling of unexpected errors."""
        # Mock an unexpected error in preprocessing
        with patch("src.processors.audio.AudioPreprocessor") as mock_preprocessor_class:
            mock_preprocessor = Mock()
            mock_preprocessor_class.return_value = mock_preprocessor
            mock_preprocessor.preprocess.side_effect = RuntimeError("Unexpected error")

            mock_job = Mock()
            mock_job.id = "test-job-123"
            mocker.patch("src.worker.pipeline.get_current_job", return_value=mock_job)

            # Mock Redis connection to use our fakeredis client
            mocker.patch("src.worker.pipeline.Redis", return_value=redis_client)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            try:
                task_params = {
                    "audio_path": audio_path,
                    "asr_backend": "whisper",
                    "features": ["clean_transcript"],
                }

                result = process_audio_task(task_params)

                assert result["status"] == "error"
                assert "error" in result
                assert result["error"]["code"] == "AUDIO_PREPROCESSING_ERROR"

                # Verify error status was stored in Redis
                task_data = redis_client.hgetall("task:test-job-123")
                assert task_data[b"status"] == b"FAILED"

            finally:
                Path(audio_path).unlink()

    def test_without_redis(self, mocker):
        """Test processing without Redis (no job context)."""
        # Mock all dependencies
        with patch("src.processors.audio.AudioPreprocessor") as mock_preprocessor_class:
            mock_preprocessor = Mock()
            mock_preprocessor_class.return_value = mock_preprocessor
            mock_preprocessor.preprocess.return_value = {
                "duration": 5.0,
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "normalized_path": None,
            }

            mock_asr_model = Mock()
            mock_asr_result = Mock()
            mock_asr_result.text = "Hello world"
            mock_asr_result.confidence = 0.95
            mock_asr_result.segments = []  # Proper empty list instead of Mock
            mock_asr_model.execute.return_value = mock_asr_result

            mock_llm_model = Mock()
            mock_llm_model.needs_enhancement.return_value = False
            mock_llm_model.generate_summary.return_value = {
                "summary": "A greeting",
                "retry_count": 0,
                "model_info": {"name": "test-llm"},
            }

            # Mock the loading functions
            mocker.patch(
                "src.worker.pipeline.load_asr_model", return_value=mock_asr_model
            )
            mocker.patch(
                "src.worker.pipeline.load_llm_model", return_value=mock_llm_model
            )
            mocker.patch("src.worker.pipeline.unload_asr_model")
            mocker.patch("src.worker.pipeline.unload_llm_model")

            # No Redis/job mocking - should work without Redis
            mocker.patch("src.worker.pipeline.get_current_job", return_value=None)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            try:
                task_params = {
                    "audio_path": audio_path,
                    "asr_backend": "whisper",
                    "features": ["clean_transcript", "summary"],
                    "template_id": "meeting_notes_v1",
                }

                result = process_audio_task(task_params)

                # Should still succeed
                assert "results" in result
                assert result["results"]["transcript"] == "Hello world"

            finally:
                Path(audio_path).unlink()

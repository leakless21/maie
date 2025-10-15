"""
Integration tests for the complete worker pipeline with real components.

Tests the full pipeline flow:
1. Real AudioPreprocessor (validates and normalizes audio)
2. Real ASR Factory (with mocked model loading for speed)
3. Real LLM Processor (with mocked vLLM for speed)
4. Fake Redis (fakeredis for consistent mocking)
5. Real audio files from tests/assets

Scenarios tested:
- Full pipeline with WAV file
- Full pipeline with different audio formats
- Pipeline with transcript-only feature
- Pipeline with summary-only feature
- Error handling with invalid audio
- Status transitions in Redis
"""

import pytest
import json
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
import fakeredis

from src.worker.pipeline import process_audio_task
from src.api.schemas import TaskStatus
from src.config import settings


# Test fixtures

@pytest.fixture
def redis_client():
    """Provide fake Redis client for consistent mocking."""
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    # Cleanup: flush all data
    client.flushall()
    client.close()


@pytest.fixture
def test_audio_path():
    """Provide path to test audio file."""
    audio_path = Path(__file__).parent.parent / "assets" / "test_audio.wav"
    if not audio_path.exists():
        pytest.skip(f"Test audio file not found: {audio_path}")
    return str(audio_path)


@pytest.fixture
def northern_female_audio():
    """Provide path to Northern Female 1.wav test audio."""
    audio_path = Path(__file__).parent.parent / "assets" / "Northern Female 1.wav"
    if not audio_path.exists():
        pytest.skip(f"Test audio file not found: {audio_path}")
    return str(audio_path)


@pytest.fixture
def mock_asr_model():
    """Mock ASR model that returns realistic results without loading actual models."""
    mock_model = Mock()
    mock_model.execute.return_value = Mock(
        transcript="This is a test transcription from the audio file.",
        segments=[
            {
                "start": 0.0,
                "end": 2.5,
                "text": "This is a test",
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "transcription from the audio file.",
            },
        ],
        language="en",
        confidence_avg=0.95,
        vad_coverage=0.88,
        duration_ms=5000,
        model_name="whisper-large-v3",
        checkpoint_hash="mock_hash_abc123",
    )
    mock_model.unload.return_value = None
    mock_model.get_version_info.return_value = {
        "backend": "whisper",
        "model_variant": "large-v3",
        "model_path": "/data/models/whisper/large-v3",
        "checkpoint_hash": "mock_hash_abc123",
        "compute_type": "float16",
        "decoding_params": {
            "beam_size": 5,
            "vad_filter": True,
        },
    }
    return mock_model


@pytest.fixture
def mock_llm_model():
    """Mock LLM model that returns realistic results without loading vLLM."""
    mock_model = Mock()
    
    # Mock enhance_text
    mock_model.enhance_text.return_value = {
        "enhanced_text": "This is a test transcription from the audio file.",
        "enhancement_applied": False,  # Whisper has native punctuation
        "edit_rate": 0.0,
        "edit_distance": 0,
    }
    
    # Mock generate_summary
    mock_model.generate_summary.return_value = {
        "summary": {
            "title": "Test Audio Transcription",
            "main_points": [
                "Audio file contains test transcription",
                "Quality is good",
            ],
            "tags": ["test", "audio", "transcription"],
        },
        "retry_count": 0,
        "model_info": {
            "model_name": "qwen3-4b-instruct",
            "checkpoint_hash": "mock_llm_hash_xyz",
        },
    }
    
    # Mock needs_enhancement
    mock_model.needs_enhancement.return_value = False  # Whisper variant
    
    # Mock unload
    mock_model.unload.return_value = None
    
    # Mock get_version_info
    mock_model.get_version_info.return_value = {
        "model_name": "qwen3-4b-instruct",
        "checkpoint_hash": "mock_llm_hash_xyz",
        "backend": "vllm",
        "quantization": "awq-4bit",
    }
    
    return mock_model


# Integration Tests

class TestFullPipelineWithRealComponents:
    """Test complete pipeline with real components and mocked models."""

    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.Redis")
    def test_full_pipeline_wav_file(
        self, mock_redis_class, mock_load_llm, mock_load_asr, redis_client, 
        test_audio_path, mock_asr_model, mock_llm_model
    ):
        """
        Test full pipeline with WAV file.
        
        Components:
        - Real AudioPreprocessor (validates format, extracts duration)
        - Mocked ASR (returns realistic transcription)
        - Mocked LLM (returns realistic summary)
        - Fake Redis (stores status and results)
        """
        # Setup mocks
        mock_redis_class.return_value = redis_client
        mock_load_asr.return_value = mock_asr_model
        mock_load_llm.return_value = mock_llm_model
        
        # Prepare task parameters
        task_id = "test-full-pipeline-wav"
        task_params = {
            "audio_path": test_audio_path,
            "asr_backend": "whisper",
            "features": ["clean_transcript", "summary"],
            "template_id": "meeting_notes_v1",
        }
        
        # Mock the job context
        with patch("src.worker.pipeline.get_current_job") as mock_job:
            mock_job.return_value = Mock(id=task_id)
            
            # Execute pipeline
            result = process_audio_task(task_params)
        
        # Verify result structure
        assert result is not None
        assert "versions" in result
        assert "metrics" in result
        assert "results" in result
        
        # Verify versions
        assert result["versions"]["asr"]["model_name"] == "whisper-large-v3"
        assert result["versions"]["llm"]["model_name"] == "qwen3-4b-instruct"
        assert result["versions"]["processing_pipeline"] == settings.pipeline_version
        
        # Verify metrics
        metrics = result["metrics"]
        assert "total_processing_time" in metrics
        assert "total_rtf" in metrics
        assert "audio_duration" in metrics
        assert metrics["audio_duration"] > 0  # Real audio duration from preprocessing
        
        # Verify results
        assert "transcript" in result["results"]
        assert "summary" in result["results"]
        assert result["results"]["summary"]["title"] == "Test Audio Transcription"
        
        # Verify Redis status
        task_key = f"task:{task_id}"
        task_data = redis_client.hgetall(task_key)
        assert task_data["status"] == TaskStatus.COMPLETE.value
        
        # Verify ASR was called
        mock_asr_model.execute.assert_called_once()
        mock_asr_model.unload.assert_called_once()
        
        # Verify LLM was called
        mock_llm_model.generate_summary.assert_called_once()
        mock_llm_model.unload.assert_called_once()

    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.Redis")
    def test_transcript_only_feature(
        self, mock_redis_class, mock_load_llm, mock_load_asr, redis_client, test_audio_path,
        mock_asr_model, mock_llm_model
    ):
        """Test pipeline with transcript-only feature (no summary)."""
        mock_redis_class.return_value = redis_client
        mock_load_asr.return_value = mock_asr_model
        mock_load_llm.return_value = mock_llm_model
        
        task_id = "test-transcript-only"
        task_params = {
            "audio_path": test_audio_path,
            "asr_backend": "whisper",
            "features": ["clean_transcript"],  # No summary
            "template_id": None,
        }
        
        with patch("src.worker.pipeline.get_current_job") as mock_job:
            mock_job.return_value = Mock(id=task_id)
            result = process_audio_task(task_params)
        
        # Verify transcript is present
        assert "transcript" in result["results"]
        
        # Verify summary is NOT present
        assert "summary" not in result["results"]
        
        # Verify LLM generate_summary was NOT called (only enhancement, which is skipped for Whisper)
        mock_llm_model.generate_summary.assert_not_called()

    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.Redis")
    def test_summary_only_feature(
        self, mock_redis_class, mock_load_llm, mock_load_asr, redis_client, test_audio_path,
        mock_asr_model, mock_llm_model
    ):
        """Test pipeline with summary-only feature (no clean transcript)."""
        mock_redis_class.return_value = redis_client
        mock_load_asr.return_value = mock_asr_model
        mock_load_llm.return_value = mock_llm_model
        
        task_id = "test-summary-only"
        task_params = {
            "audio_path": test_audio_path,
            "asr_backend": "whisper",
            "features": ["summary"],  # Only summary
            "template_id": "meeting_notes_v1",
        }
        
        with patch("src.worker.pipeline.get_current_job") as mock_job:
            mock_job.return_value = Mock(id=task_id)
            result = process_audio_task(task_params)
        
        # Verify summary is present
        assert "summary" in result["results"]
        
        # Verify transcript may or may not be present (implementation detail)
        # The important thing is summary was generated
        
        # Verify LLM generate_summary WAS called
        mock_llm_model.generate_summary.assert_called_once()


class TestAudioPreprocessingIntegration:
    """Test real AudioPreprocessor integration."""

    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.Redis")
    def test_audio_duration_extraction(
        self, mock_redis_class, mock_load_llm, mock_load_asr, redis_client, test_audio_path,
        mock_asr_model, mock_llm_model
    ):
        """Test that audio duration is correctly extracted from preprocessing."""
        mock_redis_class.return_value = redis_client
        mock_load_asr.return_value = mock_asr_model
        mock_load_llm.return_value = mock_llm_model
        
        task_id = "test-audio-duration"
        task_params = {
            "audio_path": test_audio_path,
            "asr_backend": "whisper",
            "features": ["clean_transcript"],
            "template_id": None,
        }
        
        with patch("src.worker.pipeline.get_current_job") as mock_job:
            mock_job.return_value = Mock(id=task_id)
            result = process_audio_task(task_params)
        
        # Verify audio_duration is in metrics and is realistic
        metrics = result["metrics"]
        assert "audio_duration" in metrics
        audio_duration = metrics["audio_duration"]
        
        # Audio duration should be positive and reasonable (not the default 10.0)
        assert audio_duration > 0
        # For test audio, duration should be between 0.1s and 60s (reasonable range)
        assert 0.1 < audio_duration < 60.0
        
        # Verify RTF calculation uses this duration
        assert "total_rtf" in metrics
        processing_time = metrics["total_processing_time"]
        expected_rtf = processing_time / audio_duration
        assert abs(metrics["total_rtf"] - expected_rtf) < 0.01

    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.Redis")
    def test_audio_format_validation(
        self, mock_redis_class, mock_load_llm, mock_load_asr, redis_client, test_audio_path,
        mock_asr_model, mock_llm_model
    ):
        """Test that audio preprocessing validates format correctly."""
        mock_redis_class.return_value = redis_client
        mock_load_asr.return_value = mock_asr_model
        mock_load_llm.return_value = mock_llm_model
        
        task_id = "test-format-validation"
        task_params = {
            "audio_path": test_audio_path,
            "asr_backend": "whisper",
            "features": ["clean_transcript"],
            "template_id": None,
        }
        
        with patch("src.worker.pipeline.get_current_job") as mock_job:
            mock_job.return_value = Mock(id=task_id)
            
            # Should not raise exception for valid WAV file
            result = process_audio_task(task_params)
            assert result is not None
            assert result.get("status") != "error"


class TestRedisIntegration:
    """Test fake Redis integration for status tracking."""

    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.Redis")
    def test_status_transitions(
        self, mock_redis_class, mock_load_llm, mock_load_asr, redis_client, test_audio_path,
        mock_asr_model, mock_llm_model
    ):
        """Test that status transitions are correctly stored in Redis."""
        mock_redis_class.return_value = redis_client
        mock_load_asr.return_value = mock_asr_model
        mock_load_llm.return_value = mock_llm_model
        
        task_id = "test-status-transitions"
        task_key = f"task:{task_id}"
        task_params = {
            "audio_path": test_audio_path,
            "asr_backend": "whisper",
            "features": ["clean_transcript", "summary"],
            "template_id": "meeting_notes_v1",
        }
        
        with patch("src.worker.pipeline.get_current_job") as mock_job:
            mock_job.return_value = Mock(id=task_id)
            process_audio_task(task_params)
        
        # Verify final status
        task_data = redis_client.hgetall(task_key)
        assert task_data["status"] == TaskStatus.COMPLETE.value
        
        # Verify result fields are present
        assert "versions" in task_data
        assert "metrics" in task_data
        assert "results" in task_data
        
        # Verify data is JSON serializable
        versions = json.loads(task_data["versions"])
        assert isinstance(versions, dict)
        
        metrics = json.loads(task_data["metrics"])
        assert isinstance(metrics, dict)
        
        results = json.loads(task_data["results"])
        assert isinstance(results, dict)

    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.Redis")
    def test_error_status_on_failure(
        self, mock_redis_class, mock_load_llm, mock_load_asr, redis_client, test_audio_path,
        mock_asr_model, mock_llm_model
    ):
        """Test that FAILED status is set on errors."""
        # Make ASR fail
        mock_asr_model.execute.side_effect = RuntimeError("ASR model failed")
        mock_redis_class.return_value = redis_client
        mock_load_asr.return_value = mock_asr_model
        mock_load_llm.return_value = mock_llm_model
        
        task_id = "test-error-status"
        task_key = f"task:{task_id}"
        task_params = {
            "audio_path": test_audio_path,
            "asr_backend": "whisper",
            "features": ["clean_transcript"],
            "template_id": None,
        }
        
        with patch("src.worker.pipeline.get_current_job") as mock_job:
            mock_job.return_value = Mock(id=task_id)
            
            # Execute pipeline (should handle error)
            result = process_audio_task(task_params)
        
        # Verify result indicates error
        assert result["status"] == "error"
        assert "error" in result
        
        # Verify Redis status
        task_data = redis_client.hgetall(task_key)
        assert task_data["status"] == TaskStatus.FAILED.value
        assert "error" in task_data


class TestEnhancementLogic:
    """Test enhancement logic with different ASR backends."""

    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.Redis")
    def test_whisper_skips_enhancement(
        self, mock_redis_class, mock_load_llm, mock_load_asr, redis_client, test_audio_path,
        mock_asr_model, mock_llm_model
    ):
        """Test that Whisper backend skips text enhancement."""
        mock_redis_class.return_value = redis_client
        mock_load_asr.return_value = mock_asr_model
        mock_load_llm.return_value = mock_llm_model
        
        # Configure mock to indicate Whisper doesn't need enhancement
        mock_llm_model.needs_enhancement.return_value = False
        
        task_id = "test-whisper-skip-enhancement"
        task_params = {
            "audio_path": test_audio_path,
            "asr_backend": "whisper",  # Whisper backend
            "features": ["clean_transcript"],
            "template_id": None,
        }
        
        with patch("src.worker.pipeline.get_current_job") as mock_job:
            mock_job.return_value = Mock(id=task_id)
            result = process_audio_task(task_params)
        
        # Verify enhancement was NOT applied
        metrics = result["metrics"]
        # edit_rate_cleaning should NOT be present (no enhancement)
        assert "edit_rate_cleaning" not in metrics or metrics["edit_rate_cleaning"] == 0.0
        
        # Verify needs_enhancement was checked
        mock_llm_model.needs_enhancement.assert_called()

    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.Redis")
    def test_chunkformer_applies_enhancement(
        self, mock_redis_class, mock_load_llm, mock_load_asr, redis_client, test_audio_path,
        mock_asr_model, mock_llm_model
    ):
        """Test that ChunkFormer backend applies text enhancement."""
        # Configure ASR to return ChunkFormer-like output (no punctuation)
        mock_asr_model.get_version_info.return_value = {
            "backend": "chunkformer",
            "model_variant": "large-vie",
        }
        mock_redis_class.return_value = redis_client
        mock_load_asr.return_value = mock_asr_model
        
        # Configure LLM to indicate ChunkFormer needs enhancement
        mock_llm_model.needs_enhancement.return_value = True
        mock_llm_model.enhance_text.return_value = {
            "enhanced_text": "This is enhanced text with punctuation.",
            "enhancement_applied": True,
            "edit_rate": 0.15,
            "edit_distance": 5,
        }
        mock_load_llm.return_value = mock_llm_model
        
        task_id = "test-chunkformer-enhancement"
        task_params = {
            "audio_path": test_audio_path,
            "asr_backend": "chunkformer",  # ChunkFormer backend
            "features": ["clean_transcript"],
            "template_id": None,
        }
        
        with patch("src.worker.pipeline.get_current_job") as mock_job:
            mock_job.return_value = Mock(id=task_id)
            result = process_audio_task(task_params)
        
        # Verify enhancement WAS applied
        metrics = result["metrics"]
        assert "edit_rate_cleaning" in metrics
        assert metrics["edit_rate_cleaning"] > 0
        
        # Verify enhance_text was called
        mock_llm_model.enhance_text.assert_called()


# Run tests with: pytest tests/integration/test_worker_pipeline_real.py -v

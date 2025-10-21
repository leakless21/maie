"""
Unit tests for process_audio_task happy path.

Tests the full pipeline orchestration with all mocked dependencies:
- Status transitions through all stages
- Version metadata collection
- Metrics calculation
- Result storage in Redis

Follows TDD.md section 3.2 (GPU Worker) requirements.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.api.schemas import TaskStatus
from src.worker.pipeline import process_audio_task


class TestProcessAudioTaskHappyPath:
    """Test the complete happy path for process_audio_task."""

    @patch("src.worker.pipeline.Redis")
    @patch("src.processors.audio.AudioPreprocessor")
    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.execute_asr_transcription")
    @patch("src.worker.pipeline.unload_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.execute_llm_processing")
    @patch("src.worker.pipeline.unload_llm_model")
    @patch("src.worker.pipeline.get_version_metadata")
    @patch("src.worker.pipeline.calculate_metrics")
    @patch("src.worker.pipeline._update_status")
    @patch("src.worker.pipeline.get_current_job")
    def test_full_pipeline_with_all_features(
        self,
        mock_get_job,
        mock_update_status,
        mock_calc_metrics,
        mock_get_version,
        mock_unload_llm,
        mock_exec_llm,
        mock_load_llm,
        mock_unload_asr,
        mock_exec_asr,
        mock_load_asr,
        mock_audio_preprocessor_class,
        mock_redis_class,
        mock_rq_job,
        tmp_path,
    ):
        """
        Test complete pipeline execution with all features requested.

        Expected flow:
        1. PENDING → PREPROCESSING → PROCESSING_ASR
        2. ASR: Load model → Execute → Unload → Clear GPU
        3. PROCESSING_ASR → PROCESSING_LLM
        4. LLM: Load model → Enhance text → Summarize → Unload → Clear GPU
        5. Collect versions and metrics
        6. PROCESSING_LLM → COMPLETE
        7. Store final result in Redis
        """
        # Setup mock job to be truthy (MagicMock is truthy by default)
        mock_get_job.return_value = mock_rq_job

        # Create a mock Redis instance
        mock_redis_instance = MagicMock()
        mock_redis_instance.hset = MagicMock()
        mock_redis_instance.close = MagicMock()
        mock_redis_class.return_value = mock_redis_instance

        # Mock AudioPreprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.preprocess.return_value = {
            "format": "wav",
            "duration": 45.0,  # Test audio duration in seconds
            "sample_rate": 16000,
            "channels": 1,
            "normalized_path": None,  # No normalization needed
        }
        mock_audio_preprocessor_class.return_value = mock_preprocessor

        # Setup task parameters
        task_params = {
            "audio_path": str(tmp_path / "test_audio.wav"),
            "asr_backend": "whisper",
            "features": ["clean_transcript", "summary"],
            "template_id": "meeting_notes_v1",
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 1,
        }

        # Create dummy audio file
        audio_file = Path(task_params["audio_path"])
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.write_bytes(b"dummy audio data")

        # Mock ASR results
        mock_asr_model = MagicMock()
        mock_load_asr.return_value = mock_asr_model

        # Phase 1: Now returns (ASRResult, rtf, metadata) instead of (text, rtf, confidence, metadata)
        from src.processors.base import ASRResult
        asr_transcription = "This is a test transcription from the mock ASR backend."
        asr_rtf = 0.5
        asr_confidence = 0.92
        asr_metadata = {
            "model_name": "mock-whisper-v1",
            "checkpoint_hash": "abc123",
            "duration_ms": 5000,
        }
        asr_result = ASRResult(text=asr_transcription, confidence=asr_confidence)
        mock_exec_asr.return_value = (
            asr_result,
            asr_rtf,
            asr_metadata,
        )

        # Mock LLM results
        mock_llm_model = MagicMock()
        mock_load_llm.return_value = mock_llm_model

        enhanced_transcript = "This is a test transcription from the mock ASR backend."
        structured_summary = {
            "title": "Test Meeting Summary",
            "abstract": "A brief summary of the test meeting.",
            "main_points": ["First main point", "Second main point"],
            "tags": ["test", "meeting", "summary"],
        }
        mock_exec_llm.return_value = (enhanced_transcript, structured_summary)

        # Mock version metadata
        version_metadata = {
            "pipeline_version": "1.0.0",
            "asr_backend": {
                "backend": "whisper",
                "model_variant": "erax-wow-turbo",
                "checkpoint_hash": "abc123",
            },
            "enhancement_llm": {"name": "qwen3-4b-awq", "checkpoint_hash": "xyz789"},
            "summarization_llm": {"name": "qwen3-4b-awq", "checkpoint_hash": "xyz789"},
        }
        mock_get_version.return_value = version_metadata

        # Mock metrics
        metrics = {
            "input_duration_seconds": 5.0,
            "processing_time_seconds": 25.5,
            "rtf": 5.1,
            "vad_coverage": 0.85,
            "asr_confidence_avg": 0.92,
            "edit_rate_cleaning": 0.0,
        }
        mock_calc_metrics.return_value = metrics

        # Mock Redis connection - need to ensure Redis() is called and returns our mock
        with patch("src.worker.pipeline.Redis") as mock_redis_class:
            # Configure the Redis mock to return our mock_redis_sync
            mock_redis_instance = MagicMock()
            mock_redis_instance.hset = MagicMock()
            mock_redis_instance.close = MagicMock()
            mock_redis_class.return_value = mock_redis_instance

            # Execute pipeline
            result = process_audio_task(task_params)

            # Verify status transitions (_update_status should be called at least 3 times:
        # PREPROCESSING, PROCESSING_ASR, PROCESSING_LLM, COMPLETE)
        assert mock_update_status.call_count >= 3, (
            f"Expected at least 3 status updates, got {mock_update_status.call_count}. "
            f"Calls: {mock_update_status.call_args_list}"
        )

        # Check status progression: PROCESSING_ASR → PROCESSING_LLM → COMPLETE
        # Extract status from call args: _update_status(redis_conn, task_key, status, metadata)
        # call[0] is the positional args tuple, and status is at index 2
        status_calls = [call[0][2] for call in mock_update_status.call_args_list]
        assert TaskStatus.PROCESSING_ASR in status_calls
        assert TaskStatus.PROCESSING_LLM in status_calls
        assert TaskStatus.COMPLETE in status_calls

        # Verify ASR pipeline
        mock_load_asr.assert_called_once_with("whisper")
        mock_exec_asr.assert_called_once()
        mock_unload_asr.assert_called_once_with(mock_asr_model)

        # Verify LLM pipeline
        mock_load_llm.assert_called_once()
        mock_exec_llm.assert_called_once()
        mock_unload_llm.assert_called_once_with(mock_llm_model)

        # Verify result structure
        assert result is not None
        assert "versions" in result
        assert "metrics" in result
        assert "results" in result

        # Verify versions
        assert result["versions"] == version_metadata

        # Verify metrics
        assert result["metrics"] == metrics

        # Verify results contain expected features
        assert "transcript" in result["results"]
        assert "summary" in result["results"]
        assert result["results"]["summary"] == structured_summary

    @patch("src.processors.audio.AudioPreprocessor")
    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.execute_asr_transcription")
    @patch("src.worker.pipeline.unload_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.execute_llm_processing")
    @patch("src.worker.pipeline.unload_llm_model")
    @patch("src.worker.pipeline.get_version_metadata")
    @patch("src.worker.pipeline.calculate_metrics")
    @patch("src.worker.pipeline._update_status")
    @patch("src.worker.pipeline.get_current_job")
    def test_pipeline_with_transcript_only(
        self,
        mock_get_job,
        mock_update_status,
        mock_calc_metrics,
        mock_get_version,
        mock_unload_llm,
        mock_exec_llm,
        mock_load_llm,
        mock_unload_asr,
        mock_exec_asr,
        mock_load_asr,
        mock_audio_preprocessor_class,
        mock_rq_job,
        tmp_path,
    ):
        """
        Test pipeline with only transcript feature (no summary).

        Expected behavior:
        - ASR executes normally
        - LLM may or may not load depending on enhancement needs
        - Final result contains only transcript, no summary
        """
        # Setup mock job
        mock_get_job.return_value = mock_rq_job

        # Mock AudioPreprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.preprocess.return_value = {
            "format": "wav",
            "duration": 30.0,
            "sample_rate": 16000,
            "channels": 1,
            "normalized_path": None,
        }
        mock_audio_preprocessor_class.return_value = mock_preprocessor

        # Setup task parameters
        task_params = {
            "audio_path": str(tmp_path / "test_audio.wav"),
            "asr_backend": "whisper",
            "features": ["clean_transcript"],  # Only transcript, no summary
            "template_id": None,
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 1,
        }

        # Create dummy audio file
        audio_file = Path(task_params["audio_path"])
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.write_bytes(b"dummy audio data")

        # Mock ASR results
        mock_asr_model = MagicMock()
        mock_load_asr.return_value = mock_asr_model

        # Phase 1: Now returns (ASRResult, rtf, metadata)
        from src.processors.base import ASRResult
        asr_transcription = "This is a test transcription."
        asr_metadata = {"model_name": "whisper", "checkpoint_hash": "abc123"}
        asr_result = ASRResult(text=asr_transcription, confidence=0.92)
        mock_exec_asr.return_value = (asr_result, 0.5, asr_metadata)

        # Mock LLM results (enhancement only, no summary)
        mock_llm_model = MagicMock()
        mock_load_llm.return_value = mock_llm_model
        mock_exec_llm.return_value = (asr_transcription, None)  # No summary

        # Mock version metadata
        version_metadata = {
            "pipeline_version": "1.0.0",
            "asr_backend": {"backend": "whisper"},
        }
        mock_get_version.return_value = version_metadata

        # Mock metrics
        metrics = {
            "input_duration_seconds": 5.0,
            "processing_time_seconds": 10.0,
            "rtf": 2.0,
        }
        mock_calc_metrics.return_value = metrics

        # Mock Redis connection
        with patch("src.worker.pipeline.Redis") as mock_redis_class:
            mock_redis_instance = MagicMock()
            mock_redis_instance.hset = MagicMock()
            mock_redis_instance.close = MagicMock()
            mock_redis_class.return_value = mock_redis_instance

            # Execute pipeline
            result = process_audio_task(task_params)

        # Verify ASR executed
        mock_load_asr.assert_called_once()
        mock_exec_asr.assert_called_once()
        mock_unload_asr.assert_called_once()

        # Verify result contains transcript but no summary
        assert "transcript" in result["results"]
        assert "summary" not in result["results"]

    @patch("src.processors.audio.AudioPreprocessor")
    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.execute_asr_transcription")
    @patch("src.worker.pipeline.unload_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.execute_llm_processing")
    @patch("src.worker.pipeline.unload_llm_model")
    @patch("src.worker.pipeline.get_version_metadata")
    @patch("src.worker.pipeline.calculate_metrics")
    @patch("src.worker.pipeline.update_task_status")
    @patch("src.worker.pipeline.get_current_job")
    def test_version_metadata_collection(
        self,
        mock_get_job,
        mock_update_status,
        mock_calc_metrics,
        mock_get_version,
        mock_unload_llm,
        mock_exec_llm,
        mock_load_llm,
        mock_unload_asr,
        mock_exec_asr,
        mock_load_asr,
        mock_audio_preprocessor_class,
        mock_rq_job,
        tmp_path,
    ):
        """
        Test that version metadata is properly collected from all components.

        NFR-1 requirement: Full reproducibility through comprehensive versioning.

        Expected version data:
        - pipeline_version
        - asr_backend (model variant, checkpoint hash, compute type, decoding params)
        - enhancement_llm (if used)
        - summarization_llm (if used)
        """
        # Setup mock job
        mock_get_job.return_value = mock_rq_job

        # Mock AudioPreprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.preprocess.return_value = {
            "format": "wav",
            "duration": 35.0,
            "sample_rate": 16000,
            "channels": 1,
            "normalized_path": None,
        }
        mock_audio_preprocessor_class.return_value = mock_preprocessor

        # Setup task parameters
        task_params = {
            "audio_path": str(tmp_path / "test_audio.wav"),
            "asr_backend": "whisper",
            "features": ["clean_transcript", "summary"],
            "template_id": "meeting_notes_v1",
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 1,
        }

        # Create dummy audio file
        audio_file = Path(task_params["audio_path"])
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.write_bytes(b"dummy audio data")

        # Mock ASR results
        mock_asr_model = MagicMock()
        mock_load_asr.return_value = mock_asr_model

        # Phase 1: Now returns (ASRResult, rtf, metadata)
        from src.processors.base import ASRResult
        asr_metadata = {
            "model_name": "erax-wow-turbo-v1.1",
            "checkpoint_hash": "a1b2c3d4e5f6",
            "compute_type": "int8_float16",
            "decoding_params": {"beam_size": 5, "vad_filter": True, "temperature": 0.0},
        }
        asr_result = ASRResult(text="Test transcript", confidence=0.92)
        mock_exec_asr.return_value = (asr_result, 0.5, asr_metadata)

        # Mock LLM results
        mock_llm_model = MagicMock()
        mock_load_llm.return_value = mock_llm_model
        mock_exec_llm.return_value = ("Enhanced transcript", {"title": "Summary"})

        # Mock comprehensive version metadata
        version_metadata = {
            "pipeline_version": "1.0.0",
            "asr_backend": {
                "backend": "whisper",
                "model_variant": "erax-wow-turbo",
                "model_path": "/data/models/whisper/erax-wow-turbo-v1.1",
                "checkpoint_hash": "a1b2c3d4e5f6",
                "compute_type": "int8_float16",
                "decoding_params": {
                    "beam_size": 5,
                    "vad_filter": True,
                    "temperature": 0.0,
                },
            },
            "enhancement_llm": {
                "name": "qwen3-4b-instruct-awq",
                "checkpoint_hash": "x7y8z9w0",
                "quantization": "awq-4bit",
                "task": "text_enhancement",
                "generation_params": {"temperature": 0.7, "max_tokens": 4096},
            },
            "summarization_llm": {
                "name": "qwen3-4b-instruct-awq",
                "checkpoint_hash": "x7y8z9w0",
                "quantization": "awq-4bit",
                "task": "summarization",
                "structured_output": True,
                "template_id": "meeting_notes_v1",
            },
        }
        mock_get_version.return_value = version_metadata

        # Mock metrics
        metrics = {"rtf": 1.0}
        mock_calc_metrics.return_value = metrics

        # Mock Redis connection
        with patch("src.worker.pipeline.Redis") as mock_redis_class:
            mock_redis_instance = MagicMock()
            mock_redis_instance.hset = MagicMock()
            mock_redis_instance.close = MagicMock()
            mock_redis_class.return_value = mock_redis_instance

            # Execute pipeline
            result = process_audio_task(task_params)

        # Verify version metadata structure
        assert "versions" in result
        versions = result["versions"]

        # Verify pipeline version
        assert versions["pipeline_version"] == "1.0.0"

        # Verify ASR backend versioning (NFR-1)
        assert "asr_backend" in versions
        asr_version = versions["asr_backend"]
        assert asr_version["backend"] == "whisper"
        assert asr_version["model_variant"] == "erax-wow-turbo"
        assert asr_version["checkpoint_hash"] == "a1b2c3d4e5f6"
        assert "decoding_params" in asr_version

        # Verify LLM versioning (NFR-1)
        assert "enhancement_llm" in versions
        assert versions["enhancement_llm"]["name"] == "qwen3-4b-instruct-awq"
        assert versions["enhancement_llm"]["checkpoint_hash"] == "x7y8z9w0"

        assert "summarization_llm" in versions
        assert versions["summarization_llm"]["task"] == "summarization"
        assert versions["summarization_llm"]["structured_output"] is True

    @patch("src.processors.audio.AudioPreprocessor")
    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.execute_asr_transcription")
    @patch("src.worker.pipeline.unload_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.execute_llm_processing")
    @patch("src.worker.pipeline.unload_llm_model")
    @patch("src.worker.pipeline.get_version_metadata")
    @patch("src.worker.pipeline.calculate_metrics")
    @patch("src.worker.pipeline.update_task_status")
    @patch("src.worker.pipeline.get_current_job")
    def test_metrics_calculation(
        self,
        mock_get_job,
        mock_update_status,
        mock_calc_metrics,
        mock_get_version,
        mock_unload_llm,
        mock_exec_llm,
        mock_load_llm,
        mock_unload_asr,
        mock_exec_asr,
        mock_load_asr,
        mock_audio_preprocessor_class,
        mock_rq_job,
        tmp_path,
    ):
        """
        Test that runtime metrics are properly calculated.

        FR-5 requirement: Calculate and return metrics including:
        - input_duration_seconds
        - processing_time_seconds
        - rtf (Real-Time Factor)
        - vad_coverage
        - asr_confidence_avg
        - edit_rate_cleaning (if enhancement applied)
        """
        # Setup mock job
        mock_get_job.return_value = mock_rq_job

        # Mock AudioPreprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.preprocess.return_value = {
            "format": "wav",
            "duration": 45.0,
            "sample_rate": 16000,
            "channels": 1,
            "normalized_path": None,
        }
        mock_audio_preprocessor_class.return_value = mock_preprocessor

        # Setup task parameters
        task_params = {
            "audio_path": str(tmp_path / "test_audio.wav"),
            "asr_backend": "whisper",
            "features": ["clean_transcript", "summary"],
            "template_id": "meeting_notes_v1",
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 1,
        }

        # Create dummy audio file
        audio_file = Path(task_params["audio_path"])
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.write_bytes(b"dummy audio data")

        # Mock ASR results with specific metrics
        mock_asr_model = MagicMock()
        mock_load_asr.return_value = mock_asr_model

        # Phase 1: Now returns (ASRResult, rtf, metadata)
        from src.processors.base import ASRResult
        asr_transcription = "This is a test transcription."
        asr_rtf = 0.5  # Specific RTF for ASR
        asr_confidence = 0.92  # High confidence
        asr_metadata = {"model_name": "whisper", "checkpoint_hash": "abc123"}
        asr_result = ASRResult(text=asr_transcription, confidence=asr_confidence)
        mock_exec_asr.return_value = (
            asr_result,
            asr_rtf,
            asr_metadata,
        )

        # Mock LLM results
        mock_llm_model = MagicMock()
        mock_load_llm.return_value = mock_llm_model

        enhanced_transcript = "This is a test transcription!"  # Minor enhancement
        mock_exec_llm.return_value = (enhanced_transcript, {"title": "Summary"})

        # Mock version metadata
        version_metadata = {"pipeline_version": "1.0.0"}
        mock_get_version.return_value = version_metadata

        # Mock comprehensive metrics
        metrics = {
            "input_duration_seconds": 45.0,  # 45 second audio
            "processing_time_seconds": 135.0,  # 135 seconds to process
            "rtf": 3.0,  # Total RTF: 135/45 = 3.0
            "asr_rtf": 0.5,  # ASR-specific RTF
            "vad_coverage": 0.85,  # 85% of audio contains speech
            "asr_confidence_avg": 0.92,  # Average confidence
            "edit_rate_cleaning": 0.03,  # 3% edit rate from enhancement
            "audio_duration": 45.0,
            "total_processing_time": 135.0,
        }
        mock_calc_metrics.return_value = metrics

        # Mock Redis connection
        with patch("src.worker.pipeline.Redis") as mock_redis_class:
            mock_redis_instance = MagicMock()
            mock_redis_instance.hset = MagicMock()
            mock_redis_instance.close = MagicMock()
            mock_redis_class.return_value = mock_redis_instance

            # Execute pipeline
            result = process_audio_task(task_params)

        # Verify metrics structure
        assert "metrics" in result
        result_metrics = result["metrics"]

        # Verify core metrics (FR-5)
        assert result_metrics["input_duration_seconds"] == 45.0
        assert result_metrics["processing_time_seconds"] == 135.0
        assert result_metrics["rtf"] == 3.0

        # Verify ASR metrics
        assert result_metrics["vad_coverage"] == 0.85
        assert result_metrics["asr_confidence_avg"] == 0.92

        # Verify enhancement metrics
        assert result_metrics["edit_rate_cleaning"] == 0.03

    @patch("src.processors.audio.AudioPreprocessor")
    @patch("src.worker.pipeline.load_asr_model")
    @patch("src.worker.pipeline.execute_asr_transcription")
    @patch("src.worker.pipeline.unload_asr_model")
    @patch("src.worker.pipeline.load_llm_model")
    @patch("src.worker.pipeline.execute_llm_processing")
    @patch("src.worker.pipeline.unload_llm_model")
    @patch("src.worker.pipeline.get_version_metadata")
    @patch("src.worker.pipeline.calculate_metrics")
    @patch("src.worker.pipeline._update_status")
    @patch("src.worker.pipeline.get_current_job")
    def test_result_storage_in_redis(
        self,
        mock_get_job,
        mock_update_status,
        mock_calc_metrics,
        mock_get_version,
        mock_unload_llm,
        mock_exec_llm,
        mock_load_llm,
        mock_unload_asr,
        mock_exec_asr,
        mock_load_asr,
        mock_audio_preprocessor_class,
        mock_rq_job,
        tmp_path,
    ):
        """
        Test that final results are properly stored in Redis DB 1.

        Expected storage:
        - Task key: "task:{task_id}"
        - Fields: status, versions, metrics, results
        - Database: Redis DB 1 (results database)
        """
        # Setup mock job
        mock_get_job.return_value = mock_rq_job

        # Mock AudioPreprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.preprocess.return_value = {
            "format": "wav",
            "duration": 50.0,
            "sample_rate": 16000,
            "channels": 1,
            "normalized_path": None,
        }
        mock_audio_preprocessor_class.return_value = mock_preprocessor

        # Setup task parameters
        task_params = {
            "audio_path": str(tmp_path / "test_audio.wav"),
            "asr_backend": "whisper",
            "features": ["clean_transcript", "summary"],
            "template_id": "meeting_notes_v1",
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 1,  # Results DB
        }

        # Create dummy audio file
        audio_file = Path(task_params["audio_path"])
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.write_bytes(b"dummy audio data")

        # Mock ASR results
        mock_asr_model = MagicMock()
        mock_load_asr.return_value = mock_asr_model
        # Phase 1: Now returns (ASRResult, rtf, metadata)
        from src.processors.base import ASRResult
        asr_result = ASRResult(text="Test transcript", confidence=0.92)
        mock_exec_asr.return_value = (
            asr_result,
            0.5,
            {"model_name": "whisper"},
        )

        # Mock LLM results
        mock_llm_model = MagicMock()
        mock_load_llm.return_value = mock_llm_model

        summary_result = {
            "title": "Meeting Summary",
            "abstract": "Summary text",
            "main_points": ["Point 1"],
            "tags": ["meeting", "test"],
        }
        mock_exec_llm.return_value = ("Enhanced", summary_result)

        # Mock version and metrics
        mock_get_version.return_value = {"pipeline_version": "1.0.0"}
        mock_calc_metrics.return_value = {"rtf": 1.0}

        # Mock Redis connection
        with patch("src.worker.pipeline.Redis") as mock_redis_class:
            mock_redis_instance = MagicMock()
            mock_redis_instance.hset = MagicMock()
            mock_redis_instance.close = MagicMock()
            mock_redis_class.return_value = mock_redis_instance

            # Execute pipeline
            result = process_audio_task(task_params)

        # Verify Redis was used to update status to COMPLETE
        # _update_status signature: (redis_conn, task_key, status, metadata)
        # call[0] is the positional args tuple, status is at index 2
        complete_status_call = None
        for call in mock_update_status.call_args_list:
            if len(call[0]) >= 3 and call[0][2] == TaskStatus.COMPLETE:
                complete_status_call = call
                break

        assert complete_status_call is not None, "Status should be updated to COMPLETE"

        # Verify result contains all required data
        assert result["versions"] is not None
        assert result["metrics"] is not None
        assert result["results"] is not None

        # Verify result structure matches expected format
        assert "transcript" in result["results"]
        assert "summary" in result["results"]
        assert result["results"]["summary"] == summary_result

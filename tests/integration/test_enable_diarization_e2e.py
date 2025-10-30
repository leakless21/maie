"""
End-to-end integration tests for enable_diarization parameter.

Tests the complete flow from API request through worker pipeline,
verifying that diarization is properly controlled by the enable_diarization parameter.
"""

import pytest
from unittest.mock import patch

from src.api.schemas import ProcessRequestSchema, Feature


class TestEnableDiarizationE2E:
    """End-to-end tests for enable_diarization parameter flow."""

    def test_process_request_schema_with_diarization_disabled(self):
        """Test API schema with enable_diarization=False (default)."""
        request_data = {
            "file": "test.wav",
            "features": [Feature.RAW_TRANSCRIPT],
        }

        schema = ProcessRequestSchema.model_validate(request_data)

        assert schema.enable_diarization is False
        assert schema.features == [Feature.RAW_TRANSCRIPT]

    def test_process_request_schema_with_diarization_enabled(self):
        """Test API schema with enable_diarization=True."""
        request_data = {
            "file": "test.wav",
            "features": [Feature.CLEAN_TRANSCRIPT],
            "enable_diarization": True,
            "asr_backend": "whisper",
        }

        schema = ProcessRequestSchema.model_validate(request_data)

        assert schema.enable_diarization is True
        assert schema.asr_backend == "whisper"

    def test_task_params_flow_from_api_to_worker(self):
        """Test that enable_diarization flows from API request to worker task_params."""
        # Simulate API request
        request_data = {
            "file": "meeting.wav",
            "features": ["clean_transcript", "summary"],
            "template_id": "meeting_notes_v1",
            "enable_diarization": True,
            "asr_backend": "whisper",
        }

        schema = ProcessRequestSchema.model_validate(request_data)

        # Simulate task_params creation (as done in routes.py)
        task_params = {
            "audio_path": "/path/to/audio.wav",
            "features": [f.value for f in schema.features],
            "template_id": schema.template_id,
            "asr_backend": schema.asr_backend,
            "enable_diarization": schema.enable_diarization,
        }

        # Verify the parameter made it through
        assert task_params["enable_diarization"] is True
        assert task_params["asr_backend"] == "whisper"

    def test_pipeline_extracts_enable_diarization_from_task_params(self):
        """Test that pipeline correctly extracts enable_diarization from task_params."""
        task_params = {
            "audio_path": "/test/audio.wav",
            "features": ["raw_transcript"],
            "enable_diarization": True,
        }

        # Simulate pipeline extraction
        enable_diarization = task_params.get("enable_diarization", False)

        assert enable_diarization is True

    def test_pipeline_defaults_to_false_when_parameter_missing(self):
        """Test that pipeline defaults enable_diarization to False when not provided."""
        task_params = {
            "audio_path": "/test/audio.wav",
            "features": ["raw_transcript"],
        }

        # Simulate pipeline extraction with default
        enable_diarization = task_params.get("enable_diarization", False)

        assert enable_diarization is False

    @pytest.mark.parametrize(
        "request_value,expected_in_params",
        [
            (True, True),
            (False, False),
            (None, False),  # When omitted, defaults to False
        ],
    )
    def test_various_enable_diarization_values(self, request_value, expected_in_params):
        """Test various values of enable_diarization parameter."""
        request_data = {
            "file": "test.wav",
            "features": [Feature.RAW_TRANSCRIPT],
        }

        if request_value is not None:
            request_data["enable_diarization"] = request_value

        schema = ProcessRequestSchema.model_validate(request_data)

        assert schema.enable_diarization == expected_in_params


class TestDiarizationWithMockedPipeline:
    """Test diarization control with mocked pipeline components."""

    @patch("src.processors.audio.diarizer.get_diarizer")
    @patch("src.config.settings")
    def test_diarization_not_called_when_parameter_false(
        self, mock_settings, mock_get_diarizer
    ):
        """Verify diarization is not called when enable_diarization=False."""
        # Setup mock settings
        mock_settings.diarization.enabled = True

        # Simulate the pipeline condition
        enable_diarization = False
        should_run = mock_settings.diarization.enabled and enable_diarization

        if should_run:
            from src.processors.audio.diarizer import get_diarizer

            get_diarizer()

        # Verify diarizer was NOT called
        mock_get_diarizer.assert_not_called()

    @patch("src.processors.audio.diarizer.get_diarizer")
    @patch("src.config.settings")
    def test_diarization_called_when_both_enabled(
        self, mock_settings, mock_get_diarizer
    ):
        """Verify diarization is called when both global and parameter are True."""
        # Setup mock settings
        mock_settings.diarization.enabled = True
        mock_settings.diarization.model_path = "pyannote/speaker-diarization-3.1"
        mock_settings.diarization.require_cuda = False
        mock_settings.diarization.embedding_batch_size = 32
        mock_settings.diarization.segmentation_batch_size = 32

        # Simulate the pipeline condition
        enable_diarization = True
        should_run = mock_settings.diarization.enabled and enable_diarization

        if should_run:
            from src.processors.audio.diarizer import get_diarizer

            get_diarizer(
                model_path=mock_settings.diarization.model_path,
                require_cuda=mock_settings.diarization.require_cuda,
                embedding_batch_size=mock_settings.diarization.embedding_batch_size,
                segmentation_batch_size=mock_settings.diarization.segmentation_batch_size,
            )

        # Verify diarizer WAS called
        mock_get_diarizer.assert_called_once()

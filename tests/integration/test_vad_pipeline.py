"""Integration tests for VAD processing in the pipeline."""

import pytest
from unittest.mock import patch
from src.processors.vad.base import VADResult, VADSegment
from src.processors.asr.factory import ASRFactory
from src.config.model import VADSettings


class TestVADIntegration:
    """Integration tests for VAD with other components."""

    def test_asr_factory_create_vad_backend(self):
        """Test ASRFactory can create VAD backend."""
        config = VADSettings(device="cpu")
        try:
            backend = ASRFactory.create_vad_backend(config)
            assert backend is not None
            backend.unload()
        except RuntimeError as e:
            if "silero-vad not installed" in str(e):
                pytest.skip("silero-vad not installed")
            raise

    def test_asr_factory_vad_disabled(self):
        """Test ASRFactory returns None when VAD disabled."""
        config = VADSettings(enabled=False)
        backend = ASRFactory.create_vad_backend(config)
        assert backend is None

    def test_vad_result_storage_format(self):
        """Test that VADResult can be converted to storage format."""
        segments = [
            VADSegment(0.0, 1.0, 0.0, False),
            VADSegment(1.0, 5.0, 1.0, True),
        ]
        vad_result_obj = VADResult(
            segments=segments,
            total_duration=5.0,
            speech_duration=4.0,
            speech_ratio=0.8,
            processing_time=0.05,
            backend_info={"backend": "silero", "model": "test"},
        )

        # Convert to storage format (like in pipeline)
        vad_result = {
            "total_duration": vad_result_obj.total_duration,
            "speech_duration": vad_result_obj.speech_duration,
            "speech_ratio": vad_result_obj.speech_ratio,
            "non_speech_duration": vad_result_obj.non_speech_duration(),
            "num_segments": len(vad_result_obj.segments),
            "num_speech_segments": len(vad_result_obj.get_speech_segments()),
            "num_silence_segments": len(vad_result_obj.get_silence_segments()),
            "processing_time": vad_result_obj.processing_time,
            "backend_info": vad_result_obj.backend_info,
        }

        # Verify storage format
        assert vad_result["total_duration"] == 5.0
        assert vad_result["speech_duration"] == 4.0
        assert vad_result["speech_ratio"] == 0.8
        assert vad_result["num_segments"] == 2
        assert vad_result["num_speech_segments"] == 1
        assert vad_result["num_silence_segments"] == 1

    @patch("src.worker.pipeline.time")
    def test_pipeline_vad_metrics_calculation(self, mock_time):
        """Test VAD integration with metrics calculation."""
        from src.worker.pipeline import calculate_metrics

        mock_time.time.return_value = 100.0

        vad_result = {
            "speech_ratio": 0.75,
            "num_segments": 5,
        }

        metrics = calculate_metrics(
            transcription="Test text",
            clean_transcript="Test text",
            start_time=99.0,
            audio_duration=10.0,
            asr_rtf=0.1,
            vad_result=vad_result,
        )

        assert metrics["vad_coverage"] == 0.75
        assert metrics["vad_segments"] == 5
        assert metrics["input_duration_seconds"] == 10.0

    def test_vad_threshold_override(self):
        """Test that VAD threshold can be overridden per request."""
        config = VADSettings(silero_threshold=0.5)
        assert config.silero_threshold == 0.5

        # Simulate request override
        override_threshold = 0.7
        config_with_override = VADSettings(silero_threshold=override_threshold)
        assert config_with_override.silero_threshold == 0.7


class TestVADEdgeCases:
    """Test edge cases in VAD processing."""

    def test_empty_audio_handling(self):
        """Test handling of empty/silence-only audio."""
        segments = []  # No speech detected
        result = VADResult(
            segments=segments,
            total_duration=5.0,
            speech_duration=0.0,
            speech_ratio=0.0,
            processing_time=0.05,
            backend_info={"backend": "silero"},
        )
        assert result.speech_ratio == 0.0
        assert len(result.get_speech_segments()) == 0

    def test_continuous_speech_handling(self):
        """Test handling of continuous speech audio."""
        segments = [VADSegment(0.0, 10.0, 1.0, True)]
        result = VADResult(
            segments=segments,
            total_duration=10.0,
            speech_duration=10.0,
            speech_ratio=1.0,
            processing_time=0.05,
            backend_info={"backend": "silero"},
        )
        assert result.speech_ratio == 1.0
        assert len(result.get_speech_segments()) == 1
        assert result.non_speech_duration() == 0.0

    def test_fragmented_speech_handling(self):
        """Test handling of highly fragmented speech."""
        segments = [
            VADSegment(0.0, 0.5, 0.0, False),
            VADSegment(0.5, 1.0, 1.0, True),
            VADSegment(1.0, 1.5, 0.0, False),
            VADSegment(1.5, 2.0, 1.0, True),
            VADSegment(2.0, 2.5, 0.0, False),
            VADSegment(2.5, 3.0, 1.0, True),
        ]
        result = VADResult(
            segments=segments,
            total_duration=3.0,
            speech_duration=1.5,
            speech_ratio=0.5,
            processing_time=0.05,
            backend_info={"backend": "silero"},
        )
        assert len(result.get_speech_segments()) == 3
        assert len(result.get_silence_segments()) == 3
        assert result.speech_ratio == 0.5

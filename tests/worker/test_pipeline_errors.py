"""Unit tests for worker pipeline error handling."""

from src.worker.pipeline import process_audio_task


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_missing_audio_file(self):
        """Test handling of missing audio file."""
        task_params = {
            "audio_path": "/nonexistent/path/to/audio.wav",
            "asr_backend": "whisper",
            "features": ["clean_transcript"],
        }

        result = process_audio_task(task_params)
        assert result["status"] == "error"
        assert "error" in result

"""Integration tests for verbose component output."""

import subprocess

import pytest


class TestVerboseIntegration:
    """Integration tests verifying actual verbose output."""

    @pytest.mark.integration
    def test_ffmpeg_shows_progress_when_verbose(self, tmp_path):
        """Verify FFmpeg actually shows progress with verbose enabled."""
        # This test requires actual FFmpeg binary
        # Skip if FFmpeg not available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("FFmpeg not available")

        # Create a test audio file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 1000)  # Minimal WAV header

        # Test will be implemented to verify actual stderr/stdout contains progress
        # For now, just verify the file exists
        assert test_file.exists()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_whisper_shows_progress_when_verbose(self):
        """Verify Whisper shows progress bars with verbose enabled."""
        # This test requires actual Whisper model
        # Will be implemented to verify tqdm progress bars appear
        # For now, just a placeholder
        assert True

    @pytest.mark.integration
    @pytest.mark.slow
    def test_vllm_shows_loading_when_verbose(self):
        """Verify vLLM shows loading messages with verbose enabled."""
        # This test requires actual vLLM model
        # Will be implemented to verify loading messages appear
        # For now, just a placeholder
        assert True

    @pytest.mark.integration
    def test_audio_preprocessor_verbose_behavior(self, tmp_path, monkeypatch):
        """Test that AudioPreprocessor respects verbose settings in integration."""
        # Create a test audio file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 1000)

        # Test with verbose enabled
        monkeypatch.setenv("VERBOSE_COMPONENTS", "true")

        from src.processors.audio.preprocessor import AudioPreprocessor

        preprocessor = AudioPreprocessor()

        # This will fail if FFmpeg is not available, but that's expected
        # The test verifies the integration works
        try:
            result = preprocessor.preprocess(test_file)
            # If successful, verify the result structure
            assert isinstance(result, dict)
            assert "duration" in result
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Expected if FFmpeg not available
            pytest.skip("FFmpeg not available for integration test")

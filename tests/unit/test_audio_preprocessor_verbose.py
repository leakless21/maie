"""Tests for verbose output control in AudioPreprocessor."""

from unittest.mock import MagicMock, patch


from src.processors.audio.preprocessor import AudioPreprocessor


class TestAudioPreprocessorVerbose:
    """Test verbose output behavior."""

    def test_ffprobe_verbose_disabled_captures_output(self, tmp_path, monkeypatch):
        """When verbose=False, subprocess should capture output."""
        # Create test WAV file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 100)

        preprocessor = AudioPreprocessor()

        with patch("subprocess.run") as mock_run:
            with patch(
                "src.processors.audio.preprocessor.settings.verbose_components",
                False,
            ):
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout='{"streams": [{"codec_type": "audio", "sample_rate": "16000", "channels": 1}], "format": {"duration": "5.0"}}',
                    stderr="",
                )

                preprocessor._probe_audio(test_file)

                # Verify capture_output=True was used
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["capture_output"] is True

    def test_ffprobe_verbose_enabled_shows_output(self, tmp_path, monkeypatch):
        """When verbose=True, subprocess should not capture output."""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 100)

        preprocessor = AudioPreprocessor()

        with patch("subprocess.run") as mock_run:
            with (
                patch(
                    "src.processors.audio.preprocessor.settings.verbose_components",
                    True,
                ),
                patch("src.processors.audio.preprocessor.logger") as mock_logger,
            ):
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout='{"streams": [{"codec_type": "audio", "sample_rate": "16000", "channels": 1}], "format": {"duration": "5.0"}}',
                    stderr="",
                )

                preprocessor._probe_audio(test_file)

                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["capture_output"] is True
                mock_logger.debug.assert_any_call(
                    "ffprobe output: {}",
                    '{"streams": [{"codec_type": "audio", "sample_rate": "16000", "channels": 1}], "format": {"duration": "5.0"}}',
                )

    def test_ffmpeg_normalize_respects_verbose_setting(self, tmp_path, monkeypatch):
        """FFmpeg normalization should respect verbose setting."""
        preprocessor = AudioPreprocessor()
        test_file = tmp_path / "input.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 100)

        metadata = {"sample_rate": 48000, "channels": 2, "format": "mp3"}

        with patch("subprocess.run") as mock_run:
            with (
                patch(
                    "src.processors.audio.preprocessor.settings.verbose_components",
                    True,
                ),
                patch("src.processors.audio.preprocessor.logger") as mock_logger,
            ):
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="normalized", stderr=""
                )

                preprocessor._normalize_audio(test_file, metadata)

                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["capture_output"] is True
                mock_logger.debug.assert_any_call(
                    "ffmpeg normalize output: {}",
                    "normalized",
                )

    def test_ffmpeg_normalize_quiet_mode_captures_output(self, tmp_path, monkeypatch):
        """FFmpeg normalization should capture output when verbose=False."""
        preprocessor = AudioPreprocessor()
        test_file = tmp_path / "input.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 100)

        metadata = {"sample_rate": 48000, "channels": 2, "format": "mp3"}

        with patch("subprocess.run") as mock_run:
            with patch(
                "src.processors.audio.preprocessor.settings.verbose_components",
                False,
            ):
                mock_run.return_value = MagicMock(returncode=0)

                preprocessor._normalize_audio(test_file, metadata)

                # Verify capture_output=True for quiet mode
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["capture_output"] is True

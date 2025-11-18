"""Unit tests for VAD configuration."""

import pytest
from src.config.model import VADSettings


class TestVADSettings:
    """Test VADSettings configuration."""

    def test_vad_settings_defaults(self):
        """Test default VAD settings."""
        settings = VADSettings()
        assert settings.enabled is True
        assert settings.backend == "silero"
        assert settings.silero_threshold == 0.5
        assert settings.silero_sampling_rate == 16000
        assert settings.device in {"cuda", "cpu"}

    def test_vad_settings_custom_threshold(self):
        """Test custom threshold."""
        settings = VADSettings(silero_threshold=0.7)
        assert settings.silero_threshold == 0.7

    def test_vad_settings_invalid_threshold_high(self):
        """Test threshold validation - too high."""
        with pytest.raises(ValueError):
            VADSettings(silero_threshold=1.5)

    def test_vad_settings_invalid_threshold_low(self):
        """Test threshold validation - too low."""
        with pytest.raises(ValueError):
            VADSettings(silero_threshold=-0.5)

    def test_vad_settings_threshold_boundaries(self):
        """Test threshold at valid boundaries."""
        settings_min = VADSettings(silero_threshold=0.0)
        assert settings_min.silero_threshold == 0.0

        settings_max = VADSettings(silero_threshold=1.0)
        assert settings_max.silero_threshold == 1.0

    def test_vad_settings_invalid_backend(self):
        """Test invalid backend."""
        with pytest.raises(ValueError, match="Unsupported VAD backend"):
            VADSettings(backend="unknown")

    def test_vad_settings_invalid_device(self):
        """Test invalid device."""
        with pytest.raises(ValueError, match="Invalid device"):
            VADSettings(device="tpu")

    def test_vad_settings_valid_devices(self):
        """Test valid device options."""
        for device in ["cuda", "cpu"]:
            settings = VADSettings(device=device)
            assert settings.device == device

    def test_vad_settings_disabled(self):
        """Test creating disabled VAD settings."""
        settings = VADSettings(enabled=False)
        assert settings.enabled is False

    def test_vad_settings_duration_validation(self):
        """Test duration field validation."""
        settings = VADSettings(
            min_speech_duration_ms=100,
            max_speech_duration_ms=30000,
            min_silence_duration_ms=50,
        )
        assert settings.min_speech_duration_ms == 100
        assert settings.max_speech_duration_ms == 30000
        assert settings.min_silence_duration_ms == 50

    def test_vad_settings_negative_duration_invalid(self):
        """Test that negative durations are invalid."""
        with pytest.raises(ValueError):
            VADSettings(min_speech_duration_ms=-100)

    def test_vad_settings_custom_model_path(self):
        """Test custom ONNX model path."""
        model_path = "/path/to/model.onnx"
        settings = VADSettings(silero_model_path=model_path)
        assert settings.silero_model_path == model_path

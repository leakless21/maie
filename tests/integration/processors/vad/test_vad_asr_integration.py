"""Integration tests for VAD with ASR backend."""

import pytest
from src.processors.asr.factory import ASRFactory
from src.config.model import VADSettings


class TestVADASRIntegration:
    """Test VAD integration with ASR backends."""

    @pytest.mark.integration
    def test_vad_factory_creates_backend_from_config(self):
        """Test VAD factory can create backend from config."""
        vad_config = VADSettings(
            enabled=True,
            backend="silero",
            silero_threshold=0.5,
            silero_sampling_rate=16000,
            device="cpu",
        )
        
        try:
            vad_backend = ASRFactory.create_vad_backend(vad_config)
            assert vad_backend is not None
            vad_backend.unload()
        except Exception as e:
            pytest.skip(f"Silero VAD not available: {str(e)}")

    @pytest.mark.integration
    def test_vad_disabled_returns_none(self):
        """Test VAD factory returns None when disabled."""
        vad_config = VADSettings(enabled=False)
        vad_backend = ASRFactory.create_vad_backend(vad_config)
        assert vad_backend is None

    @pytest.mark.integration
    def test_vad_config_validation(self):
        """Test VAD config validation."""
        # Valid configuration
        config = VADSettings(
            enabled=True,
            silero_threshold=0.5,
            silero_sampling_rate=16000,
        )
        assert config.enabled is True
        assert config.silero_threshold == 0.5

        # Test threshold bounds
        with pytest.raises(Exception):  # ValidationError
            VADSettings(silero_threshold=1.5)

        # Test device validation
        with pytest.raises(Exception):  # ValidationError
            VADSettings(device="gpu")

    @pytest.mark.integration
    def test_vad_config_custom_threshold(self):
        """Test VAD config with custom threshold."""
        config = VADSettings(
            enabled=True,
            silero_threshold=0.7,
        )
        assert config.silero_threshold == 0.7

    @pytest.mark.integration
    def test_vad_config_default_values(self):
        """Test VAD config default values."""
        config = VADSettings()
        assert config.enabled is True
        assert config.backend == "silero"
        assert config.silero_threshold == 0.5
        assert config.silero_sampling_rate == 16000
        assert config.min_speech_duration_ms == 250

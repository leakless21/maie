"""
Unit tests for diarizer memory management configuration.

Tests the batch size parameters that prevent memory allocation failures.
"""

import pytest

from src.processors.audio.diarizer import Diarizer, get_diarizer


class TestDiarizerMemoryConfiguration:
    """Test memory management configuration for diarizer."""

    def test_diarizer_accepts_batch_size_parameters(self):
        """Diarizer should accept and store batch size parameters."""
        diarizer = Diarizer(
            model_path="data/models/speaker-diarization-community-1",
            require_cuda=False,
            overlap_threshold=0.3,
            embedding_batch_size=16,
            segmentation_batch_size=8,
        )

        assert diarizer.embedding_batch_size == 16
        assert diarizer.segmentation_batch_size == 8

    def test_diarizer_default_batch_sizes(self):
        """Diarizer should use default batch sizes when not specified."""
        diarizer = Diarizer()

        assert diarizer.embedding_batch_size == 32  # Official default value
        assert diarizer.segmentation_batch_size == 32  # Official default value

    def test_get_diarizer_factory_passes_batch_sizes(self):
        """get_diarizer factory should pass batch size parameters."""
        diarizer = get_diarizer(
            model_path="data/models/speaker-diarization-community-1",
            require_cuda=False,
            overlap_threshold=0.3,
            embedding_batch_size=32,
            segmentation_batch_size=16,
        )

        assert diarizer is not None
        assert diarizer.embedding_batch_size == 32
        assert diarizer.segmentation_batch_size == 16

    def test_get_diarizer_factory_default_batch_sizes(self):
        """get_diarizer should use default batch sizes when not specified."""
        diarizer = get_diarizer(
            model_path="data/models/speaker-diarization-community-1",
            require_cuda=False,
        )

        assert diarizer is not None
        assert diarizer.embedding_batch_size == 32  # Official default
        assert diarizer.segmentation_batch_size == 32  # Official default

    def test_batch_size_configuration_for_different_scenarios(self):
        """Test batch size recommendations for different audio lengths."""
        # Short audio (< 5 min)
        short_diarizer = Diarizer(
            embedding_batch_size=32,
            segmentation_batch_size=32,
        )
        assert short_diarizer.embedding_batch_size == 32
        assert short_diarizer.segmentation_batch_size == 32

        # Medium audio (5-15 min)
        medium_diarizer = Diarizer(
            embedding_batch_size=16,
            segmentation_batch_size=16,
        )
        assert medium_diarizer.embedding_batch_size == 16
        assert medium_diarizer.segmentation_batch_size == 16

        # Long audio (15-60 min)
        long_diarizer = Diarizer(
            embedding_batch_size=8,
            segmentation_batch_size=8,
        )
        assert long_diarizer.embedding_batch_size == 8
        assert long_diarizer.segmentation_batch_size == 8

        # Very long audio (> 60 min)
        very_long_diarizer = Diarizer(
            embedding_batch_size=4,
            segmentation_batch_size=4,
        )
        assert very_long_diarizer.embedding_batch_size == 4
        assert very_long_diarizer.segmentation_batch_size == 4

    def test_batch_size_edge_cases(self):
        """Test batch size edge cases."""
        # Minimum batch size (1)
        min_diarizer = Diarizer(
            embedding_batch_size=1,
            segmentation_batch_size=1,
        )
        assert min_diarizer.embedding_batch_size == 1
        assert min_diarizer.segmentation_batch_size == 1

        # High batch size (for short audio with lots of memory)
        max_diarizer = Diarizer(
            embedding_batch_size=128,
            segmentation_batch_size=128,
        )
        assert max_diarizer.embedding_batch_size == 128
        assert max_diarizer.segmentation_batch_size == 128


class TestDiarizerMemoryIntegration:
    """Integration tests for memory configuration in real scenarios."""

    def test_diarizer_initialization_does_not_crash(self):
        """Diarizer should initialize without memory errors."""
        # This test ensures the basic initialization doesn't crash
        diarizer = Diarizer(
            model_path="data/models/speaker-diarization-community-1",
            require_cuda=False,
            embedding_batch_size=8,
            segmentation_batch_size=8,
        )
        assert diarizer is not None
        assert diarizer.model is None  # Model not loaded yet (lazy loading)

    def test_factory_function_returns_configured_instance(self):
        """Factory function should return properly configured diarizer."""
        diarizer = get_diarizer(
            model_path="data/models/speaker-diarization-community-1",
            require_cuda=False,
            overlap_threshold=0.4,
            embedding_batch_size=12,
            segmentation_batch_size=10,
        )

        assert diarizer is not None
        assert diarizer.embedding_batch_size == 12
        assert diarizer.segmentation_batch_size == 10
        assert diarizer.overlap_threshold == 0.4
        assert diarizer.require_cuda is False


class TestConfigurationCompatibility:
    """Test configuration system compatibility with batch sizes."""

    def test_config_model_has_batch_size_fields(self):
        """DiarizationSettings should have batch size fields."""
        from src.config.model import DiarizationSettings

        # Check fields exist
        assert "embedding_batch_size" in DiarizationSettings.model_fields
        assert "segmentation_batch_size" in DiarizationSettings.model_fields

        # Check default values (official defaults from pyannote.audio)
        settings = DiarizationSettings()
        assert settings.embedding_batch_size == 32  # Official default
        assert settings.segmentation_batch_size == 32  # Official default

    def test_config_batch_size_validation(self):
        """DiarizationSettings should validate batch size ranges."""
        from src.config.model import DiarizationSettings

        # Valid values
        valid_settings = DiarizationSettings(
            embedding_batch_size=16,
            segmentation_batch_size=16,
        )
        assert valid_settings.embedding_batch_size == 16
        assert valid_settings.segmentation_batch_size == 16

        # Edge cases
        min_settings = DiarizationSettings(
            embedding_batch_size=1,
            segmentation_batch_size=1,
        )
        assert min_settings.embedding_batch_size == 1

        max_settings = DiarizationSettings(
            embedding_batch_size=256,
            segmentation_batch_size=256,
        )
        assert max_settings.segmentation_batch_size == 256

        # Invalid values should raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            DiarizationSettings(embedding_batch_size=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            DiarizationSettings(segmentation_batch_size=300)

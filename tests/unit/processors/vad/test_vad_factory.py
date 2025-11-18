"""Unit tests for VAD factory."""

import pytest
from src.processors.vad.factory import VADFactory
from src.processors.vad.silero import SileroVADBackend
from src.processors.vad.base import VADBackend, VADResult


class TestVADFactory:
    """Test VAD factory pattern implementation."""

    def test_factory_registers_backends(self):
        """Test that factory has registered backends."""
        assert "silero" in VADFactory.BACKENDS
        assert VADFactory.BACKENDS["silero"] == SileroVADBackend

    def test_factory_register_backend(self):
        """Test registering a new backend."""

        class DummyBackend(VADBackend):
            def detect_speech(self, audio_path: str) -> VADResult:
                return VADResult(
                    segments=[],
                    total_duration=0.0,
                    speech_duration=0.0,
                    speech_ratio=0.0,
                    processing_time=0.0,
                    backend_info={}
                )

            def unload(self) -> None:
                pass

            def get_version_info(self):  # type: ignore
                pass

        initial_count = len(VADFactory.BACKENDS)
        VADFactory.register_backend("dummy", DummyBackend)
        assert len(VADFactory.BACKENDS) == initial_count + 1
        assert VADFactory.BACKENDS["dummy"] == DummyBackend

    def test_factory_create_silero_backend(self):
        """Test creating Silero VAD backend."""
        try:
            backend = VADFactory.create("silero")
            assert isinstance(backend, SileroVADBackend)
            backend.unload()
        except Exception as e:
            # Silero VAD might not be installed in test environment
            pytest.skip(f"Silero VAD not available: {str(e)}")

    def test_factory_create_unknown_backend(self):
        """Test creating unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown VAD backend type"):
            VADFactory.create("unknown_backend")

    def test_factory_create_with_kwargs(self):
        """Test creating backend with keyword arguments."""
        try:
            backend = VADFactory.create(
                "silero",
                threshold=0.7,
                sampling_rate=16000,
            )
            assert isinstance(backend, SileroVADBackend)
            backend.unload()
        except Exception as e:
            pytest.skip(f"Silero VAD not available: {str(e)}")

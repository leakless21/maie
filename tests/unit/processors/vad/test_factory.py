"""Unit tests for VAD Factory."""

import pytest
from src.processors.vad.factory import VADFactory
from src.processors.vad.silero import SileroVADBackend
from src.processors.vad.base import VADBackend


class TestVADFactory:
    """Test VADFactory registration and creation."""

    def test_factory_register_backend(self):
        """Test registering a backend."""
        initial_backends = set(VADFactory.BACKENDS.keys())
        assert "silero" in initial_backends

    def test_factory_create_silero(self):
        """Test creating Silero VAD backend."""
        try:
            backend = VADFactory.create("silero", device="cpu")
            assert isinstance(backend, SileroVADBackend)
            # Cleanup
            backend.unload()
        except RuntimeError as e:
            if "silero-vad not installed" in str(e):
                pytest.skip("silero-vad not installed")
            raise

    def test_factory_unknown_backend(self):
        """Test error on unknown backend."""
        with pytest.raises(ValueError, match="Unknown VAD backend type"):
            VADFactory.create("unknown_backend")

    def test_factory_list_backends(self):
        """Test listing available backends."""
        backends = VADFactory.BACKENDS
        assert len(backends) > 0
        assert "silero" in backends

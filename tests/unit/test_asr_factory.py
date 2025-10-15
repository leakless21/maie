# tests/unit/test_asr_factory.py
import pytest
from typing import Any, Dict

from src.processors.asr import factory as asr_factory


class DummyBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class BackendRaiser:
    def __init__(self, **kwargs):
        raise RuntimeError("init failed")


class FakePreprocessor:
    def __init__(self):
        self.processed = True


class FakeMetrics:
    def __init__(self):
        self.metrics = {}


@pytest.fixture(autouse=True)
def reset_backends():
    """Reset factory state before each test."""
    original = dict(asr_factory.ASRFactory.BACKENDS)
    asr_factory.ASRFactory.BACKENDS = {}
    yield
    asr_factory.ASRFactory.BACKENDS = original


def test_register_backend():
    asr_factory.ASRFactory.register_backend("dummy", DummyBackend)
    assert "dummy" in asr_factory.ASRFactory.BACKENDS
    assert asr_factory.ASRFactory.BACKENDS["dummy"] is DummyBackend


def test_register_backend_and_overwrite():
    asr_factory.ASRFactory.register_backend("dummy", DummyBackend)
    assert "dummy" in asr_factory.ASRFactory.BACKENDS
    assert asr_factory.ASRFactory.BACKENDS["dummy"] is DummyBackend

    # overwrite an existing backend registration
    asr_factory.ASRFactory.register_backend("dummy", BackendRaiser)
    assert asr_factory.ASRFactory.BACKENDS["dummy"] is BackendRaiser


def test_create_backend():
    asr_factory.ASRFactory.register_backend("dummy", DummyBackend)
    inst = asr_factory.ASRFactory.create("dummy", foo="bar", n=1)
    assert isinstance(inst, DummyBackend)
    assert inst.kwargs == {"foo": "bar", "n": 1}


def test_create_unknown_backend():
    with pytest.raises(ValueError, match="Unknown ASR backend"):
        asr_factory.ASRFactory.create("nope")


def test_create_backend_init_error():
    asr_factory.ASRFactory.register_backend("raiser", BackendRaiser)
    with pytest.raises(RuntimeError, match="init failed"):
        asr_factory.ASRFactory.create("raiser")


def test_create_with_audio_processing_success(monkeypatch):
    # Mock AudioPreprocessor
    class MockPreprocessor:
        def preprocess(self, audio_path):
            return {"duration": 5.0, "sample_rate": 16000}
    
    monkeypatch.setattr("src.processors.asr.factory.AudioPreprocessor", MockPreprocessor)

    asr_factory.ASRFactory.register_backend("dummy", DummyBackend)

    result = asr_factory.ASRFactory.create_with_audio_processing(
        backend_type="dummy", foo="bar"
    )
    
    assert "asr_processor" in result
    assert "audio_preprocessor" in result
    assert "audio_metrics_collector" in result
    assert isinstance(result["asr_processor"], DummyBackend)
    assert result["asr_processor"].kwargs["foo"] == "bar"


def test_registering_non_class_backend_results_in_type_error_on_creation():
    """
    register_backend does not validate types; ensure a non-callable entry leads to a TypeError
    when factory attempts to instantiate it.
    """
    asr_factory.ASRFactory.register_backend("notaclass", {"x": 1})
    with pytest.raises(TypeError):
        asr_factory.ASRFactory.create("notaclass")


def test_smoke_factory_creation_fast():
    """
    A minimal smoke test to ensure the factory can create a registered backend quickly.
    This avoids heavy external dependencies by using DummyBackend.
    """
    asr_factory.ASRFactory.register_backend("dummy", DummyBackend)
    inst = asr_factory.ASRFactory.create("dummy")
    assert isinstance(inst, DummyBackend)

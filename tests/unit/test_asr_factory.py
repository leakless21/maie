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
def clear_backends(monkeypatch):
    """
    Ensure each test runs with a clean registry and restore afterwards.
    This keeps tests deterministic and isolated from module-level registrations.
    """
    original = dict(asr_factory.ASRProcessorFactory.BACKENDS)
    asr_factory.ASRProcessorFactory.BACKENDS = {}
    yield
    asr_factory.ASRProcessorFactory.BACKENDS = original


def test_register_backend_and_overwrite():
    asr_factory.ASRProcessorFactory.register_backend("dummy", DummyBackend)
    assert "dummy" in asr_factory.ASRProcessorFactory.BACKENDS
    assert asr_factory.ASRProcessorFactory.BACKENDS["dummy"] is DummyBackend

    # overwrite an existing backend registration
    asr_factory.ASRProcessorFactory.register_backend("dummy", BackendRaiser)
    assert asr_factory.ASRProcessorFactory.BACKENDS["dummy"] is BackendRaiser


def test_create_returns_instance_with_kwargs():
    asr_factory.ASRProcessorFactory.register_backend("dummy", DummyBackend)
    inst = asr_factory.ASRProcessorFactory.create("dummy", foo="bar", n=1)
    assert isinstance(inst, DummyBackend)
    assert inst.kwargs == {"foo": "bar", "n": 1}


def test_create_raises_unknown_backend():
    with pytest.raises(ValueError) as exc:
        asr_factory.ASRProcessorFactory.create("nope")
    assert "Unknown ASR backend type" in str(exc.value)


def test_create_propagates_backend_init_exception():
    asr_factory.ASRProcessorFactory.register_backend("raiser", BackendRaiser)
    with pytest.raises(RuntimeError) as exc:
        asr_factory.ASRProcessorFactory.create("raiser")
    assert "init failed" in str(exc.value)


def test_create_with_audio_processing_integration(monkeypatch):
    """
    Integration test for create_with_audio_processing:
    - monkeypatch AudioPreprocessor and AudioMetricsCollector to lightweight fakes
    - ensure returned dict contains expected keys and types
    - ensure kwargs propagate to ASR backend
    """
    asr_factory.ASRProcessorFactory.register_backend("dummy", DummyBackend)
    monkeypatch.setattr(asr_factory, "AudioPreprocessor", FakePreprocessor)
    monkeypatch.setattr(asr_factory, "AudioMetricsCollector", FakeMetrics)

    result = asr_factory.ASRProcessorFactory.create_with_audio_processing("dummy", lang="en")
    assert isinstance(result, dict)
    assert "asr_processor" in result and "audio_preprocessor" in result and "audio_metrics_collector" in result
    assert isinstance(result["asr_processor"], DummyBackend)
    assert isinstance(result["audio_preprocessor"], FakePreprocessor)
    assert isinstance(result["audio_metrics_collector"], FakeMetrics)
    assert result["asr_processor"].kwargs.get("lang") == "en"


def test_registering_non_class_backend_results_in_type_error_on_creation():
    """
    register_backend does not validate types; ensure a non-callable entry leads to a TypeError
    when factory attempts to instantiate it.
    """
    asr_factory.ASRProcessorFactory.register_backend("notaclass", {"x": 1})
    with pytest.raises(TypeError):
        asr_factory.ASRProcessorFactory.create("notaclass")


def test_smoke_factory_creation_fast():
    """
    A minimal smoke test to ensure the factory can create a registered backend quickly.
    This avoids heavy external dependencies by using DummyBackend.
    """
    asr_factory.ASRProcessorFactory.register_backend("dummy", DummyBackend)
    inst = asr_factory.ASRProcessorFactory.create("dummy")
    assert isinstance(inst, DummyBackend)
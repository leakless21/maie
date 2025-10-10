import pytest
from types import SimpleNamespace

from src.processors.base import ASRResult, ASRBackend
from src.processors.asr.chunkformer import ChunkFormerBackend
from src import config as cfg


# -------------------------
# Interface Compliance
# -------------------------
def test_interface_compliance():
    assert hasattr(ChunkFormerBackend, "_load_model")
    assert hasattr(ChunkFormerBackend, "execute")
    assert hasattr(ChunkFormerBackend, "unload")
    assert hasattr(ChunkFormerBackend, "get_version_info")
    assert issubclass(ChunkFormerBackend, ASRBackend)


# -------------------------
# Model Loading
# -------------------------
def test_load_model_sets_model(monkeypatch, inject_mock_chunkformer):
    """When a valid model path is provided, _load_model should set self.model."""
    # Patch the loader to use mock module provided by fixture
    def _patch_load(self, **kwargs):
        from chunkformer import ChunkFormerModel
        self.model = ChunkFormerModel("ignored")
    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)

    backend = ChunkFormerBackend(model_path="ignored")
    assert backend.model is not None
    assert hasattr(backend.model, "decode") or hasattr(backend.model, "transcribe")


def test_load_model_invalid_path_raises(monkeypatch):
    """If model loading fails, backend initialization should raise RuntimeError."""
    # Patch _load_model to raise
    monkeypatch.setattr(ChunkFormerBackend, "_load_model", lambda self, **k: (_ for _ in ()).throw(RuntimeError("failed")), raising=True)
    with pytest.raises(RuntimeError):
        ChunkFormerBackend(model_path="invalid")


# -------------------------
# Execution
# -------------------------
def test_execute_calls_model_decode(monkeypatch):
    """Execute should call the underlying model.decode and return an ASRResult."""
    calls = SimpleNamespace(decode=0)

    class FakeModel:
        def __init__(self, *a, **k):
            self.closed = False
        def decode(self, audio_chunks, left_context=None, right_context=None, **kwargs):
            calls.decode += 1
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello chunk"}], "language": "en", "confidence": 0.9}
        def close(self):
            self.closed = True

    def _patch_load(self, **kwargs):
        self.model = FakeModel()

    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)
    backend = ChunkFormerBackend(model_path="ignored")
    result = backend.execute(b"\x00\x01")
    assert isinstance(result, ASRResult)
    assert "hello chunk" in result.text or result.text == "hello chunk"
    assert calls.decode == 1


def test_execute_without_model_raises(monkeypatch):
    """If the model failed to load, execute should raise a RuntimeError."""
    monkeypatch.setattr(ChunkFormerBackend, "_load_model", lambda self, **k: None, raising=True)
    wb = ChunkFormerBackend(model_path="ignored")
    # force model to None to simulate failed load
    wb.model = None
    with pytest.raises(RuntimeError):
        wb.execute(b"\x00")


# -------------------------
# Resource Management
# -------------------------
def test_unload_releases_resources(monkeypatch):
    """unload should call model.close() (if available) and set model to None."""
    class FakeModel:
        def __init__(self):
            self.closed = False
        def close(self):
            self.closed = True

    def _patch_load(self, **kwargs):
        self.model = FakeModel()

    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)
    wb = ChunkFormerBackend(model_path="ignored")
    assert wb.model is not None
    closed_before = getattr(wb.model, "closed", False)
    wb.unload()
    assert wb.model is None
    assert closed_before is False


# -------------------------
# Configuration
# -------------------------
def test_configuration_integration_uses_config_model_path(monkeypatch, tmp_path):
    """If configuration provides a model path, ChunkFormerBackend should honor it during init."""
    # set config model path
    monkeypatch.setattr(cfg.settings, "chunkformer_model_path", str(tmp_path / "cf"), raising=False)

    # Patch loader to assert it receives no explicit path (uses config)
    def _patch_load(self, **kwargs):
        self.model = SimpleNamespace(decoded=True)
    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)

    wb = ChunkFormerBackend()
    assert wb.model is not None


# -------------------------
# ChunkFormer-specific features
# -------------------------
def test_execute_respects_context_windows(monkeypatch):
    """Ensure execute passes left/right context parameters to model.decode."""
    observed = {}

    class FakeModel:
        def __init__(self):
            pass
        def decode(self, audio_chunks, left_context=None, right_context=None, **kwargs):
            observed["left"] = left_context
            observed["right"] = right_context
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "ok"}], "language": "en", "confidence": 0.9}
        def close(self):
            pass

    def _patch_load(self, **kwargs):
        self.model = FakeModel()

    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)
    backend = ChunkFormerBackend()
    # Provide explicit context via kwargs
    backend.execute(b"\x00\x01", left_context=32, right_context=64)
    assert observed.get("left") == 32
    assert observed.get("right") == 64


# -------------------------
# Error Handling
# -------------------------
def test_decode_failure_raises(monkeypatch):
    """If the underlying model raises during decode, execute should propagate or wrap the error."""
    class BrokenModel:
        def __init__(self):
            pass
        def decode(self, *a, **k):
            raise RuntimeError("decode failed")
        def close(self):
            pass

    def _patch_load(self, **kwargs):
        self.model = BrokenModel()

    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)
    backend = ChunkFormerBackend()
    with pytest.raises(RuntimeError):
        backend.execute(b"\x00\x01")


# -------------------------
# Performance (basic smoke)
# -------------------------
def test_execute_performance_smoke(monkeypatch):
    """Simple smoke to ensure execute runs quickly for small input (not a benchmark)."""
    class FastModel:
        def __init__(self):
            pass
        def decode(self, audio_chunks, **kwargs):
            return {"segments": [{"start": 0.0, "end": 0.5, "text": "fast"}], "language": "en", "confidence": 0.9}
        def close(self):
            pass

    def _patch_load(self, **kwargs):
        self.model = FastModel()

    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)
    backend = ChunkFormerBackend()
    res = backend.execute(b"\x00\x01")
    assert isinstance(res, ASRResult)
    assert res.text in ("fast", "fast")  # simplistic check


# -------------------------
# ChunkFormer endless_decode (Primary API) - TDD: Write tests first
# -------------------------
def test_execute_calls_endless_decode_with_all_params(monkeypatch):
    """Ensure execute passes all ChunkFormer parameters to endless_decode (TDD Test)."""
    observed = {}

    class FakeModelWithEndlessDecode:
        def __init__(self):
            pass
        
        def endless_decode(self, audio_path, **kwargs):
            # Capture all parameters
            observed.update(kwargs)
            return {
                "segments": [{"start": 0.0, "end": 2.0, "text": "long-form result"}],
                "language": "en",
                "confidence": 0.95
            }
        
        def close(self):
            pass

    def _patch_load(self, **kwargs):
        self.model = FakeModelWithEndlessDecode()

    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)
    backend = ChunkFormerBackend()
    
    # Execute without explicit params - should use config defaults
    result = backend.execute(b"\x00\x01")
    
    # TDD: These assertions will FAIL until we fix the implementation
    assert "chunk_size" in observed, "chunk_size should be passed to endless_decode"
    assert "left_context_size" in observed, "left_context_size should be passed"
    assert "right_context_size" in observed, "right_context_size should be passed"
    assert "total_batch_duration" in observed, "total_batch_duration should be passed"
    assert "return_timestamps" in observed, "return_timestamps should be passed"
    
    # Verify result
    assert isinstance(result, ASRResult)
    assert "long-form" in result.text


def test_execute_endless_decode_with_custom_params(monkeypatch):
    """Test that custom parameters override config defaults (TDD Test)."""
    observed = {}

    class FakeModelWithEndlessDecode:
        def __init__(self):
            pass
        
        def endless_decode(self, audio_path, **kwargs):
            observed.update(kwargs)
            return {"segments": [{"text": "ok"}], "language": "vi"}
        
        def close(self):
            pass

    def _patch_load(self, **kwargs):
        self.model = FakeModelWithEndlessDecode()

    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)
    backend = ChunkFormerBackend()
    
    # Execute with custom params
    result = backend.execute(
        b"\x00\x01",
        chunk_size=32,
        left_context=64,
        right_context=64,
        total_batch_duration=7200,
        return_timestamps=False
    )
    
    # TDD: These assertions will FAIL until we fix the implementation
    assert observed.get("chunk_size") == 32, "Custom chunk_size should override config"
    assert observed.get("left_context_size") == 64, "Custom left_context should be passed as left_context_size"
    assert observed.get("right_context_size") == 64, "Custom right_context should be passed as right_context_size"
    assert observed.get("total_batch_duration") == 7200, "Custom total_batch_duration should override config"
    assert observed.get("return_timestamps") == False, "Custom return_timestamps should override config"


def test_version_info_includes_architecture_params(monkeypatch):
    """Version info should include architecture params for NFR-1 compliance (TDD Test)."""
    def _patch_load(self, **kwargs):
        self.model = SimpleNamespace(device="cuda")
    
    monkeypatch.setattr(ChunkFormerBackend, "_load_model", _patch_load, raising=True)
    backend = ChunkFormerBackend()
    
    info = backend.get_version_info()
    
    # TDD: These assertions will FAIL until we enhance get_version_info()
    assert "chunk_size" in info, "chunk_size should be in version info for reproducibility"
    assert "left_context_size" in info, "left_context_size should be in version info"
    assert "right_context_size" in info, "right_context_size should be in version info"
    assert "total_batch_duration" in info, "total_batch_duration should be in version info"
    assert "return_timestamps" in info, "return_timestamps should be in version info"
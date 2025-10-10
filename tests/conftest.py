"""
Pytest configuration and shared fixtures for MAIE test suite.

Testing Strategy:
- Unit tests: Fast, use mocks, test logic in isolation
- Integration tests: Use real libraries when available, marked with @pytest.mark.integration
- E2E tests: Full system tests with real models

Markers:
- @pytest.mark.integration: Requires real libraries (faster-whisper, etc.)
- @pytest.mark.slow: Tests that take >5 seconds
- @pytest.mark.gpu: Requires GPU hardware

Note: Integration tests require CUDA library path configuration on Linux.
Use scripts/run_integration_tests.sh to run tests with proper environment setup.
See docs/whisper-cuda-fix.md for details.
"""
import sys
import pytest
from pathlib import Path
from types import ModuleType

# Ensure project root is on sys.path so tests can import src.*
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require real libraries)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (>5 seconds)"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU hardware"
    )


# ============================================================================
# Mock faster-whisper Module (for unit tests)
# ============================================================================

class MockWhisperModel:
    """Mock WhisperModel for testing without real library."""
    
    def __init__(self, model_size_or_path, device="cuda", compute_type="float16", **kwargs):
        self.model_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.kwargs = kwargs
        self.closed = False
        self.transcribe_call_count = 0
        
    def transcribe(self, audio, **kwargs):
        """Mock transcribe that returns realistic structure.
        
        Returns (segments_generator, info) tuple matching faster-whisper API.
        """
        self.transcribe_call_count += 1
        self.last_transcribe_kwargs = kwargs
        
        # Create mock segment
        class MockSegment:
            def __init__(self):
                self.start = 0.0
                self.end = 1.5
                self.text = "hello world from mock"
                
        def segments_generator():
            yield MockSegment()
        
        # Create mock info
        class MockInfo:
            def __init__(self):
                self.language = "en"
                self.language_probability = 0.95
                
        return segments_generator(), MockInfo()
    
    def close(self):
        """Mock cleanup."""
        self.closed = True


@pytest.fixture
def mock_faster_whisper():
    """Provide a mock faster-whisper module for unit tests."""
    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = MockWhisperModel
    # Add load_model marker to indicate test mode
    fake_fw.load_model = lambda *args, **kwargs: MockWhisperModel(*args, **kwargs)
    return fake_fw


# ============================================================================
# Mock ChunkFormer Module (for unit tests)
# ============================================================================
class MockChunkFormerModel:
    """Mock ChunkFormer model to emulate chunk-based ASR APIs."""

    def __init__(self, model_path, device="cuda", **kwargs):
        self.model_path = model_path
        self.device = device
        self.kwargs = kwargs
        self.closed = False
        self.decode_call_count = 0

    def decode(self, audio_chunks, left_context=None, right_context=None, **kwargs):
        """Decode a list/iterator of audio chunks and return structured results.

        Returns a dict with keys similar to expected production API:
        {
            "segments": [{"start": 0.0, "end": 1.5, "text": "transcript"}],
            "language": "en",
            "confidence": 0.9
        }
        """
        self.decode_call_count += 1
        self.last_decode_kwargs = {
            "left_context": left_context,
            "right_context": right_context,
            **kwargs,
        }

        # Single mock segment
        segments = [{"start": 0.0, "end": 1.5, "text": "chunkformer mock transcript"}]
        return {"segments": segments, "language": "en", "confidence": 0.92}

    # Backwards-compatible alias
    def transcribe(self, audio, **kwargs):
        """Fallback transcribe that accepts raw audio and returns similar shape."""
        self.decode_call_count += 1
        self.last_decode_kwargs = kwargs
        segments = [{"start": 0.0, "end": 1.5, "text": "chunkformer mock transcript"}]
        return segments, {"language": "en", "confidence": 0.92}

    def close(self):
        self.closed = True


@pytest.fixture
def mock_chunkformer():
    """Provide a mock chunkformer module for unit tests."""
    fake_cf = ModuleType("chunkformer")
    fake_cf.ChunkFormerModel = MockChunkFormerModel
    # Provide a compat load_model factory for tests
    fake_cf.load_model = lambda *args, **kwargs: MockChunkFormerModel(*args, **kwargs)
    return fake_cf


@pytest.fixture
def inject_mock_chunkformer(mock_chunkformer):
    """Inject mock chunkformer into sys.modules."""
    original = sys.modules.get("chunkformer")
    sys.modules["chunkformer"] = mock_chunkformer
    yield mock_chunkformer

    # Cleanup
    if original is not None:
        sys.modules["chunkformer"] = original
    else:
        sys.modules.pop("chunkformer", None)


@pytest.fixture
def inject_mock_faster_whisper(mock_faster_whisper):
    """Inject mock faster-whisper into sys.modules."""
    original = sys.modules.get("faster_whisper")
    sys.modules["faster_whisper"] = mock_faster_whisper
    yield mock_faster_whisper
    
    # Cleanup
    if original is not None:
        sys.modules["faster_whisper"] = original
    else:
        sys.modules.pop("faster_whisper", None)


# ============================================================================
# Real Library Detection
# ============================================================================

# Cache the faster_whisper and GPU availability checks
_HAS_FASTER_WHISPER = None
_HAS_GPU = None

def has_faster_whisper():
    """Check if faster-whisper is actually installed and can be imported.
    
    Result is cached to avoid repeated imports which can cause issues with
    PyTorch 2.8's module initialization.
    """
    global _HAS_FASTER_WHISPER
    
    if _HAS_FASTER_WHISPER is not None:
        return _HAS_FASTER_WHISPER
    
    try:
        import faster_whisper
        _HAS_FASTER_WHISPER = True
        return True
    except (ImportError, RuntimeError) as e:
        # ImportError: library not installed
        # RuntimeError: can occur with torch/ctranslate2 compatibility issues
        _HAS_FASTER_WHISPER = False
        return False


def has_gpu():
    """Check if GPU is available.
    
    Result is cached to avoid repeated torch imports which can cause issues with
    PyTorch 2.8's module re-initialization bug.
    """
    global _HAS_GPU
    
    if _HAS_GPU is not None:
        return _HAS_GPU
    
    try:
        import torch
        _HAS_GPU = torch.cuda.is_available()
        return _HAS_GPU
    except (ImportError, RuntimeError) as e:
        _HAS_GPU = False
        return False


@pytest.fixture
def skip_if_no_faster_whisper():
    """Skip test if faster-whisper is not installed."""
    if not has_faster_whisper():
        pytest.skip("faster-whisper not installed")


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not has_gpu():
        pytest.skip("GPU not available")


# ============================================================================
# Test Assets
# ============================================================================

@pytest.fixture
def test_assets_dir():
    """Path to test assets directory."""
    return Path(__file__).parent / "assets"


@pytest.fixture
def sample_audio_path(test_assets_dir):
    """Path to sample audio file for testing."""
    audio_file = test_assets_dir / "Northern Female 1.wav"
    if audio_file.exists():
        return audio_file
    pytest.skip(f"Test audio not found: {audio_file}")


@pytest.fixture
def sample_audio_bytes(sample_audio_path):
    """Read sample audio as bytes."""
    return sample_audio_path.read_bytes()


# ============================================================================
# Model Paths
# ============================================================================

@pytest.fixture
def whisper_model_path():
    """Path to local Whisper model (if exists)."""
    model_path = Path("data/models/era-x-wow-turbo-v1.1-ct2")
    if model_path.exists():
        return model_path
    return None


@pytest.fixture
def skip_if_no_whisper_model(whisper_model_path):
    """Skip test if local Whisper model is not available."""
    if whisper_model_path is None:
        pytest.skip("Whisper model not found at data/models/era-x-wow-turbo-v1.1-ct2")


# ============================================================================
# Temporary Files
# ============================================================================

@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file (sine wave)."""
    import wave
    import struct
    import math
    
    audio_path = tmp_path / "test_audio.wav"
    
    # Generate 1 second of 440Hz sine wave
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    amplitude = 16000
    
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        
        for i in range(int(sample_rate * duration)):
            value = int(amplitude * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack("<h", value))
    
    return audio_path


# ============================================================================
# Configuration Mocking
# ============================================================================

@pytest.fixture
def mock_config(monkeypatch):
    """Provide a fixture to easily mock config values."""
    from src import config
    
    class ConfigMocker:
        def set(self, key, value):
            monkeypatch.setattr(config.settings, key, value)
            
        def set_whisper_defaults(self):
            """Set common Whisper config defaults."""
            self.set("whisper_device", "cuda")
            self.set("whisper_compute_type", "float16")
            self.set("whisper_beam_size", 5)
            self.set("whisper_vad_filter", True)
            self.set("whisper_vad_min_silence_ms", 500)
            self.set("whisper_vad_speech_pad_ms", 400)
            self.set("whisper_condition_on_previous_text", True)
            self.set("whisper_language", None)
            self.set("whisper_cpu_fallback", True)
    
    return ConfigMocker()


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_modules():
    """Clean up sys.modules after each test to prevent cross-contamination."""
    before = dict(sys.modules)
    yield
    
    # Remove any modules added during test
    to_remove = set(sys.modules) - set(before)
    for module_name in to_remove:
        sys.modules.pop(module_name, None)
"""
Integration tests for WhisperBackend using real faster-whisper library.

These tests require:
- faster-whisper installed
- Local model at data/models/era-x-wow-turbo-v1.1-ct2 (or will download tiny.en)
- GPU recommended but CPU fallback supported

Run with: pytest tests/integration/test_whisper_real.py -v
Skip integration tests: pytest -m "not integration"
"""
import pytest
import sys
from pathlib import Path

from src.processors.asr.whisper import WhisperBackend
from src.config import settings


@pytest.mark.integration
def test_real_whisper_model_loading(skip_if_no_faster_whisper, skip_if_no_whisper_model, whisper_model_path):
    """Test loading a real Whisper model."""
    backend = WhisperBackend()
    
    # Model loads automatically in __init__
    assert backend.model is not None
    
    # Cleanup
    backend.unload()
    assert backend.model is None


@pytest.mark.integration
def test_real_transcription_with_tiny_model(skip_if_no_faster_whisper, temp_audio_file):
    """Test transcription using tiny.en model (will download if needed)."""
    # Override config to use tiny.en model for fast testing
    import os
    original_model_path = os.environ.get("WHISPER_MODEL_PATH")
    os.environ["WHISPER_MODEL_PATH"] = "tiny.en"
    
    try:
        backend = WhisperBackend()
        
        # Transcribe the sine wave (won't produce real text, but tests the API)
        result = backend.execute(str(temp_audio_file))
        
        assert result is not None
        assert "text" in result
        assert "segments" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["segments"], list)
        
        backend.unload()
        
    finally:
        # Restore original config
        if original_model_path:
            os.environ["WHISPER_MODEL_PATH"] = original_model_path
        else:
            os.environ.pop("WHISPER_MODEL_PATH", None)


@pytest.mark.integration
@pytest.mark.slow
def test_real_transcription_with_local_model(
    skip_if_no_faster_whisper, 
    skip_if_no_whisper_model, 
    whisper_model_path,
    sample_audio_path
):
    """Test transcription with local model using real audio."""
    backend = WhisperBackend()
    
    # Transcribe real audio
    result = backend.execute(str(sample_audio_path))
    
    # Validate structure
    assert result is not None
    assert "text" in result
    assert "segments" in result
    assert "language" in result
    assert "duration" in result
    
    # Validate content
    assert len(result["text"]) > 0, "Expected non-empty transcription"
    assert len(result["segments"]) > 0, "Expected at least one segment"
    
    # Validate segment structure
    first_segment = result["segments"][0]
    assert "start" in first_segment
    assert "end" in first_segment
    assert "text" in first_segment
    assert first_segment["start"] >= 0
    assert first_segment["end"] > first_segment["start"]
    
    backend.unload()


@pytest.mark.integration
def test_real_version_info(skip_if_no_faster_whisper):
    """Test that version info returns actual faster-whisper metadata."""
    backend = WhisperBackend()
    
    version_info = backend.get_version_info()
    
    # Check structure
    assert "name" in version_info
    assert "version" in version_info
    assert version_info["name"] == "faster-whisper"
    
    # Check that we get real config values (not hardcoded)
    assert "model_path" in version_info
    assert "device" in version_info
    assert "compute_type" in version_info
    
    # Newly added fields
    assert "model_variant" in version_info
    assert "beam_size" in version_info
    assert "vad_filter" in version_info
    assert "condition_on_previous_text" in version_info
    assert "language" in version_info
    
    backend.unload()


@pytest.mark.integration
def test_real_vad_filtering(skip_if_no_faster_whisper, temp_audio_file):
    """Test that VAD filtering works with real faster-whisper."""
    import os
    
    # Test with tiny.en model
    original_model = os.environ.get("WHISPER_MODEL_PATH")
    original_vad = os.environ.get("WHISPER_VAD_FILTER")
    
    try:
        os.environ["WHISPER_MODEL_PATH"] = "tiny.en"
        os.environ["WHISPER_VAD_FILTER"] = "true"
        
        backend = WhisperBackend()
        
        # Transcribe with VAD
        result = backend.execute(str(temp_audio_file))
        
        assert result is not None
        assert "segments" in result
        
        backend.unload()
        
    finally:
        # Restore
        if original_model:
            os.environ["WHISPER_MODEL_PATH"] = original_model
        else:
            os.environ.pop("WHISPER_MODEL_PATH", None)
            
        if original_vad:
            os.environ["WHISPER_VAD_FILTER"] = original_vad
        else:
            os.environ.pop("WHISPER_VAD_FILTER", None)


@pytest.mark.integration
def test_real_condition_on_previous_text(skip_if_no_faster_whisper, temp_audio_file):
    """Test that condition_on_previous_text parameter is passed correctly."""
    import os
    from unittest.mock import patch
    
    original_model = os.environ.get("WHISPER_MODEL_PATH")
    original_condition = os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT")
    
    try:
        os.environ["WHISPER_MODEL_PATH"] = "tiny.en"
        os.environ["WHISPER_CONDITION_ON_PREVIOUS_TEXT"] = "false"
        
        backend = WhisperBackend()
        
        # Spy on the transcribe call to verify parameter
        original_transcribe = backend.model.transcribe
        call_kwargs = {}
        
        def spy_transcribe(*args, **kwargs):
            call_kwargs.update(kwargs)
            return original_transcribe(*args, **kwargs)
        
        backend.model.transcribe = spy_transcribe
        
        # Execute transcription
        result = backend.execute(str(temp_audio_file))
        
        # Verify the parameter was passed
        assert "condition_on_previous_text" in call_kwargs
        assert call_kwargs["condition_on_previous_text"] is False
        
        backend.unload()
        
    finally:
        if original_model:
            os.environ["WHISPER_MODEL_PATH"] = original_model
        else:
            os.environ.pop("WHISPER_MODEL_PATH", None)
            
        if original_condition:
            os.environ["WHISPER_CONDITION_ON_PREVIOUS_TEXT"] = original_condition
        else:
            os.environ.pop("WHISPER_CONDITION_ON_PREVIOUS_TEXT", None)


@pytest.mark.integration
def test_real_language_parameter(skip_if_no_faster_whisper, temp_audio_file):
    """Test that language parameter is passed correctly to faster-whisper."""
    import os
    
    original_model = os.environ.get("WHISPER_MODEL_PATH")
    original_language = os.environ.get("WHISPER_LANGUAGE")
    
    try:
        os.environ["WHISPER_MODEL_PATH"] = "tiny.en"
        os.environ["WHISPER_LANGUAGE"] = "en"
        
        backend = WhisperBackend()
        
        # Spy on transcribe call
        original_transcribe = backend.model.transcribe
        call_kwargs = {}
        
        def spy_transcribe(*args, **kwargs):
            call_kwargs.update(kwargs)
            return original_transcribe(*args, **kwargs)
        
        backend.model.transcribe = spy_transcribe
        
        # Execute
        result = backend.execute(str(temp_audio_file))
        
        # Verify language was passed
        assert "language" in call_kwargs
        assert call_kwargs["language"] == "en"
        
        backend.unload()
        
    finally:
        if original_model:
            os.environ["WHISPER_MODEL_PATH"] = original_model
        else:
            os.environ.pop("WHISPER_MODEL_PATH", None)
            
        if original_language:
            os.environ["WHISPER_LANGUAGE"] = original_language
        else:
            os.environ.pop("WHISPER_LANGUAGE", None)


@pytest.mark.integration
@pytest.mark.gpu
def test_real_gpu_execution(skip_if_no_faster_whisper, skip_if_no_gpu, temp_audio_file):
    """Test that GPU execution works correctly."""
    import os
    
    original_model = os.environ.get("WHISPER_MODEL_PATH")
    original_device = os.environ.get("WHISPER_DEVICE")
    
    try:
        os.environ["WHISPER_MODEL_PATH"] = "tiny.en"
        os.environ["WHISPER_DEVICE"] = "cuda"
        
        backend = WhisperBackend()
        
        # Verify model loaded on GPU
        assert backend.model.device == "cuda"
        import sys
        
        # Execute transcription
        result = backend.execute(str(temp_audio_file))
        
        assert result is not None
        
        backend.unload()
        
    finally:
        if original_model:
            os.environ["WHISPER_MODEL_PATH"] = original_model
        else:
            os.environ.pop("WHISPER_MODEL_PATH", None)
            
        if original_device:
            os.environ["WHISPER_DEVICE"] = original_device
        else:
            os.environ.pop("WHISPER_DEVICE", None)


@pytest.mark.integration
def test_real_cpu_fallback(skip_if_no_faster_whisper, temp_audio_file):
    """Test that CPU fallback works when GPU fails."""
    import os
    
    original_model = os.environ.get("WHISPER_MODEL_PATH")
    original_device = os.environ.get("WHISPER_DEVICE")
    original_fallback = os.environ.get("WHISPER_CPU_FALLBACK")
    
    try:
        os.environ["WHISPER_MODEL_PATH"] = "tiny.en"
        os.environ["WHISPER_DEVICE"] = "cuda:999"  # Invalid GPU
        os.environ["WHISPER_CPU_FALLBACK"] = "true"
        
        backend = WhisperBackend()
        
        # Should have fallen back to CPU
        assert backend.model.device == "cpu"
        
        # Should still work
        result = backend.execute(str(temp_audio_file))
        assert result is not None
        
        backend.unload()
        
    finally:
        if original_model:
            os.environ["WHISPER_MODEL_PATH"] = original_model
        else:
            os.environ.pop("WHISPER_MODEL_PATH", None)
            
        if original_device:
            os.environ["WHISPER_DEVICE"] = original_device
        else:
            os.environ.pop("WHISPER_DEVICE", None)
            
        if original_fallback:
            os.environ["WHISPER_CPU_FALLBACK"] = original_fallback
        else:
            os.environ.pop("WHISPER_CPU_FALLBACK", None)

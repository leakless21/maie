"""
Integration tests for WhisperBackend using real faster-whisper library.

These tests require:
- faster-whisper installed
- Local model at data/models/era-x-wow-turbo-v1.1-ct2 (or will download tiny.en)
- GPU required, no CPU fallback

Run with: pytest tests/integration/test_whisper_real.py -v
Skip integration tests: pytest -m "not integration"
"""

from pathlib import Path

import pytest

from src.processors.asr.whisper import WhisperBackend


@pytest.fixture(scope="module")
def shared_whisper_backend():
    """Shared WhisperBackend instance for all tests in this module."""
    import os

    from tests.conftest import has_faster_whisper

    # Check if faster-whisper is available
    if not has_faster_whisper():
        pytest.skip("faster-whisper not installed")

    # Check if model exists
    model_path = Path("data/models/era-x-wow-turbo-v1.1-ct2")
    if not model_path.exists():
        pytest.skip("Whisper model not found at data/models/era-x-wow-turbo-v1.1-ct2")

    # Force language and CPU settings to avoid detection segfault
    original_lang = os.environ.get("WHISPER_LANGUAGE")
    original_device = os.environ.get("WHISPER_DEVICE")
    original_compute = os.environ.get("WHISPER_COMPUTE_TYPE")

    os.environ["WHISPER_LANGUAGE"] = "vi"
    os.environ["WHISPER_DEVICE"] = "cpu"
    os.environ["WHISPER_COMPUTE_TYPE"] = "int8"

    backend = WhisperBackend(model_path=str(model_path))
    yield backend

    backend.unload()

    # Restore
    if original_lang is not None:
        os.environ["WHISPER_LANGUAGE"] = original_lang
    else:
        os.environ.pop("WHISPER_LANGUAGE", None)

    if original_device is not None:
        os.environ["WHISPER_DEVICE"] = original_device
    else:
        os.environ.pop("WHISPER_DEVICE", None)

    if original_compute is not None:
        os.environ["WHISPER_COMPUTE_TYPE"] = original_compute
    else:
        os.environ.pop("WHISPER_COMPUTE_TYPE", None)


@pytest.mark.integration
def test_real_whisper_model_loading(
    skip_if_no_faster_whisper, skip_if_no_whisper_model, whisper_model_path
):
    """Test loading a real Whisper model."""
    backend = WhisperBackend(model_path=str(whisper_model_path))

    # Model loads automatically in __init__
    assert backend.model is not None

    # Cleanup
    backend.unload()
    assert backend.model is None


@pytest.mark.integration
@pytest.mark.slow
def test_real_transcription_with_local_model(shared_whisper_backend, sample_audio_path):
    """Test transcription with local model using real audio."""
    # Transcribe real audio with explicit language to avoid detection segfault
    result = shared_whisper_backend.execute(str(sample_audio_path), language="vi")

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


@pytest.mark.integration
def test_real_version_info(shared_whisper_backend):
    """Test that version info returns actual faster-whisper metadata."""
    version_info = shared_whisper_backend.get_version_info()

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


@pytest.mark.integration
def test_real_vad_filtering(shared_whisper_backend, temp_audio_file):
    """Test that VAD filtering works with real faster-whisper."""
    import os

    # Test with local model
    original_vad = os.environ.get("WHISPER_VAD_FILTER")

    try:
        os.environ["WHISPER_VAD_FILTER"] = "true"

        # Transcribe with VAD and explicit language to avoid detection segfault
        result = shared_whisper_backend.execute(str(temp_audio_file), language="vi")

        assert result is not None
        assert "segments" in result

    finally:
        # Restore
        if original_vad:
            os.environ["WHISPER_VAD_FILTER"] = original_vad
        else:
            os.environ.pop("WHISPER_VAD_FILTER", None)


@pytest.mark.integration
def test_real_condition_on_previous_text(shared_whisper_backend, temp_audio_file):
    """Test that condition_on_previous_text parameter is passed correctly."""
    import os

    original_condition = os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT")

    try:
        os.environ["WHISPER_CONDITION_ON_PREVIOUS_TEXT"] = "false"

        # Spy on the transcribe call to verify parameter
        original_transcribe = shared_whisper_backend.model.transcribe
        call_kwargs = {}

        def spy_transcribe(*args, **kwargs):
            call_kwargs.update(kwargs)
            return original_transcribe(*args, **kwargs)

        shared_whisper_backend.model.transcribe = spy_transcribe

        # Execute transcription with explicit language to avoid detection segfault
        shared_whisper_backend.execute(str(temp_audio_file), language="vi")

        # Verify the parameter was passed
        assert "condition_on_previous_text" in call_kwargs
        assert call_kwargs["condition_on_previous_text"] is False

    finally:
        if original_condition:
            os.environ["WHISPER_CONDITION_ON_PREVIOUS_TEXT"] = original_condition
        else:
            os.environ.pop("WHISPER_CONDITION_ON_PREVIOUS_TEXT", None)


@pytest.mark.integration
def test_real_language_parameter(
    skip_if_no_faster_whisper,
    skip_if_no_whisper_model,
    whisper_model_path,
    temp_audio_file,
):
    """Test that language parameter is passed correctly to faster-whisper."""
    import os

    original_language = os.environ.get("WHISPER_LANGUAGE")

    try:
        # Clear language to test parameter passing
        os.environ.pop("WHISPER_LANGUAGE", None)

        backend = WhisperBackend(model_path=str(whisper_model_path))

        # Spy on transcribe call
        original_transcribe = backend.model.transcribe
        call_kwargs = {}

        def spy_transcribe(*args, **kwargs):
            call_kwargs.update(kwargs)
            return original_transcribe(*args, **kwargs)

        backend.model.transcribe = spy_transcribe

        # Execute with explicit language parameter
        backend.execute(str(temp_audio_file), language="vi")

        # Verify language was passed
        assert "language" in call_kwargs
        assert call_kwargs["language"] == "vi"

        backend.unload()

    finally:
        if original_language:
            os.environ["WHISPER_LANGUAGE"] = original_language
        else:
            os.environ.pop("WHISPER_LANGUAGE", None)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skip(
    reason="GPU test requires CUDA setup and may cause segfaults. Use manual testing for GPU verification."
)
def test_real_gpu_execution(
    skip_if_no_faster_whisper,
    skip_if_no_gpu,
    skip_if_no_whisper_model,
    whisper_model_path,
    temp_audio_file,
):
    """Test that GPU execution works correctly."""
    import os

    original_device = os.environ.get("WHISPER_DEVICE")
    original_compute = os.environ.get("WHISPER_COMPUTE_TYPE")

    try:
        os.environ["WHISPER_DEVICE"] = "cuda"
        os.environ["WHISPER_COMPUTE_TYPE"] = "float16"
        os.environ["WHISPER_LANGUAGE"] = "vi"

        backend = WhisperBackend(model_path=str(whisper_model_path))

        # Verify model loaded on GPU
        assert backend.model.device == "cuda"

        # Execute transcription with explicit language to avoid detection segfault
        result = backend.execute(str(temp_audio_file), language="vi")

        assert result is not None

        backend.unload()

    finally:
        if original_device:
            os.environ["WHISPER_DEVICE"] = original_device
        else:
            os.environ.pop("WHISPER_DEVICE", None)

        if original_compute:
            os.environ["WHISPER_COMPUTE_TYPE"] = original_compute
        else:
            os.environ.pop("WHISPER_COMPUTE_TYPE", None)


@pytest.mark.integration
def test_gpu_required_no_fallback(
    skip_if_no_faster_whisper,
    skip_if_no_whisper_model,
    whisper_model_path,
    temp_audio_file,
):
    """Test that GPU is required and no CPU fallback occurs."""
    import os

    original_device = os.environ.get("WHISPER_DEVICE")
    original_fallback = os.environ.get("WHISPER_CPU_FALLBACK")

    try:
        os.environ["WHISPER_DEVICE"] = "cuda:999"  # Invalid GPU
        os.environ["WHISPER_CPU_FALLBACK"] = "false"  # Disabled

        # Should raise RuntimeError due to GPU requirement
        with pytest.raises(RuntimeError, match="Failed to load whisper model"):
            WhisperBackend(model_path=str(whisper_model_path))

    finally:
        if original_device:
            os.environ["WHISPER_DEVICE"] = original_device
        else:
            os.environ.pop("WHISPER_DEVICE", None)

        if original_fallback:
            os.environ["WHISPER_CPU_FALLBACK"] = original_fallback
        else:
            os.environ.pop("WHISPER_CPU_FALLBACK", None)

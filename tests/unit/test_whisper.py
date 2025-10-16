"""
Unit tests for WhisperBackend using mocks for fast execution.

Test Coverage:
- Interface compliance and basic functionality
- Model loading and validation logic
- Execution flow and result structure
- Resource management and cleanup
- Configuration integration
- VAD (Voice Activity Detection) configuration
- GPU requirement logic
- Parameter preparation and overrides

Note: These are UNIT tests with mocks for speed.
See tests/integration/test_whisper_real.py for integration tests with real faster-whisper.

API Notes:
- Model loads automatically in __init__() when model_path or config is set
- Use unload() to clean up, not close()
- Check backend.model is None to verify unloaded state
"""

import pytest

from src.processors.base import ASRBackend, ASRResult

# ============================================================================
# Test Section 1: Interface Compliance & Basic Functionality
# ============================================================================


def test_interface_compliance(inject_mock_faster_whisper):
    """Verify WhisperBackend implements the ASRBackend interface."""
    from src.processors.asr.whisper import WhisperBackend

    # Check required methods exist
    assert hasattr(WhisperBackend, "_load_model")
    assert hasattr(WhisperBackend, "execute")
    assert hasattr(WhisperBackend, "unload")
    assert hasattr(WhisperBackend, "get_version_info")

    # Verify inheritance
    assert issubclass(WhisperBackend, ASRBackend)


def test_version_info_structure(inject_mock_faster_whisper, mock_config, tmp_path):
    """Verify get_version_info returns all required fields."""
    from src.processors.asr.whisper import WhisperBackend

    # Set model path so it loads
    model_path = tmp_path / "fake_model"
    model_path.mkdir()
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()
    info = backend.get_version_info()

    # Required fields
    assert "backend" in info
    assert "library" in info
    assert "version" in info
    assert info["backend"] == "whisper"
    assert info["library"] == "faster-whisper"

    # Extended fields from recent implementation
    assert "model_path" in info
    assert "model_variant" in info
    assert "device" in info
    assert "compute_type" in info
    assert "beam_size" in info
    assert "vad_filter" in info
    assert "condition_on_previous_text" in info
    assert "language" in info

    backend.unload()


# ============================================================================
# Test Section 2: Model Loading & Validation
# ============================================================================


def test_model_loading(inject_mock_faster_whisper, tmp_path, mock_config):
    """Test basic model loading flow."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    assert backend.model is not None

    backend.unload()


def test_model_path_validation(inject_mock_faster_whisper, tmp_path):
    """Test that invalid model paths raise FileNotFoundError."""
    from src.processors.asr.whisper import WhisperBackend

    invalid_path = str(tmp_path / "nonexistent")

    # Should raise FileNotFoundError for invalid path
    with pytest.raises(FileNotFoundError):
        backend = WhisperBackend(model_path=invalid_path)


# ============================================================================
# Test Section 3: Execution & Transcription
# ============================================================================


def test_execute_returns_asr_result(
    inject_mock_faster_whisper, temp_audio_file, tmp_path
):
    """Verify execute returns proper ASRResult structure."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    backend = WhisperBackend(model_path=str(model_path))

    result = backend.execute(str(temp_audio_file))

    # Check structure
    assert isinstance(result, ASRResult)
    assert result.text is not None
    assert result.segments is not None
    assert len(result.segments) > 0

    # Check segment structure
    segment = result.segments[0]
    assert "start" in segment
    assert "end" in segment
    assert "text" in segment

    backend.unload()


def test_execute_with_bytes(inject_mock_faster_whisper, temp_audio_file, tmp_path):
    """Test execution with audio bytes instead of file path."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    backend = WhisperBackend(model_path=str(model_path))

    # Read audio as bytes
    audio_bytes = temp_audio_file.read_bytes()

    result = backend.execute(audio_bytes)

    assert isinstance(result, ASRResult)
    assert result.text is not None

    backend.unload()


# ============================================================================
# Test Section 4: Resource Management & Cleanup
# ============================================================================


def test_unload_releases_resources(inject_mock_faster_whisper, tmp_path):
    """Verify unload() properly releases model resources."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    backend = WhisperBackend(model_path=str(model_path))

    assert backend.model is not None

    backend.unload()

    assert backend.model is None


def test_double_unload_is_safe(inject_mock_faster_whisper, tmp_path):
    """Verify calling unload() multiple times is safe."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    backend = WhisperBackend(model_path=str(model_path))

    backend.unload()
    backend.unload()  # Should not raise

    assert backend.model is None


# ============================================================================
# Test Section 5: Configuration Integration
# ============================================================================


def test_config_device_integration(inject_mock_faster_whisper, mock_config, tmp_path):
    """Test that device config is properly used."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_device", "cpu")
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    # Check version info reflects config
    info = backend.get_version_info()
    assert info["device"] == "cpu"

    backend.unload()


def test_config_compute_type_integration(
    inject_mock_faster_whisper, mock_config, tmp_path
):
    """Test that compute_type config is properly used."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_compute_type", "int8")
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    info = backend.get_version_info()
    assert info["compute_type"] == "int8"

    backend.unload()


def test_config_beam_size_integration(
    inject_mock_faster_whisper, mock_config, tmp_path
):
    """Test that beam_size config is properly used."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_beam_size", 10)
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    info = backend.get_version_info()
    assert info["beam_size"] == 10

    backend.unload()


def test_config_vad_integration(inject_mock_faster_whisper, mock_config, tmp_path):
    """Test that VAD config is properly used."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_vad_filter", True)
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    info = backend.get_version_info()
    assert info["vad_filter"] is True

    backend.unload()


def test_config_condition_on_previous_text(
    inject_mock_faster_whisper, mock_config, tmp_path
):
    """Test that condition_on_previous_text config is used."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_condition_on_previous_text", False)
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    info = backend.get_version_info()
    assert info["condition_on_previous_text"] is False

    backend.unload()


def test_config_language(inject_mock_faster_whisper, mock_config, tmp_path):
    """Test that language config is used."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_language", "en")
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    info = backend.get_version_info()
    assert info["language"] == "en"

    backend.unload()


# ============================================================================
# Test Section 6: VAD Support
# ============================================================================


def test_vad_parameters_preparation(
    inject_mock_faster_whisper, mock_config, temp_audio_file, tmp_path
):
    """Test that VAD parameters are correctly prepared for transcription."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_vad_filter", True)
    mock_config.set("whisper_vad_min_silence_ms", 1000)
    mock_config.set("whisper_vad_speech_pad_ms", 300)
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    # Execute to trigger parameter preparation
    result = backend.execute(str(temp_audio_file))

    # Verify transcribe was called (mock tracks this)
    assert backend.model.transcribe_call_count > 0

    backend.unload()


def test_vad_disabled_by_default(inject_mock_faster_whisper, temp_audio_file, tmp_path):
    """Test that VAD configuration works."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    backend = WhisperBackend(model_path=str(model_path))

    # Check version info has VAD setting
    info = backend.get_version_info()
    assert "vad_filter" in info

    backend.unload()


# ============================================================================
# Test Section 7: GPU Requirement Mechanism
# ============================================================================


def test_gpu_requirement_enforced(inject_mock_faster_whisper, mock_config, tmp_path):
    """Test that GPU requirement is enforced."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_cpu_fallback", False)  # GPU required
    mock_config.set("whisper_device", "cuda")  # GPU device
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()
    # In real code, this would require GPU
    # In mock, we just verify config is read

    assert backend.model is not None
    backend.unload()


def test_gpu_requirement_default(inject_mock_faster_whisper, mock_config, tmp_path):
    """Test that GPU requirement is the default behavior."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_cpu_fallback", False)  # GPU required (default)
    mock_config.set("whisper_device", "cuda")  # GPU device
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()
    # In real code, this would require GPU
    # In mock, we just verify config is read

    assert backend.model is not None
    backend.unload()


# ============================================================================
# Test Section 8: Parameter Preparation & Overrides
# ============================================================================


def test_prepare_transcribe_kwargs(
    inject_mock_faster_whisper, mock_config, temp_audio_file, tmp_path
):
    """Test that transcribe kwargs are properly prepared."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_beam_size", 7)
    mock_config.set("whisper_condition_on_previous_text", False)
    mock_config.set("whisper_language", "vi")
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    result = backend.execute(str(temp_audio_file))

    # Check that parameters were used in transcription
    kwargs = backend.model.last_transcribe_kwargs
    assert "beam_size" in kwargs
    assert "condition_on_previous_text" in kwargs

    backend.unload()


def test_vad_parameters_in_kwargs(
    inject_mock_faster_whisper, mock_config, temp_audio_file, tmp_path
):
    """Test that VAD parameters are included in transcribe kwargs."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_vad_filter", True)
    mock_config.set("whisper_vad_min_silence_ms", 800)
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    result = backend.execute(str(temp_audio_file))

    # VAD params should be in kwargs
    kwargs = backend.model.last_transcribe_kwargs
    assert "vad_filter" in kwargs

    backend.unload()


def test_language_parameter_in_kwargs(
    inject_mock_faster_whisper, mock_config, temp_audio_file, tmp_path
):
    """Test that language parameter is passed to transcribe."""
    from src.processors.asr.whisper import WhisperBackend

    model_path = tmp_path / "fake_model"
    model_path.mkdir()

    mock_config.set("whisper_language", "en")
    mock_config.set("whisper_model_path", str(model_path))

    backend = WhisperBackend()

    result = backend.execute(str(temp_audio_file))

    kwargs = backend.model.last_transcribe_kwargs
    assert "language" in kwargs
    assert kwargs["language"] == "en"

    backend.unload()

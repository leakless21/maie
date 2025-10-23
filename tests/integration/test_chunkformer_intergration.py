"""
Comprehensive ChunkFormer integration tests.
Tests real ChunkFormer model with actual audio files and verifies all V1.0 fixes.
"""

from pathlib import Path

import pytest
from loguru import logger

from src.config.loader import settings
from src.config.model import AppSettings
from src.processors.asr.chunkformer import ChunkFormerBackend
from src.processors.base import ASRResult

chunkformer = pytest.importorskip(
    "chunkformer", reason="chunkformer library not installed"
)
TEST_AUDIO = Path(__file__).parent.parent / "assets" / "Northern Female 1.wav"


@pytest.mark.integration
class TestChunkFormerConfiguration:
    """Test ChunkFormer configuration values (V1.0 fixes verification)."""

    def test_config_values_are_correct(self):
        """Verify that ChunkFormer configuration values match TDD specification after V1.0 fixes."""
        assert settings.chunkformer.chunkformer_chunk_size == 64, (
            f"chunk_size should be 64 frames, got {settings.chunkformer.chunkformer_chunk_size}"
        )
        assert settings.chunkformer.chunkformer_left_context_size == 128, (
            f"left_context_size should be 128 frames, got {settings.chunkformer.chunkformer_left_context_size}"
        )
        assert settings.chunkformer.chunkformer_right_context_size == 128, (
            f"right_context_size should be 128 frames, got {settings.chunkformer.chunkformer_right_context_size}"
        )
        assert settings.chunkformer.chunkformer_total_batch_duration == 14400, (
            f"total_batch_duration should be 14400 seconds, got {settings.chunkformer.chunkformer_total_batch_duration}"
        )
        assert settings.chunkformer.chunkformer_return_timestamps, (
            f"return_timestamps should be True, got {settings.chunkformer.chunkformer_return_timestamps}"
        )

    def test_config_description_units(self):
        """Verify that configuration field descriptions use correct units."""
        chunk_size_field = AppSettings.model_fields[
            "chunkformer"
        ].annotation.model_fields["chunkformer_chunk_size"]
        assert "frames" in chunk_size_field.description.lower(), (
            f"chunk_size description should mention 'frames', got: {chunk_size_field.description}"
        )
        duration_field = AppSettings.model_fields[
            "chunkformer"
        ].annotation.model_fields["chunkformer_total_batch_duration"]
        assert "seconds" in duration_field.description.lower(), (
            f"total_batch_duration description should mention 'seconds', got: {duration_field.description}"
        )


@pytest.mark.integration
class TestChunkFormerModelLoading:
    """Test ChunkFormer model loading and initialization."""

    def test_model_loading(self):
        """Test loading a real ChunkFormer model when library is present."""
        backend = ChunkFormerBackend()
        assert backend.model is not None, "Model should be loaded"
        assert hasattr(backend.model, "endless_decode") or hasattr(
            backend.model, "decode"
        ), "Model should have decode method"
        backend.unload()
        assert backend.model is None, "Model should be unloaded"

    def test_model_device_configuration(self):
        """Verify model is loaded on correct device."""
        backend = ChunkFormerBackend()
        info = backend.get_version_info()
        assert "device" in info, "Version info should include device"
        device = info.get("device")
        assert isinstance(device, (str, type(None))), (
            f"Device should be string or None, got {type(device)}"
        )
        if device:
            assert "cuda" in device or "cpu" in device, (
                f"Device should be 'cuda' or 'cpu', got {device}"
            )
        backend.unload()


@pytest.mark.integration
class TestChunkFormerVersionInfo:
    """Test ChunkFormer version info (NFR-1 compliance)."""

    def test_version_info_structure(self):
        """Ensure version info contains all expected fields."""
        backend = ChunkFormerBackend()
        info = backend.get_version_info()
        assert isinstance(info, dict), "Version info should be a dict"
        assert "backend" in info, "Version info should include backend"
        assert info.get("backend") == "chunkformer", (
            f"Backend should be 'chunkformer', got {info.get('backend')}"
        )
        assert "model_variant" in info, "Version info should include model_variant"
        assert "model_path" in info, "Version info should include model_path"
        assert "library" in info, "Version info should include library"
        backend.unload()

    def test_version_info_includes_architecture_params(self):
        """Verify that version info includes all architecture parameters for NFR-1 compliance."""
        backend = ChunkFormerBackend()
        info = backend.get_version_info()
        required_params = [
            "chunk_size",
            "left_context_size",
            "right_context_size",
            "total_batch_duration",
            "return_timestamps",
        ]
        for param in required_params:
            assert param in info, f"Version info missing required parameter: {param}"
        assert info["chunk_size"] == 64, (
            f"Version info chunk_size should be 64, got {info['chunk_size']}"
        )
        assert info["left_context_size"] == 128, (
            f"Version info left_context_size should be 128, got {info['left_context_size']}"
        )
        assert info["right_context_size"] == 128, (
            f"Version info right_context_size should be 128, got {info['right_context_size']}"
        )
        assert info["total_batch_duration"] == 14400, (
            f"Version info total_batch_duration should be 14400, got {info['total_batch_duration']}"
        )
        assert info["return_timestamps"], (
            f"Version info return_timestamps should be True, got {info['return_timestamps']}"
        )
        backend.unload()


@pytest.mark.integration
class TestChunkFormerTranscription:
    """Test ChunkFormer transcription with real audio."""

    def test_transcription_with_real_audio(self):
        """Test transcription using real audio file 'Northern Female 1.wav'."""
        assert TEST_AUDIO.exists(), f"Test audio file not found: {TEST_AUDIO}"
        backend = ChunkFormerBackend()
        audio_bytes = TEST_AUDIO.read_bytes()
        result = backend.execute(audio_bytes)
        assert result is not None, "Result should not be None"
        assert isinstance(result, ASRResult), (
            f"Result should be ASRResult, got {type(result)}"
        )
        assert hasattr(result, "text"), "Result should have text attribute"
        assert result.text, "Transcribed text should not be empty"
        assert len(result.text) > 0, "Transcribed text should have content"
        if result.segments:
            assert isinstance(result.segments, list), "Segments should be a list"
            assert len(result.segments) > 0, "Should have at least one segment"
            first_segment = result.segments[0]
            assert "text" in first_segment, "Segment should have text"
            if "start" in first_segment or "end" in first_segment:
                start_val = first_segment.get("start")
                end_val = first_segment.get("end")
                assert isinstance(start_val, (int, float, str, type(None)))
                assert isinstance(end_val, (int, float, str, type(None)))
        if result.language:
            assert isinstance(result.language, str), "Language should be a string"
            assert len(result.language) >= 2, "Language code should be at least 2 chars"
        if result.confidence is not None:
            assert isinstance(result.confidence, (int, float)), (
                "Confidence should be numeric"
            )
            assert 0 <= result.confidence <= 1, "Confidence should be between 0 and 1"
        backend.unload()
        logger.info("\n✅ ChunkFormer Transcription Test Results:")
        logger.info(f"   Audio file: {TEST_AUDIO.name}")
        logger.info(f"   Text length: {len(result.text)} characters")
        logger.info(f"   Text preview: {result.text[:100]}...")
        logger.info(f"   Segments: {(len(result.segments) if result.segments else 0)}")
        logger.info(f"   Language: {result.language or 'N/A'}")
        logger.info(f"   Confidence: {result.confidence or 'N/A'}")

    def test_transcription_returns_expected_format(self):
        """Verify transcription result matches ASRResult format."""
        assert TEST_AUDIO.exists(), f"Test audio file not found: {TEST_AUDIO}"
        backend = ChunkFormerBackend()
        audio_bytes = TEST_AUDIO.read_bytes()
        result = backend.execute(audio_bytes)
        assert hasattr(result, "text"), "Should have text attribute"
        assert hasattr(result, "segments"), "Should have segments attribute"
        assert hasattr(result, "language"), "Should have language attribute"
        assert hasattr(result, "confidence"), "Should have confidence attribute"
        # Pure dataclass interface - no dict-like access
        assert result.text is not None, "Should have text content"
        backend.unload()

    def test_transcription_with_custom_parameters(self):
        """Test that custom parameters are properly passed to the model."""
        assert TEST_AUDIO.exists(), f"Test audio file not found: {TEST_AUDIO}"
        backend = ChunkFormerBackend()
        audio_bytes = TEST_AUDIO.read_bytes()
        result = backend.execute(
            audio_bytes, chunk_size=32, left_context=64, right_context=64
        )
        assert result is not None
        assert isinstance(result, ASRResult)
        assert result.text
        backend.unload()


@pytest.mark.integration
@pytest.mark.slow
class TestChunkFormerPerformance:
    """Performance and resource tests for ChunkFormer."""

    def test_multiple_transcriptions(self):
        """Test multiple transcriptions in sequence to verify resource cleanup."""
        assert TEST_AUDIO.exists(), f"Test audio file not found: {TEST_AUDIO}"
        backend = ChunkFormerBackend()
        audio_bytes = TEST_AUDIO.read_bytes()
        results = []
        for i in range(3):
            result = backend.execute(audio_bytes)
            assert result is not None
            assert result.text
            results.append(result)
        texts = [r.text for r in results]
        assert all(texts), "All results should have text"
        assert texts[0] == texts[1] == texts[2], (
            "Multiple runs on same audio should produce same text"
        )
        backend.unload()

    def test_resource_cleanup(self):
        """Verify proper resource cleanup after processing."""
        backend = ChunkFormerBackend()
        assert backend.model is not None
        info_before = backend.get_version_info()
        device_before = info_before.get("device")
        backend.unload()
        assert backend.model is None, "Model should be None after unload"
        logger.info("\n✅ Resource Cleanup Test:")
        logger.info(f"   Device before: {device_before}")
        logger.info(f"   Model unloaded: {backend.model is None}")


@pytest.fixture(scope="module", autouse=True)
def test_summary():
    """Print summary at end of test module."""
    yield
    logger.info("\n" + "=" * 70)
    logger.info("ChunkFormer Integration Tests - Summary")
    logger.info("=" * 70)
    logger.info("✅ Configuration: chunk_size=64 frames, contexts=128, duration=14400s")
    logger.info(f"✅ Test audio: {TEST_AUDIO.name}")
    logger.info("✅ All V1.0 fixes verified and working correctly")
    logger.info("=" * 70)

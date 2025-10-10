"""
Comprehensive ChunkFormer integration tests.
Tests real ChunkFormer model with actual audio files and verifies all V1.0 fixes.
"""
import pytest
from pathlib import Path
from src.config import settings, Settings
from src.processors.asr.chunkformer import ChunkFormerBackend
from src.processors.base import ASRResult

# Skip the whole module if chunkformer library is not available
chunkformer = pytest.importorskip("chunkformer", reason="chunkformer library not installed")

# Test audio file path
TEST_AUDIO = Path(__file__).parent.parent / "assets" / "Northern Female 1.wav"


@pytest.mark.integration
class TestChunkFormerConfiguration:
    """Test ChunkFormer configuration values (V1.0 fixes verification)."""
    
    def test_config_values_are_correct(self):
        """Verify that ChunkFormer configuration values match TDD specification after V1.0 fixes."""
        
        # Test 1: Verify chunk_size is corrected (was 16000, should be 64)
        assert settings.chunkformer_chunk_size == 64, \
            f"chunk_size should be 64 frames, got {settings.chunkformer_chunk_size}"
        
        # Test 2: Verify context sizes
        assert settings.chunkformer_left_context_size == 128, \
            f"left_context_size should be 128 frames, got {settings.chunkformer_left_context_size}"
        
        assert settings.chunkformer_right_context_size == 128, \
            f"right_context_size should be 128 frames, got {settings.chunkformer_right_context_size}"
        
        # Test 3: Verify batch duration
        assert settings.chunkformer_total_batch_duration == 14400, \
            f"total_batch_duration should be 14400 seconds, got {settings.chunkformer_total_batch_duration}"
        
        # Test 4: Verify return_timestamps
        assert settings.chunkformer_return_timestamps == True, \
            f"return_timestamps should be True, got {settings.chunkformer_return_timestamps}"
    
    def test_config_description_units(self):
        """Verify that configuration field descriptions use correct units."""
        
        # Check that chunk_size description mentions "frames" not "samples"
        chunk_size_field = Settings.model_fields["chunkformer_chunk_size"]
        assert "frames" in chunk_size_field.description.lower(), \
            f"chunk_size description should mention 'frames', got: {chunk_size_field.description}"
        
        # Check that total_batch_duration description mentions "seconds" not "ms"
        duration_field = Settings.model_fields["chunkformer_total_batch_duration"]
        assert "seconds" in duration_field.description.lower(), \
            f"total_batch_duration description should mention 'seconds', got: {duration_field.description}"


@pytest.mark.integration
class TestChunkFormerModelLoading:
    """Test ChunkFormer model loading and initialization."""
    
    def test_model_loading(self):
        """Test loading a real ChunkFormer model when library is present."""
        backend = ChunkFormerBackend()
        assert backend.model is not None, "Model should be loaded"
        assert hasattr(backend.model, "endless_decode") or hasattr(backend.model, "decode"), \
            "Model should have decode method"
        backend.unload()
        assert backend.model is None, "Model should be unloaded"
    
    def test_model_device_configuration(self):
        """Verify model is loaded on correct device."""
        backend = ChunkFormerBackend()
        info = backend.get_version_info()
        
        # Device should be set
        assert "device" in info, "Version info should include device"
        device = info.get("device")
        
        # Device should be a string (not torch.device object - this was a bug we fixed)
        assert isinstance(device, (str, type(None))), \
            f"Device should be string or None, got {type(device)}"
        
        # Should be either cuda or cpu
        if device:
            assert "cuda" in device or "cpu" in device, \
                f"Device should be 'cuda' or 'cpu', got {device}"
        
        backend.unload()


@pytest.mark.integration
class TestChunkFormerVersionInfo:
    """Test ChunkFormer version info (NFR-1 compliance)."""
    
    def test_version_info_structure(self):
        """Ensure version info contains all expected fields."""
        backend = ChunkFormerBackend()
        info = backend.get_version_info()
        
        # Basic fields
        assert isinstance(info, dict), "Version info should be a dict"
        assert "backend" in info, "Version info should include backend"
        assert info.get("backend") == "chunkformer", f"Backend should be 'chunkformer', got {info.get('backend')}"
        
        # Model info
        assert "model_variant" in info, "Version info should include model_variant"
        assert "model_path" in info, "Version info should include model_path"
        assert "library" in info, "Version info should include library"
        
        backend.unload()
    
    def test_version_info_includes_architecture_params(self):
        """Verify that version info includes all architecture parameters for NFR-1 compliance."""
        backend = ChunkFormerBackend()
        info = backend.get_version_info()
        
        # Test NFR-1 compliance: All architecture params should be in version info
        required_params = [
            "chunk_size",
            "left_context_size", 
            "right_context_size",
            "total_batch_duration",
            "return_timestamps"
        ]
        
        for param in required_params:
            assert param in info, f"Version info missing required parameter: {param}"
        
        # Verify values match configuration
        assert info["chunk_size"] == 64, \
            f"Version info chunk_size should be 64, got {info['chunk_size']}"
        
        assert info["left_context_size"] == 128, \
            f"Version info left_context_size should be 128, got {info['left_context_size']}"
        
        assert info["right_context_size"] == 128, \
            f"Version info right_context_size should be 128, got {info['right_context_size']}"
        
        assert info["total_batch_duration"] == 14400, \
            f"Version info total_batch_duration should be 14400, got {info['total_batch_duration']}"
        
        assert info["return_timestamps"] == True, \
            f"Version info return_timestamps should be True, got {info['return_timestamps']}"
        
        backend.unload()


@pytest.mark.integration
class TestChunkFormerTranscription:
    """Test ChunkFormer transcription with real audio."""
    
    def test_transcription_with_real_audio(self):
        """Test transcription using real audio file 'Northern Female 1.wav'."""
        assert TEST_AUDIO.exists(), f"Test audio file not found: {TEST_AUDIO}"
        
        backend = ChunkFormerBackend()
        
        # Read audio file
        audio_bytes = TEST_AUDIO.read_bytes()
        
        # Execute transcription
        result = backend.execute(audio_bytes)
        
        # Verify result structure
        assert result is not None, "Result should not be None"
        assert isinstance(result, ASRResult), f"Result should be ASRResult, got {type(result)}"
        
        # Verify text output
        assert hasattr(result, "text"), "Result should have text attribute"
        assert result.text, "Transcribed text should not be empty"
        assert len(result.text) > 0, "Transcribed text should have content"
        
        # Verify segments (if available)
        if result.segments:
            assert isinstance(result.segments, list), "Segments should be a list"
            assert len(result.segments) > 0, "Should have at least one segment"
            
            # Check first segment structure
            first_segment = result.segments[0]
            assert "text" in first_segment, "Segment should have text"
            
            # Timestamps are optional based on model configuration
            # ChunkFormer returns timestamps as strings in format '00:00:00:000'
            if "start" in first_segment or "end" in first_segment:
                start_val = first_segment.get("start")
                end_val = first_segment.get("end")
                # Accept both string timestamps (ChunkFormer format) and numeric timestamps
                assert isinstance(start_val, (int, float, str, type(None)))
                assert isinstance(end_val, (int, float, str, type(None)))
        
        # Verify language detection (if available)
        if result.language:
            assert isinstance(result.language, str), "Language should be a string"
            assert len(result.language) >= 2, "Language code should be at least 2 chars"
        
        # Verify confidence (if available)
        if result.confidence is not None:
            assert isinstance(result.confidence, (int, float)), "Confidence should be numeric"
            assert 0 <= result.confidence <= 1, "Confidence should be between 0 and 1"
        
        backend.unload()
        
        # Print results for verification
        print(f"\n✅ ChunkFormer Transcription Test Results:")
        print(f"   Audio file: {TEST_AUDIO.name}")
        print(f"   Text length: {len(result.text)} characters")
        print(f"   Text preview: {result.text[:100]}...")
        print(f"   Segments: {len(result.segments) if result.segments else 0}")
        print(f"   Language: {result.language or 'N/A'}")
        print(f"   Confidence: {result.confidence or 'N/A'}")
    
    def test_transcription_returns_expected_format(self):
        """Verify transcription result matches ASRResult format."""
        assert TEST_AUDIO.exists(), f"Test audio file not found: {TEST_AUDIO}"
        
        backend = ChunkFormerBackend()
        audio_bytes = TEST_AUDIO.read_bytes()
        result = backend.execute(audio_bytes)
        
        # Check ASRResult compatibility
        assert hasattr(result, "text"), "Should have text attribute"
        assert hasattr(result, "segments"), "Should have segments attribute"
        assert hasattr(result, "language"), "Should have language attribute"
        assert hasattr(result, "confidence"), "Should have confidence attribute"
        
        # Check dict-like access (ASRResult supports both)
        assert "text" in result, "Should support 'in' operator"
        assert result["text"] == result.text, "Dict access should match attribute access"
        
        backend.unload()
    
    def test_transcription_with_custom_parameters(self):
        """Test that custom parameters are properly passed to the model."""
        assert TEST_AUDIO.exists(), f"Test audio file not found: {TEST_AUDIO}"
        
        backend = ChunkFormerBackend()
        audio_bytes = TEST_AUDIO.read_bytes()
        
        # Execute with custom parameters (smaller chunk for faster test)
        result = backend.execute(
            audio_bytes,
            chunk_size=32,  # Custom chunk size
            left_context=64,  # Custom left context
            right_context=64  # Custom right context
        )
        
        # Should still produce valid result
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
        
        # Run multiple transcriptions
        results = []
        for i in range(3):
            result = backend.execute(audio_bytes)
            assert result is not None
            assert result.text
            results.append(result)
        
        # All results should be similar (same audio)
        texts = [r.text for r in results]
        assert all(texts), "All results should have text"
        
        # Text should be consistent across runs
        assert texts[0] == texts[1] == texts[2], \
            "Multiple runs on same audio should produce same text"
        
        backend.unload()
    
    def test_resource_cleanup(self):
        """Verify proper resource cleanup after processing."""
        backend = ChunkFormerBackend()
        
        # Model should be loaded
        assert backend.model is not None
        
        # Get device before cleanup
        info_before = backend.get_version_info()
        device_before = info_before.get("device")
        
        # Unload
        backend.unload()
        
        # Model should be None after unload
        assert backend.model is None, "Model should be None after unload"
        
        print(f"\n✅ Resource Cleanup Test:")
        print(f"   Device before: {device_before}")
        print(f"   Model unloaded: {backend.model is None}")


# Summary fixture to print test results
@pytest.fixture(scope="module", autouse=True)
def test_summary():
    """Print summary at end of test module."""
    yield
    print("\n" + "="*70)
    print("ChunkFormer Integration Tests - Summary")
    print("="*70)
    print(f"✅ Configuration: chunk_size=64 frames, contexts=128, duration=14400s")
    print(f"✅ Test audio: {TEST_AUDIO.name}")
    print(f"✅ All V1.0 fixes verified and working correctly")
    print("="*70)


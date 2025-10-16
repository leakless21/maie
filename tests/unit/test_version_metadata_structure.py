# tests/unit/test_version_metadata_structure.py
"""
TDD-driven tests for version metadata collection.

Verifies per TDD NFR-1:
1. ASR model metadata (model_name, checkpoint_hash)
2. LLM version info (model_name, checkpoint_hash)
3. Pipeline version
4. All config parameters used in processing
"""
import pytest

from src.worker.pipeline import get_version_metadata


class TestVersionMetadataStructure:
    """Test that version metadata has all required fields."""

    def test_basic_structure_with_asr_only(self):
        """Test version metadata structure with ASR info only."""
        asr_metadata = {
            "model_name": "whisper-large-v3",
            "checkpoint_hash": "abc123def456",
            "backend": "faster-whisper",
        }

        version_metadata = get_version_metadata(asr_metadata, llm_model=None)

        # Verify required top-level keys
        assert "asr" in version_metadata
        assert "llm" in version_metadata
        assert "maie_worker" in version_metadata
        assert "processing_pipeline" in version_metadata

    def test_asr_metadata_preserved(self):
        """Test that ASR metadata is fully preserved."""
        asr_metadata = {
            "model_name": "whisper-large-v3",
            "checkpoint_hash": "abc123",
            "backend": "faster-whisper",
            "language": "vi",
            "device": "cuda",
        }

        version_metadata = get_version_metadata(asr_metadata, llm_model=None)

        # ASR metadata should be preserved exactly
        assert version_metadata["asr"] == asr_metadata
        assert version_metadata["asr"]["model_name"] == "whisper-large-v3"
        assert version_metadata["asr"]["checkpoint_hash"] == "abc123"

    def test_llm_metadata_when_no_model(self):
        """Test LLM metadata when model is not loaded."""
        asr_metadata = {"model_name": "whisper"}

        version_metadata = get_version_metadata(asr_metadata, llm_model=None)

        # LLM should indicate model not loaded
        assert version_metadata["llm"]["model_name"] == "not_loaded"
        assert "reason" in version_metadata["llm"]

    def test_pipeline_version_present(self):
        """Test that pipeline version is included."""
        asr_metadata = {"model_name": "whisper"}

        version_metadata = get_version_metadata(asr_metadata, llm_model=None)

        assert "maie_worker" in version_metadata
        assert "processing_pipeline" in version_metadata
        # Should be semantic version format
        assert isinstance(version_metadata["maie_worker"], str)
        assert isinstance(version_metadata["processing_pipeline"], str)


class TestLLMVersionCollection:
    """Test LLM version info collection."""

    def test_llm_version_from_model(self):
        """Test that LLM version is collected from model.get_version_info()."""

        class MockLLMModel:
            def get_version_info(self):
                return {
                    "model_name": "qwen2.5-3b-instruct",
                    "checkpoint_hash": "xyz789",
                    "backend": "vllm",
                }

        asr_metadata = {"model_name": "whisper"}
        llm_model = MockLLMModel()

        version_metadata = get_version_metadata(asr_metadata, llm_model)

        # LLM version should be collected
        assert version_metadata["llm"]["model_name"] == "qwen2.5-3b-instruct"
        assert version_metadata["llm"]["checkpoint_hash"] == "xyz789"
        assert version_metadata["llm"]["backend"] == "vllm"

    def test_llm_version_handles_exception(self):
        """Test that exceptions during LLM version collection are handled."""

        class BrokenLLMModel:
            def get_version_info(self):
                raise RuntimeError("Version info unavailable")

        asr_metadata = {"model_name": "whisper"}
        llm_model = BrokenLLMModel()

        version_metadata = get_version_metadata(asr_metadata, llm_model)

        # Should handle exception gracefully
        assert version_metadata["llm"]["model_name"] == "unavailable"
        assert "error" in version_metadata["llm"]
        assert "Version info unavailable" in version_metadata["llm"]["error"]

    def test_llm_version_when_method_missing(self):
        """Test LLM version when model doesn't have get_version_info."""

        class IncompleteLLMModel:
            pass  # No get_version_info method

        asr_metadata = {"model_name": "whisper"}
        llm_model = IncompleteLLMModel()

        version_metadata = get_version_metadata(asr_metadata, llm_model)

        # Should handle missing method gracefully
        assert version_metadata["llm"]["model_name"] == "not_loaded"
        assert "reason" in version_metadata["llm"]


class TestASRMetadataCollection:
    """Test ASR metadata collection from backend."""

    def test_whisper_metadata_structure(self):
        """Test that Whisper backend metadata has required fields."""
        asr_metadata = {
            "model_name": "whisper-large-v3",
            "checkpoint_hash": "checksum_abc123",
            "backend": "faster-whisper",
            "language": "vi",
            "compute_type": "float16",
            "device": "cuda",
        }

        version_metadata = get_version_metadata(asr_metadata, None)

        asr = version_metadata["asr"]
        # Required fields per TDD NFR-1
        assert "model_name" in asr
        assert "checkpoint_hash" in asr
        assert "backend" in asr
        # Optional config fields
        assert "language" in asr
        assert "device" in asr

    def test_chunkformer_metadata_structure(self):
        """Test that ChunkFormer backend metadata has required fields."""
        asr_metadata = {
            "model_name": "chunkformer-ctc-large-vie",
            "checkpoint_hash": "hash_xyz789",
            "backend": "chunkformer",
            "language": "vi",
            "device": "cuda",
        }

        version_metadata = get_version_metadata(asr_metadata, None)

        asr = version_metadata["asr"]
        assert asr["model_name"] == "chunkformer-ctc-large-vie"
        assert asr["checkpoint_hash"] == "hash_xyz789"
        assert asr["backend"] == "chunkformer"

    def test_asr_metadata_with_minimal_fields(self):
        """Test ASR metadata with only required fields."""
        asr_metadata = {
            "model_name": "whisper-base",
        }

        version_metadata = get_version_metadata(asr_metadata, None)

        # Should work with minimal metadata
        assert version_metadata["asr"]["model_name"] == "whisper-base"


class TestCompleteVersionMetadata:
    """Test complete version metadata with all components."""

    def test_complete_metadata_with_all_components(self):
        """Test version metadata with ASR + LLM + pipeline versions."""

        class MockLLMModel:
            def get_version_info(self):
                return {
                    "model_name": "qwen2.5-3b-instruct",
                    "checkpoint_hash": "llm_hash_123",
                    "backend": "vllm",
                }

        asr_metadata = {
            "model_name": "whisper-large-v3",
            "checkpoint_hash": "asr_hash_456",
            "backend": "faster-whisper",
            "language": "vi",
            "device": "cuda",
        }

        llm_model = MockLLMModel()

        version_metadata = get_version_metadata(asr_metadata, llm_model)

        # Verify all components present
        assert version_metadata["asr"]["model_name"] == "whisper-large-v3"
        assert version_metadata["asr"]["checkpoint_hash"] == "asr_hash_456"
        assert version_metadata["llm"]["model_name"] == "qwen2.5-3b-instruct"
        assert version_metadata["llm"]["checkpoint_hash"] == "llm_hash_123"
        assert version_metadata["maie_worker"] == "1.0.0"
        assert version_metadata["processing_pipeline"] == "1.0.0"

    def test_version_metadata_serializable(self):
        """Test that version metadata can be serialized to JSON."""
        import json

        class MockLLMModel:
            def get_version_info(self):
                return {
                    "model_name": "test-model",
                    "checkpoint_hash": "hash123",
                }

        asr_metadata = {
            "model_name": "whisper",
            "checkpoint_hash": "hash456",
        }

        version_metadata = get_version_metadata(asr_metadata, MockLLMModel())

        # Should be JSON serializable (for Redis storage)
        try:
            json_str = json.dumps(version_metadata)
            assert json_str is not None
            # Should be able to deserialize back
            restored = json.loads(json_str)
            assert restored["asr"]["model_name"] == "whisper"
            assert restored["llm"]["model_name"] == "test-model"
        except (TypeError, ValueError) as e:
            pytest.fail(f"Version metadata not JSON serializable: {e}")


class TestVersionMetadataEdgeCases:
    """Test edge cases in version metadata collection."""

    def test_empty_asr_metadata(self):
        """Test with empty ASR metadata dict."""
        version_metadata = get_version_metadata({}, None)

        # Should not crash
        assert version_metadata["asr"] == {}
        assert "llm" in version_metadata
        assert "maie_worker" in version_metadata

    def test_llm_version_with_none_return(self):
        """Test when LLM get_version_info returns None."""

        class WeirdLLMModel:
            def get_version_info(self):
                return None

        asr_metadata = {"model_name": "whisper"}
        llm_model = WeirdLLMModel()

        version_metadata = get_version_metadata(asr_metadata, llm_model)

        # Should handle None return
        assert version_metadata["llm"] is None

    def test_asr_metadata_with_extra_fields(self):
        """Test that extra ASR fields are preserved."""
        asr_metadata = {
            "model_name": "whisper",
            "checkpoint_hash": "abc",
            "extra_field_1": "value1",
            "extra_field_2": 42,
            "nested": {"key": "value"},
        }

        version_metadata = get_version_metadata(asr_metadata, None)

        # All fields should be preserved
        assert version_metadata["asr"]["extra_field_1"] == "value1"
        assert version_metadata["asr"]["extra_field_2"] == 42
        assert version_metadata["asr"]["nested"]["key"] == "value"

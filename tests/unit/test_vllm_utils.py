"""
Unit tests for vLLM utility functions.

Tests cover sampling parameter management, override handling, and model versioning.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.tooling.vllm_utils import (
    apply_overrides_to_sampling,
    normalize_overrides,
    calculate_checkpoint_hash,
    get_model_info,
)


class TestApplyOverridesToSampling:
    """Test sampling parameter override functionality."""

    def test_apply_overrides_with_dict_base(self):
        """Test applying overrides when base params is a dict."""
        base_params = {"temperature": 0.7, "max_tokens": 1000}
        overrides = {"temperature": 0.3, "top_p": 0.9}

        # Mock the import to avoid torch/vllm conflicts
        mock_sampling_class = Mock()
        mock_sampling_instance = Mock()
        mock_sampling_class.return_value = mock_sampling_instance

        with patch.dict(
            "sys.modules", {"vllm": Mock(SamplingParams=mock_sampling_class)}
        ):
            result = apply_overrides_to_sampling(base_params, overrides)

            # Verify SamplingParams was called
            mock_sampling_class.assert_called_once()
            # call_args is a tuple of (args, kwargs)
            call_kwargs = mock_sampling_class.call_args.kwargs
            # Verify merged parameters
            assert call_kwargs.get("temperature") == 0.3
            assert call_kwargs.get("max_tokens") == 1000
            assert call_kwargs.get("top_p") == 0.9

    def test_apply_overrides_with_sampling_params_object(self):
        """Test applying overrides when base params has to_dict method."""
        mock_base = Mock()
        mock_base.to_dict.return_value = {"temperature": 0.7, "max_tokens": 1000}

        overrides = {"temperature": 0.3, "top_p": 0.9}

        # Mock the import to avoid torch/vllm conflicts
        mock_sampling_class = Mock()
        mock_sampling_instance = Mock()
        mock_sampling_class.return_value = mock_sampling_instance

        with patch.dict(
            "sys.modules", {"vllm": Mock(SamplingParams=mock_sampling_class)}
        ):
            result = apply_overrides_to_sampling(mock_base, overrides)

            # Verify to_dict was called and SamplingParams was called with merged params
            mock_base.to_dict.assert_called_once()
            expected_params = {"temperature": 0.3, "max_tokens": 1000, "top_p": 0.9}
            mock_sampling_class.assert_called_once_with(**expected_params)

    def test_apply_overrides_with_attribute_extraction(self):
        """Test applying overrides when extracting attributes from base params."""
        mock_base = Mock()
        mock_base.temperature = 0.7
        mock_base.max_tokens = 1000
        mock_base.top_p = 0.8
        # Simulate missing to_dict method
        del mock_base.to_dict

        overrides = {"temperature": 0.3, "top_k": 20}

        # Mock the import to avoid torch/vllm conflicts
        mock_sampling_class = Mock()
        mock_sampling_instance = Mock()
        mock_sampling_class.return_value = mock_sampling_instance

        with patch.dict(
            "sys.modules", {"vllm": Mock(SamplingParams=mock_sampling_class)}
        ):
            result = apply_overrides_to_sampling(mock_base, overrides)

            # Verify SamplingParams was called
            mock_sampling_class.assert_called_once()
            call_kwargs = mock_sampling_class.call_args[1]
            assert call_kwargs["temperature"] == 0.3
            assert call_kwargs["max_tokens"] == 1000
            assert call_kwargs["top_p"] == 0.8
            assert call_kwargs["top_k"] == 20

    def test_apply_overrides_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        mock_base = Mock()
        mock_base.to_dict.side_effect = Exception("Test error")

        overrides = {"temperature": 0.3}

        with patch("src.tooling.vllm_utils.logger") as mock_logger:
            result = apply_overrides_to_sampling(mock_base, overrides)

            # Should return base params on error
            assert result == mock_base
            mock_logger.error.assert_called_once()


class TestNormalizeOverrides:
    """Test override normalization functionality."""

    def test_normalize_simple_overrides(self):
        """Test normalizing simple parameter overrides."""
        overrides = {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9, "top_k": 20}

        result = normalize_overrides(overrides)
        assert result == overrides

    def test_normalize_filters_none_values(self):
        """Test that None values are filtered out."""
        overrides = {
            "temperature": 0.7,
            "max_tokens": None,
            "top_p": 0.9,
            "custom": None,
        }

        result = normalize_overrides(overrides)
        expected = {"temperature": 0.7, "top_p": 0.9}
        assert result == expected

    def test_normalize_converts_lists_to_strings(self):
        """Test that list values are converted to strings."""
        overrides = {"stop": ["</s>", "<|endoftext|>"], "temperature": 0.7}

        result = normalize_overrides(overrides)
        expected = {"stop": ["</s>", "<|endoftext|>"], "temperature": 0.7}
        assert result == expected

    def test_normalize_converts_complex_types(self):
        """Test that complex types are converted appropriately."""
        overrides = {
            "temperature": 0.7,
            "custom_dict": {"key": "value"},
            "custom_set": {1, 2, 3},
        }

        result = normalize_overrides(overrides)

        assert result["temperature"] == 0.7
        assert result["custom_dict"] == {"key": "value"}  # Dicts kept as-is
        assert result["custom_set"] == [1, 2, 3]  # Sets converted to sorted lists

    def test_normalize_handles_json_serialization_error(self):
        """Test handling of JSON serialization errors."""

        # Create an object that can't be JSON serialized
        class Unserializable:
            def __str__(self):
                return "unserializable"

        overrides = {"temperature": 0.7, "custom": Unserializable()}

        result = normalize_overrides(overrides)

        assert result["temperature"] == 0.7
        assert result["custom"] == "unserializable"


class TestCalculateCheckpointHash:
    """Test model checkpoint hash calculation."""

    def test_calculate_hash_with_safetensors_file(self, tmp_path):
        """Test hash calculation with model.safetensors file."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create a mock safetensors file
        safetensors_file = model_dir / "model.safetensors"
        safetensors_file.write_bytes(b"mock model weights data")

        hash_value = calculate_checkpoint_hash(model_dir)

        # Should be a valid SHA-256 hash (64 hex characters)
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_calculate_hash_with_pytorch_model_file(self, tmp_path):
        """Test hash calculation with pytorch_model.bin file."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create a mock pytorch model file
        model_file = model_dir / "pytorch_model.bin"
        model_file.write_bytes(b"mock pytorch model weights")

        hash_value = calculate_checkpoint_hash(model_dir)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_calculate_hash_falls_back_to_directory_hash(self, tmp_path):
        """Test fallback to directory hash when no weight files found."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create some files but no standard weight files
        (model_dir / "config.json").write_text('{"model_type": "test"}')
        (model_dir / "tokenizer.json").write_text('{"vocab": {}}')

        hash_value = calculate_checkpoint_hash(model_dir)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_calculate_hash_raises_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent path."""
        non_existent_path = Path("/non/existent/path")

        with pytest.raises(FileNotFoundError):
            calculate_checkpoint_hash(non_existent_path)

    def test_calculate_hash_raises_value_error_for_file(self, tmp_path):
        """Test that ValueError is raised when path is a file, not directory."""
        file_path = tmp_path / "not_a_directory"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="Model path is not a directory"):
            calculate_checkpoint_hash(file_path)

    def test_calculate_hash_handles_large_files(self, tmp_path):
        """Test hash calculation with large files."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create a large file (simulate large model weights)
        large_file = model_dir / "model.safetensors"
        large_data = b"x" * (1024 * 1024)  # 1MB of data
        large_file.write_bytes(large_data)

        hash_value = calculate_checkpoint_hash(model_dir)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_calculate_hash_handles_file_read_error(self, tmp_path):
        """Test handling of file read errors."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create a file that will cause read error
        weight_file = model_dir / "model.safetensors"
        weight_file.write_bytes(b"test")

        with patch("builtins.open", side_effect=IOError("Read error")):
            with patch("src.tooling.vllm_utils.logger") as mock_logger:
                # Should fall back to directory hash
                hash_value = calculate_checkpoint_hash(model_dir)

                assert len(hash_value) == 64
                mock_logger.error.assert_called_once()


class TestGetModelInfo:
    """Test model information extraction."""

    def test_get_model_info_basic(self, tmp_path):
        """Test basic model info extraction."""
        model_dir = tmp_path / "qwen3-4b-awq"
        model_dir.mkdir()

        # Create a weight file
        (model_dir / "model.safetensors").write_bytes(b"mock weights")

        info = get_model_info(model_dir)

        assert info["model_name"] == "qwen3-4b-awq"
        assert info["model_path"] == str(model_dir)
        assert "checkpoint_hash" in info
        assert len(info["checkpoint_hash"]) == 64

    def test_get_model_info_with_config(self, tmp_path):
        """Test model info extraction with config.json."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create weight file
        (model_dir / "model.safetensors").write_bytes(b"mock weights")

        # Create config.json
        config = {
            "model_type": "qwen",
            "architectures": ["Qwen2ForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 24,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        info = get_model_info(model_dir)

        assert info["model_name"] == "test_model"
        assert info["model_type"] == "qwen"
        assert info["architectures"] == ["Qwen2ForCausalLM"]
        assert info["vocab_size"] == 32000
        assert info["hidden_size"] == 4096
        assert info["num_attention_heads"] == 32
        assert info["num_hidden_layers"] == 24

    def test_get_model_info_handles_config_error(self, tmp_path):
        """Test handling of config.json loading errors."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create weight file
        (model_dir / "model.safetensors").write_bytes(b"mock weights")

        # Create invalid config.json
        (model_dir / "config.json").write_text("invalid json")

        with patch("src.tooling.vllm_utils.logger") as mock_logger:
            info = get_model_info(model_dir)

            # Should still return basic info
            assert info["model_name"] == "test_model"
            assert "checkpoint_hash" in info

            # Should log warning about config error
            mock_logger.warning.assert_called_once()

    def test_get_model_info_missing_config(self, tmp_path):
        """Test model info extraction without config.json."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create weight file
        (model_dir / "model.safetensors").write_bytes(b"mock weights")

        info = get_model_info(model_dir)

        assert info["model_name"] == "test_model"
        assert info["model_path"] == str(model_dir)
        assert "checkpoint_hash" in info
        # Config-related fields should not be present
        assert "model_type" not in info
        assert "architectures" not in info

"""
Tests for LLM generation configuration management.

Tests the hierarchical configuration system that supports:
- vLLM SamplingParams conversion
- HuggingFace generation_config.json loading
- Priority chain: Runtime > Environment > Model > Library
"""

import json
import tempfile
from pathlib import Path

# Import the module we're testing (will be created)
from src.processors.llm.config import (
    GenerationConfig,
    build_generation_config,
    calculate_dynamic_max_tokens,
    get_library_defaults,
    load_model_generation_config,
)


class TestGenerationConfigDataclass:
    """Test GenerationConfig dataclass fields and initialization."""

    def test_all_fields_optional_with_none_defaults(self):
        """All fields should be optional and default to None."""
        config = GenerationConfig()

        assert config.temperature is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.max_tokens is None
        assert config.repetition_penalty is None
        assert config.presence_penalty is None
        assert config.frequency_penalty is None
        assert config.min_p is None
        assert config.stop is None
        assert config.seed is None

    def test_initialization_with_values(self):
        """Should accept values for all fields."""
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_tokens=1000,
            repetition_penalty=1.1,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            min_p=0.05,
            stop=["</s>", "\n\n"],
            seed=42,
        )

        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.top_k == 50
        assert config.max_tokens == 1000
        assert config.repetition_penalty == 1.1
        assert config.presence_penalty == 0.1
        assert config.frequency_penalty == 0.1
        assert config.min_p == 0.05
        assert config.stop == ["</s>", "\n\n"]
        assert config.seed == 42

    def test_partial_initialization(self):
        """Should allow partial initialization with some fields."""
        config = GenerationConfig(temperature=0.8, max_tokens=500)

        assert config.temperature == 0.8
        assert config.max_tokens == 500
        assert config.top_p is None
        assert config.top_k is None


class TestGenerationConfigMerging:
    """Test merge_with() logic."""

    def test_merge_self_values_take_priority(self):
        """Self values should take priority over other values."""
        config1 = GenerationConfig(temperature=0.7, top_p=0.9)
        config2 = GenerationConfig(temperature=0.5, top_p=0.95, max_tokens=1000)

        merged = config1.merge_with(config2)

        # Self values should win
        assert merged.temperature == 0.7
        assert merged.top_p == 0.9
        # Other values should be used when self is None
        assert merged.max_tokens == 1000

    def test_merge_none_propagation(self):
        """None values should propagate correctly."""
        config1 = GenerationConfig(temperature=None, top_p=0.9)
        config2 = GenerationConfig(temperature=0.5, top_p=None, max_tokens=1000)

        merged = config1.merge_with(config2)

        # None in self should use other value
        assert merged.temperature == 0.5
        # Non-None in self should win
        assert merged.top_p == 0.9
        # None in other should result in None
        assert merged.max_tokens == 1000

    def test_merge_immutable(self):
        """merge_with() should return new instance, not mutate original."""
        config1 = GenerationConfig(temperature=0.7)
        config2 = GenerationConfig(temperature=0.5)

        merged = config1.merge_with(config2)

        # Original should be unchanged
        assert config1.temperature == 0.7
        assert config2.temperature == 0.5
        # Merged should have self value
        assert merged.temperature == 0.7
        # Should be different objects
        assert merged is not config1
        assert merged is not config2

    def test_merge_empty_configs(self):
        """Merging empty configs should work."""
        config1 = GenerationConfig()
        config2 = GenerationConfig()

        merged = config1.merge_with(config2)

        # All fields should be None
        assert merged.temperature is None
        assert merged.top_p is None
        assert merged.max_tokens is None


class TestGenerationConfigSamplingParams:
    """Test to_sampling_params() conversion."""

    def test_to_sampling_params_filters_none_values(self):
        """Should only include non-None values in output."""
        config = GenerationConfig(
            temperature=0.7,
            top_p=None,  # Should be filtered out
            max_tokens=1000,
            seed=None,  # Should be filtered out
        )

        params = config.to_sampling_params()

        assert "temperature" in params
        assert params["temperature"] == 0.7
        assert "max_tokens" in params
        assert params["max_tokens"] == 1000
        assert "top_p" not in params
        assert "seed" not in params

    def test_to_sampling_params_all_none(self):
        """Should return empty dict when all values are None."""
        config = GenerationConfig()

        params = config.to_sampling_params()

        assert params == {}

    def test_to_sampling_params_all_values(self):
        """Should include all non-None values."""
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_tokens=1000,
            repetition_penalty=1.1,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            min_p=0.05,
            stop=["</s>"],
            seed=42,
        )

        params = config.to_sampling_params()

        expected = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 1000,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "min_p": 0.05,
            "stop": ["</s>"],
            "seed": 42,
        }

        assert params == expected

    def test_to_sampling_params_vllm_compatible(self):
        """Output should be compatible with vLLM SamplingParams."""
        config = GenerationConfig(temperature=0.7, max_tokens=1000)
        params = config.to_sampling_params()

        # Should be able to create SamplingParams from the dict
        # (We'll test this integration later)
        assert isinstance(params, dict)
        assert all(isinstance(k, str) for k in params.keys())


class TestGenerationConfigRepr:
    """Test __repr__() method."""

    def test_repr_shows_only_non_none_fields(self):
        """Should show only non-None fields in repr."""
        config = GenerationConfig(temperature=0.7, top_p=0.95)

        repr_str = repr(config)

        assert "temperature=0.7" in repr_str
        assert "top_p=0.95" in repr_str
        assert "max_tokens" not in repr_str  # Should not show None values

    def test_repr_empty_config(self):
        """Should handle empty config gracefully."""
        config = GenerationConfig()

        repr_str = repr(config)

        assert "GenerationConfig" in repr_str
        # Should not crash and should be readable


class TestLibraryDefaults:
    """Test get_library_defaults() function."""

    def test_library_defaults_returns_vllm_defaults(self):
        """Should return vLLM default values."""
        defaults = get_library_defaults()

        assert isinstance(defaults, GenerationConfig)
        assert defaults.temperature == 1.0
        assert defaults.top_p == 1.0
        assert defaults.top_k == -1
        assert defaults.max_tokens is None  # Changed from 512 to None
        assert defaults.repetition_penalty == 1.0
        assert defaults.presence_penalty == 0.0
        assert defaults.frequency_penalty == 0.0

    def test_library_defaults_immutable(self):
        """Should return new instance each time."""
        defaults1 = get_library_defaults()
        defaults2 = get_library_defaults()

        assert defaults1 is not defaults2
        assert defaults1.temperature == defaults2.temperature


class TestLoadModelConfig:
    """Test loading from generation_config.json."""

    def test_load_model_config_file_not_exists(self):
        """Should return empty config when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "nonexistent_model"

            config = load_model_generation_config(model_path)

            assert isinstance(config, GenerationConfig)
            assert config.temperature is None
            assert config.max_tokens is None

    def test_load_model_config_valid_json(self):
        """Should load and map valid generation_config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            # Create generation_config.json
            config_data = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 1000,  # Should map to max_tokens
                "repetition_penalty": 1.1,
            }

            config_file = model_path / "generation_config.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            config = load_model_generation_config(model_path)

            assert config.temperature == 0.7
            assert config.top_p == 0.95
            assert config.top_k == 50
            assert config.max_tokens == 1000  # Mapped from max_new_tokens
            assert config.repetition_penalty == 1.1

    def test_load_model_config_max_length_mapping(self):
        """Should map max_length to max_tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            config_data = {"max_length": 2000}  # Should map to max_tokens

            config_file = model_path / "generation_config.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            config = load_model_generation_config(model_path)

            assert config.max_tokens == 2000

    def test_load_model_config_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            # Create invalid JSON file
            config_file = model_path / "generation_config.json"
            with open(config_file, "w") as f:
                f.write("invalid json content")

            config = load_model_generation_config(model_path)

            # Should return empty config and not crash
            assert isinstance(config, GenerationConfig)
            assert config.temperature is None

    def test_load_model_config_unknown_fields_ignored(self):
        """Should ignore unknown fields in JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            config_data = {
                "temperature": 0.7,
                "unknown_field": "should_be_ignored",
                "another_unknown": 123,
            }

            config_file = model_path / "generation_config.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            config = load_model_generation_config(model_path)

            assert config.temperature == 0.7
            # Unknown fields should not cause errors


class TestBuildGenerationConfig:
    """Test full hierarchy with build_generation_config()."""

    def test_build_config_hierarchy_order(self):
        """Should respect hierarchy: runtime > env > model > library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            # Model config
            model_config_data = {"temperature": 0.8, "max_tokens": 2000}
            config_file = model_path / "generation_config.json"
            with open(config_file, "w") as f:
                json.dump(model_config_data, f)

            # Environment overrides
            env_config = GenerationConfig(
                temperature=0.6,  # Should override model
                top_p=0.9,  # Should override library default
            )

            # Runtime overrides
            runtime_config = GenerationConfig(
                temperature=0.5,  # Should override env
                max_tokens=1000,  # Should override model
            )

            final_config = build_generation_config(
                model_path=model_path,
                env_overrides=env_config,
                runtime_overrides=runtime_config,
            )

            # Runtime should win
            assert final_config.temperature == 0.5
            assert final_config.max_tokens == 1000
            # Env should win over model
            assert final_config.top_p == 0.9
            # Library defaults for unspecified fields
            assert final_config.repetition_penalty == 1.0

    def test_build_config_no_runtime_overrides(self):
        """Should work without runtime overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            # Model config
            model_config_data = {"temperature": 0.8}
            config_file = model_path / "generation_config.json"
            with open(config_file, "w") as f:
                json.dump(model_config_data, f)

            env_config = GenerationConfig(top_p=0.9)

            final_config = build_generation_config(
                model_path=model_path, env_overrides=env_config, runtime_overrides=None
            )

            # Model should win for temperature
            assert final_config.temperature == 0.8
            # Env should win for top_p
            assert final_config.top_p == 0.9
            # Library defaults for others
            assert final_config.repetition_penalty == 1.0

    def test_build_config_no_model_file(self):
        """Should work without model generation_config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()
            # No generation_config.json file

            env_config = GenerationConfig(temperature=0.7)

            final_config = build_generation_config(
                model_path=model_path, env_overrides=env_config, runtime_overrides=None
            )

            # Env should win
            assert final_config.temperature == 0.7
            # Library defaults for others
            assert final_config.top_p == 1.0

    def test_build_config_empty_overrides(self):
        """Should work with empty override configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            # Model config
            model_config_data = {"temperature": 0.8}
            config_file = model_path / "generation_config.json"
            with open(config_file, "w") as f:
                json.dump(model_config_data, f)

            empty_env = GenerationConfig()
            empty_runtime = GenerationConfig()

            final_config = build_generation_config(
                model_path=model_path,
                env_overrides=empty_env,
                runtime_overrides=empty_runtime,
            )

            # Model should win
            assert final_config.temperature == 0.8
            # Library defaults for others
            assert final_config.top_p == 1.0

    def test_build_config_none_semantics(self):
        """None values should skip levels in hierarchy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            # Model config
            model_config_data = {"temperature": 0.8}
            config_file = model_path / "generation_config.json"
            with open(config_file, "w") as f:
                json.dump(model_config_data, f)

            # Env config with None temperature (should skip to model)
            env_config = GenerationConfig(temperature=None, top_p=0.9)

            final_config = build_generation_config(
                model_path=model_path, env_overrides=env_config, runtime_overrides=None
            )

            # Model should win (env had None)
            assert final_config.temperature == 0.8
            # Env should win for top_p
            assert final_config.top_p == 0.9


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_configuration_scenario(self):
        """Test a realistic configuration scenario with all levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "qwen3-4b-instruct"
            model_path.mkdir()

            # Model has some defaults
            model_config_data = {
                "temperature": 0.7,
                "max_new_tokens": 2048,
                "repetition_penalty": 1.05,
            }
            config_file = model_path / "generation_config.json"
            with open(config_file, "w") as f:
                json.dump(model_config_data, f)

            # Environment overrides for deployment
            env_config = GenerationConfig(
                temperature=0.1,  # More deterministic for production
                top_p=0.9,  # Override library default
                max_tokens=1000,  # Override model default
            )

            # Runtime overrides for specific request
            runtime_config = GenerationConfig(
                temperature=0.05,  # Even more deterministic for this request
                seed=42,  # Reproducible generation
            )

            # Build final config
            final_config = build_generation_config(
                model_path=model_path,
                env_overrides=env_config,
                runtime_overrides=runtime_config,
            )

            # Verify hierarchy worked correctly
            assert final_config.temperature == 0.05  # Runtime wins
            assert final_config.top_p == 0.9  # Env wins
            assert final_config.max_tokens == 1000  # Env wins over model
            assert final_config.repetition_penalty == 1.05  # Model wins
            assert final_config.seed == 42  # Runtime wins
            # Library defaults for unspecified fields
            assert final_config.presence_penalty == 0.0
            assert final_config.frequency_penalty == 0.0

            # Verify SamplingParams conversion works
            sampling_params = final_config.to_sampling_params()
            assert sampling_params["temperature"] == 0.05
            assert sampling_params["top_p"] == 0.9
            assert sampling_params["max_tokens"] == 1000
            assert sampling_params["repetition_penalty"] == 1.05
            assert sampling_params["seed"] == 42
            assert sampling_params["presence_penalty"] == 0.0
            assert sampling_params["frequency_penalty"] == 0.0

            # Should not include None values
            assert "stop" not in sampling_params
            assert "min_p" not in sampling_params

    def test_vllm_sampling_params_compatibility(self):
        """Test that output is compatible with vLLM SamplingParams."""
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_tokens=1000,
            repetition_penalty=1.1,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            min_p=0.05,
            stop=["</s>", "\n\n"],
            seed=42,
        )

        sampling_params_dict = config.to_sampling_params()

        # Verify all expected keys are present
        expected_keys = {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "min_p",
            "stop",
            "seed",
        }
        assert set(sampling_params_dict.keys()) == expected_keys

        # Verify types are correct for vLLM
        assert isinstance(sampling_params_dict["temperature"], float)
        assert isinstance(sampling_params_dict["top_p"], float)
        assert isinstance(sampling_params_dict["top_k"], int)
        assert isinstance(sampling_params_dict["max_tokens"], int)
        assert isinstance(sampling_params_dict["repetition_penalty"], float)
        assert isinstance(sampling_params_dict["presence_penalty"], float)
        assert isinstance(sampling_params_dict["frequency_penalty"], float)
        assert isinstance(sampling_params_dict["min_p"], float)
        assert isinstance(sampling_params_dict["stop"], list)
        assert isinstance(sampling_params_dict["seed"], int)

        # Verify values are reasonable
        assert 0.0 <= sampling_params_dict["temperature"] <= 2.0
        assert 0.0 <= sampling_params_dict["top_p"] <= 1.0
        assert sampling_params_dict["top_k"] > 0
        assert sampling_params_dict["max_tokens"] > 0
        assert sampling_params_dict["repetition_penalty"] > 0.0
        assert -2.0 <= sampling_params_dict["presence_penalty"] <= 2.0
        assert -2.0 <= sampling_params_dict["frequency_penalty"] <= 2.0
        assert 0.0 <= sampling_params_dict["min_p"] <= 1.0
        assert len(sampling_params_dict["stop"]) > 0
        assert sampling_params_dict["seed"] > 0


class TestCalculateDynamicMaxTokens:
    """Test dynamic max_tokens calculation based on input length and task."""

    def test_user_override_takes_priority(self):
        """User override should always be used when provided."""
        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3, 4, 5]  # 5 tokens
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=32768,
            user_override=999
        )
        
        assert result == 999

    def test_enhancement_task_calculation(self):
        """Enhancement should use 1:1 ratio + 10% buffer + 64, but respect minimum."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 100  # 100 tokens
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: max(256, 100 * 1.1 + 64) = max(256, 174) = 256
        assert result == 256

    def test_enhancement_task_calculation_large_input(self):
        """Enhancement should use 1:1 ratio + 10% buffer + 64 for large inputs."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 500  # 500 tokens (above minimum)
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: 500 * 1.1 + 64 = 614
        assert result == 614

    def test_summarization_task_calculation(self):
        """Summarization should use 30% compression ratio."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 1000  # 1000 tokens
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="summary",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: 1000 * 0.3 = 300
        assert result == 300

    def test_enhancement_minimum_bound(self):
        """Enhancement should respect minimum 256 tokens."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 10  # 10 tokens (very short)
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: max(256, 10 * 1.1 + 64) = max(256, 75) = 256
        assert result == 256

    def test_summarization_minimum_bound(self):
        """Summarization should respect minimum 128 tokens."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 10  # 10 tokens (very short)
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test",
            tokenizer=tokenizer,
            task="summary",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: max(128, 10 * 0.3) = max(128, 3) = 128
        assert result == 128

    def test_enhancement_maximum_bound(self):
        """Enhancement should respect dynamic maximum based on available context."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 10000  # 10000 tokens (very long)
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: max(256, min(32768 - 10000 - 128, 10000 * 1.1 + 64)) 
        # = max(256, min(22640, 11064)) = max(256, 11064) = 11064
        assert result == 11064

    def test_summarization_maximum_bound(self):
        """Summarization should respect dynamic maximum based on model context."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 10000  # 10000 tokens (very long)
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="summary",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: min(32768 * 0.5, 10000 * 0.3) = min(16384, 3000) = 3000
        # But then constrained by context: min(3000, 32768 - 10000 - 128) = min(3000, 22640) = 3000
        assert result == 3000

    def test_context_window_safety_margin(self):
        """Should respect model context window with 128 token safety margin."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 30000  # 30000 tokens (near context limit)
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: min(8192, 32768 - 30000 - 128) = min(8192, 2640) = 2640
        assert result == 2640

    def test_context_window_exceeds_available(self):
        """Should fall back to minimum when context window is too small."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 32000  # 32000 tokens (very close to context limit)
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: max(256, min(8192, 32768 - 32000 - 128)) = max(256, 640) = 640
        assert result == 640

    def test_invalid_task_defaults_to_summarization(self):
        """Invalid task should default to summarization behavior."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 1000  # 1000 tokens
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="invalid_task",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: 1000 * 0.3 = 300 (summarization behavior)
        assert result == 300

    def test_empty_input_text(self):
        """Empty input should still return minimum tokens."""
        class MockTokenizer:
            def encode(self, text):
                return []  # 0 tokens
        
        tokenizer = MockTokenizer()
        
        result = calculate_dynamic_max_tokens(
            input_text="",
            tokenizer=tokenizer,
            task="summary",
            max_model_len=32768,
            user_override=None
        )
        
        # Expected: max(128, 0 * 0.3) = 128
        assert result == 128

    def test_enhancement_respects_smaller_context_window(self):
        """Enhancement with input near context limit should be heavily constrained."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 2000  # 2000 tokens (relatively large input)
        
        tokenizer = MockTokenizer()
        
        # Small model with 4096 token context
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=4096,
            user_override=None
        )
        
        # Expected: max(256, min(4096 - 2000 - 128, 2000 * 1.1 + 64))
        # = max(256, min(1968, 2264)) = max(256, 1968) = 1968
        assert result == 1968

    def test_summarization_less_constrained_by_context(self):
        """Summarization with large input should still get reasonable output budget."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 2000  # 2000 tokens (relatively large input)
        
        tokenizer = MockTokenizer()
        
        # Same 4096 context limit
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="summary",
            max_model_len=4096,
            user_override=None
        )
        
        # Expected: min(1024, 2000 * 0.3, 4096 - 2000 - 128) 
        # = min(1024, 600, 1968) = 600
        assert result == 600

    def test_dynamic_max_tokens_small_model(self):
        """Test dynamic max_tokens with a small model context."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 100  # 100 tokens
        
        tokenizer = MockTokenizer()
        
        # Small model with 1024 token context
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=1024,
            user_override=None
        )
        
        # Expected: max(256, min(1024 * 0.75, 100 * 1.1 + 64)) = max(256, min(768, 174)) = max(256, 174) = 256
        # But then constrained by context: min(256, 1024 - 100 - 128) = min(256, 796) = 256
        assert result == 256

    def test_dynamic_max_tokens_large_input_small_model(self):
        """Test dynamic max_tokens with large input on small model."""
        class MockTokenizer:
            def encode(self, text):
                return [1] * 500  # 500 tokens (large input)
        
        tokenizer = MockTokenizer()
        
        # Small model with 1024 token context
        result = calculate_dynamic_max_tokens(
            input_text="test text",
            tokenizer=tokenizer,
            task="enhancement",
            max_model_len=1024,
            user_override=None
        )
        
        # Expected: max(256, min(1024 - 500 - 128, 500 * 1.1 + 64)) 
        # = max(256, min(396, 614)) = max(256, 396) = 396
        assert result == 396

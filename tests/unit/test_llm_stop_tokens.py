"""
Unit tests for LLM stop tokens configuration.

Tests verify that stop tokens are properly configured to prevent
chat template echo issue (BUGFIX_LLM_CHAT_TEMPLATE_ECHO.md).
"""

import pytest

from src.processors.llm.config import GenerationConfig, get_library_defaults


def test_library_defaults_include_stop_tokens():
    """
    Test that library defaults include stop tokens.

    This prevents the LLM from echoing chat template markers
    in its output (BUGFIX_LLM_CHAT_TEMPLATE_ECHO.md).
    """
    config = get_library_defaults()

    assert config.stop is not None, "Stop tokens should be configured"
    assert isinstance(config.stop, list), "Stop should be a list"
    assert len(config.stop) > 0, "Stop list should not be empty"
    assert "<|im_end|>" in config.stop, "Should include Qwen3 chat end marker"


def test_stop_tokens_in_sampling_params():
    """Test that stop tokens are included in SamplingParams dict."""
    config = get_library_defaults()
    sampling_params = config.to_sampling_params()

    assert "stop" in sampling_params, "Stop should be in sampling params"
    assert "<|im_end|>" in sampling_params["stop"], "Should include <|im_end|>"


def test_stop_tokens_merge_behavior():
    """
    Test that stop tokens merge correctly in config hierarchy.

    Runtime overrides should be able to extend or replace stop tokens.
    """
    base_config = get_library_defaults()
    runtime_config = GenerationConfig(stop=["custom_stop"])

    # Runtime overrides should take priority
    merged = runtime_config.merge_with(base_config)

    assert merged.stop == ["custom_stop"], "Runtime stop tokens should override"


def test_stop_tokens_preserve_in_merge():
    """Test that stop tokens are preserved when merging configs without stop."""
    base_config = get_library_defaults()
    runtime_config = GenerationConfig(temperature=0.5)  # No stop specified

    # Base stop tokens should be preserved
    merged = runtime_config.merge_with(base_config)

    assert merged.stop is not None, "Stop tokens should be preserved"
    assert "<|im_end|>" in merged.stop, "Base stop tokens should remain"
    assert merged.temperature == 0.5, "Runtime temperature should apply"


def test_generation_config_none_values():
    """Test that None values in config don't override stop tokens."""
    base_config = get_library_defaults()
    empty_config = GenerationConfig()  # All fields are None

    merged = empty_config.merge_with(base_config)

    assert merged.stop is not None, "Stop tokens should come from base"
    assert merged.stop == base_config.stop, "Stop tokens should be preserved"


def test_multiple_stop_tokens():
    """Test that multiple stop tokens can be configured."""
    config = GenerationConfig(stop=["<|im_end|>", "<|endoftext|>", "###"])
    params = config.to_sampling_params()

    assert len(params["stop"]) == 3, "Should support multiple stop tokens"
    assert all(
        token in params["stop"] for token in ["<|im_end|>", "<|endoftext|>", "###"]
    ), "All stop tokens should be present"


def test_empty_stop_list_handling():
    """Test that empty stop list is handled correctly."""
    config = GenerationConfig(stop=[])
    params = config.to_sampling_params()

    # Empty list should be in params (not filtered out)
    assert "stop" in params, "Empty stop list should be in params"
    assert params["stop"] == [], "Empty list should be preserved"


@pytest.mark.parametrize(
    "stop_token",
    [
        "<|im_end|>",
        "<|endoftext|>",
        "\n\n\n",  # Multiple newlines can also be a stop signal
    ],
)
def test_common_stop_tokens(stop_token):
    """Test various common stop token formats."""
    config = GenerationConfig(stop=[stop_token])
    params = config.to_sampling_params()

    assert stop_token in params["stop"], f"Should support {stop_token}"


def test_stop_tokens_repr():
    """Test that stop tokens appear in config repr."""
    config = GenerationConfig(stop=["<|im_end|>"])
    repr_str = repr(config)

    assert "stop" in repr_str, "Stop should appear in repr"
    assert "<|im_end|>" in repr_str, "Stop value should appear in repr"

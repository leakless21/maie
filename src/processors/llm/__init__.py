"""
LLM processing module for MAIE.

This module provides hierarchical configuration management for LLM generation parameters,
supporting vLLM SamplingParams with priority chain: Runtime > Environment > Model > Library.
"""


# Lazy imports to avoid loading vLLM on module import (Jetson memory optimization)
def _get_vllm_utils():
    from src.tooling.vllm_utils import calculate_checkpoint_hash, get_model_info

    return calculate_checkpoint_hash, get_model_info


from .config import (
    GenerationConfig,
    build_generation_config,
    get_library_defaults,
    load_model_generation_config,
)
from .processor import LLMProcessor
from .schema_validator import (
    create_validation_summary,
    load_template_schema,
    retry_with_lower_temperature,
    validate_llm_output,
    validate_schema_completeness,
    validate_tags_field,
)

__all__ = [
    "LLMProcessor",
    "calculate_checkpoint_hash",
    "get_model_info",
    # Re-export vLLM-facing symbols for tests to patch
    "LLM",
    "SamplingParams",
    "GuidedDecodingParams",
    "GenerationConfig",
    "get_library_defaults",
    "load_model_generation_config",
    "build_generation_config",
    "load_template_schema",
    "validate_llm_output",
    "validate_tags_field",
    "retry_with_lower_temperature",
    "create_validation_summary",
    "validate_schema_completeness",
]


# Lazy attribute getter for vLLM imports and utils
def __getattr__(name):
    if name in ["calculate_checkpoint_hash", "get_model_info"]:
        calculate_checkpoint_hash, get_model_info = _get_vllm_utils()
        return locals()[name]
    elif name in ["LLM", "SamplingParams", "GuidedDecodingParams"]:
        try:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import GuidedDecodingParams

            return locals()[name]
        except ImportError:
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

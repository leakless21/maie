"""
LLM processing module for MAIE.

This module provides hierarchical configuration management for LLM generation parameters,
supporting vLLM SamplingParams with priority chain: Runtime > Environment > Model > Library.
"""

from .config import (
    GenerationConfig,
    get_library_defaults,
    load_model_generation_config,
    build_generation_config,
)
from .processor import LLMProcessor
from src.tooling.vllm_utils import (
    calculate_checkpoint_hash,
    get_model_info,
)
from .schema_validator import (
    load_template_schema,
    validate_llm_output,
    validate_tags_field,
    retry_with_lower_temperature,
    create_validation_summary,
    validate_schema_completeness,
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

# Re-export vLLM classes for test patching (with safe import)
try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    # Allow imports to fail for testing without vLLM
    LLM = None
    SamplingParams = None
    GuidedDecodingParams = None

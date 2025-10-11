"""
Hierarchical configuration management for LLM generation parameters.

This module provides a configuration system that supports:
- vLLM SamplingParams conversion
- HuggingFace generation_config.json loading  
- Priority chain: Runtime > Environment > Model > Library defaults
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class GenerationConfig:
    """
    Configuration for LLM generation parameters.
    
    All fields are optional and default to None, meaning "use lower priority value".
    This enables a hierarchical configuration system where None values propagate
    down the priority chain.
    
    Fields correspond to vLLM SamplingParams parameters:
    - temperature: Controls randomness (0.0 = deterministic, 1.0 = default)
    - top_p: Nucleus sampling threshold (0.0-1.0)
    - top_k: Top-k sampling (number of tokens to consider)
    - max_tokens: Maximum tokens to generate
    - repetition_penalty: Penalty for repetition (>1.0 = penalize)
    - presence_penalty: Penalty for presence of tokens (-2.0 to 2.0)
    - frequency_penalty: Penalty for frequency of tokens (-2.0 to 2.0)
    - min_p: Minimum probability threshold for token selection
    - stop: List of stop sequences
    - seed: Random seed for reproducibility
    """
    
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    min_p: Optional[float] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None

    def to_sampling_params(self) -> Dict[str, Any]:
        """
        Convert to dictionary suitable for vLLM SamplingParams.
        
        Returns:
            Dictionary with only non-None values, ready for SamplingParams(**dict)
        """
        # Get all fields as dict
        params = asdict(self)
        
        # Filter out None values
        return {k: v for k, v in params.items() if v is not None}

    def merge_with(self, other: "GenerationConfig") -> "GenerationConfig":
        """
        Merge with another GenerationConfig, with self values taking priority.
        
        Args:
            other: Other GenerationConfig to merge with
            
        Returns:
            New GenerationConfig with merged values (self > other)
        """
        # For each field: use self value if not None, else use other value
        merged_data = {}
        for field_name in asdict(self).keys():
            self_value = getattr(self, field_name)
            other_value = getattr(other, field_name)
            merged_data[field_name] = self_value if self_value is not None else other_value
        
        return GenerationConfig(**merged_data)

    def __repr__(self) -> str:
        """Show only non-None fields for clean debugging output."""
        non_none_fields = []
        for field_name, value in asdict(self).items():
            if value is not None:
                non_none_fields.append(f"{field_name}={value!r}")
        
        if non_none_fields:
            return f"GenerationConfig({', '.join(non_none_fields)})"
        else:
            return "GenerationConfig()"


def get_library_defaults() -> GenerationConfig:
    """
    Get vLLM library default values.
    
    Returns:
        GenerationConfig with vLLM default values
    """
    return GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=16,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )


def load_model_generation_config(model_path: Path) -> GenerationConfig:
    """
    Load generation configuration from HuggingFace model's generation_config.json.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        GenerationConfig with model defaults, or empty config if file not found/invalid
    """
    config_file = model_path / "generation_config.json"
    
    if not config_file.exists():
        logger.debug(f"No generation_config.json found at {config_file}")
        return GenerationConfig()
    
    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        # Map HuggingFace fields to our fields
        config_data = {}
        
        # Direct mappings
        for hf_field, our_field in [
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("repetition_penalty", "repetition_penalty"),
            ("presence_penalty", "presence_penalty"),
            ("frequency_penalty", "frequency_penalty"),
            ("min_p", "min_p"),
            ("stop", "stop"),
            ("seed", "seed"),
        ]:
            if hf_field in data:
                config_data[our_field] = data[hf_field]
        
        # Special mappings for max_tokens
        if "max_new_tokens" in data:
            config_data["max_tokens"] = data["max_new_tokens"]
        elif "max_length" in data:
            config_data["max_tokens"] = data["max_length"]
        
        logger.debug(f"Loaded model config from {config_file}: {config_data}")
        return GenerationConfig(**config_data)
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to load generation_config.json from {config_file}: {e}")
        return GenerationConfig()


def build_generation_config(
    model_path: Path,
    env_overrides: GenerationConfig,
    runtime_overrides: Optional[GenerationConfig] = None
) -> GenerationConfig:
    """
    Build final generation configuration using hierarchical priority.
    
    Priority order (highest to lowest):
    1. Runtime overrides (if provided)
    2. Environment overrides  
    3. Model generation_config.json
    4. Library defaults (vLLM)
    
    Args:
        model_path: Path to the model directory
        env_overrides: Environment-level configuration overrides
        runtime_overrides: Runtime-level configuration overrides (optional)
        
    Returns:
        Final GenerationConfig with all levels merged
    """
    # Start with library defaults
    config = get_library_defaults()
    
    # Merge model config
    model_config = load_model_generation_config(model_path)
    config = model_config.merge_with(config)
    
    # Merge environment overrides
    config = env_overrides.merge_with(config)
    
    # Merge runtime overrides if provided
    if runtime_overrides is not None:
        config = runtime_overrides.merge_with(config)
    
    logger.debug(f"Final generation config: {config}")
    return config

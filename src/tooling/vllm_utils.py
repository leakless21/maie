"""
Utility functions for vLLM integration.

This module provides helper functions for working with vLLM, including
sampling parameter management, override handling, and model versioning.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from src.config.logging import get_module_logger

# Disable vLLM telemetry at module level
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

# Create module-bound logger for better debugging
logger = get_module_logger(__name__)


def apply_overrides_to_sampling(base_params: Any, overrides: Dict[str, Any]) -> Any:
    """
    Merge config overrides into vLLM SamplingParams.

    Args:
        base_params: Base SamplingParams instance
        overrides: Dictionary of parameter overrides

    Returns:
        New SamplingParams with overrides applied

    Example:
        >>> from vllm import SamplingParams
        >>> base = SamplingParams(temperature=0.7)
        >>> overrides = {"temperature": 0.3, "max_tokens": 1000}
        >>> new_params = apply_overrides_to_sampling(base, overrides)
    """
    try:
        # Get current parameters as dict
        if isinstance(base_params, dict):
            # base_params is already a dict
            current_params = base_params
        elif hasattr(base_params, "to_dict"):
            # base_params has a to_dict method
            current_params = base_params.to_dict()
        else:
            # Fallback: extract common parameters as attributes
            current_params = {}
            for attr in [
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
            ]:
                if hasattr(base_params, attr):
                    current_params[attr] = getattr(base_params, attr)

        # Merge overrides
        merged_params = {**current_params, **overrides}

        # Create new SamplingParams
        from vllm import SamplingParams

        return SamplingParams(**merged_params)

    except Exception as e:
        logger.error(f"Failed to apply overrides to sampling params: {e}")
        # Return base params if merge fails
        return base_params


def normalize_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize override dictionary for logging and metadata.

    Converts non-serializable values to strings and filters out None values.

    Args:
        overrides: Raw override dictionary

    Returns:
        Normalized dictionary safe for JSON serialization

    Example:
        >>> overrides = {"temperature": 0.7, "stop": ["</s>"], "custom": None}
        >>> normalized = normalize_overrides(overrides)
        >>> # Result: {"temperature": 0.7, "stop": ["</s>"]}
    """
    normalized = {}

    for key, value in overrides.items():
        if value is None:
            continue

        # Convert non-serializable types to strings
        if isinstance(value, (list, tuple)):
            # Keep lists as-is (serializable)
            normalized[key] = list(value)
        elif isinstance(value, set):
            # Convert sets to lists (sorted for consistency)
            normalized[key] = sorted(list(value))
        elif isinstance(value, dict):
            # Keep dicts as-is (serializable)
            normalized[key] = value
        elif isinstance(value, (str, int, float, bool)):
            # Keep primitives as-is
            normalized[key] = value
        else:
            # Convert other types to strings
            normalized[key] = str(value)

    return normalized


def calculate_checkpoint_hash(model_path: Path) -> str:
    """
    Calculate SHA-256 hash of model checkpoint for versioning (NFR-1).

    This function attempts to find and hash the model weights file.
    For HuggingFace models, it looks for common weight files like:
    - model.safetensors
    - pytorch_model.bin
    - model.bin

    Args:
        model_path: Path to model directory

    Returns:
        SHA-256 hash as hexadecimal string

    Raises:
        FileNotFoundError: If no model weights found
        ValueError: If model_path is not a directory

    Example:
        >>> model_path = Path("data/models/qwen3-4b-awq")
        >>> hash_value = calculate_checkpoint_hash(model_path)
        >>> print(f"Model hash: {hash_value}")
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if not model_path.is_dir():
        raise ValueError(f"Model path is not a directory: {model_path}")

    # Common model weight file patterns (in order of preference)
    weight_patterns = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "pytorch_model-00001-of-00001.bin",
        "model-00001-of-00001.safetensors",
    ]

    weight_file = None
    for pattern in weight_patterns:
        candidate = model_path / pattern
        if candidate.exists():
            weight_file = candidate
            break

    if weight_file is None:
        # Fallback: hash the entire directory contents
        logger.warning(
            f"No standard weight file found in {model_path}, hashing directory"
        )
        return _hash_directory(model_path)

    # Calculate SHA-256 hash of the weight file
    sha256_hash = hashlib.sha256()

    try:
        with open(weight_file, "rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        hash_value = sha256_hash.hexdigest()
        logger.debug(f"Calculated hash for {weight_file}: {hash_value[:16]}...")
        return hash_value

    except Exception as e:
        logger.error(f"Failed to calculate hash for {weight_file}: {e}")
        # Fallback to directory hash
        return _hash_directory(model_path)


def _hash_directory(directory: Path) -> str:
    """
    Calculate hash of directory contents as fallback.

    Args:
        directory: Directory to hash

    Returns:
        SHA-256 hash as hexadecimal string
    """
    sha256_hash = hashlib.sha256()

    # Get all files in directory, sorted for consistency
    files = sorted(directory.rglob("*"))
    files = [f for f in files if f.is_file()]

    for file_path in files:
        # Include file path in hash
        sha256_hash.update(str(file_path.relative_to(directory)).encode())

        # Include file size
        sha256_hash.update(str(file_path.stat().st_size).encode())

        # Include file modification time
        sha256_hash.update(str(file_path.stat().st_mtime).encode())

    return sha256_hash.hexdigest()


def get_model_info(model_path: Path) -> Dict[str, Any]:
    """
    Extract model information for versioning metadata.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary with model metadata

    Example:
        >>> model_path = Path("data/models/qwen3-4b-awq")
        >>> info = get_model_info(model_path)
        >>> print(info["model_name"])  # "qwen3-4b-awq"
    """
    info = {
        "model_name": model_path.name,
        "model_path": str(model_path),
        "checkpoint_hash": calculate_checkpoint_hash(model_path),
    }

    # Try to load config.json for additional metadata
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Extract relevant fields
            info.update(
                {
                    "model_type": config.get("model_type", "unknown"),
                    "architectures": config.get("architectures", []),
                    "vocab_size": config.get("vocab_size"),
                    "hidden_size": config.get("hidden_size"),
                    "num_attention_heads": config.get("num_attention_heads"),
                    "num_hidden_layers": config.get("num_hidden_layers"),
                }
            )

        except Exception as e:
            logger.warning(f"Failed to load config.json from {model_path}: {e}")

    return info

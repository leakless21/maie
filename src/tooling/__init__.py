"""
Tooling utilities for MAIE.

This module provides utility functions for various integrations including vLLM.
Note: vllm_utils functions are lazily imported to avoid memory issues on Jetson.
"""

__all__ = [
    "apply_overrides_to_sampling",
    "normalize_overrides",
    "calculate_checkpoint_hash",
]


# Lazy imports to avoid loading vLLM unnecessarily
def __getattr__(name):
    if name in __all__:
        from .vllm_utils import (
            apply_overrides_to_sampling,
            calculate_checkpoint_hash,
            normalize_overrides,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

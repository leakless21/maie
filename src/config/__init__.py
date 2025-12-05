"""
Configuration package exposing application settings and logging helpers.
"""

import os

# Ensure PyTorch uses expandable CUDA segments to reduce fragmentation warnings.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# CRITICAL: Disable ALL telemetry and analytics for FULLY OFFLINE operation
# This prevents network calls to:
# - otel.pyannote.ai (OpenTelemetry telemetry from pyannote)
# - HuggingFace Hub (transformers/HuggingFace telemetry)
os.environ.setdefault("PYANNOTE_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYANNOTE_NO_ANALYTICS", "1")
os.environ.setdefault("HF_TRANSFER_DISABLE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from .model import AppSettings
from .loader import get_settings, reset_settings_cache, settings
from .logging import (
    bind_correlation_id,
    clear_correlation_id,
    configure_logging,
    correlation_id,
    generate_correlation_id,
    get_logger,
    get_module_logger,
)  # noqa: E402
from .profiles import (
    PROFILES,
    JETSON_PROFILE,
    EDGE_PROFILE,
    get_profile,
    apply_profile,
    is_jetson_environment,
    is_edge_environment,
    has_llm_support,
    has_diarization_support,
    has_redis_support,
)

__all__ = [
    "AppSettings",
    "settings",
    "get_settings",
    "reset_settings_cache",
    "configure_logging",
    "get_logger",
    "bind_correlation_id",
    "clear_correlation_id",
    "correlation_id",
    "generate_correlation_id",
    "get_module_logger",
    # Profile exports
    "PROFILES",
    "JETSON_PROFILE",
    "EDGE_PROFILE",
    "get_profile",
    "apply_profile",
    "is_jetson_environment",
    "is_edge_environment",
    "has_llm_support",
    "has_diarization_support",
    "has_redis_support",
]

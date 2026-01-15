"""
Configuration profiles for different deployment environments.

This module defines pre-configured settings profiles for various deployment
scenarios. Profiles can be applied to override default settings without
modifying environment variables.

Usage:
    from src.config.profiles import JETSON_PROFILE, apply_profile
    settings = apply_profile(get_settings(), JETSON_PROFILE)

Or via environment:
    ENVIRONMENT=jetson python -m src.api.edge_main
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .model import AppSettings

# =============================================================================
# Profile Definitions
# =============================================================================

# Jetson Nano/Orin profile - ASR only, no LLM, no diarization
JETSON_PROFILE: Dict[str, Any] = {
    "environment": "jetson",
    "debug": False,
    "verbose_components": False,
    # Feature flags - disable heavy components
    "features": {
        "enable_llm": False,
        "enable_diarization": False,
        "enable_enhancement": False,
        "enable_redis_queue": False,
    },
    # ASR config - use efficient models
    "asr": {
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_cpu_fallback": True,
    },
    # Chunkformer config
    "chunkformer": {
        "chunkformer_device": "cuda",
        "chunkformer_cpu_fallback": True,
    },
    # Diarization disabled
    "diarization": {
        "enabled": False,
    },
    # VAD can still be used for preprocessing
    "vad": {
        "enabled": True,
        "device": "cuda",
    },
    # Conservative API limits for edge device
    "api": {
        "max_file_size_mb": 100.0,
    },
    # Minimal logging retention
    "logging": {
        "log_rotation": "50 MB",
        "log_retention": "3 days",
        "log_level": "INFO",
    },
    # Shorter cleanup intervals for edge
    "cleanup": {
        "audio_retention_days": 1,
        "logs_retention_days": 3,
    },
    # Single-threaded worker (not used in edge mode, but set for completeness)
    "worker": {
        "worker_concurrency": 1,
        "worker_prefetch_multiplier": 1,
    },
}

# Edge profile - generic edge deployment (similar to Jetson but less specific)
EDGE_PROFILE: Dict[str, Any] = {
    "environment": "edge",
    "debug": False,
    "verbose_components": False,
    "features": {
        "enable_llm": False,
        "enable_diarization": False,
        "enable_enhancement": False,
        "enable_redis_queue": False,
    },
    "api": {
        "max_file_size_mb": 100.0,
    },
    "logging": {
        "log_rotation": "50 MB",
        "log_retention": "7 days",
    },
    "worker": {
        "worker_concurrency": 1,
    },
}

# Development profile (default)
DEVELOPMENT_PROFILE: Dict[str, Any] = {
    "environment": "development",
    "debug": True,
    "verbose_components": True,
    "features": {
        "enable_llm": True,
        "enable_diarization": True,
        "enable_enhancement": True,
        "enable_redis_queue": True,
    },
    "logging": {
        "log_level": "DEBUG",
    },
}

# Production profile
PRODUCTION_PROFILE: Dict[str, Any] = {
    "environment": "production",
    "debug": False,
    "verbose_components": False,
    "features": {
        "enable_llm": True,
        "enable_diarization": True,
        "enable_enhancement": True,
        "enable_redis_queue": True,
    },
    "logging": {
        "log_level": "INFO",
    },
}

# Registry of all profiles
PROFILES: Dict[str, Dict[str, Any]] = {
    "development": DEVELOPMENT_PROFILE,
    "production": PRODUCTION_PROFILE,
    "edge": EDGE_PROFILE,
    "jetson": JETSON_PROFILE,
}


# =============================================================================
# Profile Application
# =============================================================================


def get_profile(environment: str) -> Dict[str, Any]:
    """
    Get a profile by environment name.

    Args:
        environment: Environment name (development, production, edge, jetson)

    Returns:
        Profile dictionary, or empty dict if not found
    """
    return PROFILES.get(environment.lower(), {})


def apply_profile(settings: "AppSettings", profile: Dict[str, Any]) -> "AppSettings":
    """
    Apply a profile to an AppSettings instance.

    Uses the settings.apply_profile method which respects environment variable
    precedence - env vars always win over profile values.

    Args:
        settings: Current AppSettings instance
        profile: Profile dictionary to apply

    Returns:
        New AppSettings instance with profile applied
    """
    if not profile:
        return settings
    return settings.apply_profile(profile)


def is_jetson_environment(settings: "AppSettings") -> bool:
    """Check if running in Jetson environment."""
    return settings.environment == "jetson"


def is_edge_environment(settings: "AppSettings") -> bool:
    """Check if running in any edge environment (jetson or generic edge)."""
    return settings.environment in ("jetson", "edge")


def has_llm_support(settings: "AppSettings") -> bool:
    """Check if LLM features are enabled and available."""
    return settings.features.enable_llm


def has_diarization_support(settings: "AppSettings") -> bool:
    """Check if diarization is enabled and available."""
    return settings.features.enable_diarization and settings.diarization.enabled


def has_redis_support(settings: "AppSettings") -> bool:
    """Check if Redis queue is enabled."""
    return settings.features.enable_redis_queue

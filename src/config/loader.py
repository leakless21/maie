from __future__ import annotations

import os
from typing import Dict, Optional

from .model import AppSettings
from .profiles import PROFILES

DEFAULT_ENVIRONMENT = "development"

_ENV_ALIASES = {
    "dev": "development",
    "development": "development",
    "prod": "production",
    "production": "production",
}

_SETTINGS_CACHE: Dict[str, AppSettings] = {}


def _normalize_environment(value: Optional[str]) -> str:
    candidate = (value or os.getenv("ENVIRONMENT") or DEFAULT_ENVIRONMENT).strip()
    if not candidate:
        return DEFAULT_ENVIRONMENT
    normalized = candidate.lower()
    return _ENV_ALIASES.get(normalized, DEFAULT_ENVIRONMENT)


def _build_settings(environment: str) -> AppSettings:
    profile = PROFILES.get(environment, PROFILES[DEFAULT_ENVIRONMENT])
    settings = AppSettings().apply_profile(profile)
    if settings.environment != environment:
        settings = settings.model_copy(update={"environment": environment})
    return settings


def get_settings(
    environment: Optional[str] = None,
    *,
    reload: bool = False,
) -> AppSettings:
    env_key = _normalize_environment(environment)
    if reload:
        _SETTINGS_CACHE.pop(env_key, None)
    if env_key not in _SETTINGS_CACHE:
        _SETTINGS_CACHE[env_key] = _build_settings(env_key)
    return _SETTINGS_CACHE[env_key]


def reset_settings_cache() -> None:
    _SETTINGS_CACHE.clear()


settings = get_settings()

__all__ = ["get_settings", "reset_settings_cache", "settings", "AppSettings"]

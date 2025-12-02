from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import dotenv_values

from .model import AppSettings

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


def _load_environment_file(environment: str) -> None:
    """
    Load environment-specific .env file and merge into os.environ.
    
    Only loads the file matching the current environment (e.g., .env.development
    when ENVIRONMENT=development). If both .env.development and .env.production
    exist, only the one matching the current ENVIRONMENT variable is loaded.
    
    Precedence (highest to lowest):
    1. os.environ (existing environment variables)
    2. .env.{environment} (environment-specific file, e.g., .env.development)
    3. .env (base file, loaded by pydantic-settings)
    4. Default values from model fields
    
    Only sets values that don't already exist in os.environ to maintain precedence.
    This ensures environment variables can override file values.
    """
    env_file = Path(f".env.{environment}")
    
    # Load environment-specific file if it exists
    # Note: Only the file matching the current environment is loaded.
    # If both .env.development and .env.production exist, only one is loaded
    # based on the ENVIRONMENT variable.
    if env_file.exists():
        env_values = dotenv_values(env_file, encoding="utf-8")
        # Only set values that don't already exist in os.environ
        # This preserves the precedence: env vars > env file > .env > defaults
        for key, value in env_values.items():
            if value is not None and key not in os.environ:
                os.environ[key] = value


def _build_settings(environment: str) -> AppSettings:
    """
    Build AppSettings instance for the specified environment.
    
    Loads .env.{environment} file if it exists (e.g., .env.development when
    environment="development"), then creates AppSettings which will read from:
    1. os.environ (highest priority - includes values from .env.{environment})
    2. .env file (base/fallback file loaded by pydantic-settings)
    3. Default values from model fields (lowest priority)
    
    If both .env.development and .env.production exist, only the one matching
    the current environment is loaded. Use .env for shared values and
    .env.{environment} for environment-specific overrides.
    """
    # Load environment-specific file (merges into os.environ)
    # This only loads the file matching the current environment
    _load_environment_file(environment)
    
    # Create AppSettings - it will read from os.environ (highest priority),
    # then .env file (fallback), then defaults
    settings = AppSettings()
    
    # Ensure environment field matches the requested environment
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

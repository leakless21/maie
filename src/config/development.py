"""Development environment overrides for MAIE settings."""

from .base import BaseAppSettings


class DevelopmentSettings(BaseAppSettings):
    debug: bool = True
    log_level: str = "DEBUG"


__all__ = ["DevelopmentSettings"]




"""Production environment overrides for MAIE settings."""

from .base import BaseAppSettings


class ProductionSettings(BaseAppSettings):
    debug: bool = False
    log_level: str = "INFO"


__all__ = ["ProductionSettings"]




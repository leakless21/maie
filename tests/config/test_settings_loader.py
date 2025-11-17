import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.loader import get_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _reset_loader_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


class TestSettingsCaching:
    def test_get_settings_caches_instance(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            first = get_settings()
            second = get_settings()

        assert first is second

    def test_reload_returns_new_instance(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            first = get_settings()
            second = get_settings(reload=True)

        assert first is not second
        assert second.environment == "development"


class TestEnvironmentProfiles:
    def test_development_profile(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            settings = get_settings(reload=True)

        assert settings.environment == "development"
        assert settings.debug is True
        assert settings.logging.log_level == "DEBUG"
        assert settings.worker.job_timeout == 360
        assert settings.paths.audio_dir == Path("data/audio")

    def test_production_profile(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            settings = get_settings(reload=True)

        assert settings.environment == "production"
        assert settings.debug is False
        assert settings.logging.log_console_serialize is True
        assert settings.paths.audio_dir == Path("/app/data/audio")
        assert settings.worker.job_timeout == 600


class TestEnvironmentFallbacks:
    def test_unknown_environment_falls_back_to_development(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            settings = get_settings(reload=True)

        assert settings.environment == "development"
        assert settings.logging.log_level == "DEBUG"

    def test_environment_variable_overrides_profile(self):
        env_vars = {
            "ENVIRONMENT": "production",
            "APP_LOGGING__LOG_LEVEL": "debug",
            "APP_WORKER__JOB_TIMEOUT": "120",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings(reload=True)

        assert settings.environment == "production"
        assert settings.logging.log_level == "DEBUG"
        assert settings.worker.job_timeout == 120

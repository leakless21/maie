import os
import tempfile
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
    def test_development_profile(self, tmp_path, monkeypatch):
        """Test that .env.development file is loaded correctly."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create .env.development file
        env_file = tmp_path / ".env.development"
        env_file.write_text("""APP_ENVIRONMENT=development
APP_DEBUG=true
APP_VERBOSE_COMPONENTS=true
APP_LOGGING__LOG_LEVEL=DEBUG
APP_WORKER__JOB_TIMEOUT=360
""")
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            settings = get_settings(reload=True)

        assert settings.environment == "development"
        assert settings.debug is True
        assert settings.logging.log_level == "DEBUG"
        assert settings.worker.job_timeout == 360
        assert settings.paths.audio_dir == Path("data/audio")  # Default value

    def test_production_profile(self, tmp_path, monkeypatch):
        """Test that .env.production file is loaded correctly."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create .env.production file
        env_file = tmp_path / ".env.production"
        env_file.write_text("""APP_ENVIRONMENT=production
APP_DEBUG=false
APP_VERBOSE_COMPONENTS=false
APP_LOGGING__LOG_LEVEL=INFO
APP_LOGGING__LOG_CONSOLE_SERIALIZE=true
APP_PATHS__AUDIO_DIR=/app/data/audio
APP_WORKER__JOB_TIMEOUT=600
""")
        
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            settings = get_settings(reload=True)

        assert settings.environment == "production"
        assert settings.debug is False
        assert settings.logging.log_console_serialize is True
        assert settings.paths.audio_dir == Path("/app/data/audio")
        assert settings.worker.job_timeout == 600


class TestEnvironmentFallbacks:
    def test_unknown_environment_falls_back_to_defaults(self, tmp_path, monkeypatch):
        """Test that unknown environment falls back to default values (not development profile)."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            settings = get_settings(reload=True)

        # Should normalize to development (via _ENV_ALIASES fallback)
        assert settings.environment == "development"
        # Should use default values from model, not development profile
        assert settings.logging.log_level == "INFO"  # Default from LoggingSettings

    def test_environment_variable_overrides_env_file(self, tmp_path, monkeypatch):
        """Test that environment variables override .env file values."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create .env.production file
        env_file = tmp_path / ".env.production"
        env_file.write_text("""APP_ENVIRONMENT=production
APP_LOGGING__LOG_LEVEL=INFO
APP_WORKER__JOB_TIMEOUT=600
""")
        
        env_vars = {
            "ENVIRONMENT": "production",
            "APP_LOGGING__LOG_LEVEL": "debug",  # Override env file
            "APP_WORKER__JOB_TIMEOUT": "120",  # Override env file
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings(reload=True)

        assert settings.environment == "production"
        assert settings.logging.log_level == "DEBUG"  # From env var, not file
        assert settings.worker.job_timeout == 120  # From env var, not file

    def test_only_matching_environment_file_loaded(self, tmp_path, monkeypatch):
        """Test that when both .env.development and .env.production exist, only the matching one is loaded."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create both environment files with different values
        dev_file = tmp_path / ".env.development"
        dev_file.write_text("""APP_ENVIRONMENT=development
APP_LOGGING__LOG_LEVEL=DEBUG
APP_WORKER__JOB_TIMEOUT=360
""")
        
        prod_file = tmp_path / ".env.production"
        prod_file.write_text("""APP_ENVIRONMENT=production
APP_LOGGING__LOG_LEVEL=INFO
APP_WORKER__JOB_TIMEOUT=600
""")
        
        # Test development environment - should only load .env.development
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            settings = get_settings(reload=True)
            assert settings.environment == "development"
            assert settings.logging.log_level == "DEBUG"  # From .env.development
            assert settings.worker.job_timeout == 360  # From .env.development
        
        # Test production environment - should only load .env.production
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            settings = get_settings(reload=True)
            assert settings.environment == "production"
            assert settings.logging.log_level == "INFO"  # From .env.production
            assert settings.worker.job_timeout == 600  # From .env.production

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.model import AppSettings


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure environment state isolation between tests."""
    with patch.dict(os.environ, {}, clear=True):
        yield


class TestAppSettingsStructure:
    def test_defaults_expose_nested_sections(self):
        settings = AppSettings()

        assert settings.pipeline_version == "1.0.0"
        assert settings.environment == "development"
        assert settings.logging.log_level == "INFO"
        assert settings.api.port == 8000
        assert settings.paths.audio_dir == Path("data/audio")

    def test_env_overrides_with_nested_delimiter(self):
        env_vars = {
            "APP_LOGGING__LOG_LEVEL": "warning",
            "APP_LOGGING__LOG_CONSOLE_SERIALIZE": "true",
            "APP_API__PORT": "9000",
            "APP_ASR__WHISPER_BEAM_SIZE": "7",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = AppSettings()

        assert settings.logging.log_level == "WARNING"
        assert settings.logging.log_console_serialize is True
        assert settings.api.port == 9000
        assert settings.asr.whisper_beam_size == 7



class TestDirectoryManagement:
    def test_directories_created_on_demand(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_dir = Path(tmp_dir) / "audio"
            models_dir = Path(tmp_dir) / "models"
            templates_dir = Path(tmp_dir) / "templates"

            env_vars = {
                "APP_PATHS__AUDIO_DIR": str(audio_dir),
                "APP_PATHS__MODELS_DIR": str(models_dir),
                "APP_PATHS__TEMPLATES_DIR": str(templates_dir),
            }

            with patch.dict(os.environ, env_vars, clear=True):
                settings = AppSettings()

            assert not audio_dir.exists()
            assert not models_dir.exists()
            assert not templates_dir.exists()

            created = settings.ensure_directories()

            assert created == {
                "audio_dir": audio_dir,
                "models_dir": models_dir,
                "templates_dir": templates_dir,
            }
            assert audio_dir.exists()
            assert models_dir.exists()
            assert templates_dir.exists()

    def test_directory_creation_errors_propagate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_dir = Path(tmp_dir) / "audio"

            env_vars = {"APP_PATHS__AUDIO_DIR": str(audio_dir)}

            with patch.dict(os.environ, env_vars, clear=True):
                settings = AppSettings()

            with patch.object(Path, "mkdir", side_effect=PermissionError):
                with pytest.raises(PermissionError):
                    settings.ensure_directories()


class TestValidation:
    def test_invalid_log_level_raises_value_error(self):
        with patch.dict(os.environ, {"APP_LOGGING__LOG_LEVEL": "verbose"}, clear=True):
            with pytest.raises(ValueError):
                AppSettings()

    def test_invalid_port_raises_value_error(self):
        with patch.dict(os.environ, {"APP_API__PORT": "70000"}, clear=True):
            with pytest.raises(ValueError):
                AppSettings()

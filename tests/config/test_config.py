"""Concise test docstrings for src.config.Settings (RED-style expectations)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Settings, settings


class TestSettingsDefaults:
    """Defaults for Settings."""

    def test_core_settings_defaults(self):
        """RED: pipeline_version '1.0.0', environment 'development', debug False."""
        with patch.dict(os.environ, {}, clear=True):
            config = Settings()
            assert config.pipeline_version == "1.0.0"
            assert config.environment == "development"
            assert config.debug is False

    def test_api_settings_defaults(self):
        """RED: api_host '0.0.0.0', api_port 8000, secret_api_key placeholder, max_file_size_mb 500."""
        with patch.dict(os.environ, {}, clear=True):
            config = Settings()
            assert config.api_host == "0.0.0.0"
            assert config.api_port == 8000
            assert config.secret_api_key == "your_secret_api_key_here"
            assert config.max_file_size_mb == 500

    def test_redis_settings_defaults(self):
        """RED: redis_url 'redis://localhost:6379/0', redis_results_db 1, max_queue_depth 50."""
        with patch.dict(os.environ, {}, clear=True):
            config = Settings()
            assert config.redis_url == "redis://localhost:6379/0"
            assert config.redis_results_db == 1
            assert config.max_queue_depth == 50

    def test_asr_settings_defaults(self):
        """RED: whisper_model_variant 'erax-wow-turbo', beam_size 5, vad_filter True."""
        with patch.dict(os.environ, {}, clear=True):
            config = Settings()
            assert config.whisper_model_variant == "erax-wow-turbo"
            assert config.whisper_beam_size == 5
            assert config.whisper_vad_filter is True
            assert config.whisper_compute_type == "float16"
            assert config.whisper_device == "cuda"

    def test_llm_enhance_settings_defaults(self):
        """RED: llm_enhance model, gpu util 0.95, max_model_len 32768, sampling defaults are None."""
        with patch.dict(os.environ, {}, clear=True):
            config = Settings()
            assert config.llm_enhance_model == "data/models/qwen3-4b-instruct-2507-awq"
            assert config.llm_enhance_gpu_memory_utilization == 0.95
            assert config.llm_enhance_max_model_len == 32768
            assert config.llm_enhance_temperature == 0.0
            assert config.llm_enhance_top_p is None
            assert config.llm_enhance_top_k is None
            assert config.llm_enhance_max_tokens is None

    def test_llm_sum_settings_defaults(self):
        """RED: summarization model defaults, temperature 0.7, sampling defaults are None."""
        with patch.dict(os.environ, {}, clear=True):
            config = Settings()
            assert config.llm_sum_model == "cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit"
            assert config.llm_sum_gpu_memory_utilization == 0.95
            assert config.llm_sum_max_model_len == 32768
            assert config.llm_sum_temperature == 0.7
            assert config.llm_sum_top_p is None
            assert config.llm_sum_top_k is None
            assert config.llm_sum_max_tokens is None

    def test_file_path_defaults(self):
        """RED: default paths set for audio, models, templates, chat-templates."""
        with patch.dict(os.environ, {}, clear=True):
            config = Settings()
            assert config.audio_dir == Path("data/audio")
            assert config.models_dir == Path("data/models")
            assert config.templates_dir == Path("templates")

    def test_worker_settings_defaults(self):
        """RED: worker_name 'maie-worker', job_timeout 600, result_ttl 86400."""
        with patch.dict(os.environ, {}, clear=True):
            config = Settings()
            assert config.worker_name == "maie-worker"
            assert config.job_timeout == 600
            assert config.result_ttl == 86400


class TestEnvironmentVariableLoading:
    """Environment variable parsing for Settings."""

    def test_core_settings_from_env(self):
        """RED: DEBUG 'true' -> True; version and environment parsed."""
        env_vars = {
            "PIPELINE_VERSION": "2.0.0",
            "ENVIRONMENT": "production",
            "DEBUG": "true",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = Settings()
            assert config.pipeline_version == "2.0.0"
            assert config.environment == "production"
            assert config.debug is True

    def test_api_settings_from_env(self):
        """RED: API_PORT and MAX_FILE_SIZE_MB cast to int; values match env."""
        env_vars = {
            "API_HOST": "127.0.0.1",
            "API_PORT": "9000",
            "SECRET_API_KEY": "test_secret_key",
            "MAX_FILE_SIZE_MB": "1000",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = Settings()
            assert config.api_host == "127.0.0.1"
            assert config.api_port == 9000
            assert config.secret_api_key == "test_secret_key"
            assert config.max_file_size_mb == 1000

    def test_redis_settings_from_env(self):
        """RED: redis env strings parsed to expected types and values."""
        env_vars = {
            "REDIS_URL": "redis://redis.example.com:6380/1",
            "REDIS_RESULTS_DB": "2",
            "MAX_QUEUE_DEPTH": "100",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = Settings()
            assert config.redis_url == "redis://redis.example.com:6380/1"
            assert config.redis_results_db == 2
            assert config.max_queue_depth == 100

    def test_file_paths_from_env(self):
        """RED: path env vars converted to Path and assigned."""
        env_vars = {
            "AUDIO_DIR": "/tmp/audio",
            "MODELS_DIR": "/tmp/models",
            "TEMPLATES_DIR": "/tmp/templates",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = Settings()
            assert config.audio_dir == Path("/tmp/audio")
            assert config.models_dir == Path("/tmp/models")
            assert config.templates_dir == Path("/tmp/templates")


class TestFieldValidators:
    """Validators related to filesystem fields."""

    def test_directory_creation_validator(self):
        """RED: Validators create missing directories and set paths accordingly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_audio_dir = Path(temp_dir) / "audio"
            test_models_dir = Path(temp_dir) / "models"
            test_templates_dir = Path(temp_dir) / "templates"

            env_vars = {
                "AUDIO_DIR": str(test_audio_dir),
                "MODELS_DIR": str(test_models_dir),
                "TEMPLATES_DIR": str(test_templates_dir),
            }

            with patch.dict(os.environ, env_vars, clear=True):
                assert not test_audio_dir.exists()
                assert not test_models_dir.exists()
                assert not test_templates_dir.exists()

                config = Settings()

                assert test_audio_dir.exists()
                assert test_models_dir.exists()
                assert test_templates_dir.exists()

                assert config.audio_dir == test_audio_dir
                assert config.models_dir == test_models_dir
                assert config.templates_dir == test_templates_dir

    def test_directory_creation_with_parents(self):
        """RED: mkdir(parents=True) should create nested parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nested" / "deep" / "audio"

            env_vars = {"AUDIO_DIR": str(nested_dir)}

            with patch.dict(os.environ, env_vars, clear=True):
                config = Settings()

                assert nested_dir.exists()
                assert nested_dir.parent.exists()
                assert nested_dir.parent.parent.exists()

    @patch("pathlib.Path.mkdir")
    def test_directory_creation_error_handling(self, mock_mkdir):
        """RED: propagation of PermissionError from mkdir is expected."""
        mock_mkdir.side_effect = PermissionError("Cannot create directory")

        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test"
            env_vars = {"AUDIO_DIR": str(test_dir)}

            with patch.dict(os.environ, env_vars, clear=True):
                with pytest.raises(PermissionError):
                    Settings()


class TestSettingsMethods:
    """Helpers on Settings return expected Paths."""

    def test_get_model_path(self):
        """RED: get_model_path('whisper') -> MODELS_DIR/whisper."""
        with patch.dict(os.environ, {"MODELS_DIR": "/test/models"}, clear=True):
            config = Settings()
            result = config.get_model_path("whisper")
            expected = Path("/test/models/whisper")
            assert result == expected

    def test_get_template_path(self):
        """RED: get_template_path('summary') -> TEMPLATES_DIR/summary.json."""
        with patch.dict(os.environ, {"TEMPLATES_DIR": "/test/templates"}, clear=True):
            config = Settings()
            result = config.get_template_path("summary")
            expected = Path("/test/templates/summary.json")
            assert result == expected

    def test_get_model_path_with_nested_model(self):
        """RED: nested model id appended correctly under MODELS_DIR."""
        with patch.dict(os.environ, {"MODELS_DIR": "/test/models"}, clear=True):
            config = Settings()
            result = config.get_model_path("asr/whisper/large")
            expected = Path("/test/models/asr/whisper/large")
            assert result == expected

    def test_get_template_path_with_complex_id(self):
        """RED: complex template id returns nested .json path."""
        with patch.dict(os.environ, {"TEMPLATES_DIR": "/test/templates"}, clear=True):
            config = Settings()
            result = config.get_template_path("v2/interview_transcript")
            expected = Path("/test/templates/v2/interview_transcript.json")
            assert result == expected


class TestEdgeCasesAndErrorConditions:
    """Validation error conditions for Settings."""

    def test_invalid_environment_value(self):
        """RED: invalid ENVIRONMENT raises ValueError."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_invalid_gpu_memory_utilization_too_low(self):
        """RED: too-low GPU utilization raises ValueError."""
        with patch.dict(
            os.environ, {"LLM_ENHANCE_GPU_MEMORY_UTILIZATION": "0.05"}, clear=True
        ):
            with pytest.raises(ValueError):
                Settings()

    def test_invalid_gpu_memory_utilization_too_high(self):
        """RED: too-high GPU utilization raises ValueError."""
        with patch.dict(
            os.environ, {"LLM_ENHANCE_GPU_MEMORY_UTILIZATION": "1.5"}, clear=True
        ):
            with pytest.raises(ValueError):
                Settings()

    def test_invalid_temperature_too_low(self):
        """RED: temperature below allowed range raises ValueError."""
        with patch.dict(os.environ, {"LLM_ENHANCE_TEMPERATURE": "-0.1"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_invalid_temperature_too_high(self):
        """RED: temperature above allowed range raises ValueError."""
        with patch.dict(os.environ, {"LLM_ENHANCE_TEMPERATURE": "2.1"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_invalid_top_p_too_low(self):
        """RED: top_p <0 raises ValueError."""
        with patch.dict(os.environ, {"LLM_ENHANCE_TOP_P": "-0.1"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_invalid_top_p_too_high(self):
        """RED: top_p >1 raises ValueError."""
        with patch.dict(os.environ, {"LLM_ENHANCE_TOP_P": "1.1"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_invalid_top_k_too_low(self):
        """RED: top_k <1 raises ValueError."""
        with patch.dict(os.environ, {"LLM_ENHANCE_TOP_K": "0"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_invalid_port_number(self):
        """RED: non-numeric API_PORT raises ValueError."""
        with patch.dict(os.environ, {"API_PORT": "invalid"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_negative_port_number(self):
        """RED: negative API_PORT raises ValueError."""
        with patch.dict(os.environ, {"API_PORT": "-1"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_large_port_number(self):
        """RED: API_PORT >65535 raises ValueError."""
        with patch.dict(os.environ, {"API_PORT": "65536"}, clear=True):
            with pytest.raises(ValueError):
                Settings()


class TestSmokeTests:
    """Basic smoke tests for Settings."""

    def test_settings_instantiation(self):
        """RED: Settings() constructs without error and returns instance."""
        config = Settings()
        assert config is not None
        assert isinstance(config, Settings)

    def test_global_settings_instance(self):
        """RED: module-level 'settings' is a Settings instance."""
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_settings_with_env_file(self):
        """RED: Settings() resilient to presence/absence of .env; does not crash."""
        try:
            config = Settings()
            assert config is not None
        except Exception as e:
            pytest.fail(f"Settings instantiation failed: {e}")

    def test_config_model_attributes(self):
        """RED: Settings exposes expected attributes across config sections."""
        config = Settings()

        # Core settings
        assert hasattr(config, "pipeline_version")
        assert hasattr(config, "environment")
        assert hasattr(config, "debug")

        # API settings
        assert hasattr(config, "api_host")
        assert hasattr(config, "api_port")
        assert hasattr(config, "secret_api_key")
        assert hasattr(config, "max_file_size_mb")

        # Redis settings
        assert hasattr(config, "redis_url")
        assert hasattr(config, "redis_results_db")
        assert hasattr(config, "max_queue_depth")

        # ASR settings
        assert hasattr(config, "whisper_model_variant")
        assert hasattr(config, "whisper_beam_size")
        assert hasattr(config, "whisper_vad_filter")
        assert hasattr(config, "whisper_compute_type")

        # LLM settings
        assert hasattr(config, "llm_enhance_model")
        assert hasattr(config, "llm_sum_model")

        # File paths
        assert hasattr(config, "audio_dir")
        assert hasattr(config, "models_dir")
        assert hasattr(config, "templates_dir")

        # Worker settings
        assert hasattr(config, "worker_name")
        assert hasattr(config, "job_timeout")
        assert hasattr(config, "result_ttl")


class TestPydanticSettingsBestPractices:
    """Test Pydantic Settings v2 best practices."""

    def test_settings_config_has_validate_default_true(self):
        """RED: SettingsConfigDict should have validate_default=True for early error detection."""
        config = Settings.model_config
        assert "validate_default" in config or hasattr(config, "validate_default")
        # If present as dict or attribute, it should be True
        if isinstance(config, dict):
            assert config.get("validate_default", False) is True
        else:
            assert getattr(config, "validate_default", False) is True

    def test_settings_config_has_env_nested_delimiter(self):
        """RED: SettingsConfigDict should have env_nested_delimiter for nested config support."""
        config = Settings.model_config
        # Check for env_nested_delimiter (commonly '__' for nested env vars)
        if isinstance(config, dict):
            assert "env_nested_delimiter" in config
            assert config.get("env_nested_delimiter") == "__"
        else:
            assert hasattr(config, "env_nested_delimiter")
            assert getattr(config, "env_nested_delimiter") == "__"

    def test_settings_validates_defaults_on_instantiation(self):
        """RED: When validate_default=True, invalid defaults should raise errors immediately."""
        # This test verifies that default values are validated
        # If a default violates constraints, it should fail on instantiation
        with patch.dict(os.environ, {}, clear=True):
            try:
                config = Settings()
                # If we get here, all defaults are valid
                assert config is not None
            except ValueError:
                pytest.fail(
                    "Default values should be valid, but validation raised ValueError"
                )

    def test_nested_env_vars_with_delimiter(self):
        """RED: env_nested_delimiter='__' allows nested configuration like REDIS__POOL__SIZE."""
        # Example: REDIS__POOL__SIZE should map to redis.pool.size
        # For now, test that delimiter is configured
        config = Settings.model_config
        if isinstance(config, dict):
            delimiter = config.get("env_nested_delimiter")
        else:
            delimiter = getattr(config, "env_nested_delimiter", None)

        assert delimiter == "__", f"Expected '__' but got {delimiter}"

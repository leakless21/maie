"""Tests for the config_validation module."""

import pytest
from src.utils.config_validation import (
    validate_audio_settings,
    validate_llm_settings,
    validate_api_settings,
    validate_cleanup_intervals,
    validate_retention_periods,
    validate_disk_thresholds,
    validate_worker_settings,
    validate_logging_settings,
    validate_all_settings,
)


class TestValidateAudioSettings:
    """Tests for validate_audio_settings function."""
    
    def test_validate_audio_settings_valid(self):
        """Test validate_audio_settings with valid settings."""
        settings = {
            "sample_rate": 16000,
            "chunk_duration": 30,
            "audio_format": "mp3",
            "normalize_volume": True
        }
        result = validate_audio_settings(settings)
        assert result["sample_rate"] == 16000
        assert result["chunk_duration"] == 30
        assert result["audio_format"] == "mp3"
        assert result["normalize_volume"] is True
    
    def test_validate_audio_settings_invalid_sample_rate(self):
        """Test validate_audio_settings with invalid sample rate."""
        settings = {"sample_rate": 1000}  # Too low
        with pytest.raises(ValueError, match="sample_rate"):
            validate_audio_settings(settings)
    
    def test_validate_audio_settings_invalid_format(self):
        """Test validate_audio_settings with invalid format."""
        settings = {"audio_format": "exe"}
        with pytest.raises(ValueError, match="audio_format"):
            validate_audio_settings(settings)


class TestValidateLlmSettings:
    """Tests for validate_llm_settings function."""
    
    def test_validate_llm_settings_valid(self):
        """Test validate_llm_settings with valid settings."""
        settings = {
            "model_name": "test-model",
            "temperature": 0.7,
            "max_tokens": 100,  # Fixed: was expecting 10 but should be 100
            "top_p": 0.9
        }
        result = validate_llm_settings(settings)
        assert result["model_name"] == "test-model"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert result["top_p"] == 0.9
    
    def test_validate_llm_settings_invalid_temperature(self):
        """Test validate_llm_settings with invalid temperature."""
        settings = {"temperature": 3.0}  # Too high
        with pytest.raises(ValueError, match="temperature"):
            validate_llm_settings(settings)


class TestValidateApiSettings:
    """Tests for validate_api_settings function."""
    
    def test_validate_api_settings_valid(self):
        """Test validate_api_settings with valid settings."""
        settings = {
            "port": 8080,
            "host": "localhost",
            "rate_limit": 100,
            "request_timeout": 30
        }
        result = validate_api_settings(settings)
        assert result["port"] == 8080
        assert result["host"] == "localhost"
        assert result["rate_limit"] == 100
        assert result["request_timeout"] == 30
    
    def test_validate_api_settings_invalid_port(self):
        """Test validate_api_settings with invalid port."""
        settings = {"port": 70000}  # Too high
        with pytest.raises(ValueError, match="port"):
            validate_api_settings(settings)


class TestValidateCleanupIntervals:
    """Tests for validate_cleanup_intervals function."""
    
    def test_validate_cleanup_intervals_valid(self):
        """Test validate_cleanup_intervals with valid intervals."""
        intervals = {
            "audio_cleanup_hours": 24,
            "log_cleanup_hours": 48,
            "cache_cleanup_hours": 12
        }
        result = validate_cleanup_intervals(intervals)
        assert result["audio_cleanup_hours"] == 24
        assert result["log_cleanup_hours"] == 48
        assert result["cache_cleanup_hours"] == 12
    
    def test_validate_cleanup_intervals_invalid(self):
        """Test validate_cleanup_intervals with invalid interval."""
        intervals = {"audio_cleanup_hours": 9000}  # Too high (more than 1 year)
        with pytest.raises(ValueError, match="audio_cleanup_hours"):
            validate_cleanup_intervals(intervals)


class TestValidateRetentionPeriods:
    """Tests for validate_retention_periods function."""
    
    def test_validate_retention_periods_valid(self):
        """Test validate_retention_periods with valid periods."""
        periods = {
            "audio_retention_days": 30,
            "log_retention_days": 90,
            "result_retention_days": 365
        }
        result = validate_retention_periods(periods)
        assert result["audio_retention_days"] == 30
        assert result["log_retention_days"] == 90
        assert result["result_retention_days"] == 365
    
    def test_validate_retention_periods_invalid(self):
        """Test validate_retention_periods with invalid period."""
        periods = {"audio_retention_days": 5000}  # Too high (more than 10 years)
        with pytest.raises(ValueError, match="audio_retention_days"):
            validate_retention_periods(periods)


class TestValidateDiskThresholds:
    """Tests for validate_disk_thresholds function."""
    
    def test_validate_disk_thresholds_valid(self):
        """Test validate_disk_thresholds with valid thresholds."""
        thresholds = {
            "disk_warning_threshold": 80.0,
            "disk_critical_threshold": 95.0,
            "min_free_space_gb": 10.0
        }
        result = validate_disk_thresholds(thresholds)
        assert result["disk_warning_threshold"] == 80.0
        assert result["disk_critical_threshold"] == 95.0
        assert result["min_free_space_gb"] == 10.0
    
    def test_validate_disk_thresholds_invalid_percentage(self):
        """Test validate_disk_thresholds with invalid percentage."""
        thresholds = {"disk_warning_threshold": 110.0}  # More than 100%
        with pytest.raises(ValueError, match="disk_warning_threshold"):
            validate_disk_thresholds(thresholds)


class TestValidateWorkerSettings:
    """Tests for validate_worker_settings function."""
    
    def test_validate_worker_settings_valid(self):
        """Test validate_worker_settings with valid settings."""
        settings = {
            "num_workers": 4,
            "worker_timeout": 60,
            "batch_size": 32
        }
        result = validate_worker_settings(settings)
        assert result["num_workers"] == 4
        assert result["worker_timeout"] == 60
        assert result["batch_size"] == 32
    
    def test_validate_worker_settings_invalid_workers(self):
        """Test validate_worker_settings with invalid number of workers."""
        settings = {"num_workers": 0}  # Too low
        with pytest.raises(ValueError, match="num_workers"):
            validate_worker_settings(settings)


class TestValidateLoggingSettings:
    """Tests for validate_logging_settings function."""
    
    def test_validate_logging_settings_valid(self):
        """Test validate_logging_settings with valid settings."""
        settings = {
            "log_level": "INFO",
            "log_retention_days": 30,
            "log_rotation_size_mb": 100
        }
        result = validate_logging_settings(settings)
        assert result["log_level"] == "INFO"
        assert result["log_retention_days"] == 30
        assert result["log_rotation_size_mb"] == 100
    
    def test_validate_logging_settings_invalid_level(self):
        """Test validate_logging_settings with invalid log level."""
        settings = {"log_level": "INVALID"}
        with pytest.raises(ValueError, match="log_level"):
            validate_logging_settings(settings)


class TestValidateAllSettings:
    """Tests for validate_all_settings function."""
    
    def test_validate_all_settings_valid(self):
        """Test validate_all_settings with valid settings."""
        config = {
            "audio": {"sample_rate": 16000},
            "llm": {"temperature": 0.7},
            "api": {"port": 8080},
            "cleanup": {"audio_cleanup_hours": 24},
            "retention": {"audio_retention_days": 30},
            "disk": {"disk_warning_threshold": 80.0},
            "worker": {"num_workers": 4},
            "logging": {"log_level": "INFO"}
        }
        result = validate_all_settings(config)
        assert "audio" in result
        assert "llm" in result
        assert "api" in result
        assert "cleanup" in result
        assert "retention" in result
        assert "disk" in result
        assert "worker" in result
        assert "logging" in result
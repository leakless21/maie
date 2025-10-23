"""Tests for cleanup configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.model import CleanupSettings


class TestCleanupSettingsValidation:
    """Tests for CleanupSettings validation."""

    def test_valid_cleanup_settings(self):
        """Test valid cleanup settings."""
        settings = CleanupSettings(
            audio_cleanup_interval=3600,
            log_cleanup_interval=86400,
            cache_cleanup_interval=1800,
            disk_monitor_interval=300,
            audio_retention_days=7,
            logs_retention_days=7,
            disk_threshold_pct=80,
            emergency_cleanup=False,
            check_dir="."
        )
        
        assert settings.audio_cleanup_interval == 3600
        assert settings.disk_threshold_pct == 80

    def test_cleanup_interval_too_small(self):
        """Test that cleanup intervals must be at least 10 seconds."""
        with pytest.raises(ValidationError) as exc_info:
            CleanupSettings(audio_cleanup_interval=5)
        
        assert "at least 10 seconds" in str(exc_info.value)

    def test_cleanup_interval_too_large(self):
        """Test that cleanup intervals should not exceed 1 day."""
        with pytest.raises(ValidationError) as exc_info:
            CleanupSettings(audio_cleanup_interval=100000)
        
        assert "should not exceed 86400" in str(exc_info.value)

    def test_retention_days_negative(self):
        """Test that retention days cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            CleanupSettings(audio_retention_days=-1)
        
        assert "cannot be negative" in str(exc_info.value)

    def test_retention_days_too_large(self):
        """Test that retention days should not exceed 365."""
        with pytest.raises(ValidationError) as exc_info:
            CleanupSettings(audio_retention_days=400)
        
        assert "should not exceed 365" in str(exc_info.value)

    def test_disk_threshold_out_of_range(self):
        """Test that disk threshold must be 0-100."""
        with pytest.raises(ValidationError) as exc_info:
            CleanupSettings(disk_threshold_pct=150)
        
        assert "between 0 and 100" in str(exc_info.value)

    def test_disk_threshold_negative(self):
        """Test that disk threshold cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            CleanupSettings(disk_threshold_pct=-10)
        
        assert "between 0 and 100" in str(exc_info.value)

    def test_check_dir_empty_string(self):
        """Test that check_dir cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            CleanupSettings(check_dir="")
        
        assert "non-empty string" in str(exc_info.value)

    def test_all_intervals_valid_range(self):
        """Test various valid interval values."""
        # Minimum valid
        settings = CleanupSettings(
            audio_cleanup_interval=10,
            log_cleanup_interval=10,
            cache_cleanup_interval=10,
            disk_monitor_interval=10
        )
        assert settings.audio_cleanup_interval == 10

        # Maximum valid
        settings = CleanupSettings(
            audio_cleanup_interval=86400,
            log_cleanup_interval=86400,
            cache_cleanup_interval=86400,
            disk_monitor_interval=86400
        )
        assert settings.audio_cleanup_interval == 86400

    def test_all_retention_days_valid_range(self):
        """Test various valid retention day values."""
        # Zero days (immediate cleanup)
        settings = CleanupSettings(
            audio_retention_days=0,
            logs_retention_days=0
        )
        assert settings.audio_retention_days == 0

        # Maximum valid
        settings = CleanupSettings(
            audio_retention_days=365,
            logs_retention_days=365
        )
        assert settings.audio_retention_days == 365

    def test_disk_threshold_boundaries(self):
        """Test disk threshold boundary values."""
        # Minimum valid
        settings = CleanupSettings(disk_threshold_pct=0)
        assert settings.disk_threshold_pct == 0

        # Maximum valid
        settings = CleanupSettings(disk_threshold_pct=100)
        assert settings.disk_threshold_pct == 100

        # Common values
        settings = CleanupSettings(disk_threshold_pct=80)
        assert settings.disk_threshold_pct == 80

    def test_default_values(self):
        """Test that default values are reasonable."""
        settings = CleanupSettings()
        
        assert settings.audio_cleanup_interval == 3600  # 1 hour
        assert settings.log_cleanup_interval == 86400  # 24 hours
        assert settings.cache_cleanup_interval == 1800  # 30 minutes
        assert settings.disk_monitor_interval == 300  # 5 minutes
        assert settings.audio_retention_days == 7  # 7 days
        assert settings.logs_retention_days == 7  # 7 days
        assert settings.disk_threshold_pct == 80  # 80%
        assert settings.emergency_cleanup is False
        assert settings.check_dir == "."

    def test_emergency_cleanup_flag(self):
        """Test emergency_cleanup flag configuration."""
        settings_disabled = CleanupSettings(emergency_cleanup=False)
        assert settings_disabled.emergency_cleanup is False

        settings_enabled = CleanupSettings(emergency_cleanup=True)
        assert settings_enabled.emergency_cleanup is True

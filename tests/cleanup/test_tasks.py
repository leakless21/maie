"""Tests for cleanup tasks."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import redis


@pytest.fixture
def mock_settings():
    """Mock settings fixture."""
    settings = MagicMock()
    settings.paths.audio_dir = Path(tempfile.mkdtemp())
    settings.logging.log_dir = "logs"
    settings.redis.url = "redis://localhost:6379"
    settings.redis.results_db = 1
    settings.cleanup.logs_retention_days = 7
    settings.cleanup.disk_threshold_pct = 80
    settings.cleanup.check_dir = "."
    settings.cleanup.emergency_cleanup = False
    return settings


@pytest.fixture
def temp_audio_dir():
    """Create temporary audio directory."""
    base_dir = Path(tempfile.mkdtemp())
    yield base_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(base_dir, ignore_errors=True)


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory."""
    base_dir = Path(tempfile.mkdtemp())
    yield base_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(base_dir, ignore_errors=True)


class TestCleanupAudioFiles:
    """Tests for cleanup_audio_files function."""

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    def test_audio_cleanup_no_directory(self, mock_redis_from_url, mock_get_job):
        """Test audio cleanup when directory doesn't exist."""
        from src.cleanup.tasks import cleanup_audio_files

        mock_get_job.return_value = None

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.paths.audio_dir = Path("/nonexistent/directory")
            
            result = cleanup_audio_files(dry_run=False)
            
            assert result["checked"] == 0
            assert result["deleted"] == 0
            assert result["skipped"] == 0

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    def test_audio_cleanup_dry_run(self, mock_redis_from_url, mock_get_job, temp_audio_dir):
        """Test audio cleanup dry run mode."""
        from src.cleanup.tasks import cleanup_audio_files

        mock_get_job.return_value = None

        # Create test directories and files
        task_id = "test-task-123"
        task_dir = temp_audio_dir / task_id
        task_dir.mkdir(parents=True)
        preprocessed_file = task_dir / "preprocessed.wav"
        preprocessed_file.write_text("test audio data")

        # Mock Redis connection
        mock_redis_conn = MagicMock()
        mock_redis_conn.hget.return_value = b"COMPLETE"
        mock_redis_from_url.return_value = mock_redis_conn

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.paths.audio_dir = temp_audio_dir
            mock_settings.redis.url = "redis://localhost:6379"
            mock_settings.redis.results_db = 1

            result = cleanup_audio_files(dry_run=True)
            
            # In dry run, file should not be deleted
            assert preprocessed_file.exists()
            assert result["checked"] == 1
            assert result["deleted"] == 0

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    def test_audio_cleanup_deletes_complete_task_files(
        self, mock_redis_from_url, mock_get_job, temp_audio_dir
    ):
        """Test that audio cleanup deletes files for COMPLETE tasks."""
        from src.cleanup.tasks import cleanup_audio_files

        mock_get_job.return_value = None

        # Create test directories and files
        task_id = "test-task-456"
        task_dir = temp_audio_dir / task_id
        task_dir.mkdir(parents=True)
        preprocessed_file = task_dir / "preprocessed.wav"
        preprocessed_file.write_text("test audio data")

        # Mock Redis connection
        mock_redis_conn = MagicMock()
        mock_redis_conn.hget.return_value = b"COMPLETE"
        mock_redis_from_url.return_value = mock_redis_conn

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.paths.audio_dir = temp_audio_dir
            mock_settings.redis.url = "redis://localhost:6379"
            mock_settings.redis.results_db = 1

            result = cleanup_audio_files(dry_run=False)
            
            # File should be deleted
            assert not preprocessed_file.exists()
            assert result["checked"] == 1
            assert result["deleted"] == 1

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    def test_audio_cleanup_deletes_failed_task_files(
        self, mock_redis_from_url, mock_get_job, temp_audio_dir
    ):
        """Test that audio cleanup deletes files for FAILED tasks."""
        from src.cleanup.tasks import cleanup_audio_files

        mock_get_job.return_value = None

        # Create test directories and files
        task_id = "test-task-789"
        task_dir = temp_audio_dir / task_id
        task_dir.mkdir(parents=True)
        preprocessed_file = task_dir / "preprocessed.wav"
        preprocessed_file.write_text("test audio data")

        # Mock Redis connection
        mock_redis_conn = MagicMock()
        mock_redis_conn.hget.return_value = b"FAILED"
        mock_redis_from_url.return_value = mock_redis_conn

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.paths.audio_dir = temp_audio_dir
            mock_settings.redis.url = "redis://localhost:6379"
            mock_settings.redis.results_db = 1

            result = cleanup_audio_files(dry_run=False)
            
            # File should be deleted
            assert not preprocessed_file.exists()
            assert result["deleted"] == 1

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    def test_audio_cleanup_skips_pending_task_files(
        self, mock_redis_from_url, mock_get_job, temp_audio_dir
    ):
        """Test that audio cleanup skips files for PENDING tasks."""
        from src.cleanup.tasks import cleanup_audio_files

        mock_get_job.return_value = None

        # Create test directories and files
        task_id = "test-task-pending"
        task_dir = temp_audio_dir / task_id
        task_dir.mkdir(parents=True)
        preprocessed_file = task_dir / "preprocessed.wav"
        preprocessed_file.write_text("test audio data")

        # Mock Redis connection
        mock_redis_conn = MagicMock()
        mock_redis_conn.hget.return_value = b"PENDING"
        mock_redis_from_url.return_value = mock_redis_conn

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.paths.audio_dir = temp_audio_dir
            mock_settings.redis.url = "redis://localhost:6379"
            mock_settings.redis.results_db = 1

            result = cleanup_audio_files(dry_run=False)
            
            # File should NOT be deleted
            assert preprocessed_file.exists()
            assert result["deleted"] == 0
            assert result["skipped"] == 1

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    def test_audio_cleanup_redis_error(
        self, mock_redis_from_url, mock_get_job, temp_audio_dir
    ):
        """Test audio cleanup handles Redis connection errors."""
        from src.cleanup.tasks import cleanup_audio_files

        mock_get_job.return_value = None

        # Mock Redis connection that fails
        mock_redis_conn = MagicMock()
        mock_redis_conn.ping.side_effect = redis.ConnectionError("Connection failed")
        mock_redis_from_url.return_value = mock_redis_conn

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.paths.audio_dir = temp_audio_dir
            mock_settings.redis.url = "redis://localhost:6379"
            mock_settings.redis.results_db = 1

            result = cleanup_audio_files(dry_run=False)
            
            assert "error" in result
            assert result["error"] == "redis_unavailable"


class TestCleanupLogs:
    """Tests for cleanup_logs function."""

    @patch("src.cleanup.tasks.get_current_job")
    def test_log_cleanup_no_directory(self, mock_get_job):
        """Test log cleanup when directory doesn't exist."""
        from src.cleanup.tasks import cleanup_logs

        mock_get_job.return_value = None

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.logging.log_dir = "/nonexistent/logs"
            mock_settings.cleanup.logs_retention_days = 7

            result = cleanup_logs(dry_run=False)
            
            assert result["checked"] == 0
            assert result["deleted"] == 0

    @patch("src.cleanup.tasks.get_current_job")
    def test_log_cleanup_dry_run(self, mock_get_job, temp_log_dir):
        """Test log cleanup dry run mode."""
        from src.cleanup.tasks import cleanup_logs

        mock_get_job.return_value = None

        # Create old log file (10 days old)
        old_log = temp_log_dir / "app.log"
        old_log.write_text("old log data")
        old_time = time.time() - (10 * 24 * 3600)
        Path(old_log).touch()
        import os
        os.utime(old_log, (old_time, old_time))

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.logging.log_dir = temp_log_dir
            mock_settings.cleanup.logs_retention_days = 7

            result = cleanup_logs(dry_run=True)
            
            # In dry run, file should not be deleted
            assert old_log.exists()
            assert result["deleted"] == 0

    @patch("src.cleanup.tasks.get_current_job")
    def test_log_cleanup_deletes_old_files(self, mock_get_job, temp_log_dir):
        """Test that log cleanup deletes old log files."""
        from src.cleanup.tasks import cleanup_logs

        mock_get_job.return_value = None

        # Create old log file (10 days old)
        old_log = temp_log_dir / "app.log.1"
        old_log.write_text("old log data")
        old_time = time.time() - (10 * 24 * 3600)
        import os
        os.utime(old_log, (old_time, old_time))

        # Create recent log file (2 days old)
        recent_log = temp_log_dir / "app.log"
        recent_log.write_text("recent log data")
        recent_time = time.time() - (2 * 24 * 3600)
        os.utime(recent_log, (recent_time, recent_time))

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.logging.log_dir = temp_log_dir
            mock_settings.cleanup.logs_retention_days = 7

            result = cleanup_logs(dry_run=False)
            
            # Old file should be deleted, recent file should remain
            assert not old_log.exists()
            assert recent_log.exists()
            assert result["deleted"] == 1

    @patch("src.cleanup.tasks.get_current_job")
    def test_log_cleanup_calculates_space_freed(self, mock_get_job, temp_log_dir):
        """Test that log cleanup calculates space freed correctly."""
        from src.cleanup.tasks import cleanup_logs

        mock_get_job.return_value = None

        # Create old log file with known size
        old_log = temp_log_dir / "app.log.old"
        old_log.write_text("a" * 1024 * 100)  # 100 KB
        old_time = time.time() - (10 * 24 * 3600)
        import os
        os.utime(old_log, (old_time, old_time))

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.logging.log_dir = temp_log_dir
            mock_settings.cleanup.logs_retention_days = 7

            result = cleanup_logs(dry_run=False)
            
            assert result["deleted"] == 1
            assert "space_freed_mb" in result
            # Should be approximately 0.1 MB
            assert 0.09 < result["space_freed_mb"] < 0.11


class TestCleanupCache:
    """Tests for cleanup_cache function."""

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    def test_cache_cleanup_redis_error(self, mock_redis_from_url, mock_get_job):
        """Test cache cleanup handles Redis connection errors."""
        from src.cleanup.tasks import cleanup_cache

        mock_get_job.return_value = None

        # Mock Redis connection that fails
        mock_redis_conn = MagicMock()
        mock_redis_conn.ping.side_effect = redis.ConnectionError("Connection failed")
        mock_redis_from_url.return_value = mock_redis_conn

        result = cleanup_cache(dry_run=False)
        
        assert "error" in result
        assert result["error"] == "redis_unavailable"

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    def test_cache_cleanup_basic(self, mock_redis_from_url, mock_get_job):
        """Test basic cache cleanup functionality."""
        from src.cleanup.tasks import cleanup_cache

        mock_get_job.return_value = None

        # Mock Redis connections
        mock_redis_conn = MagicMock()
        mock_redis_conn.ping.return_value = True
        mock_redis_conn.dbsize.return_value = 100
        mock_redis_conn.scan_iter.return_value = []
        mock_redis_from_url.return_value = mock_redis_conn

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.redis.url = "redis://localhost:6379"
            mock_settings.redis.results_db = 1

            result = cleanup_cache(dry_run=False)
            
            assert "error" not in result
            assert result["rq_jobs_cleaned"] == 0
            assert "results_db_keys_before" in result
            assert "queue_db_keys_before" in result

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    @patch("src.cleanup.tasks.time")
    def test_cache_cleanup_deletes_old_jobs(
        self, mock_time_module, mock_redis_from_url, mock_get_job
    ):
        """Test cache cleanup deletes RQ jobs older than 24 hours."""
        from src.cleanup.tasks import cleanup_cache

        mock_get_job.return_value = None

        # Mock time module
        current_time = 1000000
        mock_time_module.time.return_value = current_time

        # Mock Redis connections
        mock_redis_conn = MagicMock()
        mock_redis_conn.ping.return_value = True
        mock_redis_conn.dbsize.return_value = 100

        # Mock old job (25 hours old)
        old_job_key = b"rq:job:old-job-123"
        old_job_time = current_time - (25 * 3600)
        old_job_data = {
            b"created_at": str(old_job_time).encode(),
            b"status": b"finished"
        }

        # Mock scan_iter to return our test job
        mock_redis_conn.scan_iter.return_value = [old_job_key]
        mock_redis_conn.hgetall.return_value = old_job_data
        mock_redis_from_url.return_value = mock_redis_conn

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.redis.url = "redis://localhost:6379"
            mock_settings.redis.results_db = 1

            result = cleanup_cache(dry_run=False)
            
            # Old job should be deleted
            assert result["rq_jobs_cleaned"] == 1
            mock_redis_conn.delete.assert_called_once()

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.redis.from_url")
    @patch("src.cleanup.tasks.time")
    def test_cache_cleanup_keeps_recent_jobs(
        self, mock_time_module, mock_redis_from_url, mock_get_job
    ):
        """Test cache cleanup keeps RQ jobs less than 24 hours old."""
        from src.cleanup.tasks import cleanup_cache

        mock_get_job.return_value = None

        # Mock time module
        current_time = 1000000
        mock_time_module.time.return_value = current_time

        # Mock Redis connections
        mock_redis_conn = MagicMock()
        mock_redis_conn.ping.return_value = True
        mock_redis_conn.dbsize.return_value = 100

        # Mock recent job (2 hours old)
        recent_job_key = b"rq:job:recent-job-456"
        recent_job_time = current_time - (2 * 3600)
        recent_job_data = {
            b"created_at": str(recent_job_time).encode(),
            b"status": b"finished"
        }

        # Mock scan_iter to return our test job
        mock_redis_conn.scan_iter.return_value = [recent_job_key]
        mock_redis_conn.hgetall.return_value = recent_job_data
        mock_redis_from_url.return_value = mock_redis_conn

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.redis.url = "redis://localhost:6379"
            mock_settings.redis.results_db = 1

            result = cleanup_cache(dry_run=False)
            
            # Recent job should NOT be deleted
            assert result["rq_jobs_cleaned"] == 0
            mock_redis_conn.delete.assert_not_called()


class TestDiskMonitor:
    """Tests for disk_monitor function."""

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.shutil.disk_usage")
    def test_disk_monitor_within_limits(self, mock_disk_usage, mock_get_job):
        """Test disk monitor when usage is within limits."""
        from src.cleanup.tasks import disk_monitor

        mock_get_job.return_value = None

        # Mock disk usage: 60% used
        mock_usage = MagicMock()
        mock_usage.used = 600
        mock_usage.total = 1000
        mock_disk_usage.return_value = mock_usage

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.cleanup.check_dir = "."
            mock_settings.cleanup.disk_threshold_pct = 80
            mock_settings.cleanup.emergency_cleanup = False

            result = disk_monitor()
            
            assert result["usage_percent"] == 60.0
            assert "alert" not in result or result.get("alert") is False

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.shutil.disk_usage")
    def test_disk_monitor_exceeds_threshold(self, mock_disk_usage, mock_get_job):
        """Test disk monitor when usage exceeds threshold."""
        from src.cleanup.tasks import disk_monitor

        mock_get_job.return_value = None

        # Mock disk usage: 85% used
        mock_usage = MagicMock()
        mock_usage.used = 850
        mock_usage.total = 1000
        mock_disk_usage.return_value = mock_usage

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.cleanup.check_dir = "."
            mock_settings.cleanup.disk_threshold_pct = 80
            mock_settings.cleanup.emergency_cleanup = False

            result = disk_monitor()
            
            assert result["usage_percent"] == 85.0
            assert result.get("alert") is True

    @patch("src.cleanup.tasks.get_current_job")
    @patch("src.cleanup.tasks.cleanup_audio_files")
    @patch("src.cleanup.tasks.cleanup_logs")
    @patch("src.cleanup.tasks.shutil.disk_usage")
    def test_disk_monitor_emergency_cleanup(
        self, mock_disk_usage, mock_cleanup_logs, mock_cleanup_audio, mock_get_job
    ):
        """Test disk monitor triggers emergency cleanup when threshold exceeded."""
        from src.cleanup.tasks import disk_monitor

        mock_get_job.return_value = None

        # Mock disk usage before cleanup: 90% used
        mock_usage_before = MagicMock()
        mock_usage_before.used = 900
        mock_usage_before.total = 1000

        # Mock disk usage after cleanup: 75% used
        mock_usage_after = MagicMock()
        mock_usage_after.used = 750
        mock_usage_after.total = 1000

        mock_disk_usage.side_effect = [mock_usage_before, mock_usage_after]

        mock_cleanup_audio.return_value = {"deleted": 10}
        mock_cleanup_logs.return_value = {"deleted": 5}

        with patch("src.config.loader.settings") as mock_settings:
            mock_settings.cleanup.check_dir = "."
            mock_settings.cleanup.disk_threshold_pct = 80
            mock_settings.cleanup.emergency_cleanup = True

            result = disk_monitor()
            
            assert result.get("alert") is True
            assert "emergency_cleanup" in result
            assert mock_cleanup_audio.called
            assert mock_cleanup_logs.called
            assert result["cleanup_effective"] is True

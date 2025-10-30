"""
Unit tests for RQ Scheduler functionality.

Tests scheduler creation, job scheduling, and main loop execution.
Uses mocks for fast execution without requiring Redis.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fakeredis import FakeStrictRedis


class TestScheduler:
    """Test RQ Scheduler setup and job scheduling."""

    @pytest.fixture
    def fake_redis(self):
        """Provide a fake Redis client for testing."""
        client = FakeStrictRedis(decode_responses=False)
        yield client
        client.close()

    @pytest.fixture
    def mock_settings(self):
        """Mock application settings for testing."""
        mock_settings = MagicMock()
        mock_settings.redis.url = "redis://localhost:6379"
        mock_settings.redis.results_db = 1
        mock_settings.cleanup.audio_cleanup_interval = 3600  # 1 hour
        mock_settings.cleanup.log_cleanup_interval = 86400  # 1 day
        mock_settings.cleanup.cache_cleanup_interval = 1800  # 30 minutes
        mock_settings.cleanup.disk_monitor_interval = 300  # 5 minutes
        return mock_settings

    @pytest.fixture
    def mock_cleanup_tasks(self):
        """Mock cleanup task functions."""
        with patch("src.scheduler.main.cleanup_audio_files") as mock_audio, \
             patch("src.scheduler.main.cleanup_logs") as mock_logs, \
             patch("src.scheduler.main.cleanup_cache") as mock_cache, \
             patch("src.scheduler.main.disk_monitor") as mock_disk:
            yield {
                "cleanup_audio_files": mock_audio,
                "cleanup_logs": mock_logs,
                "cleanup_cache": mock_cache,
                "disk_monitor": mock_disk,
            }

    def test_create_scheduler(self, fake_redis, mock_settings):
        """Test scheduler creation with proper configuration."""
        from src.scheduler.main import create_scheduler

        with patch("src.scheduler.main.settings", mock_settings), \
             patch("src.scheduler.main.redis.from_url", return_value=fake_redis):

            scheduler = create_scheduler()

            # Verify scheduler was created with correct parameters
            assert scheduler is not None
            assert scheduler.connection is fake_redis
            assert scheduler.queue_name == "default"
            assert scheduler._interval == 60

    def test_schedule_cleanup_jobs_calls_scheduler_with_datetime(self, fake_redis, mock_settings, mock_cleanup_tasks):
        """Test that schedule_cleanup_jobs calls scheduler.schedule with datetime, not None."""
        from src.scheduler.main import create_scheduler, schedule_cleanup_jobs

        with patch("src.scheduler.main.settings", mock_settings), \
             patch("src.scheduler.main.redis.from_url", return_value=fake_redis), \
             patch("src.config.loader.settings", mock_settings):  # Also patch the import inside the function

            scheduler = create_scheduler()

            # Mock the scheduler.schedule method to capture calls
            scheduler.schedule = MagicMock()

            schedule_cleanup_jobs(scheduler)

            # Verify scheduler.schedule was called 4 times (one for each cleanup job)
            assert scheduler.schedule.call_count == 4

            # Verify each call used a datetime for scheduled_time (not None)
            for call in scheduler.schedule.call_args_list:
                args, kwargs = call
                # scheduled_time should be first positional arg or in kwargs
                scheduled_time = kwargs.get("scheduled_time") or (args[0] if args else None)
                assert isinstance(scheduled_time, datetime), f"Expected datetime, got {type(scheduled_time)}: {scheduled_time}"
                assert scheduled_time.tzinfo is not None, "scheduled_time should be timezone-aware"

            # Verify the specific job configurations
            calls = scheduler.schedule.call_args_list

            # Audio cleanup - hourly
            audio_call = calls[0]
            assert audio_call[1]["func"] == mock_cleanup_tasks["cleanup_audio_files"]
            assert audio_call[1]["args"] == [False]  # Not dry run
            assert audio_call[1]["interval"] == 3600
            assert audio_call[1]["id"] == "cleanup_audio_files"
            assert audio_call[1]["queue_name"] == "cleanup"

            # Log cleanup - daily
            log_call = calls[1]
            assert log_call[1]["func"] == mock_cleanup_tasks["cleanup_logs"]
            assert log_call[1]["args"] == [False]
            assert log_call[1]["interval"] == 86400
            assert log_call[1]["id"] == "cleanup_logs"

            # Cache cleanup - every 30 minutes
            cache_call = calls[2]
            assert cache_call[1]["func"] == mock_cleanup_tasks["cleanup_cache"]
            assert cache_call[1]["args"] == [False]
            assert cache_call[1]["interval"] == 1800
            assert cache_call[1]["id"] == "cleanup_cache"

            # Disk monitor - every 5 minutes
            disk_call = calls[3]
            assert disk_call[1]["func"] == mock_cleanup_tasks["disk_monitor"]
            assert disk_call[1]["args"] == []
            assert disk_call[1]["interval"] == 300
            assert disk_call[1]["id"] == "disk_monitor"

    def test_run_scheduler_calls_scheduler_run(self, fake_redis, mock_settings, mock_cleanup_tasks):
        """Test that run_scheduler calls scheduler.run() to start the main loop."""
        from src.scheduler.main import run_scheduler

        with patch("src.scheduler.main.settings", mock_settings), \
             patch("src.scheduler.main.redis.from_url", return_value=fake_redis), \
             patch("src.scheduler.main.signal.signal"), \
             patch("src.scheduler.main.sys.exit") as mock_exit:

            # Mock scheduler to avoid actually running the loop
            mock_scheduler = MagicMock()
            # Don't set side_effect to KeyboardInterrupt, just mock the method
            mock_scheduler.run = MagicMock()

            with patch("src.scheduler.main.create_scheduler", return_value=mock_scheduler), \
                 patch("src.scheduler.main.schedule_cleanup_jobs"):

                # Mock the scheduler.run to raise SystemExit to simulate normal shutdown
                mock_scheduler.run.side_effect = SystemExit(0)

                # Expect SystemExit to be raised
                with pytest.raises(SystemExit):
                    run_scheduler()

                # Verify scheduler.run was called
                mock_scheduler.run.assert_called_once()

    def test_scheduler_shutdown_handler(self, fake_redis, mock_settings):
        """Test that signal handlers properly shut down the scheduler."""
        from src.scheduler.main import run_scheduler

        with patch("src.scheduler.main.settings", mock_settings), \
             patch("src.scheduler.main.redis.from_url", return_value=fake_redis), \
             patch("src.scheduler.main.signal.signal") as mock_signal, \
             patch("src.scheduler.main.sys.exit") as mock_exit:

            # Mock scheduler
            mock_scheduler = MagicMock()
            mock_scheduler.run = MagicMock(side_effect=SystemExit(0))  # Use SystemExit instead of KeyboardInterrupt

            with patch("src.scheduler.main.create_scheduler", return_value=mock_scheduler), \
                 patch("src.scheduler.main.schedule_cleanup_jobs"):

                # Expect SystemExit from scheduler.run
                with pytest.raises(SystemExit):
                    run_scheduler()

                # Verify signal handlers were registered
                assert mock_signal.call_count == 2  # SIGINT and SIGTERM

                # Get the signal handler function
                signal_handler = mock_signal.call_args_list[0][0][1]

                # Simulate calling the signal handler
                signal_handler(2, None)  # SIGINT

                # Verify scheduler shutdown methods were called
                mock_scheduler.register_death.assert_called_once()
                mock_scheduler.remove_lock.assert_called_once()
                mock_exit.assert_called_once_with(0)
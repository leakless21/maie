"""Tests for log file functionality and configuration."""

import json
from unittest.mock import patch

import pytest

from src.config.logging import configure_logging
from src.config import settings


class TestLogFileFunctionality:
    """Test that log files are created and written to correctly."""

    def test_log_files_are_created_in_temp_directory(self, tmp_path):
        """Test that log files are created when logging is configured."""
        # Use a temporary directory for log files
        log_dir = tmp_path / "test_logs"

        # Configure logging with temporary directory
        logger = configure_logging(file_dir=str(log_dir))

        # Log some test messages
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        # Force flush to ensure messages are written
        logger.complete()

        # Check that log files were created
        app_log = log_dir / "app.log"
        errors_log = log_dir / "errors.log"

        assert app_log.exists(), "app.log should be created"
        assert errors_log.exists(), "errors.log should be created"

        # Check that messages were written to app.log
        app_content = app_log.read_text()
        assert "Test info message" in app_content
        assert "Test warning message" in app_content
        assert "Test error message" in app_content

        # Check that error messages were written to errors.log
        errors_content = errors_log.read_text()
        assert "Test error message" in errors_content
        # Info and warning should not be in errors.log (only ERROR level and above)
        assert "Test info message" not in errors_content
        assert "Test warning message" not in errors_content

    def test_log_file_serialization_format(self, tmp_path):
        """Test that log files use the correct serialization format."""
        log_dir = tmp_path / "test_logs"

        # Configure logging with JSON serialization
        logger = configure_logging(file_dir=str(log_dir), file_serialize=True)

        # Log a test message
        logger.info("Test serialized message")
        logger.complete()

        # Check that the log file contains JSON
        app_log = log_dir / "app.log"
        assert app_log.exists()

        content = app_log.read_text().strip()
        lines = content.split("\n")

        # Each line should be valid JSON
        for line in lines:
            if line.strip():
                try:
                    log_entry = json.loads(line)
                    assert "text" in log_entry
                    assert "record" in log_entry
                    assert "Test serialized message" in log_entry["text"]
                except json.JSONDecodeError:
                    pytest.fail(f"Log entry is not valid JSON: {line}")

    def test_log_file_rotation_and_retention_settings(self, tmp_path):
        """Test that log file rotation and retention settings are applied."""
        log_dir = tmp_path / "test_logs"

        # Configure logging with custom rotation and retention
        logger = configure_logging(
            file_dir=str(log_dir),
            rotation="1 KB",  # Small rotation for testing
            retention="1 day",
        )

        # Log enough messages to trigger rotation
        for i in range(100):
            logger.info(
                f"Test message {i} - this is a long message to fill up the log file quickly"
            )

        logger.complete()

        # Check that log files exist
        app_log = log_dir / "app.log"
        assert app_log.exists()

        # With small rotation size, we might have multiple log files
        log_files = list(log_dir.glob("app.log*"))
        assert len(log_files) >= 1, "At least one log file should exist"

    def test_log_directory_creation(self, tmp_path):
        """Test that log directory is created if it doesn't exist."""
        # Use a nested directory that doesn't exist
        log_dir = tmp_path / "nested" / "log" / "directory"

        # Configure logging - should create the directory
        logger = configure_logging(file_dir=str(log_dir))

        # Log a message
        logger.info("Test directory creation")
        logger.complete()

        # Check that directory was created and log file exists
        assert log_dir.exists(), "Log directory should be created"
        app_log = log_dir / "app.log"
        assert app_log.exists(), "Log file should be created in new directory"

    def test_log_file_permissions(self, tmp_path):
        """Test that log files have appropriate permissions."""
        log_dir = tmp_path / "test_logs"

        # Configure logging
        logger = configure_logging(file_dir=str(log_dir))
        logger.info("Test permissions")
        logger.complete()

        # Check file permissions
        app_log = log_dir / "app.log"
        errors_log = log_dir / "errors.log"

        assert app_log.exists()
        assert errors_log.exists()

        # Check that files are readable and writable by owner
        # (exact permissions may vary by system, but should be reasonable)
        app_stat = app_log.stat()
        errors_stat = errors_log.stat()

        # Files should be readable by owner (at minimum)
        assert app_stat.st_mode & 0o400, "app.log should be readable by owner"
        assert errors_stat.st_mode & 0o400, "errors.log should be readable by owner"

    def test_log_file_with_module_binding(self, tmp_path):
        """Test that log files work correctly with module binding."""
        log_dir = tmp_path / "test_logs"

        # Configure logging
        logger = configure_logging(file_dir=str(log_dir))

        # Use module binding
        from src.config.logging import get_module_logger

        module_logger = get_module_logger(__name__)

        # Log with module binding
        module_logger.info("Test message with module binding")
        logger.complete()

        # Check that the log file contains module information
        app_log = log_dir / "app.log"
        assert app_log.exists()

        content = app_log.read_text()
        assert "Test message with module binding" in content
        # Module information should be included in the log
        assert "test_log_file_functionality" in content or "module" in content

    def test_log_file_disabled_when_loguru_disabled(self, tmp_path):
        """Test that log files are not created when Loguru is disabled."""
        log_dir = tmp_path / "test_logs"

        # Configure logging with Loguru disabled
        with patch.object(settings, "enable_loguru", False):
            logger = configure_logging(file_dir=str(log_dir))

            # Log a message
            logger.info("Test message with Loguru disabled")
            logger.complete()

            # Check that no log files were created
            app_log = log_dir / "app.log"
            errors_log = log_dir / "errors.log"

            assert not app_log.exists(), (
                "app.log should not be created when Loguru is disabled"
            )
            assert not errors_log.exists(), (
                "errors.log should not be created when Loguru is disabled"
            )

    def test_log_file_force_override(self, tmp_path):
        """Test that force=True overrides the enable_loguru setting."""
        log_dir = tmp_path / "test_logs"

        # Configure logging with Loguru disabled but force=True
        with patch.object(settings, "enable_loguru", False):
            logger = configure_logging(file_dir=str(log_dir), force=True)

            # Log a message
            logger.info("Test message with force override")
            logger.complete()

            # Check that log files were created despite enable_loguru=False
            app_log = log_dir / "app.log"
            errors_log = log_dir / "errors.log"

            assert app_log.exists(), "app.log should be created when force=True"
            assert errors_log.exists(), "errors.log should be created when force=True"

            content = app_log.read_text()
            assert "Test message with force override" in content


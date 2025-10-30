"""Tests for stdlib logging interception into Loguru."""

import logging
from pathlib import Path
from unittest.mock import patch

from src.config.logging import configure_logging


def test_stdlib_logging_is_intercepted(tmp_path: Path):
    """Test that stdlib logs reach Loguru sinks after configure_logging()."""
    from src.config import settings
    
    log_dir = tmp_path / "logs"
    original_log_dir = settings.logging.log_dir
    original_log_level = settings.logging.log_level

    try:
        # Mock settings to use tmp_path for logs
        with patch.object(settings.logging, "log_dir", log_dir):
            with patch.object(settings.logging, "log_level", "DEBUG"):
                logger = configure_logging()  # No parameter needed
                logging.getLogger("uvicorn").warning("interop warning")
                try:
                    logger.complete()
                except Exception:
                    # complete() may not exist on some logger references; ignore
                    pass
                content = (log_dir / "app.log").read_text()
                assert "interop warning" in content
    finally:
        # Restore original settings
        with patch.object(settings.logging, "log_dir", original_log_dir):
            with patch.object(settings.logging, "log_level", original_log_level):
                configure_logging()

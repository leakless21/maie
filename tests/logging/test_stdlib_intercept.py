"""Tests for stdlib logging interception into Loguru."""

import logging
from pathlib import Path
from unittest.mock import patch

from src.config.logging import configure_logging


def test_stdlib_logging_is_intercepted(tmp_path: Path):
    """Test that stdlib logs reach Loguru sinks after configure_logging()."""
    log_dir = tmp_path / "logs"
    
    # Mock settings to use tmp_path for logs
    with patch("src.config.logging.settings") as mock_settings:
        mock_settings.log_dir = str(log_dir)
        mock_settings.log_level = "DEBUG"
        mock_settings.loguru_enabled = True
        
        logger = configure_logging()  # No parameter needed
        logging.getLogger("uvicorn").warning("interop warning")
        try:
            logger.complete()
        except Exception:
            # complete() may not exist on some logger references; ignore
            pass
        content = (log_dir / "app.log").read_text()
        assert "interop warning" in content

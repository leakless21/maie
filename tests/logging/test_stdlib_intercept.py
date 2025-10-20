"""Tests for stdlib logging interception into Loguru."""

import logging
from pathlib import Path

from src.config.logging import configure_logging


def test_stdlib_logging_is_intercepted(tmp_path: Path):
    """Test that stdlib logs reach Loguru sinks after configure_logging()."""
    log_dir = tmp_path / "logs"
    logger = configure_logging(file_dir=str(log_dir))
    logging.getLogger("uvicorn").warning("interop warning")
    try:
        logger.complete()
    except Exception:
        # complete() may not exist on some logger references; ignore
        pass
    content = (log_dir / "app.log").read_text()
    assert "interop warning" in content

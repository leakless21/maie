"""
Simplified logging configuration for the MAIE project.

This module provides a clean, straightforward logging setup using Loguru
that focuses on error pinpointing and human-readable logs.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from loguru import logger as _loguru_logger

from .loader import settings

# Context variable for correlation IDs to track requests across components
correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

def generate_correlation_id(prefix: str = "") -> str:
    """Generate a UUID-based correlation id for request tracing."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:8]

def bind_correlation_id(value: Optional[str] = None) -> None:
    """Bind a correlation id to the current context."""
    if value is None:
        value = generate_correlation_id()
    correlation_id.set(value)

def clear_correlation_id() -> None:
    """Clear the correlation id from the current context."""
    correlation_id.set(None)

class InterceptHandler(logging.Handler):
    """Handler to intercept standard logging and forward to loguru."""
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Forward to loguru
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def intercept_stdlib_logging(level: int | str = logging.INFO) -> None:
    """Intercept standard library loggers to use loguru."""
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(
        level if isinstance(level, int)
        else getattr(logging, str(level).upper(), logging.INFO)
    )
    
    # Intercept common third-party loggers
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access", "vLLM", "asyncio"]:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

def get_logger():
    """Return the configured Loguru logger."""
    return _loguru_logger

def configure_logging() -> Any:
    """
    Apply a simplified logging configuration using Loguru.
    
    This configuration focuses on:
    - Clean, human-readable console output
    - UTC timestamps for consistency
    - Automatic correlation ID injection
    - Error-focused file logging
    """
    # Remove default handlers to avoid duplicates
    try:
        _loguru_logger.remove()
    except ValueError:
        pass

    # Get settings
    level = settings.log_level.upper()
    log_dir = Path(settings.log_dir)
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console format - clean and readable with correlation ID
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[module]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{extra[correlation_id]} | "
        "<level>{message}</level>"
    )

    # Add console handler with clean formatting
    _loguru_logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=settings.loguru_backtrace,
        diagnose=settings.loguru_diagnose,
        filter=lambda record: _inject_context(record)
    )

    # Add file handler for all logs with rotation
    _loguru_logger.add(
        log_dir / "app.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{extra[module]}:{function}:{line} | {extra[correlation_id]} | {message}"
        ),
        level=level,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression=settings.log_compression,
        backtrace=settings.loguru_backtrace,
        diagnose=False,
        filter=lambda record: _inject_context(record)
    )

    # Add file handler for errors only with extended retention and structured data
    _loguru_logger.add(
        log_dir / "errors.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{extra[module]}:{function}:{line} | {extra[correlation_id]} | {message} | {exception} | "
            "STRUCTURED: {extra}"
        ),
        level="ERROR",
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression=settings.log_compression,
        backtrace=True,
        diagnose=False,
        filter=lambda record: _inject_context(record)
    )

    # Intercept standard library loggers
    intercept_stdlib_logging(level=level)

    return _loguru_logger


def _inject_context(record):
    """
    Inject context information into log records.
    
    This function is used as a filter to add correlation IDs and module info
    to each log record.
    """
    # Inject correlation ID
    cid = correlation_id.get()
    record["extra"]["correlation_id"] = cid or "no-correlation-id"
    
    # Inject module info
    module_name = record["name"]
    if module_name.startswith("src."):
        module_name = module_name[4:]  # Remove 'src.' prefix
    elif module_name.startswith("tests."):
        module_name = module_name[6:]  # Remove 'tests.' prefix
    record["extra"]["module"] = module_name
    
    # Add basic ML context fields if available in the record
    # These would typically be added via logger.bind() in ML operations
    ml_context_fields = [
        "model_name",
        "inference_time_ms",
        "audio_duration_sec",
        "tokens_processed",
        "gpu_memory_mb",
        "task_id"
    ]
    
    # Check if any ML context fields are already bound to the logger
    for field in ml_context_fields:
        if field not in record["extra"]:
            # Set default values for missing ML context fields to ensure consistency
            record["extra"][field] = record["extra"].get(field, None)
    
    return True
def get_module_logger(module_name: str) -> Any:
    """
    Get a logger with module context for better debugging.
    
    Usage:
        logger = get_module_logger(__name__)
        logger.info("Processing started")
    """
    # Clean up module name for readability
    if module_name.startswith("src."):
        module_name = module_name[4:]  # Remove 'src.' prefix
    elif module_name.startswith("tests."):
        module_name = module_name[6:]  # Remove 'tests.' prefix
    
    return _loguru_logger.bind(module=module_name)

__all__ = [
    "get_logger",
    "configure_logging",
    "bind_correlation_id",
    "clear_correlation_id",
    "correlation_id",
    "generate_correlation_id",
    "get_module_logger",
]

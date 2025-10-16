"""
Loguru configuration helpers used across the application.

This module wires Loguru using centralized settings while remaining import-safe.
Entry points must call `configure_logging()` explicitly.
"""

from __future__ import annotations

import contextvars
import os
import re
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger as _loguru_logger  # type: ignore

from .settings import settings

correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def generate_correlation_id(prefix: str = "") -> str:
    """Generate a UUID-based correlation id. Caller may add prefixes for easier grouping."""
    return (prefix + "-") * bool(prefix) + uuid.uuid4().hex


def bind_correlation_id(value: Optional[str] = None) -> None:
    """
    Bind a correlation id to the current context.

    Example:
        bind_correlation_id(generate_correlation_id("req"))
        logger.info("processing")
    """
    if value is None:
        value = generate_correlation_id()
    correlation_id.set(value)


def clear_correlation_id() -> None:
    """Clear the correlation id from the current context."""
    correlation_id.set(None)


_SENSITIVE_FIELD_NAMES = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
}
_SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"(?i)(bearer\s+[A-Za-z0-9\-\._~+/]+=*)"),
    re.compile(r"(?i)(api[_-]?key[:=]\s*[A-Za-z0-9\-\._~+/]+=*)"),
    re.compile(r"[A-Za-z0-9-_]{32,}"),
    re.compile(r"(?i)password[:=]\s*\S+"),
    re.compile(r"(?i)secret[:=]\s*\S+"),
]


def _mask_value(value: Any) -> Any:
    """Return a masked version of value for simple primitives."""
    if value is None:
        return None
    try:
        string_value = str(value)
    except Exception:
        return "REDACTED"
    if len(string_value) <= 4:
        return "REDACTED"
    return string_value[:2] + ("*" * (len(string_value) - 4)) + string_value[-2:]


def _redact_text(text: str) -> str:
    """Redact sensitive value patterns inside text using defined regex patterns."""
    if not text:
        return text
    redacted = text
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        redacted = pattern.sub(lambda match: "REDACTED", redacted)
    return redacted


def _redact_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Redact sensitive keys/values from `extra` dictionary."""
    if not isinstance(extra, dict):
        return extra
    output: Dict[str, Any] = {}
    for key, value in extra.items():
        if key.lower() in _SENSITIVE_FIELD_NAMES:
            output[key] = "REDACTED"
        elif isinstance(value, dict):
            output[key] = _redact_extra(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            string_value = str(value) if value is not None else ""
            output[key] = (
                _mask_value(value)
                if len(string_value) < 200
                else _redact_text(string_value)
            )
        else:
            output[key] = f"<{type(value).__name__}>"
    return output


def _secure_opener(path: str, flags: int):
    """
    Open a file with secure permissions (0o600) to avoid accidental world-readable logs.

    This function matches the signature expected by ``loguru.logger.add(..., opener=...)``.
    """
    return os.open(path, flags, 0o600)


def _ensure_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def get_logger():
    """Return the Loguru logger (Loguru is mandatory)."""
    return _loguru_logger


def configure_logging(
    *,
    environment: Optional[str] = None,
    log_level: Optional[str] = None,
    file_dir: Optional[str] = None,
    rotation: Optional[str] = None,
    retention: Optional[str] = None,
    compression: Optional[str] = None,
    console_serialize: Optional[bool] = None,
    file_serialize: Optional[bool] = None,
    diagnose: Optional[bool] = None,
    backtrace: Optional[bool] = None,
    force: bool = False,
):
    """
    Apply a safe, environment-aware logging configuration using Loguru.

    Configuration source precedence:
    1) Explicit function arguments
    2) Centralized settings from `src.config.settings`
    """
    logger = get_logger()

    if not settings.enable_loguru and not force:
        return logger

    level = (log_level or settings.log_level).upper()
    dirpath = Path(file_dir) if file_dir is not None else settings.log_dir
    rot = rotation or settings.log_rotation
    ret = retention or settings.log_retention
    comp = compression or settings.log_compression
    console_ser = (
        console_serialize
        if console_serialize is not None
        else settings.log_console_serialize
    )
    file_ser = (
        file_serialize if file_serialize is not None else settings.log_file_serialize
    )
    diag = diagnose if diagnose is not None else settings.loguru_diagnose
    btrace = backtrace if backtrace is not None else settings.loguru_backtrace

    try:
        _loguru_logger.remove()
    except Exception:
        pass

    def _filter_and_redact(record: Dict[str, Any]) -> bool:
        """
        Mutate the record in place:
        - Inject `request_id` from context if not present in extra.
        - Redact sensitive keys and value patterns.
        Returns True to allow the record to be logged.
        """
        try:
            request_id_value = correlation_id.get()
            extra = record.get("extra", {})
            if request_id_value and "request_id" not in extra:
                extra["request_id"] = request_id_value
            record["extra"] = _redact_extra(extra)

            message = record.get("message")
            if isinstance(message, str) and message:
                record["message"] = _redact_text(message)
        except Exception:
            try:
                record["extra"] = {}
            except Exception:
                pass
        return True

    console_sink = sys.stdout
    try:
        if console_ser:
            _loguru_logger.add(
                console_sink,
                level=level,
                serialize=True,
                diagnose=diag,
                backtrace=btrace,
                filter=_filter_and_redact,
                enqueue=True,
            )
        else:
            console_format = (
                settings.loguru_format
                or "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message} | <cyan>{extra}</cyan>"
            )
            _loguru_logger.add(
                console_sink,
                level=level,
                format=console_format,
                colorize=True,
                diagnose=diag,
                backtrace=btrace,
                filter=_filter_and_redact,
                enqueue=True,
            )
    except Exception:
        pass

    if dirpath is not None:
        try:
            _ensure_dir(dirpath / "app.log")
            _ensure_dir(dirpath / "errors.log")
            app_log_path = str(dirpath / "app.log")
            _loguru_logger.add(
                app_log_path,
                level=level,
                rotation=rot,
                retention=ret,
                compression=comp,
                serialize=file_ser,
                diagnose=False,
                backtrace=btrace,
                filter=_filter_and_redact,
                opener=_secure_opener,
                enqueue=True,
            )
            errors_log_path = str(dirpath / "errors.log")
            _loguru_logger.add(
                errors_log_path,
                level="ERROR",
                rotation=os.getenv("LOG_ROTATION_ERRORS", "100 MB"),
                retention=os.getenv("LOG_RETENTION_ERRORS", "90 days"),
                compression=os.getenv("LOG_COMPRESSION_ERRORS", comp),
                serialize=False,
                diagnose=False,
                backtrace=True,
                filter=_filter_and_redact,
                opener=_secure_opener,
                enqueue=True,
            )
        except Exception:
            pass

    return _loguru_logger


@contextmanager
def correlation_scope(name: Optional[str] = None):
    """
    Context manager to create a scoped correlation id.

    Example:
        with correlation_scope("req"):
            logger.info("running")
    """
    previous = correlation_id.get()
    try:
        bind_correlation_id(
            name and generate_correlation_id(name) or generate_correlation_id()
        )
        yield correlation_id.get()
    finally:
        correlation_id.set(previous)


def logger_with_context(**extra) -> Any:
    """
    Return a logger bound with the current correlation id and provided extra.

    This lets callers migrate incrementally:
        logger = logger_with_context(user_id=user.id)
        logger.info("event")
    """
    logger = get_logger()
    try:
        request_id_value = correlation_id.get()
        if request_id_value:
            extra.setdefault("request_id", request_id_value)
        return logger.bind(**extra)
    except Exception:
        return logger


__all__ = [
    "get_logger",
    "configure_logging",
    "bind_correlation_id",
    "clear_correlation_id",
    "correlation_id",
    "generate_correlation_id",
    "correlation_scope",
    "logger_with_context",
]

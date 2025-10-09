"""Dependency injection for the MAIE API."""
from __future__ import annotations

import hmac
import mimetypes
from pathlib import Path
from typing import Any, Optional

from litestar import Request
from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException

from src.config import Settings
from src.api.schemas import ProcessRequestSchema, Feature


async def api_key_auth(conn: ASGIConnection, request: Request) -> Optional[bool]:
    """
    API key authentication dependency.

    Uses a timing-safe comparison to avoid leaking key information.

    Raises:
        NotAuthorizedException: on missing/invalid/format errors
    """
    # accept common header casings
    api_key = None
    if hasattr(conn, "headers"):
        api_key = conn.headers.get("x-api-key") or conn.headers.get("X-API-Key")

    if not api_key:
        raise NotAuthorizedException("API key is required")

    if not isinstance(api_key, str) or len(api_key) < 32:
        # Message must hint at format/length for tests
        raise NotAuthorizedException("Invalid API key format or length")

    # Use timing-safe comparison
    try:
        # Ensure both sides are strings
        expected = getattr(Settings, "secret_api_key", None)
        if expected is None:
            # Defensive: if settings not configured, deny access
            raise NotAuthorizedException("API key validation not configured")
        if not hmac.compare_digest(str(api_key), str(expected)):
            raise NotAuthorizedException(detail="Invalid API key")
    except NotAuthorizedException:
        raise
    except Exception as e:  # pragma: no cover - defensive
        raise NotAuthorizedException(f"Authentication error: {e}")

    # Return a truthy marker to indicate successful auth (Litestar DI can accept this)
    return True


async def validate_request_data(data: Any) -> bool:
    """
    Validate request data for /v1/process.

    Accepts either a raw dict or a pydantic model (which will be converted via model_dump()).

    Validation checks:
    - payload is a dict-like structure
    - file content_type or extension is an allowed audio format
    - file size does not exceed Settings.max_file_size_mb
    - template_id (if provided) exists in Settings.available_templates
    - features contains only allowed values

    Raises:
        ValueError: on validation failures with clear message
    """
    # Unpack pydantic model instances
    if hasattr(data, "model_dump") and callable(getattr(data, "model_dump")):
        try:
            data = data.model_dump()
        except Exception as e:
            raise ValueError(f"Failed to extract data from model: {e}")

    if not isinstance(data, dict):
        raise ValueError("Request payload must be a dict")

    # Basic required field: file
    file_obj = data.get("file")
    if not file_obj or not isinstance(file_obj, dict):
        raise ValueError("Missing or invalid 'file' field")
    assert isinstance(file_obj, dict)

    filename = file_obj.get("filename") or file_obj.get("name") or ""
    content_type = file_obj.get("content_type") or ""
    size = file_obj.get("size")

    # Validate content type / extension
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac"}
    is_audio_ct = False
    if isinstance(content_type, str) and content_type.startswith("audio/"):
        is_audio_ct = True

    ext = Path(filename).suffix.lower()
    if not is_audio_ct and ext not in allowed_extensions:
        # try guess from filename if content_type missing
        guessed, _ = mimetypes.guess_type(filename)
        if not guessed or not guessed.startswith("audio/") or ext not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {filename}")

    # Validate size (bytes)
    if size is None:
        raise ValueError("File size is required")
    try:
        size_int = int(size)
    except Exception:
        raise ValueError("Invalid file size value")

    max_bytes = int(getattr(Settings, "max_file_size_mb", 500) * 1024 * 1024)
    if size_int > max_bytes:
        raise ValueError(f"File too large: {size_int} bytes (max {max_bytes} bytes)")

    # Validate template_id if provided
    template_id = data.get("template_id")
    if template_id:
        available = getattr(Settings, "available_templates", None)
        if available is None:
            # If templates not configured, be permissive
            pass
        else:
            if template_id not in available:
                raise ValueError(f"template_id '{template_id}' not found")

    # Validate features
    features = data.get("features", [])
    if features is None:
        features = []
    if not isinstance(features, (list, tuple)):
        raise ValueError("features must be a list")

    # Allowed feature values: those from Feature enum and a small legacy alias set
    allowed = {f.value for f in Feature}

    for f in features:
        # normalize enum/objects that provide .value
        val = getattr(f, "value", f)
        if val not in allowed:
            raise ValueError(f"Unsupported feature: {val}")

    # Integrate with ProcessRequestSchema to ensure downstream schema compatibility
    try:
        ProcessRequestSchema.model_validate(data)
    except Exception as e:
        # Normalize pydantic/validation errors to ValueError as tests expect
        raise ValueError(f"Schema validation failed: {e}")

    return True

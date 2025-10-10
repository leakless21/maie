"""Dependency injection for the MAIE API."""
from __future__ import annotations

import hmac
import mimetypes
from pathlib import Path
from typing import Any

from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException
from litestar.handlers.base import BaseRouteHandler
from redis.asyncio import Redis as AsyncRedis
from redis import Redis as SyncRedis
from rq import Queue

from src.config import settings
from src.api.schemas import ProcessRequestSchema, Feature


# ============================================================================
# Redis Dependencies (Official redis-py 5.x patterns)
# ============================================================================

async def get_redis_client() -> AsyncRedis:
    """
    Get async Redis client for general API operations.
   
    Returns connection pool-backed client for efficient async operations.
    
    Returns:
        AsyncRedis: Async Redis client instance
    """
    client = AsyncRedis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )
    return client


async def get_results_redis() -> AsyncRedis:
    """
    Get async Redis client specifically for results storage.
    
    Separate client allows different connection pool configuration
    for result operations (which may have different timeout/retry needs).
    
    Returns:
        AsyncRedis: Async Redis client for results
    """
    client = AsyncRedis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=10.0,  # Longer timeout for large results
        socket_connect_timeout=5.0
    )
    return client


def get_sync_redis() -> SyncRedis:
    """
    Get synchronous Redis client for RQ task queue operations.
    
    RQ (python-rq) requires a synchronous Redis client.
    This client should NOT be used for async API operations.
    Use get_redis_client() or get_results_redis() for async ops.
    
    Returns:
        SyncRedis: Synchronous Redis client for RQ
    """
    client = SyncRedis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )
    return client


def get_rq_queue(name: str = "default") -> Queue:
    """
    Get RQ Queue instance for task enqueueing.
    
    Queue uses synchronous Redis client (get_sync_redis).
    When enqueueing from async endpoints, wrap with asyncio.to_thread():
    
        queue = get_rq_queue()
        job = await asyncio.to_thread(queue.enqueue, task_func, *args)
    
    Args:
        name: Queue name (default: "default")
    
    Returns:
        Queue: RQ Queue instance
    """
    redis_conn = get_sync_redis()
    return Queue(name=name, connection=redis_conn)


# ============================================================================
# Authentication Guard (Official Litestar signature)
# ============================================================================

async def api_key_guard(connection: ASGIConnection, route_handler: BaseRouteHandler) -> None:
    """
    API key authentication guard.
    
    Official Litestar guard signature: (ASGIConnection, BaseRouteHandler) -> None
    Guards raise NotAuthorizedException on failure (do not return False).
    
    Uses timing-safe comparison to avoid leaking key information.
    
    Args:
        connection: ASGI connection with request data
        route_handler: Route handler being guarded (for context-aware auth)
    
    Raises:
        NotAuthorizedException: On missing/invalid/format errors
    """
    # Accept common header casings
    api_key = connection.headers.get("x-api-key") or connection.headers.get("X-API-Key")
    
    if not api_key:
        raise NotAuthorizedException("API key is required")
    
    if not isinstance(api_key, str) or len(api_key) < 32:
        raise NotAuthorizedException("Invalid API key format or length")
    
    # Use timing-safe comparison
    try:
        expected = getattr(settings, "secret_api_key", None)
        if expected is None:
            raise NotAuthorizedException("API key validation not configured")
        
        if not hmac.compare_digest(str(api_key), str(expected)):
            raise NotAuthorizedException(detail="Invalid API key")
    except NotAuthorizedException:
        raise
    except Exception as e:  # pragma: no cover
        raise NotAuthorizedException(f"Authentication error: {e}")
    
    # Guard returns None on success


# ============================================================================
# Request Validation
# ============================================================================


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

    max_bytes = int(getattr(settings, "max_file_size_mb", 500) * 1024 * 1024)
    if size_int > max_bytes:
        raise ValueError(f"File too large: {size_int} bytes (max {max_bytes} bytes)")

    # Validate template_id if provided
    template_id = data.get("template_id")
    if template_id:
        available = getattr(settings, "available_templates", None)
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

    # Allowed feature values: those from Feature enum
    allowed = {f.value for f in Feature}

    for f in features:
        # normalize enum/objects that provide .value
        val = getattr(f, "value", f)
        if val not in allowed:
            raise ValueError(f"Unsupported feature: {val}")


    try:
        ProcessRequestSchema.model_validate(data)
    except Exception as e:
        # Normalize pydantic/validation errors to ValueError as tests expect
        raise ValueError(f"Schema validation failed: {e}")

    return True

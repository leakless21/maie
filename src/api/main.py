"""Main Litestar application entry point for the Modular Audio Intelligence Engine (MAIE) API.

This module exposes a preconfigured Litestar `app` object with:
- OpenAPI metadata
- registered route controllers
- lightweight health endpoint
- dependency registrations for validate_request_data
- conservative CORS configuration (exposed as app.cors_config)
- exception handlers for authorization and generic errors

The implementation avoids performing network I/O at import time to keep unit tests deterministic.
"""

from __future__ import annotations

from typing import Any, Dict

from litestar import Litestar, Response, get
from litestar.exceptions import HTTPException, NotAuthorizedException, ValidationException
from litestar.openapi import OpenAPIConfig
from litestar.status_codes import HTTP_422_UNPROCESSABLE_ENTITY

from src.config import configure_logging, get_logger
from src.config.logging import get_module_logger

try:
    from litestar.middleware.cors import CORSConfig
except Exception:
    CORSConfig = None

from src.api.dependencies import validate_request_data
from src.api.routes import route_handlers
from src.api.schemas import HealthResponse
from src.config import settings

# Always configure Loguru at API startup.
_logger = configure_logging()
logger = _logger if _logger is not None else get_logger()
logger = get_module_logger(__name__)
logger.info("Loguru configuration active (phase1) - api")
try:
    from src.config import settings as _settings_for_log
    logger.info("verbose_components={} debug={}", _settings_for_log.verbose_components, _settings_for_log.debug)
except Exception:
    pass
openapi_config = OpenAPIConfig(
    title="Modular Audio Intelligence Engine (MAIE) API",
    version="1.0.0",
    description="API for the Modular Audio Intelligence Engine providing audio processing capabilities",
)


def _handle_not_authorized(_: Any, exc: Exception) -> Response:
    return Response({"detail": str(exc)}, status_code=401)


def _handle_generic(_: Any, exc: Exception) -> Response:
    # Log full exception details server-side; keep generic client response
    try:
        logger.opt(exception=exc).error("Unhandled exception during request: {}", str(exc))
    except Exception:
        pass
    return Response({"detail": "Internal Server Error"}, status_code=500)


def _handle_http_exception(_: Any, exc: HTTPException) -> Response:
    # Preserve HTTPException semantics (status + detail)
    try:
        logger.warning("HTTPException {}: {}", getattr(exc, "status_code", 500), getattr(exc, "detail", ""))
    except Exception:
        pass
    return Response({"detail": getattr(exc, "detail", "")}, status_code=getattr(exc, "status_code", 500))


def _handle_validation_exception(_: Any, exc: ValidationException) -> Response:
    payload: Dict[str, Any] = {"detail": getattr(exc, "detail", "Validation error")}
    extra = getattr(exc, "extra", None)
    if extra is not None:
        payload["extra"] = extra
    return Response(payload, status_code=HTTP_422_UNPROCESSABLE_ENTITY)


@get("/health", summary="API health", tags=["Health"])
async def health() -> HealthResponse:
    """Return a comprehensive health response suitable for orchestration checks."""
    from src.api.dependencies import get_results_redis, get_rq_queue
    from rq import Worker
    
    # Check Redis connection
    redis_connected = False
    try:
        redis_client = await get_results_redis()
        await redis_client.ping()
        redis_connected = True
        await redis_client.aclose()
    except Exception:
        pass
    
    # Get queue depth
    queue_depth = 0
    try:
        queue = get_rq_queue()
        queue_depth = queue.count
    except Exception:
        pass
    
    # Check for active workers
    worker_active = False
    try:
        from src.api.dependencies import get_sync_redis
        sync_redis = get_sync_redis()
        worker_count = Worker.count(connection=sync_redis)
        worker_active = worker_count > 0
    except Exception:
        pass
    
    # Determine overall status
    status = "healthy" if redis_connected and worker_active else "unhealthy"
    
    return HealthResponse(
        status=status,
        version=getattr(settings, "pipeline_version", "unknown"),
        redis_connected=redis_connected,
        queue_depth=queue_depth,
        worker_active=worker_active,
    )


litestar_kwargs: Dict[str, Any] = {
    "route_handlers": [*route_handlers, health],
    "openapi_config": openapi_config,
    "dependencies": {"validate_request_data": validate_request_data},
    "exception_handlers": {
        NotAuthorizedException: _handle_not_authorized,
        HTTPException: _handle_http_exception,
        ValidationException: _handle_validation_exception,
        Exception: _handle_generic,
    },
    "debug": settings.debug,
    "request_max_body_size": settings.api.max_file_size_mb * 1024 * 1024,
}
if CORSConfig is not None:
    litestar_kwargs["cors_config"] = CORSConfig(
        allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"]
    )
app = Litestar(**litestar_kwargs)
if not getattr(app, "cors_config", None):

    class _MinimalCors:
        allow_origins = ["*"]
        allow_methods = ["GET", "POST", "OPTIONS"]

    app.cors_config = _MinimalCors()
if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Configure uvicorn to use our logging system
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = []
    uvicorn_logger.propagate = True
    
    # Suppress uvicorn's default logging to avoid duplicate messages
    uvicorn.run(
        app, 
        host=settings.api.host,
        port=settings.api.port,
        log_config=None,  # Disable uvicorn's default logging config
        access_log=False  # Disable access logs to reduce noise
    )

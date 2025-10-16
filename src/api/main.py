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
from litestar.openapi import OpenAPIConfig

from src.config import configure_logging, get_logger

try:
    from litestar.middleware.cors import CORSConfig
except Exception:
    CORSConfig = None
from litestar.exceptions import NotAuthorizedException

from src.api.dependencies import validate_request_data
from src.api.routes import route_handlers
from src.api.schemas import HealthResponse
from src.config import settings

# Always configure Loguru at API startup.
_logger = configure_logging()
logger = _logger if _logger is not None else get_logger()
logger.info("Loguru configuration active (phase1) - api")
openapi_config = OpenAPIConfig(
    title="Modular Audio Intelligence Engine (MAIE) API",
    version="1.0.0",
    description="API for the Modular Audio Intelligence Engine providing audio processing capabilities",
)


def _handle_not_authorized(_: Any, exc: Exception) -> Response:
    return Response({"detail": str(exc)}, status_code=401)


def _handle_generic(_: Any, exc: Exception) -> Response:
    return Response({"detail": "Internal Server Error"}, status_code=500)


@get("/health", summary="API health", tags=["Health"])
async def health() -> HealthResponse:
    """Return a conservative health response suitable for orchestration checks."""
    redis_configured = bool(getattr(settings, "redis_url", None))
    return HealthResponse(
        status="healthy",
        version=getattr(settings, "pipeline_version", "unknown"),
        redis_connected=redis_configured,
        queue_depth=0,
        worker_active=False,
    )


litestar_kwargs: Dict[str, Any] = {
    "route_handlers": [*route_handlers, health],
    "openapi_config": openapi_config,
    "dependencies": {"validate_request_data": validate_request_data},
    "exception_handlers": {
        NotAuthorizedException: _handle_not_authorized,
        Exception: _handle_generic,
    },
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

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)

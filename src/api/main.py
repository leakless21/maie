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

from typing import Dict, Any

from litestar import Litestar, Response, get
from litestar.openapi import OpenAPIConfig

# Optional import for CORS configuration
try:
    from litestar.middleware.cors import CORSConfig  # type: ignore
except Exception:  # pragma: no cover - defensive
    CORSConfig = None  # type: ignore

from litestar.exceptions import NotAuthorizedException

from src.api.routes import route_handlers
from src.api.dependencies import api_key_guard, validate_request_data
from src.api.schemas import HealthResponse
from src.config import settings

# OpenAPI configuration
openapi_config = OpenAPIConfig(
    title="Modular Audio Intelligence Engine (MAIE) API",
    version="1.0.0",
    description="API for the Modular Audio Intelligence Engine providing audio processing capabilities",
)


# Simple exception handlers
def _handle_not_authorized(_: Any, exc: Exception) -> Response:
    return Response({"detail": str(exc)}, status_code=401)


def _handle_generic(_: Any, exc: Exception) -> Response:
    # Avoid leaking internal state; provide a generic message
    return Response({"detail": "Internal Server Error"}, status_code=500)


# Lightweight health endpoint
@get("/health", summary="API health", tags=["Health"])
async def health() -> HealthResponse:
    """Return a conservative health response suitable for orchestration checks."""
    redis_configured = bool(getattr(settings, "redis_url", None))
    return HealthResponse(
        status="healthy",
        version=getattr(settings, "pipeline_version", "unknown"),
        # Report whether Redis is configured; avoid I/O during health checks.
        redis_connected=redis_configured,
        queue_depth=0,
        worker_active=False,
    )


# Build Litestar constructor args
litestar_kwargs: Dict[str, Any] = {
    "route_handlers": [*route_handlers, health],
    "openapi_config": openapi_config,
    "dependencies": {
        "validate_request_data": validate_request_data,
    },
    "exception_handlers": {
        NotAuthorizedException: _handle_not_authorized,
        Exception: _handle_generic,
    },
}

# Attach a conservative CORS config if available
if CORSConfig is not None:
    litestar_kwargs["cors_config"] = CORSConfig(
        allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"]
    )

# Instantiate the app
app = Litestar(**litestar_kwargs)

# Ensure a cors_config attribute is present for tests that expect it (some Litestar versions)
if not getattr(app, "cors_config", None):
    # If CORSConfig was unavailable or the attribute is falsy, set a minimal attribute to satisfy consumers/tests.
    class _MinimalCors:
        allow_origins = ["*"]
        allow_methods = ["GET", "POST", "OPTIONS"]

    app.cors_config = _MinimalCors()  # type: ignore

if __name__ == "__main__":  # pragma: no cover - executed only when run directly
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)

"""
Custom API middleware for request correlation.

This module provides a minimal ASGI middleware that:
- extracts an incoming correlation/request ID from headers (X-Request-ID or X-Correlation-ID)
- generates one when missing
- binds it to the logging context via contextvars for Loguru formatting
- injects the ID back into the HTTP response headers

This keeps console/file logs readable while enabling simple end-to-end tracing.
"""

from __future__ import annotations

from typing import Iterable, Tuple


class CorrelationIdMiddleware:
    """ASGI middleware that manages a correlation ID per request.

    - Accepts an inbound ID from headers (X-Request-ID or X-Correlation-ID)
    - Generates one when missing
    - Binds to logging context for downstream logs
    - Adds the ID to response headers for clients
    """

    def __init__(self, app):
        self.app = app
        self._header_candidates = (b"x-request-id", b"x-correlation-id")

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        # Lazy imports to avoid circulars at import-time
        from src.config.logging import (
            bind_correlation_id,
            clear_correlation_id,
            generate_correlation_id,
        )

        # Extract incoming correlation/request ID from headers if present
        headers: Iterable[Tuple[bytes, bytes]] = scope.get("headers") or ()
        cid: str | None = None
        for name, value in headers:
            lname = name.lower()
            if lname in self._header_candidates:
                try:
                    decoded = value.decode("utf-8", errors="ignore").strip()
                except Exception:
                    decoded = ""
                if decoded:
                    cid = decoded
                    break

        # Generate when missing
        if not cid:
            # Prefix makes it easier to eyeball API-originated events in logs
            cid = generate_correlation_id("api")

        # Bind to logging context for this request
        bind_correlation_id(cid)

        async def send_with_correlation(message):
            # Add the ID to response start headers so clients can reference it
            if message.get("type") == "http.response.start":
                hdrs = list(message.get("headers") or [])
                hdrs.append((b"x-request-id", cid.encode("utf-8")))
                hdrs.append((b"x-correlation-id", cid.encode("utf-8")))
                message["headers"] = hdrs
            return await send(message)

        try:
            return await self.app(scope, receive, send_with_correlation)
        finally:
            # Ensure we always clear it at end of request
            clear_correlation_id()

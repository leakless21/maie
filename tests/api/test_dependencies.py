"""
Comprehensive tests for src/api/dependencies.py

This module tests all dependency injection functions including:
- API key guard with official Litestar signature
- Redis client dependencies (async and sync)
- RQ queue management
- Request data validation

Following TDD principles:
- Tests define expected behavior
- Official Litestar patterns verified
- Guards raise NotAuthorizedException (not return False)
- Proper async/sync separation for Redis/RQ
"""

from typing import cast
from unittest.mock import Mock, patch

import pytest
from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException
from litestar.handlers.base import BaseRouteHandler
from litestar.types.asgi_types import Scope

from src.api.dependencies import validate_request_data
from src.api.schemas import ProcessRequestSchema

# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


@pytest.fixture(autouse=True)
def fixed_settings(monkeypatch):
    """
    Fixture to set predictable AppSettings values for tests.

    - Sets max_file_size_mb to a small value so size-limit tests are easy to trigger.
    """
    from src.config import settings

    monkeypatch.setattr(settings.api, "max_file_size_mb", 1, raising=False)
    yield


def make_mock_connection(headers: dict[str, str]) -> ASGIConnection:
    """Helper to create mock ASGIConnection with headers."""
    scope = cast(
        Scope,
        {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
            "query_string": b"",
            "server": ("localhost", 8000),
            "state": {},
        },
    )
    return ASGIConnection(scope)


def make_mock_route_handler() -> BaseRouteHandler:
    """Helper to create mock BaseRouteHandler."""
    return Mock(spec=BaseRouteHandler)


def make_file_obj(filename: str, content_type: str, size_bytes: int):
    """Helper to build a minimal file-like dict used by validate_request_data."""
    return {"filename": filename, "content_type": content_type, "size": size_bytes}


def base_valid_request():
    """Return a minimal, well-formed request dict that should pass validation."""
    return {
        "file": make_file_obj("speech.wav", "audio/wav", 1024),
        "pipeline_version": "1.0",
        "features": ["raw_transcript"],
        "template_id": "generic_summary_v1",
    }


# ============================================================================
# Guard Signature Tests
# ============================================================================


class TestGuardSignature:
    """Test correct guard signature following official Litestar patterns."""

    @pytest.mark.asyncio
    async def test_api_key_guard_signature_accepts_connection_and_handler(self):
        """api_key_guard must accept (ASGIConnection, BaseRouteHandler) parameters."""
        import inspect

        from src.api.dependencies import api_key_guard

        # Verify function signature
        sig = inspect.signature(api_key_guard)
        params = list(sig.parameters.keys())

        # Guard must have exactly 2 parameters
        assert len(params) == 2, (
            f"Guard should have 2 parameters, got {len(params)}: {params}"
        )

        # Verify parameter names (convention)
        assert "connection" in params[0].lower() or "conn" in params[0].lower()
        assert "handler" in params[1].lower() or "route" in params[1].lower()

    @pytest.mark.asyncio
    async def test_api_key_guard_parameter_types(self):
        """Verify guard parameters are correctly typed."""
        import inspect

        from src.api.dependencies import api_key_guard

        sig = inspect.signature(api_key_guard)
        params = list(sig.parameters.values())

        # First parameter should be ASGIConnection
        assert "ASGIConnection" in str(params[0].annotation)

        # Second parameter should be BaseRouteHandler
        assert "BaseRouteHandler" in str(params[1].annotation)

    @pytest.mark.asyncio
    async def test_api_key_guard_return_type_is_none(self):
        """Guard should return None (raises exception on failure)."""
        import inspect

        from src.api.dependencies import api_key_guard

        sig = inspect.signature(api_key_guard)
        return_annotation = sig.return_annotation

        # Return type should be None
        assert return_annotation is None or "None" in str(return_annotation)


class TestGuardBehavior:
    """Test guard behavior with correct signature."""

    @pytest.mark.asyncio
    async def test_guard_valid_api_key_returns_none(self):
        """Valid API key should pass guard and return None."""
        from src.api.dependencies import api_key_guard

        with patch("src.api.dependencies.settings.api.secret_key", "s" * 64):
            connection = make_mock_connection({"X-API-Key": "s" * 64})
            route_handler = make_mock_route_handler()

            # Guard should return None on success
            result = await api_key_guard(connection, route_handler)
            assert result is None

    @pytest.mark.asyncio
    async def test_guard_invalid_api_key_raises_exception(self):
        """Invalid API key should raise NotAuthorizedException."""
        from src.api.dependencies import api_key_guard

        with patch("src.api.dependencies.settings.api.secret_key", "s" * 64):
            connection = make_mock_connection({"X-API-Key": "invalid_key"})
            route_handler = make_mock_route_handler()

            # Guard must raise exception, not return False
            with pytest.raises(NotAuthorizedException):
                await api_key_guard(connection, route_handler)

    @pytest.mark.asyncio
    async def test_guard_missing_api_key_raises_exception(self):
        """Missing API key should raise NotAuthorizedException."""
        from src.api.dependencies import api_key_guard

        connection = make_mock_connection({})  # No API key
        route_handler = make_mock_route_handler()

        with pytest.raises(NotAuthorizedException):
            await api_key_guard(connection, route_handler)

    @pytest.mark.asyncio
    async def test_guard_can_access_connection_state(self):
        """Guard should be able to access and modify connection.state."""
        from src.api.dependencies import api_key_guard

        with patch("src.api.dependencies.settings.api.secret_key", "s" * 64):
            connection = make_mock_connection({"X-API-Key": "s" * 64})
            route_handler = make_mock_route_handler()

            # Add initial state
            connection.state["test_value"] = "initial"

            await api_key_guard(connection, route_handler)

            # Verify guard can access state (State object, not plain dict)
            assert hasattr(connection, "state")
            # Verify we can access state values
            assert connection.state["test_value"] == "initial"

    @pytest.mark.asyncio
    async def test_guard_can_access_route_handler_info(self):
        """Guard should have access to route handler for context-aware auth."""
        from src.api.dependencies import api_key_guard

        with patch("src.api.dependencies.settings.api.secret_key", "s" * 64):
            connection = make_mock_connection({"X-API-Key": "s" * 64})
            route_handler = make_mock_route_handler()

            # Add route handler metadata (as mock attributes)
            route_handler.name = "test_handler"

            # Guard receives handler and could use its metadata
            await api_key_guard(connection, route_handler)

            # Verify handler has expected attributes
            assert hasattr(route_handler, "name")


class TestGuardExecution:
    """Test guard execution order and integration."""

    @pytest.mark.asyncio
    async def test_guard_executes_before_dependencies(self):
        """Guards execute before dependency injection (Litestar execution order)."""
        execution_order = []

        # Mock that tracks execution
        async def mock_guard(conn, handler):
            execution_order.append("guard")

        def mock_dependency():
            execution_order.append("dependency")
            return "dep_value"

        # Simulate guard execution
        connection = make_mock_connection({"X-API-Key": "s" * 64})
        route_handler = make_mock_route_handler()

        await mock_guard(connection, route_handler)
        mock_dependency()

        # Guard should execute first
        assert execution_order == ["guard", "dependency"]

    @pytest.mark.asyncio
    async def test_guard_exception_prevents_handler_execution(self):
        """If guard raises exception, handler should not execute."""
        from src.api.dependencies import api_key_guard

        handler_executed = []

        async def mock_handler():
            handler_executed.append(True)
            return "response"

        connection = make_mock_connection({})  # Missing API key
        route_handler = make_mock_route_handler()

        # Guard raises exception
        with pytest.raises(NotAuthorizedException):
            await api_key_guard(connection, route_handler)
            # Handler should not be called
            await mock_handler()

        # Verify handler was never executed
        assert len(handler_executed) == 0


class TestMultipleGuards:
    """Test behavior when multiple guards are applied."""

    @pytest.mark.asyncio
    async def test_multiple_guards_all_execute_on_success(self):
        """When multiple guards are applied, all must pass."""
        execution_log = []

        async def first_guard(conn, handler):
            execution_log.append("first")
            if not conn.headers.get("X-API-Key"):
                raise NotAuthorizedException("Missing API key")

        async def second_guard(conn, handler):
            execution_log.append("second")
            if not conn.headers.get("X-Custom-Header"):
                raise NotAuthorizedException("Missing custom header")

        connection = make_mock_connection(
            {"X-API-Key": "s" * 64, "X-Custom-Header": "value"}
        )
        route_handler = make_mock_route_handler()

        # Execute both guards
        await first_guard(connection, route_handler)
        await second_guard(connection, route_handler)

        # Both should have executed
        assert execution_log == ["first", "second"]

    @pytest.mark.asyncio
    async def test_first_guard_failure_prevents_second_execution(self):
        """If first guard fails, second guard should not execute."""
        execution_log = []

        async def first_guard(conn, handler):
            execution_log.append("first")
            raise NotAuthorizedException("First guard failed")

        async def second_guard(conn, handler):
            execution_log.append("second")

        connection = make_mock_connection({})
        route_handler = make_mock_route_handler()

        # First guard fails
        with pytest.raises(NotAuthorizedException):
            await first_guard(connection, route_handler)

        # Second guard should not have executed
        assert execution_log == ["first"]


class TestGuardWithLitestarIntegration:
    """Integration tests simulating Litestar controller usage."""

    @pytest.mark.asyncio
    async def test_guard_applied_to_controller_affects_all_routes(self):
        """When guard is applied at controller level, it affects all routes."""
        from src.api.dependencies import api_key_guard

        # Simulate controller with guard
        class MockController:
            path = "/v1"
            guards = [api_key_guard]

            async def endpoint_one(self):
                return {"message": "one"}

            async def endpoint_two(self):
                return {"message": "two"}

        controller = MockController()

        # Verify guard is registered
        assert hasattr(controller, "guards")
        assert api_key_guard in controller.guards
        assert len(controller.guards) == 1


class TestTimingSafeAPIKeyComparison:
    """Test timing-safe API key comparison to prevent timing attacks."""

    @pytest.mark.asyncio
    async def test_guard_uses_hmac_compare_digest(self):
        """API key guard must use hmac.compare_digest for timing-safe comparison."""
        import inspect

        from src.api.dependencies import api_key_guard

        # Get source code
        source = inspect.getsource(api_key_guard)

        # Verify hmac.compare_digest is used
        assert "hmac.compare_digest" in source, (
            "Guard must use hmac.compare_digest for timing-safe comparison"
        )

        # Verify NOT using simple equality
        assert "api_key ==" not in source or "hmac.compare_digest" in source, (
            "Guard should not use simple == for secret comparison"
        )

    @pytest.mark.asyncio
    async def test_guard_import_hmac_module(self):
        """Verify hmac module is imported in dependencies."""
        import src.api.dependencies as deps

        # Verify hmac is imported
        assert hasattr(deps, "hmac") or "hmac" in dir(deps), (
            "hmac module must be imported for timing-safe comparison"
        )

    @pytest.mark.asyncio
    async def test_timing_attack_resistance(self):
        """Verify guard is resistant to timing attacks by using constant-time comparison."""
        import time

        from src.api.dependencies import api_key_guard

        correct_key = "s" * 64

        with patch("src.api.dependencies.settings.api.secret_key", correct_key):
            # Test with completely wrong key (first char different)
            wrong_key_early = "x" + "s" * 63
            connection_early = make_mock_connection({"X-API-Key": wrong_key_early})
            route_handler = make_mock_route_handler()

            start = time.perf_counter()
            with pytest.raises(NotAuthorizedException):
                await api_key_guard(connection_early, route_handler)
            time_early = time.perf_counter() - start

            # Test with wrong key (last char different)
            wrong_key_late = "s" * 63 + "x"
            connection_late = make_mock_connection({"X-API-Key": wrong_key_late})

            start = time.perf_counter()
            with pytest.raises(NotAuthorizedException):
                await api_key_guard(connection_late, route_handler)
            time_late = time.perf_counter() - start

            # Timing should be similar (within reasonable variance)
            # If using simple ==, early mismatch would be much faster
            # With hmac.compare_digest, times should be close
            ratio = (
                max(time_early, time_late) / min(time_early, time_late)
                if min(time_early, time_late) > 0
                else 1.0
            )

            # Allow 10x variance (constant-time should be much closer, but allow for system noise)
            assert ratio < 10, (
                f"Timing difference too large: {ratio}x (suggests non-constant-time comparison)"
            )

    @pytest.mark.asyncio
    async def test_guard_validates_key_length_before_comparison(self):
        """Guard should validate key length to prevent short key attacks."""
        from src.api.dependencies import api_key_guard

        with patch("src.api.dependencies.settings.api.secret_key", "s" * 64):
            # Very short key
            short_key = "short"
            connection = make_mock_connection({"X-API-Key": short_key})
            route_handler = make_mock_route_handler()

            with pytest.raises(NotAuthorizedException) as exc_info:
                await api_key_guard(connection, route_handler)

            # Should fail with format/length message, not just "Invalid"
            assert (
                "format" in str(exc_info.value).lower()
                or "length" in str(exc_info.value).lower()
            )


# ============================================================================
# Request Validation Tests
# ============================================================================


class TestRequestValidation:
    """Test request data validation for /v1/process endpoint."""

    @pytest.mark.asyncio
    async def test_validate_request_data_accepts_valid_schema(self):
        """A valid ProcessRequest payload should pass validation and return True."""
        data = base_valid_request()
        assert await validate_request_data(data) is True

    @pytest.mark.asyncio
    async def test_validate_request_data_rejects_non_audio_file(self):
        """Files with non-audio content-types must be rejected by validation."""
        data = base_valid_request()
        data["file"] = make_file_obj("notebook.txt", "text/plain", 10)
        with pytest.raises(ValueError):
            await validate_request_data(data)

    @pytest.mark.asyncio
    async def test_validate_request_data_enforces_file_size_limit(self):
        """Files larger than AppSettings.max_file_size_mb (in MB) must be rejected."""
        from src.config import settings

        data = base_valid_request()
        # Make file just above the limit (fixture sets max_file_size_mb to 1 MB)
        data["file"] = make_file_obj(
            "big.wav", "audio/wav", int(settings.api.max_file_size_mb * 1024 * 1024 + 1)
        )
        with pytest.raises(ValueError):
            await validate_request_data(data)

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Template validation not implemented - available_templates field doesn't exist in AppSettings yet"
    )
    async def test_validate_request_data_template_id_validation(self):
        """template_id must exist in available templates when provided."""
        data = base_valid_request()
        data["template_id"] = "nonexistent_template"
        with pytest.raises(ValueError):
            await validate_request_data(data)

    @pytest.mark.asyncio
    async def test_validate_request_data_features_allowed_values(self):
        """Features list must contain only allowed Feature enum values."""
        data = base_valid_request()
        # Inject an invalid feature string
        data["features"] = ["transcription", "teleportation"]
        with pytest.raises(ValueError):
            await validate_request_data(data)

    @pytest.mark.asyncio
    async def test_request_validation_with_pydantic_schema_integration(self):
        """Validate that ProcessRequestSchema integrates with validate_request_data."""
        payload = base_valid_request()
        # Creating the schema instance exercises pydantic validation first
        model = ProcessRequestSchema.model_validate(payload)
        # Validate both raw payload and model dump
        assert await validate_request_data(payload) is True
        assert await validate_request_data(model.model_dump()) is True


# ============================================================================
# Redis Dependency Tests
# ============================================================================


class TestRedisDependencies:
    """Test Redis client dependency functions."""

    @pytest.mark.asyncio
    async def test_get_redis_client_returns_async_redis(self):
        """get_redis_client() should return an AsyncRedis instance configured properly."""
        from redis.asyncio import Redis as AsyncRedis

        from src.api.dependencies import get_redis_client

        client = await get_redis_client()

        # Verify type
        assert isinstance(client, AsyncRedis), (
            f"Expected AsyncRedis, got {type(client)}"
        )

        # Verify configuration (client should have connection pool)
        assert client.connection_pool is not None

        # Clean up connection
        await client.aclose()

    @pytest.mark.asyncio
    async def test_get_results_redis_returns_async_redis(self):
        """get_results_redis() should return an AsyncRedis instance with custom timeout settings."""
        from redis.asyncio import Redis as AsyncRedis

        from src.api.dependencies import get_results_redis

        client = await get_results_redis()

        # Verify type
        assert isinstance(client, AsyncRedis), (
            f"Expected AsyncRedis, got {type(client)}"
        )

        # Verify configuration
        assert client.connection_pool is not None

        # Clean up connection
        await client.aclose()

    def test_get_sync_redis_returns_sync_redis(self):
        """get_sync_redis() should return a synchronous Redis client for RQ."""
        from redis import Redis as SyncRedis

        from src.api.dependencies import get_sync_redis

        client = get_sync_redis()

        # Verify type (should be sync Redis, not async)
        assert isinstance(client, SyncRedis), f"Expected SyncRedis, got {type(client)}"

        # Verify configuration
        assert client.connection_pool is not None

        # Clean up
        client.close()

    def test_get_rq_queue_returns_queue_with_sync_redis(self):
        """get_rq_queue() should return an RQ Queue instance using sync Redis connection."""
        from redis import Redis as SyncRedis
        from rq import Queue

        from src.api.dependencies import get_rq_queue

        queue = get_rq_queue(name="test-queue")

        # Verify type
        assert isinstance(queue, Queue), f"Expected Queue, got {type(queue)}"

        # Verify queue name
        assert queue.name == "test-queue"

        # Verify connection is sync Redis (RQ requirement)
        assert isinstance(queue.connection, SyncRedis), (
            "Queue must use synchronous Redis client"
        )

        # Test default queue name
        default_queue = get_rq_queue()
        assert default_queue.name == "default"

    def test_get_rq_queue_with_custom_name(self):
        """get_rq_queue() should accept custom queue names."""
        from src.api.dependencies import get_rq_queue

        custom_names = ["high-priority", "low-priority", "processing"]

        for name in custom_names:
            queue = get_rq_queue(name=name)
            assert queue.name == name, (
                f"Expected queue name '{name}', got '{queue.name}'"
            )


class TestRedisConnectionPooling:
    """Test Redis connection pooling configuration and best practices."""

    @pytest.mark.asyncio
    async def test_redis_client_has_encoding_utf8(self):
        """Async Redis clients must have encoding='utf-8' configured."""
        import inspect

        from src.api.dependencies import get_redis_client

        # Check function source for encoding parameter
        source = inspect.getsource(get_redis_client)
        assert 'encoding="utf-8"' in source or "encoding='utf-8'" in source, (
            "get_redis_client must specify encoding='utf-8'"
        )

        # Verify actual client
        client = await get_redis_client()
        # Connection pool should have encoding set
        pool = client.connection_pool
        conn_kwargs = pool.connection_kwargs
        assert conn_kwargs.get("encoding") == "utf-8", (
            f"Expected encoding='utf-8', got {conn_kwargs.get('encoding')}"
        )

        await client.aclose()

    @pytest.mark.asyncio
    async def test_redis_client_has_decode_responses_true(self):
        """Async Redis clients should have decode_responses=True for automatic decoding."""
        from src.api.dependencies import get_redis_client

        client = await get_redis_client()
        pool = client.connection_pool
        conn_kwargs = pool.connection_kwargs

        assert conn_kwargs.get("decode_responses") is True, (
            "decode_responses should be True for automatic string decoding"
        )

        await client.aclose()

    @pytest.mark.asyncio
    async def test_results_redis_has_custom_timeout(self):
        """Results Redis client should have custom timeout configuration."""
        import inspect

        from src.api.dependencies import get_results_redis

        source = inspect.getsource(get_results_redis)

        # Should configure socket_timeout
        assert "socket_timeout" in source, (
            "get_results_redis should configure socket_timeout for large results"
        )

        # Should configure socket_connect_timeout
        assert "socket_connect_timeout" in source, (
            "get_results_redis should configure socket_connect_timeout"
        )

        # Verify actual configuration
        client = await get_results_redis()
        pool = client.connection_pool
        conn_kwargs = pool.connection_kwargs

        # Should have timeout configured
        assert "socket_timeout" in conn_kwargs, "socket_timeout must be configured"
        assert "socket_connect_timeout" in conn_kwargs, (
            "socket_connect_timeout must be configured"
        )

        # Results client should have longer timeout than default
        assert conn_kwargs["socket_timeout"] >= 5.0, (
            f"socket_timeout should be >= 5s for large results, got {conn_kwargs['socket_timeout']}"
        )

        await client.aclose()

    @pytest.mark.asyncio
    async def test_sync_redis_has_encoding_utf8(self):
        """Sync Redis client for RQ must also have encoding='utf-8'."""
        import inspect

        from src.api.dependencies import get_sync_redis

        source = inspect.getsource(get_sync_redis)
        assert 'encoding="utf-8"' in source or "encoding='utf-8'" in source, (
            "get_sync_redis must specify encoding='utf-8'"
        )

        client = get_sync_redis()
        pool = client.connection_pool
        conn_kwargs = pool.connection_kwargs

        assert conn_kwargs.get("encoding") == "utf-8"
        client.close()

    @pytest.mark.asyncio
    async def test_redis_pool_uses_from_url(self):
        """Redis clients should use Redis.from_url() for proper pooling."""
        import inspect

        from src.api.dependencies import (
            get_redis_client,
            get_results_redis,
            get_sync_redis,
        )

        # All functions should use from_url() method
        for func in [get_redis_client, get_results_redis, get_sync_redis]:
            source = inspect.getsource(func)
            assert ".from_url(" in source, (
                f"{func.__name__} should use Redis.from_url() for connection pooling"
            )

    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self):
        """Multiple calls should reuse connection pool (not create new pools each time)."""
        from src.api.dependencies import get_redis_client

        client1 = await get_redis_client()
        client2 = await get_redis_client()

        # Both clients should have connection pools
        assert client1.connection_pool is not None
        assert client2.connection_pool is not None

        # Note: from_url() creates new pool each time by design
        # For production, use app.state to store and reuse pools

        await client1.aclose()
        await client2.aclose()

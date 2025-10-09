import pytest
import hmac
from typing import Any, Dict, cast
from litestar import Request
from litestar.connection import ASGIConnection
from litestar.datastructures import Headers
from litestar.exceptions import NotAuthorizedException
from litestar.types.asgi_types import Scope

from src.config import Settings
from src.api.dependencies import api_key_auth, validate_request_data
from src.api.schemas import ProcessRequestSchema


class MockASGIConnection(ASGIConnection[Any, Any, Any, Any]):
    """Mock ASGIConnection for testing."""
    def __init__(self, headers: Dict[str, str]):
        self._headers = Headers(headers)
        scope = cast(Scope, {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
            "query_string": b"",
            "server": ("localhost", 8000),
        })
        super().__init__(scope)

    @property
    def headers(self) -> Headers:
        return self._headers


class MockRequest(Request[Any, Any, Any]):
    """Mock Request for testing."""
    def __init__(self, headers: Dict[str, str] | None = None):
        if headers is None:
            headers = {}
        scope = cast(Scope, {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
            "query_string": b"",
            "server": ("localhost", 8000),
        })
        super().__init__(scope)


"""
Tests for src/api/dependencies.py

These tests follow the TDD "red" phase: they are intentionally written to fail
against the current placeholder implementations in dependencies.py.

They cover:
- API key authentication behavior and expected security practices
- Request data validation for the /v1/process endpoint
- Integration-style checks combining the Pydantic schemas and dependency helpers

Each test includes a short docstring explaining the intended behavior it asserts.
"""

@pytest.fixture(autouse=True)
def fixed_settings(monkeypatch):
    """
    Fixture to set predictable Settings values for tests.

    - Ensures secret_api_key is a long string we control.
    - Sets max_file_size_mb to a small value so size-limit tests are easy to trigger.
    """
    monkeypatch.setattr(Settings, "secret_api_key", "s" * 64, raising=False)
    monkeypatch.setattr(Settings, "max_file_size_mb", 1, raising=False)
    # Provide a minimal templates list for template_id validation tests
    monkeypatch.setattr(Settings, "available_templates", {"generic_summary_v1": "Generic Summary"}, raising=False)
    yield


# --------------------------
# Authentication tests
# --------------------------


@pytest.mark.asyncio
async def test_api_key_auth_valid_returns_true():
    """Valid API key should authenticate and return True."""
    conn = MockASGIConnection(headers={"x-api-key": Settings.secret_api_key})
    result = await api_key_auth(conn, MockRequest(headers=dict(conn.headers)))
    assert result is True


@pytest.mark.asyncio
async def test_api_key_auth_missing_raises_not_authorized():
    """Missing API key must raise NotAuthorizedException."""
    conn = MockASGIConnection(headers={})
    with pytest.raises(NotAuthorizedException):
        await api_key_auth(conn, MockRequest(headers=dict(conn.headers)))


@pytest.mark.asyncio
async def test_api_key_auth_invalid_raises_not_authorized():
    """Invalid API key (wrong value) must raise NotAuthorizedException."""
    conn = MockASGIConnection(headers={"x-api-key": "wrong" * 16})
    with pytest.raises(NotAuthorizedException):
        await api_key_auth(conn, MockRequest(headers=dict(conn.headers)))


@pytest.mark.asyncio
async def test_api_key_auth_short_key_raises_format_error():
    """API keys shorter than the expected minimum should raise a format validation error."""
    conn = MockASGIConnection(headers={"x-api-key": "short_key"})
    with pytest.raises(NotAuthorizedException) as excinfo:
        await api_key_auth(conn, MockRequest(headers=dict(conn.headers)))
    # Expect a message hinting at invalid format / length
    assert "format" in str(excinfo.value).lower() or "length" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_api_key_auth_uses_timing_safe_compare(monkeypatch):
    """
    Authentication should use a timing-attack-resistant comparison (hmac.compare_digest).
    This test patches a spy into the dependencies module's hmac.compare_digest and
    expects it to be called during authentication.
    """
    calls = {"called": False}

    def fake_compare(a, b):
        calls["called"] = True
        # behave like a real compare_digest for correctness
        return hmac.compare_digest(a, b)

    # Inject our fake into the module under test. If the implementation does not
    # use hmac.compare_digest this spy will never be called and the test will fail.
    monkeypatch.setattr("src.api.dependencies.hmac.compare_digest", fake_compare, raising=False)

    conn = MockASGIConnection(headers={"x-api-key": Settings.secret_api_key})
    # We don't assert the return here; we assert that the secure compare was invoked.
    with pytest.raises(Exception) as _:
        # Depending on implementation detail the function may raise or return;
        # call it to exercise the code path.
        await api_key_auth(conn, MockRequest(headers=dict(conn.headers)))

    assert calls["called"], "Expected hmac.compare_digest to be used for API key comparison"


# --------------------------
# Request validation tests
# --------------------------


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


@pytest.mark.asyncio
async def test_validate_request_data_accepts_valid_schema():
    """A valid ProcessRequest payload should pass validation and return True."""
    data = base_valid_request()
    assert await validate_request_data(data) is True


@pytest.mark.asyncio
async def test_validate_request_data_rejects_non_audio_file():
    """Files with non-audio content-types must be rejected by validation."""
    data = base_valid_request()
    data["file"] = make_file_obj("notebook.txt", "text/plain", 10)
    with pytest.raises(ValueError):
        await validate_request_data(data)


@pytest.mark.asyncio
async def test_validate_request_data_enforces_file_size_limit():
    """Files larger than Settings.max_file_size_mb (in MB) must be rejected."""
    data = base_valid_request()
    # Make file just above the limit: Settings.max_file_size_mb MB + 1 byte
    data["file"] = make_file_obj("big.wav", "audio/wav", Settings.max_file_size_mb * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        await validate_request_data(data)


@pytest.mark.asyncio
async def test_validate_request_data_template_id_validation():
    """template_id must exist in available templates when provided."""
    data = base_valid_request()
    data["template_id"] = "nonexistent_template"
    with pytest.raises(ValueError):
        await validate_request_data(data)


@pytest.mark.asyncio
async def test_validate_request_data_features_allowed_values():
    """Features list must contain only allowed Feature enum values."""
    data = base_valid_request()
    # Inject an invalid feature string
    data["features"] = ["transcription", "teleportation"]
    with pytest.raises(ValueError):
        await validate_request_data(data)


# --------------------------
# Integration-style tests
# --------------------------


@pytest.mark.asyncio
async def test_authentication_context_availability_after_auth():
    """
    The authentication dependency should make authentication context/state available.

    In a full Litestar integration the dependency would attach auth info to the request
    or connection context. This test asserts that after a successful call we can obtain
    an authentication indicator from the dependency's return value or side-effects.
    """
    conn = MockASGIConnection(headers={"x-api-key": Settings.secret_api_key})
    result = await api_key_auth(conn, MockRequest(headers=dict(conn.headers)))
    # The current implementation returns True; a more complete implementation might
    # attach a user/context object. We assert a truthy authentication marker here.
    assert result, "Expected an authentication marker (truthy) after successful api_key_auth"


@pytest.mark.asyncio
async def test_request_validation_with_pydantic_schema_integration():
    """
    Validate that a ProcessRequestSchema instance that is valid according to the
    pydantic model also passes the request validation helper.
    """
    # This constructs a ProcessRequestSchema using the same minimal fields used above.
    payload = base_valid_request()
    # Creating the schema instance exercises pydantic validation first.
    model = ProcessRequestSchema.model_validate(payload)
    # The dependencies.validate_request_data should accept the raw payload (dict) or
    # the model itself depending on implementation. We assert both code paths here.
    assert await validate_request_data(payload) is True
    assert await validate_request_data(model.model_dump()) is True

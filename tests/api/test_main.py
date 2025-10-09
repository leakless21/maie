import pytest
from litestar import Litestar

from src.api import main as api_main


def test_app_is_litestar():
    """App object should be a Litestar instance and importable from src.api.main."""
    assert hasattr(api_main, "app"), "Expected 'app' in src.api.main"
    assert isinstance(api_main.app, Litestar)


def test_openapi_title():
    """OpenAPI configuration should carry the project title."""
    cfg = getattr(api_main.app, "openapi_config", None)
    assert cfg is not None
    assert getattr(cfg, "title", "").startswith("Modular Audio Intelligence Engine (MAIE)")


def test_health_route_registered():
    """A /health endpoint must be registered on the application routes."""
    routes = []
    for r in getattr(api_main.app, "routes", []):
        # Litestar route objects expose different attrs depending on version
        path = getattr(r, "path", None) or getattr(r, "path_format", None) or getattr(r, "route", None)
        if path:
            routes.append(path)
        else:
            # fallback to string representation if path attribute not present
            routes.append(repr(r))
    assert any("/health" in r for r in routes), f"Health route not found in: {routes}"


def test_basic_dependencies_registered():
    """Application should register the API auth and validation dependencies."""
    deps = getattr(api_main.app, "dependencies", None)
    assert isinstance(deps, dict), "Expected app.dependencies to be a dict"
    assert "api_key_auth" in deps, "Missing 'api_key_auth' dependency registration"
    assert "validate_request_data" in deps, "Missing 'validate_request_data' dependency registration"


def test_cors_config_present():
    """CORS configuration should be present (conservative check)."""
    cors = getattr(api_main.app, "cors_config", None)
    assert cors is not None, "Expected app to expose a cors_config attribute"
import json
import shutil
from pathlib import Path
from typing import Generator

import pytest
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED, HTTP_404_NOT_FOUND
from litestar.testing import TestClient

from src.api.main import app
from src.config import settings

# Test data
TEST_TEMPLATE_ID = "test_template_crud"
TEST_SCHEMA = {
    "title": "Test Template",
    "description": "A test template",
    "type": "object",
    "properties": {"summary": {"type": "string"}},
}
TEST_PROMPT = "Summarize this: {{ text }}"
TEST_EXAMPLE = {"summary": "This is a summary."}


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create a test client for the app."""
    with TestClient(app=app) as client:
        yield client


@pytest.fixture(autouse=True)
def cleanup_templates():
    """Cleanup test templates before and after tests."""
    templates_dir = settings.paths.templates_dir
    paths = [
        templates_dir / "schemas" / f"{TEST_TEMPLATE_ID}.json",
        templates_dir / "prompts" / f"{TEST_TEMPLATE_ID}.jinja",
        templates_dir / "examples" / f"{TEST_TEMPLATE_ID}.example.json",
    ]
    
    # Cleanup before
    for p in paths:
        if p.exists():
            p.unlink()
            
    yield
    
    # Cleanup after
    for p in paths:
        if p.exists():
            p.unlink()


def test_create_template(test_client: TestClient):
    """Test creating a new template."""
    payload = {
        "id": TEST_TEMPLATE_ID,
        "schema_data": TEST_SCHEMA,
        "prompt_template": TEST_PROMPT,
        "example": TEST_EXAMPLE,
    }
    
    # We need to mock the API key or disable the guard for testing
    # For simplicity in this environment, we assume the test client might bypass or we provide the key if needed.
    # Checking routes.py, it uses api_key_guard.
    # We can inject the key in headers.
    headers = {"X-API-Key": settings.api.secret_key.get_secret_value()}

    response = test_client.post("/v1/templates", json=payload, headers=headers)
    assert response.status_code == HTTP_201_CREATED
    
    data = response.json()
    assert data["id"] == TEST_TEMPLATE_ID
    assert data["name"] == "Test Template"
    assert data["prompt_template"] == TEST_PROMPT


def test_get_template(test_client: TestClient):
    """Test retrieving a template."""
    # First create one
    headers = {"X-API-Key": settings.api.secret_key.get_secret_value()}
    payload = {
        "id": TEST_TEMPLATE_ID,
        "schema_data": TEST_SCHEMA,
        "prompt_template": TEST_PROMPT,
    }
    test_client.post("/v1/templates", json=payload, headers=headers)

    # Get it
    response = test_client.get(f"/v1/templates/{TEST_TEMPLATE_ID}")
    assert response.status_code == HTTP_200_OK
    data = response.json()
    assert data["id"] == TEST_TEMPLATE_ID
    assert data["prompt_template"] == TEST_PROMPT


def test_update_template(test_client: TestClient):
    """Test updating a template."""
    # Create
    headers = {"X-API-Key": settings.api.secret_key.get_secret_value()}
    payload = {
        "id": TEST_TEMPLATE_ID,
        "schema_data": TEST_SCHEMA,
        "prompt_template": TEST_PROMPT,
    }
    test_client.post("/v1/templates", json=payload, headers=headers)

    # Update
    new_prompt = "New prompt content"
    update_payload = {"prompt_template": new_prompt}
    response = test_client.put(f"/v1/templates/{TEST_TEMPLATE_ID}", json=update_payload, headers=headers)
    assert response.status_code == HTTP_200_OK
    
    # Verify
    response = test_client.get(f"/v1/templates/{TEST_TEMPLATE_ID}")
    assert response.json()["prompt_template"] == new_prompt


def test_delete_template(test_client: TestClient):
    """Test deleting a template."""
    # Create
    headers = {"X-API-Key": settings.api.secret_key.get_secret_value()}
    payload = {
        "id": TEST_TEMPLATE_ID,
        "schema_data": TEST_SCHEMA,
        "prompt_template": TEST_PROMPT,
    }
    test_client.post("/v1/templates", json=payload, headers=headers)

    # Delete
    response = test_client.delete(f"/v1/templates/{TEST_TEMPLATE_ID}", headers=headers)
    assert response.status_code == HTTP_200_OK or response.status_code == 204

    # Verify gone
    response = test_client.get(f"/v1/templates/{TEST_TEMPLATE_ID}")
    assert response.status_code == HTTP_404_NOT_FOUND

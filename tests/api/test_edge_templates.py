"""Tests for template endpoints on the Jetson edge API."""

from __future__ import annotations

import json
from typing import Dict

import pytest
from litestar.testing import TestClient

from src.api.edge_main import create_edge_app
from src.config import settings


@pytest.fixture
def edge_templates_dir(tmp_path, monkeypatch):
    """Point the app's template directory at a temporary location."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings.paths, "templates_dir", templates_dir)
    return templates_dir


@pytest.fixture
def edge_template_setup(edge_templates_dir) -> Dict[str, object]:
    """Create a sample template for read-only tests."""
    template_id = "edge_template"
    bundle_dir = edge_templates_dir / template_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    schema = {
        "title": "Edge Template",
        "description": "Test template available on edge API",
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "action_items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }
    prompt = "Summarize the transcript and list action items."
    example = {
        "summary": "Short summary",
        "action_items": ["Follow up", "Schedule review"],
    }

    (bundle_dir / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (bundle_dir / "prompt.jinja").write_text(prompt, encoding="utf-8")
    (bundle_dir / "example.json").write_text(
        json.dumps(example, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "templates_dir": edge_templates_dir,
        "template_id": template_id,
        "schema": schema,
        "prompt": prompt,
        "example": example,
    }


@pytest.fixture
def edge_client(edge_templates_dir) -> TestClient:
    """Provide a Litestar test client bound to the edge API."""
    app = create_edge_app()
    with TestClient(app=app) as client:
        yield client


def test_list_templates_available_on_edge(edge_client: TestClient, edge_template_setup):
    response = edge_client.get("/v1/templates")
    assert response.status_code == 200
    payload = response.json()
    assert "templates" in payload
    assert len(payload["templates"]) == 1

    template = payload["templates"][0]
    assert template["id"] == edge_template_setup["template_id"]
    assert template["schema_url"].endswith(
        f"/v1/templates/{edge_template_setup['template_id']}/schema"
    )
    assert template["example"] == edge_template_setup["example"]


def test_template_detail_and_schema(edge_client: TestClient, edge_template_setup):
    template_id = edge_template_setup["template_id"]

    detail_response = edge_client.get(f"/v1/templates/{template_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["prompt_template"] == edge_template_setup["prompt"]
    assert detail["schema_data"] == edge_template_setup["schema"]

    schema_response = edge_client.get(f"/v1/templates/{template_id}/schema")
    assert schema_response.status_code == 200
    assert schema_response.json() == edge_template_setup["schema"]


def test_create_update_delete_template(edge_client: TestClient, edge_templates_dir):
    template_id = "crud_template"
    payload = {
        "id": template_id,
        "schema_data": {
            "title": "CRUD Template",
            "description": "Created via edge API",
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        },
        "prompt_template": "Summarize: {{ transcript }}",
        "example": {"summary": "hi"},
    }

    create_response = edge_client.post("/v1/templates", json=payload)
    assert create_response.status_code == 201
    body = create_response.json()
    assert body["id"] == template_id
    assert body["prompt_template"] == payload["prompt_template"]

    updated_prompt = "Updated: {{ transcript }}"
    update_response = edge_client.put(
        f"/v1/templates/{template_id}",
        json={"prompt_template": updated_prompt},
    )
    assert update_response.status_code == 200
    assert update_response.json()["prompt_template"] == updated_prompt

    delete_response = edge_client.delete(f"/v1/templates/{template_id}")
    assert delete_response.status_code == 204
    get_response = edge_client.get(f"/v1/templates/{template_id}")
    assert get_response.status_code == 404

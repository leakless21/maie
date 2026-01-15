"""Shared helpers for template endpoints across MAIE API variants."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from litestar.exceptions import HTTPException, NotFoundException
from litestar.status_codes import (
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_ENTITY,
)

from src.api.schemas import (
    TemplateDetailSchema,
    TemplateInfoSchema,
    TemplatesResponseSchema,
)
from src.config import settings
from src.config.logging import get_module_logger
from src.utils.template_manager import TemplateManager

logger = get_module_logger(__name__)


def scan_templates_directory(templates_dir: Path | None = None) -> TemplatesResponseSchema:
    """Discover templates by scanning the configured templates directory."""
    templates: List[TemplateInfoSchema] = []
    templates_dir = templates_dir or settings.paths.templates_dir

    try:
        template_dirs = sorted(p for p in templates_dir.iterdir() if p.is_dir())
    except Exception as exc:
        logger.error("Failed to scan templates directory {}: {}", templates_dir, exc)
        return TemplatesResponseSchema(templates=[])

    for bundle_dir in template_dirs:
        template_id = bundle_dir.name
        if template_id.startswith(".") or template_id in {"schemas", "prompts", "examples"}:
            continue

        schema_path = bundle_dir / "schema.json"
        if not schema_path.exists():
            continue

        try:
            with schema_path.open("r", encoding="utf-8") as fp:
                schema_data = json.load(fp)
        except Exception as exc:
            logger.error(
                "Failed to load schema",
                extra={
                    "template_id": template_id,
                    "path": str(schema_path),
                    "error": str(exc),
                },
            )
            continue

        raw_name = schema_data.get("title") or template_id.replace("_", " ").title()
        description = schema_data.get(
            "description",
            "Auto-discovered template based on JSON schema.",
        )

        example: Dict[str, Any] | None = None
        example_path = bundle_dir / "example.json"
        if example_path.exists():
            try:
                with example_path.open("r", encoding="utf-8") as example_file:
                    example = json.load(example_file)
            except Exception as exc:
                logger.warning(
                    "Failed to load example JSON",
                    extra={"template_id": template_id, "error": str(exc)},
                )

        templates.append(
            TemplateInfoSchema(
                id=template_id,
                name=raw_name,
                description=description,
                schema_url=f"/v1/templates/{template_id}/schema",
                parameters=schema_data.get("properties", {}),
                example=example,
            )
        )

    return TemplatesResponseSchema(templates=templates)


def load_template_schema(template_id: str, templates_dir: Path | None = None) -> Dict[str, Any]:
    """Return the JSON schema for a given template ID."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", template_id):
        raise NotFoundException("Invalid template ID")

    templates_dir = templates_dir or settings.paths.templates_dir
    schema_path = templates_dir / template_id / "schema.json"

    if not schema_path.exists() or not schema_path.is_file():
        raise NotFoundException(f"Schema not found for template: {template_id}")

    try:
        with schema_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid schema JSON for template {template_id}: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Failed to load schema for template {template_id}: {exc}",
        ) from exc


async def load_template_detail(
    template_id: str,
    manager: TemplateManager | None = None,
) -> TemplateDetailSchema:
    """Load full template details via TemplateManager."""
    manager = manager or TemplateManager()
    try:
        content = await manager.get_template_content(template_id)
    except FileNotFoundError as exc:
        raise NotFoundException(f"Template {template_id} not found") from exc

    schema_data = content["schema"]
    raw_name = schema_data.get("title") or template_id.replace("_", " ").title()
    description = schema_data.get("description", "Template")

    return TemplateDetailSchema(
        id=template_id,
        name=raw_name,
        description=description,
        schema_url=f"/v1/templates/{template_id}/schema",
        parameters={},
        example=content.get("example"),
        prompt_template=content["prompt"],
        schema_data=schema_data,
    )

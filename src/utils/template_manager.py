"""
Utility for managing template files (schemas, prompts, examples).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import aiofiles
from src.config import settings
from src.config.logging import get_module_logger
from src.utils.json_utils import safe_parse_json

logger = get_module_logger(__name__)


class TemplateManager:
    """
    Manages template files on disk using asynchronous I/O.
    
    A template consists of:
    1. Schema: JSON file in templates/schemas/{id}.json
    2. Prompt: Jinja2 file in templates/prompts/{id}.jinja
    3. Example: (Optional) JSON file in templates/examples/{id}.example.json
    """

    def __init__(self):
        self.templates_dir = settings.paths.templates_dir
        # Ensure templates directory exists
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def _validate_id(self, template_id: str) -> None:
        """Validate template ID format."""
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", template_id):
            raise ValueError("Invalid template ID. Use alphanumeric, underscore, or dash.")

    def get_paths(self, template_id: str) -> Tuple[Path, Path, Path]:
        """Get paths for schema, prompt, and example files."""
        self._validate_id(template_id)
        bundle_dir = self.templates_dir / template_id
        return (
            bundle_dir / "schema.json",
            bundle_dir / "prompt.jinja",
            bundle_dir / "example.json",
        )

    def exists(self, template_id: str) -> bool:
        """Check if a template exists (schema is the source of truth)."""
        schema_path, _, _ = self.get_paths(template_id)
        return schema_path.exists()

    async def create_template(
        self,
        template_id: str,
        schema: Dict[str, Any],
        prompt: str,
        example: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a new template."""
        if self.exists(template_id):
            raise FileExistsError(f"Template {template_id} already exists")

        self._validate_id(template_id)
        bundle_dir = self.templates_dir / template_id
        bundle_dir.mkdir(exist_ok=True)

        schema_path, prompt_path, example_path = self.get_paths(template_id)

        # Write files
        try:
            async with aiofiles.open(schema_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(schema, indent=2, ensure_ascii=False))
            
            async with aiofiles.open(prompt_path, "w", encoding="utf-8") as f:
                await f.write(prompt)
            
            if example:
                async with aiofiles.open(example_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(example, indent=2, ensure_ascii=False))
                    
            logger.info(f"Created template {template_id}")
            
        except Exception as e:
            # Cleanup on failure
            await self.delete_template(template_id)
            raise e

    async def update_template(
        self,
        template_id: str,
        schema: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        example: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update an existing template."""
        if not self.exists(template_id):
            raise FileNotFoundError(f"Template {template_id} not found")

        schema_path, prompt_path, example_path = self.get_paths(template_id)

        if schema is not None:
            async with aiofiles.open(schema_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(schema, indent=2, ensure_ascii=False))

        if prompt is not None:
            async with aiofiles.open(prompt_path, "w", encoding="utf-8") as f:
                await f.write(prompt)

        if example is not None:
            async with aiofiles.open(example_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(example, indent=2, ensure_ascii=False))

        logger.info(f"Updated template {template_id}")

    async def delete_template(self, template_id: str) -> None:
        """Delete a template and its associated files."""
        if not self.exists(template_id):
            return

        bundle_dir = self.templates_dir / template_id
        
        # Remove entire directory
        try:
            # shutil.rmtree is sync, but acceptable for directory removal
            # For strict async, use run_in_executor
            import shutil
            shutil.rmtree(bundle_dir)
        except OSError as e:
            logger.warning(f"Failed to delete {bundle_dir}: {e}")

        logger.info(f"Deleted template {template_id}")

    async def get_template_content(self, template_id: str) -> Dict[str, Any]:
        """Get full content of a template."""
        if not self.exists(template_id):
            raise FileNotFoundError(f"Template {template_id} not found")

        schema_path, prompt_path, example_path = self.get_paths(template_id)

        content = {"id": template_id}

        # Load Schema
        async with aiofiles.open(schema_path, "r", encoding="utf-8") as f:
            parsed_schema, error = safe_parse_json(await f.read())
            if error:
                raise ValueError(f"Invalid schema JSON for {template_id}: {error}")
            content["schema"] = parsed_schema

        # Load Prompt
        if prompt_path.exists():
            async with aiofiles.open(prompt_path, "r", encoding="utf-8") as f:
                content["prompt"] = await f.read()
        else:
            content["prompt"] = ""

        # Load Example
        if example_path.exists():
            async with aiofiles.open(example_path, "r", encoding="utf-8") as f:
                parsed_example, error = safe_parse_json(await f.read())
                if error:
                    logger.warning(f"Invalid example JSON for {template_id}: {error}")
                    content["example"] = None
                else:
                    content["example"] = parsed_example
        else:
            content["example"] = None

        return content

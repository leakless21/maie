"""
JSON schema validation for LLM structured outputs.

This module provides functions for loading and validating JSON schemas,
ensuring LLM outputs conform to expected formats (FR-4, FR-6).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jsonschema import ValidationError

from src.config.logging import get_module_logger
from src.utils.json_utils import (
    safe_parse_json,
    validate_json_schema,
    extract_validation_error,
)
from src.utils.logging_utils import log_json_parse_error, log_validation_error

# Create module-bound logger for better debugging
logger = get_module_logger(__name__)


def load_template_schema(template_id: str, templates_dir: Path) -> Dict[str, Any]:
    """
    Load and validate template JSON schema.

    Args:
        template_id: Template identifier (e.g., "meeting_notes_v1")
        templates_dir: Path to templates directory

    Returns:
        Loaded JSON schema dictionary

    Raises:
        FileNotFoundError: If template file doesn't exist
        ValueError: If template JSON is invalid or missing required fields

    Example:
        >>> schema = load_template_schema("meeting_notes_v1", Path("templates"))
        >>> print(schema["type"])  # "object"
    """
    template_file = templates_dir / "schemas" / f"{template_id}.json"

    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    try:
        with open(template_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in template file {template_file}: {e}")
    except (OSError, IOError) as e:
        raise ValueError(f"Failed to load template {template_file}: {e}") from e

    # Validate schema structure
    if not isinstance(schema, dict):
        raise ValueError(f"Template schema must be a JSON object, got {type(schema)}")

    if "type" not in schema:
        raise ValueError(f"Template schema missing 'type' field: {template_file}")

    if schema["type"] != "object":
        raise ValueError(
            f"Template schema must be type 'object', got '{schema['type']}'"
        )

    # Validate required 'tags' field (FR-6)
    if not validate_tags_field(schema):
        raise ValueError(
            f"Template schema missing required 'tags' field: {template_file}"
        )

    logger.debug(f"Loaded template schema: {template_id}")
    return schema


def validate_llm_output(
    output: str, schema: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate LLM output against JSON schema.

    Args:
        output: Raw LLM output string
        schema: JSON schema to validate against

    Returns:
        Tuple of (parsed_data, error_message)
        - If valid: (parsed_json, None)
        - If invalid: (None, error_description)

    Example:
        >>> schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        >>> data, error = validate_llm_output('{"title": "Test"}', schema)
        >>> print(data)  # {"title": "Test"}
        >>> print(error)  # None
    """
    # Log the raw output for debugging (truncated if too long)
    output_preview = output.strip()[:200] + ("..." if len(output.strip()) > 200 else "")
    logger.debug(f"Validating LLM output: {output_preview!r}")

    # First, try to parse as JSON using the consolidated utility
    parsed_data, parse_error = safe_parse_json(output.strip())
    if parse_error:
        # Enhanced logging for JSON parse failures
        log_json_parse_error(
            json.JSONDecodeError("JSON decode error", output, 0),
            {
                "raw_output": output.strip(),
                "output_length": len(output.strip()),
                "output_preview": output_preview,
            },
        )
        logger.error(
            "JSON parsing failed",
            extra={
                "error_type": "json_parse_error",
                "error_message": parse_error,
                "raw_output": output.strip(),
                "output_length": len(output.strip()),
                "output_preview": output_preview,
            },
        )
        return None, parse_error

    if parsed_data is not None:
        logger.debug(
            f"Successfully parsed JSON with {len(parsed_data)} top-level keys: {list(parsed_data.keys())}"
        )

        # Then validate against schema using the consolidated utility
        try:
            is_valid, validation_error = validate_json_schema(parsed_data, schema)
        except Exception as e:
            # Handle unexpected errors during validation
            error_msg = f"Unexpected validation error: {str(e)}"
            logger.error(
                "Unexpected validation error",
                extra={
                    "error_type": "unexpected_validation_error",
                    "error_message": str(e),
                    "raw_output": output.strip(),
                    "output_preview": output_preview,
                    "parsed_data": parsed_data,
                },
            )
            return None, error_msg

        if not is_valid:
            # Enhanced logging for schema validation failures
            error_msg = f"Schema validation failed: {validation_error.get('error', 'Unknown validation error') if validation_error else 'Unknown validation error'}"
            if validation_error and "path" in validation_error:
                error_msg += f" (at path: {' -> '.join(str(p) for p in validation_error['path'])})"

            log_validation_error(
                ValidationError(error_msg),
                {
                    "raw_output": output.strip(),
                    "output_preview": output_preview,
                    "parsed_data": parsed_data,
                    "schema_summary": {
                        "type": schema.get("type"),
                        "properties": list(schema.get("properties", {}).keys()),
                        "required": schema.get("required", []),
                    },
                },
            )

            logger.error(
                "Schema validation failed",
                extra={
                    "error_type": "schema_validation_error",
                    "error_message": validation_error.get(
                        "error", "Unknown validation error"
                    )
                    if validation_error
                    else "Unknown validation error",
                    "validation_path": validation_error.get("path", [])
                    if validation_error
                    else [],
                    "validation_absolute_path": validation_error.get(
                        "absolute_path", []
                    )
                    if validation_error
                    else [],
                    "validator": validation_error.get("validator", "unknown")
                    if validation_error
                    else "unknown",
                    "validator_value": validation_error.get(
                        "validator_value", "unknown"
                    )
                    if validation_error
                    else "unknown",
                    "failed_instance": validation_error.get("instance", "unknown")
                    if validation_error
                    else "unknown",
                    "schema_path": validation_error.get("schema_path", [])
                    if validation_error
                    else [],
                    "raw_output": output.strip(),
                    "output_preview": output_preview,
                    "parsed_data": parsed_data,
                    "schema_summary": {
                        "type": schema.get("type"),
                        "properties": list(schema.get("properties", {}).keys()),
                        "required": schema.get("required", []),
                    },
                },
            )
            return None, error_msg

        logger.debug("LLM output validated successfully against schema")
        return parsed_data, None
    else:
        # This case should not happen since safe_parse_json returns (None, error_msg) on failure
        # But added for safety
        error_msg = "Failed to parse JSON output"
        logger.error(
            "JSON parsing failed",
            extra={
                "error_type": "json_parse_error",
                "error_message": error_msg,
                "raw_output": output.strip(),
                "output_length": len(output.strip()),
                "output_preview": output_preview,
            },
        )
        return None, error_msg


def validate_tags_field(schema: Dict[str, Any]) -> bool:
    """
    Verify schema has required 'tags' field (FR-6).

    The tags field must be:
    - Present in properties
    - Type array
    - MinItems 1, MaxItems 10
    - Items type string

    Args:
        schema: JSON schema to validate

    Returns:
        True if tags field is valid, False otherwise

    Example:
        >>> schema = {"properties": {"tags": {"type": "array", "minItems": 1, "maxItems": 10}}}
        >>> validate_tags_field(schema)  # True
    """
    properties = schema.get("properties", {})
    tags_field = properties.get("tags")

    if not tags_field:
        return False

    if not isinstance(tags_field, dict):
        return False

    # Check type is array
    if tags_field.get("type") != "array":
        return False

    # Check min/max items constraints
    min_items = tags_field.get("minItems", 0)
    max_items = tags_field.get("maxItems", float("inf"))

    if min_items < 1 or max_items > 10:
        return False

    # Check items type is string
    items = tags_field.get("items", {})
    if isinstance(items, dict) and items.get("type") != "string":
        return False

    return True


def extract_validation_errors(validation_error: ValidationError) -> Dict[str, Any]:
    """
    Extract detailed validation error information.

    Args:
        validation_error: jsonschema ValidationError

    Returns:
        Dictionary with error details for debugging

    Example:
        >>> try:
        ...     jsonschema.validate(data, schema)
        ... except ValidationError as e:
        ...     error_info = extract_validation_errors(e)
        ...     print(error_info["message"])
    """
    return extract_validation_error(validation_error)


def create_validation_summary(
    output: str,
    schema: Dict[str, Any],
    parsed_data: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a comprehensive validation summary for logging/debugging.

    Args:
        output: Raw LLM output
        schema: JSON schema used for validation
        parsed_data: Parsed JSON data (if successful)
        error_message: Error message (if validation failed)

    Returns:
        Dictionary with validation summary

    Example:
        >>> summary = create_validation_summary(
        ...     output='{"title": "Test"}',
        ...     schema=schema,
        ...     parsed_data={"title": "Test"},
        ...     error_message=None
        ... )
        >>> print(summary["valid"])  # True
    """
    summary = {
        "valid": error_message is None,
        "output_length": len(output),
        "schema_type": schema.get("type"),
        "schema_properties": list(schema.get("properties", {}).keys()),
        "has_tags_field": validate_tags_field(schema),
    }

    if parsed_data:
        summary.update(
            {
                "parsed_keys": list(parsed_data.keys()),
                "parsed_size": len(str(parsed_data)),
            }
        )

        # Check if tags field is present and valid
        tags = parsed_data.get("tags", [])
        if isinstance(tags, list):
            summary.update(
                {
                    "tags_count": len(tags),
                    "tags_valid": 1 <= len(tags) <= 5,
                }
            )

    if error_message:
        summary.update(
            {
                "error_message": error_message,
                "error_type": (
                    "validation"
                    if "Schema validation failed" in error_message
                    else "json_parse"
                ),
            }
        )

    return summary


def retry_with_lower_temperature(
    current_temperature: float, retry_count: int, max_retries: int = 2
) -> float:
    """
    Calculate temperature for retry attempts.

    Reduces temperature on each retry to encourage more deterministic output.

    Args:
        current_temperature: Current temperature setting
        retry_count: Number of retries attempted (0-based)
        max_retries: Maximum number of retries allowed

    Returns:
        New temperature value for retry

    Example:
        >>> temp = retry_with_lower_temperature(0.7, 1, 2)
        >>> print(temp)  # 0.35
    """
    if retry_count >= max_retries:
        return current_temperature

    # Reduce temperature by half on each retry
    reduction_factor = 0.5 ** (retry_count + 1)
    new_temperature = current_temperature * reduction_factor

    # Don't go below 0.1
    return max(new_temperature, 0.1)


def validate_schema_completeness(schema: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Validate that schema has all required fields for MAIE templates.

    Args:
        schema: JSON schema to validate

    Returns:
        Tuple of (is_complete, missing_fields)

    Example:
        >>> complete, missing = validate_schema_completeness(schema)
        >>> if not complete:
        ...     print(f"Missing fields: {missing}")
    """
    required_fields = ["type", "properties", "required"]
    missing_fields = []

    for field in required_fields:
        if field not in schema:
            missing_fields.append(field)

    # Check properties structure
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict):
            missing_fields.append("properties (must be object)")
        else:
            # Check for tags field
            if not validate_tags_field(schema):
                missing_fields.append("tags field (required for FR-6)")

    # Check required array
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list):
            missing_fields.append("required (must be array)")

    return len(missing_fields) == 0, missing_fields

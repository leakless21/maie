"""JSON processing and schema validation utilities for the MAIE project.

This module provides safe JSON parsing, schema validation, and error handling
for JSON-related operations across the codebase.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from jsonschema import validate, ValidationError
from .types import JSONDict, JSONParseResult, ErrorContext, ValidationResult


def safe_parse_json(
    json_str: str, *, error_context: Optional[ErrorContext] = None
) -> JSONParseResult:
    """Safely parse JSON string with detailed error context.

    Args:
        json_str: JSON string to parse
        error_context: Optional context dictionary for error reporting

    Returns:
        Tuple of (parsed_data, error_message) where parsed_data is None if parsing fails

    Examples:
        >>> safe_parse_json('{"key": "value"}')
        ({'key': 'value'}, None)
        >>> safe_parse_json('invalid json')
        (None, "JSON decode error: Expecting value: line 1 column 1 (char 0)")
    """
    try:
        parsed_data = json.loads(json_str)
        return parsed_data, None
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {str(e)}"
        if error_context:
            error_msg += f" (context: {error_context})"
        return None, error_msg


def validate_json_schema(
    data: JSONDict, schema: JSONDict, *, error_context: Optional[ErrorContext] = None
) -> ValidationResult:
    """Validate JSON data against schema with detailed error reporting.

    Args:
        data: JSON data to validate
        schema: Schema to validate against
        error_context: Optional context dictionary for error reporting

    Returns:
        Tuple of (is_valid, error_details) where error_details is None if valid

    Examples:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> validate_json_schema({"name": "test"}, schema)
        (True, None)
        >>> validate_json_schema({"name": 123}, schema)
        (False, {'error': "123 is not of type 'string'", 'path': ['name'], 'context': {}})
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        error_details = extract_validation_error(e)
        if error_context:
            error_details["context"] = error_context
        else:
            error_details["context"] = {}
        return False, error_details


def extract_validation_error(validation_error: ValidationError) -> Dict[str, Any]:
    """Extract detailed validation error information for debugging.

    Args:
        validation_error: ValidationError instance to extract information from

    Returns:
        Dictionary with detailed error information

    Examples:
        >>> from jsonschema import ValidationError
        >>> try:
        ...     validate(instance=123, schema={"type": "string"})
        ... except ValidationError as e:
        ...     extract_validation_error(e)
        {'error': "123 is not of type 'string'", 'path': [], 'schema_path': [], 'validator': 'type', 'validator_value': 'string', 'instance': 123, 'schema': {'type': 'string'}}
    """
    return {
        "error": validation_error.message,
        "path": list(validation_error.absolute_path),
        "schema_path": list(validation_error.schema_path),
        "validator": validation_error.validator,
        "validator_value": validation_error.validator_value,
        "instance": validation_error.instance,
        "schema": validation_error.schema,
    }


def create_validation_summary(
    output: str,
    schema: JSONDict,
    parsed_data: Optional[JSONDict],
    error_message: Optional[str],
) -> Dict[str, Any]:
    """Create comprehensive validation summary for logging/debugging.

    Args:
        output: Original output string that was validated
        schema: Schema used for validation
        parsed_data: Parsed data (if parsing succeeded)
        error_message: Error message (if validation failed)

    Returns:
        Dictionary with comprehensive validation summary

    Examples:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> create_validation_summary('{"name": "test"}', schema, {"name": "test"}, None)
        {'success': True, 'output_length': 17, 'parsed_keys': ['name'], 'error': None}
    """
    summary = {
        "success": error_message is None,
        "output_length": len(output),
        "parsed_keys": list(parsed_data.keys()) if parsed_data else [],
        "error": error_message,
    }

    if parsed_data:
        summary["parsed_size"] = len(json.dumps(parsed_data))

    return summary


def load_json_schema(schema_path: Path) -> JSONDict:
    """Load JSON schema from file with validation.

    Args:
        schema_path: Path to the JSON schema file

    Returns:
        Loaded schema as dictionary

    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If schema file contains invalid JSON
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Basic validation that it's a valid schema
    if not isinstance(schema, dict):
        raise ValueError(f"Schema file must contain a JSON object: {schema_path}")

    return schema


def validate_llm_output(output: str, schema: JSONDict) -> JSONParseResult:
    """Validate LLM JSON output against schema with retry logic support.

    Args:
        output: Raw output from LLM that should be JSON
        schema: Schema to validate the output against

    Returns:
        Tuple of (parsed_valid_data, error_message) where parsed_valid_data is None if validation fails

    Examples:
        >>> schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        >>> validate_llm_output('{"result": "success"}', schema)
        ({'result': 'success'}, None)
        >>> validate_llm_output('invalid json', schema)
        (None, 'JSON decode error: Expecting value: line 1 column 1 (char 0)')
    """
    # First, try to parse the JSON
    parsed_data, parse_error = safe_parse_json(output)
    if parse_error:
        return None, parse_error

    # Then validate against schema if parsing succeeded
    if parsed_data is not None:
        is_valid, validation_error = validate_json_schema(parsed_data, schema)
        if not is_valid:
            error_msg = f"Schema validation failed: {validation_error.get('error', 'Unknown validation error') if validation_error else 'Unknown validation error'}"
            return None, error_msg

        return parsed_data, None
    else:
        # This shouldn't happen given the logic above, but added for safety
        return None, "JSON parsing failed"


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string, returning a default value on failure.

    Args:
        json_str: JSON string to parse
        default: Default value to return if parsing fails

    Returns:
        Parsed JSON data or default value

    Examples:
        >>> safe_json_loads('{"key": "value"}', {})
        {'key': 'value'}
        >>> safe_json_loads('invalid', {})
        {}
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def json_dumps_safe(obj: Any, **kwargs) -> str:
    """Safely serialize object to JSON string with error handling.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments to pass to json.dumps

    Returns:
        JSON string representation of the object

    Raises:
        TypeError: If object is not JSON serializable

    Examples:
        >>> json_dumps_safe({"key": "value"})
        '{"key": "value"}'
    """
    # Set default arguments
    if "ensure_ascii" not in kwargs:
        kwargs["ensure_ascii"] = False
    if "indent" not in kwargs:
        kwargs["indent"] = 2

    return json.dumps(obj, **kwargs)


def extract_nested_value(data: JSONDict, path: str, default: Any = None) -> Any:
    """Extract a value from nested JSON using dot notation path.

    Args:
        data: JSON data to extract from
        path: Dot notation path (e.g., "user.profile.name")
        default: Default value if path doesn't exist

    Returns:
        Extracted value or default

    Examples:
        >>> data = {"user": {"profile": {"name": "John"}}}
        >>> extract_nested_value(data, "user.profile.name")
        'John'
        >>> extract_nested_value(data, "user.profile.age", 25)
        25
    """
    keys = path.split(".")
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def validate_json_string(json_str: str, schema: JSONDict) -> ValidationResult:
    """Validate a JSON string directly against a schema.

    Args:
        json_str: JSON string to validate
        schema: Schema to validate against

    Returns:
        Tuple of (is_valid, error_details) where error_details is None if valid

    Examples:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> validate_json_string('{"name": "test"}', schema)
        (True, None)
        >>> validate_json_string('invalid', schema)
        (False, {'error': 'JSON decode error: Expecting value: line 1 column 1 (char 0)', 'context': {}})
    """
    parsed_data, parse_error = safe_parse_json(json_str)
    if parse_error:
        return False, {"error": parse_error, "context": {}}

    # Only validate if parsing was successful and we have data
    if parsed_data is not None:
        return validate_json_schema(parsed_data, schema)
    else:
        return False, {"error": "Failed to parse JSON string", "context": {}}

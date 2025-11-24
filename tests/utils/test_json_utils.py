"""Tests for the json_utils module."""

from jsonschema import ValidationError
from src.utils.json_utils import (
    safe_parse_json,
    validate_json_schema,
    extract_validation_error,
    create_validation_summary,
    validate_llm_output,
    safe_json_loads,
    json_dumps_safe,
    extract_nested_value,
    validate_json_string,
)


class TestSafeParseJson:
    """Tests for safe_parse_json function."""

    def test_safe_parse_json_valid(self):
        """Test safe_parse_json with valid JSON."""
        data, error = safe_parse_json('{"key": "value"}')
        assert data == {"key": "value"}
        assert error is None

    def test_safe_parse_json_invalid(self):
        """Test safe_parse_json with invalid JSON."""
        data, error = safe_parse_json("invalid json")
        assert data is None
        assert error is not None
        assert "JSON decode error" in error

    def test_safe_parse_json_with_context(self):
        """Test safe_parse_json with error context."""
        data, error = safe_parse_json(
            "invalid json", error_context={"template_id": "test"}
        )
        assert data is None
        assert error is not None  # error should not be None
        assert "template_id" in error

    def test_safe_parse_json_with_markdown_fence(self):
        """Test safe_parse_json handles Markdown-fenced JSON payloads."""
        fenced_json = """```json
{
  "key": "value"
}
```"""
        data, error = safe_parse_json(fenced_json)
        assert data == {"key": "value"}
        assert error is None


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""

    def test_validate_json_schema_valid(self):
        """Test validate_json_schema with valid data."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        data = {"name": "test"}
        is_valid, error_details = validate_json_schema(data, schema)
        assert is_valid is True
        assert error_details is None

    def test_validate_json_schema_invalid(self):
        """Test validate_json_schema with invalid data."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        data = {"name": 123}
        is_valid, error_details = validate_json_schema(data, schema)
        assert is_valid is False
        assert error_details is not None
        assert "error" in error_details

    def test_validate_json_schema_with_context(self):
        """Test validate_json_schema with error context."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        data = {"name": 123}
        is_valid, error_details = validate_json_schema(
            data, schema, error_context={"user_id": "123"}
        )
        assert is_valid is False
        assert error_details is not None  # error_details should not be None
        assert "context" in error_details
        assert error_details["context"]["user_id"] == "123"


class TestExtractValidationError:
    """Tests for extract_validation_error function."""

    def test_extract_validation_error(self):
        """Test extract_validation_error function."""
        from jsonschema import validate

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        try:
            validate(instance={"name": 123}, schema=schema)
        except ValidationError as e:
            error_details = extract_validation_error(e)
            assert "error" in error_details
            assert "path" in error_details
            assert "schema_path" in error_details


class TestCreateValidationSummary:
    """Tests for create_validation_summary function."""

    def test_create_validation_summary_success(self):
        """Test create_validation_summary for successful validation."""
        summary = create_validation_summary(
            '{"key": "value"}', {}, {"key": "value"}, None
        )
        assert summary["success"] is True
        # Fixed: The actual length is 16, not 17
        assert summary["output_length"] == 16  # '{"key": "value"}' has 16 characters
        assert "key" in summary["parsed_keys"]
        assert summary["error"] is None

    def test_create_validation_summary_failure(self):
        """Test create_validation_summary for failed validation."""
        summary = create_validation_summary(
            '{"key": "value"}', {}, {"key": "value"}, "Some error"
        )
        assert summary["success"] is False
        assert summary["error"] == "Some error"


class TestValidateLlmOutput:
    """Tests for validate_llm_output function."""

    def test_validate_llm_output_valid(self):
        """Test validate_llm_output with valid JSON and schema."""
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        data, error = validate_llm_output('{"result": "success"}', schema)
        assert data == {"result": "success"}
        assert error is None

    def test_validate_llm_output_invalid_json(self):
        """Test validate_llm_output with invalid JSON."""
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        data, error = validate_llm_output("invalid json", schema)
        assert data is None
        assert error is not None

    def test_validate_llm_output_invalid_schema(self):
        """Test validate_llm_output with valid JSON but invalid schema."""
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        data, error = validate_llm_output('{"result": 123}', schema)
        assert data is None
        assert error is not None


class TestSafeJsonLoads:
    """Tests for safe_json_loads function."""

    def test_safe_json_loads_valid(self):
        """Test safe_json_loads with valid JSON."""
        result = safe_json_loads('{"key": "value"}', {})
        assert result == {"key": "value"}

    def test_safe_json_loads_invalid(self):
        """Test safe_json_loads with invalid JSON."""
        result = safe_json_loads("invalid json", {})
        assert result == {}

    def test_safe_json_loads_with_default(self):
        """Test safe_json_loads with default value."""
        result = safe_json_loads("invalid json", {"default": True})
        assert result == {"default": True}


class TestJsonDumpsSafe:
    """Tests for json_dumps_safe function."""

    def test_json_dumps_safe(self):
        """Test json_dumps_safe function."""
        result = json_dumps_safe({"key": "value"})
        assert '"key": "value"' in result
        assert isinstance(result, str)


class TestExtractNestedValue:
    """Tests for extract_nested_value function."""

    def test_extract_nested_value(self):
        """Test extract_nested_value function."""
        data = {"user": {"profile": {"name": "John", "age": 30}}}
        result = extract_nested_value(data, "user.profile.name")
        assert result == "John"

    def test_extract_nested_value_with_default(self):
        """Test extract_nested_value function with default."""
        data = {"user": {"profile": {"name": "John"}}}
        result = extract_nested_value(data, "user.profile.age", 25)
        assert result == 25


class TestValidateJsonString:
    """Tests for validate_json_string function."""

    def test_validate_json_string_valid(self):
        """Test validate_json_string with valid JSON."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        is_valid, error_details = validate_json_string('{"name": "test"}', schema)
        assert is_valid is True
        assert error_details is None

    def test_validate_json_string_invalid_json(self):
        """Test validate_json_string with invalid JSON."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        is_valid, error_details = validate_json_string("invalid json", schema)
        assert is_valid is False
        assert error_details is not None
        assert "JSON decode error" in error_details["error"]

    def test_validate_json_string_invalid_schema(self):
        """Test validate_json_string with valid JSON but invalid schema."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        is_valid, error_details = validate_json_string('{"name": 123}', schema)
        assert is_valid is False
        assert error_details is not None

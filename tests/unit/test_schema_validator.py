"""
Unit tests for JSON schema validation functionality.

Tests cover schema loading, validation, error handling, and retry logic.
"""

import json
from pathlib import Path

import jsonschema
import pytest
from jsonschema import ValidationError

from src.processors.llm.schema_validator import (
    create_validation_summary,
    extract_validation_errors,
    load_template_schema,
    retry_with_lower_temperature,
    validate_llm_output,
    validate_schema_completeness,
    validate_tags_field,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "templates"
TEMPLATE_SCHEMA_IDS = sorted(
    path.stem for path in (TEMPLATES_DIR / "schemas").glob("*.json")
)


class TestLoadTemplateSchema:
    """Test template schema loading functionality."""

    def test_load_valid_schema(self, tmp_path):
        """Test loading a valid template schema."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        schemas_dir = templates_dir / "schemas"
        schemas_dir.mkdir()

        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "main_points": {"type": "array", "items": {"type": "string"}},
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "main_points", "tags"],
        }

        template_file = schemas_dir / "meeting_notes_v1.json"
        template_file.write_text(json.dumps(schema, indent=2))

        loaded_schema = load_template_schema("meeting_notes_v1", templates_dir)

        assert loaded_schema == schema

    def test_load_nonexistent_template(self, tmp_path):
        """Test loading non-existent template raises FileNotFoundError."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        schemas_dir = templates_dir / "schemas"
        schemas_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_template_schema("nonexistent", templates_dir)

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises ValueError."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        schemas_dir = templates_dir / "schemas"
        schemas_dir.mkdir()

        template_file = schemas_dir / "invalid.json"
        template_file.write_text("{ invalid json }")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_template_schema("invalid", templates_dir)

    def test_load_non_object_schema(self, tmp_path):
        """Test loading non-object schema raises ValueError."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        schemas_dir = templates_dir / "schemas"
        schemas_dir.mkdir()

        template_file = schemas_dir / "invalid.json"
        template_file.write_text('["array", "not", "object"]')

        with pytest.raises(ValueError, match="Template schema must be a JSON object"):
            load_template_schema("invalid", templates_dir)

    def test_load_schema_missing_type(self, tmp_path):
        """Test loading schema without type field raises ValueError."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        schemas_dir = templates_dir / "schemas"
        schemas_dir.mkdir()

        schema = {"properties": {"title": {"type": "string"}}}
        template_file = schemas_dir / "invalid.json"
        template_file.write_text(json.dumps(schema))

        with pytest.raises(ValueError, match="Template schema missing 'type' field"):
            load_template_schema("invalid", templates_dir)

    def test_load_schema_wrong_type(self, tmp_path):
        """Test loading schema with wrong type raises ValueError."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        schemas_dir = templates_dir / "schemas"
        schemas_dir.mkdir()

        schema = {"type": "array", "items": {"type": "string"}}
        template_file = schemas_dir / "invalid.json"
        template_file.write_text(json.dumps(schema))

        with pytest.raises(ValueError, match="Template schema must be type 'object'"):
            load_template_schema("invalid", templates_dir)

    def test_load_schema_missing_tags_field(self, tmp_path):
        """Test loading schema without tags field raises ValueError."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        schemas_dir = templates_dir / "schemas"
        schemas_dir.mkdir()

        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        }
        template_file = schemas_dir / "invalid.json"
        template_file.write_text(json.dumps(schema))

        with pytest.raises(
            ValueError, match="Template schema missing required 'tags' or 'thẻ' field"
        ):
            load_template_schema("invalid", templates_dir)


class TestValidateLLMOutput:
    """Test LLM output validation functionality."""

    def test_validate_valid_output(self):
        """Test validation of valid JSON output."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "tags": {"type": "array", "minItems": 1, "maxItems": 10},
            },
            "required": ["title", "tags"],
        }

        output = '{"title": "Test Meeting", "tags": ["meeting", "test"]}'

        data, error = validate_llm_output(output, schema)

        assert data == {"title": "Test Meeting", "tags": ["meeting", "test"]}
        assert error is None

    def test_validate_invalid_json(self):
        """Test validation of invalid JSON."""
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        output = '{"title": "Test", "incomplete": }'

        data, error = validate_llm_output(output, schema)

        assert data is None
        assert "JSON decode error" in error

    def test_validate_schema_mismatch(self):
        """Test validation of output that doesn't match schema."""
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "count": {"type": "number"}},
            "required": ["title", "count"],
        }

        output = '{"title": "Test", "count": "not_a_number"}'

        data, error = validate_llm_output(output, schema)

        assert data is None
        assert "Schema validation failed" in error

    def test_validate_missing_required_field(self):
        """Test validation of output missing required fields."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "required_field": {"type": "string"},
            },
            "required": ["title", "required_field"],
        }

        output = '{"title": "Test"}'

        data, error = validate_llm_output(output, schema)

        assert data is None
        assert "Schema validation failed" in error
        assert "required_field" in error

    def test_validate_with_whitespace(self):
        """Test validation handles whitespace in output."""
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        output = '  \n  {"title": "Test"}  \n  '

        data, error = validate_llm_output(output, schema)

        assert data == {"title": "Test"}
        assert error is None


class TestValidateTagsField:
    """Test tags field validation functionality."""

    def test_validate_valid_tags_field(self):
        """Test validation of valid tags field."""
        schema = {
            "properties": {
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {"type": "string"},
                }
            }
        }

        assert validate_tags_field(schema) is True

    def test_validate_missing_tags_field(self):
        """Test validation of schema without tags field."""
        schema = {"properties": {"title": {"type": "string"}}}

        assert validate_tags_field(schema) is False

    def test_validate_tags_field_wrong_type(self):
        """Test validation of tags field with wrong type."""
        schema = {"properties": {"tags": {"type": "string"}}}  # Should be array

        assert validate_tags_field(schema) is False

    def test_validate_tags_field_invalid_constraints(self):
        """Test validation of tags field with invalid constraints."""
        # Too few min items
        schema1 = {
            "properties": {
                "tags": {
                    "type": "array",
                    "minItems": 0,  # Should be >= 1
                    "maxItems": 10,
                }
            }
        }
        assert validate_tags_field(schema1) is False

        # Too many max items
        schema2 = {
            "properties": {
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 11,  # Should be <= 10
                }
            }
        }
        assert validate_tags_field(schema2) is False

    def test_validate_tags_field_wrong_item_type(self):
        """Test validation of tags field with wrong item type."""
        schema = {
            "properties": {
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {"type": "number"},  # Should be string
                }
            }
        }

        assert validate_tags_field(schema) is False

    def test_validate_tags_field_non_dict(self):
        """Test validation of tags field that's not a dict."""
        schema = {"properties": {"tags": "not_a_dict"}}

        assert validate_tags_field(schema) is False


class TestExtractValidationErrors:
    """Test validation error extraction functionality."""

    def test_extract_validation_error(self):
        """Test extraction of validation error details."""
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        data = {"title": 123}  # Wrong type

        try:
            jsonschema.validate(data, schema)
        except ValidationError as e:
            error_info = extract_validation_errors(e)

            assert "error" in error_info
            assert "path" in error_info
            assert "absolute_path" in error_info
            assert "schema_path" in error_info
            assert "validator" in error_info
            assert "validator_value" in error_info
            assert "instance" in error_info

            assert error_info["path"] == ["title"]
            assert error_info["validator"] == "type"


class TestCreateValidationSummary:
    """Test validation summary creation functionality."""

    def test_create_summary_valid_output(self):
        """Test creating summary for valid output."""
        output = '{"title": "Test", "tags": ["tag1"]}'
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        parsed_data = {"title": "Test", "tags": ["tag1"]}

        summary = create_validation_summary(output, schema, parsed_data, None)

        assert summary["valid"] is True
        assert summary["output_length"] == len(output)
        assert summary["parsed_keys"] == ["title", "tags"]
        assert summary["tags_count"] == 1
        assert summary["tags_valid"] is True

    def test_create_summary_invalid_output(self):
        """Test creating summary for invalid output."""
        output = '{"title": "Test"}'
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        error_message = "Schema validation failed"

        summary = create_validation_summary(output, schema, None, error_message)

        assert summary["valid"] is False
        assert summary["error_message"] == error_message
        assert summary["error_type"] == "validation"

    def test_create_summary_json_parse_error(self):
        """Test creating summary for JSON parse error."""
        output = '{"title": "Test", "incomplete": }'
        schema = {"type": "object"}
        error_message = "Invalid JSON format"

        summary = create_validation_summary(output, schema, None, error_message)

        assert summary["valid"] is False
        assert summary["error_message"] == error_message
        assert summary["error_type"] == "json_parse"


class TestRetryWithLowerTemperature:
    """Test temperature retry logic functionality."""

    def test_retry_temperature_reduction(self):
        """Test temperature reduction on retries."""
        # First retry (retry_count=0)
        temp1 = retry_with_lower_temperature(0.7, 0, 2)
        assert temp1 == 0.35  # 0.7 * 0.5

        # Second retry (retry_count=1)
        temp2 = retry_with_lower_temperature(0.7, 1, 2)
        assert temp2 == 0.175  # 0.7 * 0.25

        # Third retry (retry_count=2) - should return original
        temp3 = retry_with_lower_temperature(0.7, 2, 2)
        assert temp3 == 0.7

    def test_retry_temperature_minimum(self):
        """Test temperature doesn't go below minimum."""
        # Very high temperature with many retries
        temp = retry_with_lower_temperature(10.0, 10, 20)
        assert temp == 0.1  # Minimum temperature

    def test_retry_temperature_max_retries(self):
        """Test temperature returns original when max retries reached."""
        temp = retry_with_lower_temperature(0.7, 5, 2)
        assert temp == 0.7


class TestValidateSchemaCompleteness:
    """Test schema completeness validation functionality."""

    def test_validate_complete_schema(self):
        """Test validation of complete schema."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "tags"],
        }

        complete, missing = validate_schema_completeness(schema)

        assert complete is True
        assert missing == []

    def test_validate_incomplete_schema_missing_fields(self):
        """Test validation of schema missing required fields."""
        schema = {"properties": {"title": {"type": "string"}}}

        complete, missing = validate_schema_completeness(schema)

        assert complete is False
        assert "type" in missing
        assert "required" in missing

    def test_validate_incomplete_schema_missing_tags(self):
        """Test validation of schema missing tags field."""
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        }

        complete, missing = validate_schema_completeness(schema)

        assert complete is False
        assert "tags field (required for FR-6)" in missing

    def test_validate_incomplete_schema_wrong_properties_type(self):
        """Test validation of schema with wrong properties type."""
        schema = {
            "type": "object",
            "properties": "not_an_object",
            "required": ["title"],
        }

        complete, missing = validate_schema_completeness(schema)

        assert complete is False
        assert "properties (must be object)" in missing

    def test_validate_incomplete_schema_wrong_required_type(self):
        """Test validation of schema with wrong required type."""
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": "not_an_array",
        }

        complete, missing = validate_schema_completeness(schema)

        assert complete is False
        assert "required (must be array)" in missing


class TestRepositoryTemplates:
    """Ensure built-in template schemas keep the required tags structure."""

    @pytest.mark.parametrize("template_id", TEMPLATE_SCHEMA_IDS)
    def test_schema_tags_field_is_valid(self, template_id: str):
        schema = load_template_schema(template_id, TEMPLATES_DIR)
        assert (
            validate_tags_field(schema) is True
        ), f"{template_id} should define tags as 1-10 short strings"

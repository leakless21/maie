"""
Test enhanced schema validation logging functionality.
"""

from unittest.mock import patch

from src.processors.llm.schema_validator import validate_llm_output


class TestEnhancedSchemaValidationLogging:
    """Test enhanced logging for schema validation failures."""

    def test_json_parse_error_with_enhanced_logging(self, caplog):
        """Test that JSON parse errors include detailed context in logs."""
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        invalid_json = '{"title": "incomplete json'

        with patch("src.processors.llm.schema_validator.logger") as mock_logger:
            data, error = validate_llm_output(invalid_json, schema)

            # Should return None and error
            assert data is None
            assert error is not None
            assert "JSON decode error" in error

            # Verify enhanced logging was called with detailed context
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args

            # Check the extra data contains expected fields
            extra_data = call_args[1]["extra"]
            assert extra_data["error_type"] == "json_parse_error"
            assert extra_data["raw_output"] == invalid_json.strip()
            assert "error_message" in extra_data
            assert "output_length" in extra_data
            assert "output_preview" in extra_data

    def test_schema_validation_error_with_enhanced_logging(self, caplog):
        """Test that schema validation errors include detailed context in logs."""
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "count": {"type": "number"}},
            "required": ["title", "count"],
        }
        invalid_data = '{"title": "Test", "count": "not_a_number"}'

        with patch("src.processors.llm.schema_validator.logger") as mock_logger:
            data, error = validate_llm_output(invalid_data, schema)

            # Should return None and error
            assert data is None
            assert error is not None
            assert "Schema validation failed" in error

            # Verify enhanced logging was called with detailed context
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args

            # Check the extra data contains expected fields
            extra_data = call_args[1]["extra"]
            assert extra_data["error_type"] == "schema_validation_error"
            assert extra_data["raw_output"] == invalid_data.strip()
            assert "validation_path" in extra_data
            assert "validator" in extra_data
            assert "failed_instance" in extra_data
            assert "schema_summary" in extra_data

            # Check schema summary
            schema_summary = extra_data["schema_summary"]
            assert schema_summary["type"] == "object"
            assert "title" in schema_summary["properties"]
            assert "count" in schema_summary["properties"]
            assert "title" in schema_summary["required"]
            assert "count" in schema_summary["required"]

    def test_empty_output_logging(self):
        """Test logging of empty LLM output."""
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        empty_output = ""

        with patch("src.processors.llm.schema_validator.logger") as mock_logger:
            data, error = validate_llm_output(empty_output, schema)

            # Should return None and error
            assert data is None
            assert error is not None
            assert "JSON decode error" in error

            # Verify empty output is logged properly
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            extra_data = call_args[1]["extra"]

            assert extra_data["raw_output"] == ""
            assert extra_data["output_length"] == 0
            assert extra_data["output_preview"] == ""

    def test_long_output_truncation(self):
        """Test that very long outputs are truncated in preview."""
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        long_output = "x" * 1000  # Very long output

        with patch("src.processors.llm.schema_validator.logger") as mock_logger:
            data, error = validate_llm_output(long_output, schema)

            # Should return None and error
            assert data is None
            assert error is not None
            assert "JSON decode error" in error

            # Verify output preview is truncated
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            extra_data = call_args[1]["extra"]

            assert extra_data["output_length"] == 1000
            assert extra_data["output_preview"].endswith("...")
            assert len(extra_data["output_preview"]) <= 203  # 200 + "..."

    def test_successful_validation_no_error_logging(self):
        """Test that successful validation doesn't trigger error logging."""
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        valid_json = '{"title": "Test"}'

        with patch("src.processors.llm.schema_validator.logger") as mock_logger:
            data, error = validate_llm_output(valid_json, schema)

            # Should return valid data and no error
            assert data is not None
            assert error is None

            # Should not call error logger
            mock_logger.error.assert_not_called()
            # Should call debug logger for success
            mock_logger.debug.assert_called()

    def test_unexpected_error_logging(self):
        """Test logging of unexpected errors during validation."""
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        valid_json = '{"title": "Test"}'

        # Mock jsonschema.validate to raise an unexpected error
        with patch(
            "src.utils.json_utils.validate"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Unexpected error")

            with patch("src.processors.llm.schema_validator.logger") as mock_logger:
                data, error = validate_llm_output(valid_json, schema)

                # Should return None and error
                assert data is None
                assert error is not None
                assert "Unexpected validation error" in error

                # Verify unexpected error is logged properly
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                extra_data = call_args[1]["extra"]

                assert extra_data["error_type"] == "unexpected_validation_error"
                assert extra_data["raw_output"] == valid_json.strip()
                assert "error_message" in extra_data

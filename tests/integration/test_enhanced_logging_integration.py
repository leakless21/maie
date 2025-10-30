"""
Integration test for enhanced schema validation logging.
Tests that the enhanced logging actually appears in log files.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.logging import configure_logging, bind_correlation_id
from src.config import settings
from src.processors.llm.schema_validator import validate_llm_output


class TestEnhancedLoggingIntegration:
    """Test enhanced logging in integration scenarios."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_json_parse_error_logged_to_file(self, temp_log_dir):
        """Test that JSON parse errors are logged with structured data to file."""
        # Configure logging to use temp directory
        with patch.object(settings.logging, "log_dir", temp_log_dir):
            configure_logging()
            bind_correlation_id("integration-test-json-parse")

            schema = {"type": "object", "properties": {"title": {"type": "string"}}}
            invalid_json = '{"title": "incomplete json'

            # Trigger the error
            data, error = validate_llm_output(invalid_json, schema)

            # Verify the result
            assert data is None
            assert error is not None
            assert "JSON decode error" in error

            # Check that the error was logged to file
            error_log_file = temp_log_dir / "errors.log"
            assert error_log_file.exists()

            with open(error_log_file, "r") as f:
                log_content = f.read()

                # Verify basic error message is present
                assert "JSON parsing failed" in log_content
                assert "integration-test-json-parse" in log_content

                # Verify structured data is present
                assert "STRUCTURED:" in log_content
                assert "error_type" in log_content
                assert "json_parse_error" in log_content
                assert "raw_output" in log_content
                assert "incomplete json" in log_content
                assert "output_length" in log_content
                assert "error_line" in log_content
                assert "error_col" in log_content
                assert "error_pos" in log_content

    def test_schema_validation_error_logged_to_file(self, temp_log_dir):
        """Test that schema validation errors are logged with structured data to file."""
        # Configure logging to use temp directory
        with patch.object(settings.logging, "log_dir", temp_log_dir):
            configure_logging()
            bind_correlation_id("integration-test-schema-validation")

            schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "count": {"type": "number"},
                },
                "required": ["title", "count"],
            }
            invalid_data = '{"title": "Test", "count": "not_a_number"}'

            # Trigger the error
            data, error = validate_llm_output(invalid_data, schema)

            # Verify the result
            assert data is None
            assert error is not None
            assert "Schema validation failed" in error

            # Check that the error was logged to file
            error_log_file = temp_log_dir / "errors.log"
            assert error_log_file.exists()

            with open(error_log_file, "r") as f:
                log_content = f.read()

                # Verify basic error message is present
                assert "Schema validation failed" in log_content
                assert "integration-test-schema-validation" in log_content

                # Verify structured data is present
                assert "STRUCTURED:" in log_content
                assert "error_type" in log_content
                assert "schema_validation_error" in log_content
                assert "validation_path" in log_content
                assert "validator" in log_content
                assert "failed_instance" in log_content
                assert "schema_summary" in log_content
                assert "parsed_data" in log_content

    def test_successful_validation_no_error_log(self, temp_log_dir):
        """Test that successful validation doesn't create error logs."""
        # Configure logging to use temp directory
        with patch.object(settings.logging, "log_dir", temp_log_dir):
            configure_logging()
            bind_correlation_id("integration-test-success")

            schema = {"type": "object", "properties": {"title": {"type": "string"}}}
            valid_json = '{"title": "Test"}'

            # Trigger successful validation
            data, error = validate_llm_output(valid_json, schema)

            # Verify the result
            assert data is not None
            assert error is None

            # Check that no error was logged to file
            error_log_file = temp_log_dir / "errors.log"
            if error_log_file.exists():
                with open(error_log_file, "r") as f:
                    log_content = f.read()
                    # Should not contain our correlation ID
                    assert "integration-test-success" not in log_content

    def test_raw_llm_output_capture(self, temp_log_dir):
        """Test that raw LLM output is captured even when it contains problematic content."""
        # Configure logging to use temp directory
        with patch.object(settings.logging, "log_dir", temp_log_dir):
            configure_logging()
            bind_correlation_id("integration-test-raw-output")

            schema = {"type": "object", "properties": {"content": {"type": "string"}}}

            # Test various problematic outputs that might come from LLMs
            problematic_outputs = [
                '{"content": "unclosed string',
                '{"content": "null reference: ',
                '{"content": "🚨 Special chars: \\n\\t\\"',
                '{"content": "Very long output that should be truncated but still captured in raw_output field for debugging purposes. '
                + "x" * 1000,
                '{"incomplete": true',
                "Just plain text without JSON",
                "",
                '{"nested": {"incomplete": true',
            ]

            for i, problematic_output in enumerate(problematic_outputs):
                # Trigger the error
                data, error = validate_llm_output(problematic_output, schema)

                # Should fail for all these cases
                assert data is None
                assert error is not None

            # Check that all errors were logged to file
            error_log_file = temp_log_dir / "errors.log"
            assert error_log_file.exists()

            with open(error_log_file, "r") as f:
                log_content = f.read()

                # Should contain error entries - there may be multiple log lines per error
                # (one from log_json_parse_error, one from main logger)
                assert "integration-test-raw-output" in log_content

                # Should contain structured data for each error
                structured_count = log_content.count("STRUCTURED:")
                # Each error generates at least 1-2 structured log entries
                assert structured_count >= len(problematic_outputs)

                # Should contain raw_output field in structured data for each error
                # The data is serialized as Python dict string, so look for 'raw_output'
                assert "'raw_output'" in log_content

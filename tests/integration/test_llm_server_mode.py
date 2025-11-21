"""
Integration tests for LLM server mode.
"""
import json
import pytest
from unittest.mock import Mock, patch
import urllib.request

from src.config import settings
from src.config.model import LlmBackendType
from src.processors.llm.processor import LLMProcessor


@pytest.mark.integration
class TestLLMServerMode:
    """Test LLM processor in server mode."""

    def setup_method(self):
        """Setup test environment."""
        self.original_backend = settings.llm_backend
        settings.llm_backend = LlmBackendType.VLLM_SERVER
        settings.llm_server.enhance_base_url = "http://test-server/v1"
        settings.llm_server.enhance_model_name = "test-model"

    def teardown_method(self):
        """Restore test environment."""
        settings.llm_backend = self.original_backend

    @patch("urllib.request.urlopen")
    def test_enhance_text_server_mode(self, mock_urlopen):
        """Test text enhancement in server mode."""
        # Mock server response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": "Enhanced text."},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        processor = LLMProcessor()
        result = processor.enhance_text("raw text")

        assert result["enhanced_text"] == "Enhanced text."
        assert result["enhancement_applied"] is True
        assert result["model_info"]["backend"] == "vllm_server"
        
        # Verify request
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://test-server/v1/chat/completions"
        data = json.loads(req.data)
        assert data["model"] == "test-model"
        # Check that messages contain the prompt
        assert len(data["messages"]) > 0
        assert "raw text" in data["messages"][-1]["content"] or "raw text" in str(data["messages"])

    @patch("urllib.request.urlopen")
    def test_generate_summary_server_mode(self, mock_urlopen):
        """Test summary generation in server mode."""
        # Mock server response
        summary_json = {"summary": "Test summary", "key_points": ["Point 1"]}
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": json.dumps(summary_json)},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20
            }
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        processor = LLMProcessor()
        
        # Mock template loading
        with patch("src.processors.llm.processor.load_template_schema") as mock_schema:
            mock_schema.return_value = {"type": "object"}
            
            # Mock validation to pass
            with patch("src.processors.llm.processor.validate_llm_output") as mock_validate:
                mock_validate.return_value = (summary_json, None)
                
                result = processor.generate_summary("transcript", template_id="meeting_notes_v1")

        assert result["summary"] == summary_json
        assert result["model_info"]["backend"] == "vllm_server"
        
        # Verify request
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        data = json.loads(req.data)
        # Check that structured_outputs body is present for JSON schema enforcement
        assert "extra_body" in data
        assert "structured_outputs" in data["extra_body"]
        assert "json" in data["extra_body"]["structured_outputs"]

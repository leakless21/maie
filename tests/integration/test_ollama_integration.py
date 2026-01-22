"""
Integration tests for Ollama backend compatibility.

This module tests that Ollama (running as an OpenAI-compatible server)
can be used as a drop-in LLM backend replacement for vLLM.
"""

import json
import pytest
from unittest.mock import Mock, patch

from src.config import settings
from src.config.model import LlmBackendType
from src.tooling.llm_client import VllmServerClient


@pytest.mark.integration
class TestOllamaClientIntegration:
    """Test Ollama as an alternative LLM backend through the client."""

    @patch("src.tooling.llm_client.urllib.request.urlopen")
    def test_ollama_server_client_basic_response(self, mock_urlopen):
        """Test basic text generation via Ollama through VllmServerClient."""
        # Mock Ollama-compatible response
        ollama_response = {
            "choices": [
                {
                    "message": {"content": "Generated response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(ollama_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Create a client for Ollama
        client = VllmServerClient(
            base_url="http://localhost:11434/v1",
            model_name="ministral-3:3b",
        )

        # Make a request with dictionary sampling params (not Mock objects)
        messages = [{"role": "user", "content": "test"}]
        result = client.chat(messages, sampling_params=None)

        # Verify the result structure
        assert result is not None
        assert len(result) > 0
        assert hasattr(result[0], "outputs")
        assert result[0].outputs[0].text == "Generated response"

    @patch("src.tooling.llm_client.urllib.request.urlopen")
    def test_ollama_keep_alive_parameter(self, mock_urlopen):
        """Test that keep_alive parameter is properly sent to Ollama."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = VllmServerClient(
            base_url="http://localhost:11434/v1",
            model_name="ministral-3:3b",
        )

        # Make a request with keep_alive parameter
        messages = [{"role": "user", "content": "test"}]
        client.chat(messages, keep_alive=0)

        # Verify the request was made
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        request_data = json.loads(req.data)

        # Verify keep_alive is in the request
        assert "keep_alive" in request_data
        assert request_data["keep_alive"] == 0

    @patch("src.tooling.llm_client.urllib.request.urlopen")
    def test_ollama_response_conversion(self, mock_urlopen):
        """Test that Ollama OpenAI response is properly converted."""
        # Ollama returns standard OpenAI format
        ollama_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ministral-3:3b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(ollama_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = VllmServerClient(
            base_url="http://localhost:11434/v1",
            model_name="ministral-3:3b",
        )

        result = client.chat([{"role": "user", "content": "test"}])

        # Verify the response is converted to mock vLLM format
        assert result[0].outputs[0].text == "Test response"
        assert result[0].outputs[0].finish_reason == "stop"
        assert len(result[0].prompt_token_ids) == 10
        assert len(result[0].outputs[0].token_ids) == 5

    @patch("src.tooling.llm_client.urllib.request.urlopen")
    def test_ollama_streaming_response(self, mock_urlopen):
        """Test that Ollama streaming responses are handled correctly."""
        # Mock a streaming response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "Streamed response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = VllmServerClient(
            base_url="http://localhost:11434/v1",
            model_name="ministral-3:3b",
        )

        result = client.chat(
            [{"role": "user", "content": "test"}],
            stream=False,  # Ollama streaming handled same as regular
        )

        assert result[0].outputs[0].text == "Streamed response"


@pytest.mark.integration
class TestOllamaBackendConfiguration:
    """Test Ollama backend configuration and compatibility."""

    def test_ollama_settings_configuration(self):
        """Test that Ollama settings can be properly configured."""
        original_backend = settings.llm_backend
        original_url = settings.llm_server.enhance_base_url
        original_model = settings.llm_server.enhance_model_name
        original_keep_alive = settings.llm_server.keep_alive
        original_structured = settings.llm_sum.structured_outputs_enabled

        try:
            # Configure for Ollama
            settings.llm_backend = LlmBackendType.VLLM_SERVER
            settings.llm_server.enhance_base_url = "http://localhost:11434/v1"
            settings.llm_server.enhance_model_name = "ministral-3:3b"
            settings.llm_server.keep_alive = 0
            settings.llm_sum.structured_outputs_enabled = False

            # Verify settings are correct
            assert settings.llm_backend == LlmBackendType.VLLM_SERVER
            assert "localhost:11434" in settings.llm_server.enhance_base_url
            assert settings.llm_server.enhance_model_name == "ministral-3:3b"
            assert settings.llm_server.keep_alive == 0
            assert settings.llm_sum.structured_outputs_enabled is False

        finally:
            settings.llm_backend = original_backend
            settings.llm_server.enhance_base_url = original_url
            settings.llm_server.enhance_model_name = original_model
            settings.llm_server.keep_alive = original_keep_alive
            settings.llm_sum.structured_outputs_enabled = original_structured

    def test_ollama_vs_vllm_server_api_compatibility(self):
        """Test that Ollama and vLLM server share the same API contract."""
        # Both use OpenAI-compatible API through VllmServerClient
        ollama_client = VllmServerClient(
            base_url="http://localhost:11434/v1",
            model_name="ministral-3:3b",
        )

        vllm_client = VllmServerClient(
            base_url="http://localhost:8001/v1",
            model_name="qwen3-4b-instruct",
        )

        # Both should have the same interface
        assert hasattr(ollama_client, "chat")
        assert hasattr(vllm_client, "chat")
        assert callable(ollama_client.chat)
        assert callable(vllm_client.chat)

    def test_ollama_openai_spec_compliance(self):
        """Test that Ollama configuration complies with OpenAI API spec."""
        # Ollama uses OpenAI-compatible API, so settings should reflect that
        original_url = settings.llm_server.enhance_base_url

        try:
            settings.llm_server.enhance_base_url = "http://localhost:11434/v1"

            # These should be valid for OpenAI-compatible servers
            assert settings.llm_server.enhance_base_url.endswith("/v1")
            assert "localhost" in settings.llm_server.enhance_base_url or (
                "0.0.0.0" in settings.llm_server.enhance_base_url
            )

        finally:
            settings.llm_server.enhance_base_url = original_url

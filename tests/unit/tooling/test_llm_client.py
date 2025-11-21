"""
Unit tests for LLM clients.
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import urllib.error

from src.tooling.llm_client import LocalVllmClient, VllmServerClient


class TestLocalVllmClient:
    """Test LocalVllmClient."""

    def test_chat_delegates_to_model(self):
        """Test that chat delegates to the model's chat method."""
        mock_model = Mock()
        client = LocalVllmClient(mock_model)
        
        messages = [{"role": "user", "content": "hello"}]
        sampling_params = Mock()
        
        client.chat(messages, sampling_params=sampling_params)
        
        mock_model.chat.assert_called_once_with(messages, sampling_params=sampling_params)


from types import SimpleNamespace

class TestVllmServerClient:
    """Test VllmServerClient."""

    @pytest.fixture
    def client(self):
        return VllmServerClient(base_url="http://test-server/v1", api_key="test-key", model_name="test-model")

    @patch("urllib.request.urlopen")
    def test_chat_success(self, mock_urlopen, client):
        """Test successful chat request."""
        # Mock response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": "response text"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20
            }
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        messages = [{"role": "user", "content": "hello"}]
        # Use SimpleNamespace to avoid Mock serialization issues
        sampling_params = SimpleNamespace(temperature=0.7, max_tokens=100)

        outputs = client.chat(messages, sampling_params=sampling_params)

        # Verify request
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://test-server/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer test-key"
        assert req.headers["Content-type"] == "application/json"
        
        data = json.loads(req.data)
        assert data["messages"] == messages
        assert data["model"] == "test-model"
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 100

        # Verify output conversion
        assert len(outputs) == 1
        assert outputs[0].outputs[0].text == "response text"
        assert outputs[0].outputs[0].finish_reason == "stop"
        assert len(outputs[0].prompt_token_ids) == 10
        assert len(outputs[0].outputs[0].token_ids) == 20

    @patch("urllib.request.urlopen")
    def test_chat_error(self, mock_urlopen, client):
        """Test chat request error handling."""
        mock_error = urllib.error.HTTPError(
            url="http://test", code=500, msg="Internal Server Error", hdrs={}, fp=Mock()
        )
        mock_error.read = Mock(return_value=b"Error details")
        mock_urlopen.side_effect = mock_error

        with pytest.raises(RuntimeError, match="LLM Server HTTP Error: 500"):
            client.chat([{"role": "user", "content": "hello"}])

    @patch("urllib.request.urlopen")
    def test_chat_structured_outputs(self, mock_urlopen, client):
        """Test chat request with structured outputs (JSON schema) in request."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"choices": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # New API: pass structured_outputs to request
        structured_outputs = {"json": {"type": "object"}}
        client.chat([{"role": "user", "content": "hello"}], structured_outputs=structured_outputs)

        req = mock_urlopen.call_args[0][0]
        data = json.loads(req.data)
        assert "extra_body" in data
        assert data["extra_body"]["structured_outputs"]["json"] == {"type": "object"}

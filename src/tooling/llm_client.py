"""
LLM Client abstraction for MAIE.
Provides a unified interface for interacting with LLM backends (local vLLM or remote server).
"""

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Protocol, Union
from pathlib import Path

from src.config import settings
from src.config.logging import get_module_logger

logger = get_module_logger(__name__)


class ChatCompletionClient(Protocol):
    """Protocol for LLM chat completion clients."""

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """
        Generate a chat completion.

        Args:
            messages: List of messages in OpenAI format [{"role": "...", "content": "..."}]
            **kwargs: Additional generation parameters (sampling params, etc.)

        Returns:
            The completion result (format depends on implementation, but generally vLLM-like)
        """
        ...


class LocalVllmClient:
    """Client for interacting with an in-process vLLM instance."""

    def __init__(self, model_instance: Any):
        """
        Initialize with a loaded vLLM model instance.

        Args:
            model_instance: The vLLM LLM instance
        """
        self.model = model_instance

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """
        Delegate to local vLLM model.chat().
        """
        # Extract sampling params from kwargs
        sampling_params = kwargs.get("sampling_params")
        
        # vLLM's chat() expects messages as the first argument
        # structured_outputs should be in sampling_params, not as a separate kwarg
        return self.model.chat(messages, sampling_params=sampling_params)


class VllmServerClient:
    """Client for interacting with a remote vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the server client.

        Args:
            base_url: Base URL of the vLLM server (e.g., "http://localhost:8001/v1")
            api_key: Optional API key
            model_name: Default model name to use in requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """
        Send chat completion request to the server.
        """
        url = f"{self.base_url}/chat/completions"
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Prepare payload
        # Map vLLM SamplingParams to OpenAI API parameters
        sampling_params = kwargs.get("sampling_params")
        payload = {
            "messages": messages,
            "model": self.model_name or "default",  # Server often requires a model name
        }

        if sampling_params:
            # Extract attributes from SamplingParams object
            # We iterate over common OpenAI params and extract them if present
            params_map = {
                "temperature": "temperature",
                "top_p": "top_p",
                "top_k": "top_k",
                "max_tokens": "max_tokens",
                "stop": "stop",
                "presence_penalty": "presence_penalty",
                "frequency_penalty": "frequency_penalty",
                "repetition_penalty": "repetition_penalty",
                "seed": "seed",
            }
            
            for sp_attr, api_param in params_map.items():
                if hasattr(sampling_params, sp_attr):
                    val = getattr(sampling_params, sp_attr)
                    if val is not None:
                        payload[api_param] = val

        # Handle structured_outputs (preferred) if present in kwargs or sampling_params
        # For OpenAI compatible server, we pass it in extra_body
        structured_outputs = kwargs.get("structured_outputs")
        
        # Also check if it's embedded in sampling_params (for offline/online consistency)
        if not structured_outputs and sampling_params and hasattr(sampling_params, "structured_outputs"):
            structured_outputs = sampling_params.structured_outputs

        if structured_outputs:
            # If it's a StructuredOutputsParams object, we need to convert it to the API format
            # The API expects a dictionary like {"json": schema} or {"choice": [...]}
            # We extract the set field from the params object
            
            structured_output_payload = {}
            
            # Check for known fields in StructuredOutputsParams
            # Note: In vLLM < 0.6.0 this might be different, but we target latest
            if hasattr(structured_outputs, "json") and structured_outputs.json:
                structured_output_payload["json"] = structured_outputs.json
            elif hasattr(structured_outputs, "regex") and structured_outputs.regex:
                structured_output_payload["regex"] = structured_outputs.regex
            elif hasattr(structured_outputs, "choice") and structured_outputs.choice:
                structured_output_payload["choice"] = structured_outputs.choice
            elif hasattr(structured_outputs, "grammar") and structured_outputs.grammar:
                structured_output_payload["grammar"] = structured_outputs.grammar
            elif isinstance(structured_outputs, dict):
                # If it's already a dict, pass it through (e.g. from kwargs directly)
                structured_output_payload = structured_outputs
                
            if structured_output_payload:
                payload["extra_body"] = {"structured_outputs": structured_output_payload}
                logger.debug("Added structured_outputs to extra_body")

        # Deprecated: 'guided_decoding' param removed. Use 'structured_outputs' instead.

        logger.debug(f"Sending request to {url}", extra={"payload_keys": list(payload.keys())})

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            
            with urllib.request.urlopen(req, timeout=settings.llm_server.request_timeout_seconds) as response:
                if response.status != 200:
                    raise RuntimeError(f"Server returned status {response.status}")
                
                response_body = response.read().decode("utf-8")
                response_json = json.loads(response_body)
                
                # Convert OpenAI response to vLLM-like output structure
                # vLLM local returns a list of RequestOutput objects
                # We need to mimic that structure enough for the processor to work
                return self._convert_response(response_json)

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"HTTP Error from LLM server: {e.code} - {error_body}")
            raise RuntimeError(f"LLM Server HTTP Error: {e.code}") from e
        except Exception as e:
            logger.error(f"Error communicating with LLM server: {e}")
            raise

    def _convert_response(self, response_json: Dict[str, Any]) -> List[Any]:
        """
        Convert OpenAI API response to a mock vLLM RequestOutput list.
        """
        # We need to return a list containing one object that has an 'outputs' attribute,
        # which is a list of objects with 'text', 'token_ids', 'finish_reason'.
        # Also 'prompt_token_ids'.
        
        choices = response_json.get("choices", [])
        if not choices:
            return []

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "unknown")
        
        usage = response_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # Create a mock object structure
        class MockCompletionOutput:
            def __init__(self, text, token_ids, finish_reason):
                self.text = text
                self.token_ids = token_ids # We might not have actual IDs, just count
                self.finish_reason = finish_reason

        class MockRequestOutput:
            def __init__(self, outputs, prompt_token_ids):
                self.outputs = outputs
                self.prompt_token_ids = prompt_token_ids

        # We don't have actual token IDs from standard OpenAI API, so we mock them with a list of zeros of correct length
        # This is sufficient for metrics that count tokens.
        mock_completion_output = MockCompletionOutput(
            text=content,
            token_ids=[0] * completion_tokens,
            finish_reason=finish_reason
        )
        
        mock_request_output = MockRequestOutput(
            outputs=[mock_completion_output],
            prompt_token_ids=[0] * prompt_tokens
        )
        
        return [mock_request_output]

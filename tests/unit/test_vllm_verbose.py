"""Tests for verbose output control in vLLM processor."""

import sys
from unittest.mock import MagicMock, patch
import logging


from src.processors.llm.processor import LLMProcessor


class TestVLLMVerbose:
    """Test verbose output behavior in vLLM."""

    def _test_vllm_verbose_setting(self, verbose_components, expected_log_level):
        """Helper method to test vLLM verbose setting."""
        with patch("logging.getLogger") as mock_get_logger:
            with patch(
                "src.config.loader.settings.verbose_components", verbose_components
            ):
                # Mock the vllm module
                mock_vllm = MagicMock()
                mock_llm = MagicMock()
                mock_vllm.LLM.return_value = mock_llm

                # Store original module
                original_module = sys.modules.get("vllm")
                sys.modules["vllm"] = mock_vllm

                try:
                    mock_logger = MagicMock()
                    mock_get_logger.return_value = mock_logger

                    processor = LLMProcessor()
                    processor._load_model()

                    # Check that vLLM logger was set to expected level
                    mock_get_logger.assert_called_with("vLLM")
                    mock_logger.setLevel.assert_called_with(expected_log_level)
                finally:
                    # Restore original module
                    if original_module is not None:
                        sys.modules["vllm"] = original_module
                    else:
                        sys.modules.pop("vllm", None)

    def test_vllm_verbose_disabled_in_production(self, monkeypatch):
        """vLLM should not show verbose output in production."""
        self._test_vllm_verbose_setting(False, logging.WARNING)

    def test_vllm_verbose_enabled_in_development(self, monkeypatch):
        """vLLM should show verbose output in development."""
        self._test_vllm_verbose_setting(True, logging.INFO)

    def test_vllm_verbose_respects_environment_variable(self, monkeypatch):
        """vLLM should respect VERBOSE_COMPONENTS environment variable."""
        self._test_vllm_verbose_setting(True, logging.INFO)

    def test_vllm_verbose_disabled_when_env_var_false(self, monkeypatch):
        """vLLM should not show verbose when VERBOSE_COMPONENTS=false."""
        self._test_vllm_verbose_setting(False, logging.WARNING)

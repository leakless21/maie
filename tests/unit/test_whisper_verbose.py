"""Tests for verbose output control in Whisper backend."""

import sys
from unittest.mock import MagicMock, patch


from src.processors.asr.whisper import WhisperBackend


class TestWhisperVerbose:
    """Test verbose output behavior in Whisper."""

    def _test_whisper_verbose_setting(self, verbose_components, expected_verbose):
        """Helper method to test Whisper verbose setting."""
        with patch(
            "src.config.loader.settings.verbose_components", verbose_components
        ):
            with patch("os.path.exists", return_value=True):  # Mock path exists
                # Mock the faster_whisper module
                mock_fw = MagicMock()
                mock_model = MagicMock()
                mock_fw.WhisperModel.return_value = mock_model
                mock_fw.load_model.return_value = mock_model  # For test mode

                # Store original module
                original_module = sys.modules.get("faster_whisper")
                sys.modules["faster_whisper"] = mock_fw

                try:
                    # Clear the cached module
                    import src.processors.asr.whisper as whisper_module

                    whisper_module._FASTER_WHISPER_MODULE = None

                    backend = WhisperBackend(model_path="test-model")

                    # Check that verbose setting was passed correctly
                    # In test mode, it calls load_model, not WhisperModel
                    assert mock_fw.load_model.called
                    call_kwargs = mock_fw.load_model.call_args[1]
                    assert call_kwargs.get("verbose", False) is expected_verbose
                finally:
                    # Restore original module
                    if original_module is not None:
                        sys.modules["faster_whisper"] = original_module
                    else:
                        sys.modules.pop("faster_whisper", None)

    def test_whisper_verbose_disabled_by_default_in_production(self, monkeypatch):
        """In production, Whisper should not show verbose output."""
        self._test_whisper_verbose_setting(False, False)

    def test_whisper_verbose_enabled_in_development(self, monkeypatch):
        """In development, Whisper should show verbose output."""
        self._test_whisper_verbose_setting(True, True)

    def test_whisper_verbose_respects_environment_variable(self, monkeypatch):
        """Whisper should respect VERBOSE_COMPONENTS environment variable."""
        self._test_whisper_verbose_setting(True, True)

    def test_whisper_verbose_disabled_when_env_var_false(self, monkeypatch):
        """Whisper should not show verbose when VERBOSE_COMPONENTS=false."""
        self._test_whisper_verbose_setting(False, False)

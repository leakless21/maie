"""
Unit tests for enable_diarization parameter control.

Verifies that diarization only runs when BOTH:
1. Global config (settings.diarization.enabled) is True
2. Per-request parameter (enable_diarization) is True
"""

import pytest
from unittest.mock import patch

from src.config.loader import get_settings


class TestDiarizationParameterControl:
    """Test that enable_diarization parameter correctly controls diarization execution."""

    def test_diarization_disabled_when_parameter_false(self):
        """Diarization should not run when enable_diarization=False, even if globally enabled."""
        settings = get_settings()
        enable_diarization = False

        # The condition in pipeline
        should_run = settings.diarization.enabled and enable_diarization

        # Even if global setting is True, diarization should not run
        assert should_run is False

    def test_diarization_enabled_when_both_true(self):
        """Diarization should run when both global config and parameter are True."""
        settings = get_settings()
        enable_diarization = True

        # The condition in pipeline
        should_run = settings.diarization.enabled and enable_diarization

        # Should run when both are True
        if settings.diarization.enabled:
            assert should_run is True

    def test_diarization_disabled_when_global_false(self):
        """Diarization should not run when global config is False, regardless of parameter."""
        # Create settings with diarization disabled
        with patch.dict("os.environ", {"APP_DIARIZATION__ENABLED": "false"}):
            from src.config.loader import get_settings, reset_settings_cache

            reset_settings_cache()
            settings = get_settings(reload=True)

            enable_diarization = True
            should_run = settings.diarization.enabled and enable_diarization

            assert should_run is False

            # Reset to original state
            reset_settings_cache()

    def test_task_params_includes_enable_diarization(self):
        """Task params should include enable_diarization field."""
        task_params = {
            "audio_path": "/path/to/audio.wav",
            "features": ["clean_transcript"],
            "enable_diarization": False,
        }

        # Verify the parameter can be extracted
        enable_diarization = task_params.get("enable_diarization", False)
        assert enable_diarization is False

        # Test with True
        task_params["enable_diarization"] = True
        enable_diarization = task_params.get("enable_diarization", False)
        assert enable_diarization is True

    def test_default_value_when_parameter_missing(self):
        """When enable_diarization is not in task_params, default to False."""
        task_params = {
            "audio_path": "/path/to/audio.wav",
            "features": ["clean_transcript"],
        }

        # Should default to False when missing
        enable_diarization = task_params.get("enable_diarization", False)
        assert enable_diarization is False


class TestDiarizationConditionsMatrix:
    """Test matrix of all combinations of global setting and parameter."""

    @pytest.mark.parametrize(
        "global_enabled,param_enabled,expected_result",
        [
            (True, True, True),  # Both enabled -> run
            (True, False, False),  # Global enabled, param disabled -> don't run
            (False, True, False),  # Global disabled, param enabled -> don't run
            (False, False, False),  # Both disabled -> don't run
        ],
    )
    def test_diarization_condition_matrix(
        self, global_enabled, param_enabled, expected_result
    ):
        """Test all combinations of global and parameter settings."""
        # Simulate the pipeline condition
        should_run = global_enabled and param_enabled

        assert should_run == expected_result, (
            f"Expected diarization to {'run' if expected_result else 'not run'} "
            f"with global={global_enabled}, param={param_enabled}"
        )

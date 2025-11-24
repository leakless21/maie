"""
Unit tests for the FilteringASRBackend wrapper.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.processors.base import ASRBackend, ASRResult
from src.processors.asr.filtering import FilteringASRBackend


class TestFilteringASRBackend:
    """Tests for the FilteringASRBackend class."""

    @pytest.fixture
    def mock_inner_backend(self):
        """Create a mock inner ASR backend."""
        backend = MagicMock(spec=ASRBackend)
        backend.get_version_info.return_value = {"name": "mock_backend"}
        return backend

    @pytest.fixture
    def mock_filter(self):
        """Create a mock hallucination filter."""
        with patch("src.processors.asr.filtering.create_filter_from_config") as mock_create:
            filter_instance = MagicMock()
            mock_create.return_value = filter_instance
            yield filter_instance

    @pytest.fixture
    def mock_settings(self):
        """Mock application settings to enable filtering."""
        with patch("src.processors.asr.filtering.cfg.settings") as mock_settings:
            mock_settings.asr.hallucination.enabled = True
            yield mock_settings

    def test_initialization_enables_filter(self, mock_inner_backend, mock_settings):
        """Test that filter is initialized when enabled in config."""
        with patch("src.processors.asr.filtering.create_filter_from_config") as mock_create:
            backend = FilteringASRBackend(mock_inner_backend)
            
            assert backend.filter is not None
            mock_create.assert_called_once()

    def test_initialization_disables_filter(self, mock_inner_backend):
        """Test that filter is NOT initialized when disabled in config."""
        with patch("src.processors.asr.filtering.cfg.settings") as mock_settings:
            mock_settings.asr.hallucination.enabled = False
            
            backend = FilteringASRBackend(mock_inner_backend)
            
            assert backend.filter is None

    def test_execute_delegates_and_filters(self, mock_inner_backend, mock_filter, mock_settings):
        """Test that execute calls inner backend and then applies filter."""
        # Setup inner backend result
        original_segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.0},
            {"text": "thank you", "start": 1.0, "end": 2.0},
        ]
        mock_inner_backend.execute.return_value = ASRResult(
            text="Hello world thank you",
            segments=original_segments,
            language="en",
            duration=2.0
        )

        # Setup filter result (remove "thank you")
        filtered_segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.0},
        ]
        mock_filter.filter_segments.return_value = filtered_segments

        backend = FilteringASRBackend(mock_inner_backend)
        result = backend.execute(b"audio_data")

        # Verify inner backend called
        mock_inner_backend.execute.assert_called_once_with(b"audio_data")

        # Verify filter called
        mock_filter.filter_segments.assert_called_once_with(original_segments)

        # Verify result is filtered
        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "Hello world"

    def test_execute_no_filtering_if_disabled(self, mock_inner_backend, mock_settings):
        """Test that execute just returns inner result if filter is not initialized."""
        # Disable filter
        mock_settings.asr.hallucination.enabled = False
        
        backend = FilteringASRBackend(mock_inner_backend)
        
        # Setup inner result
        inner_result = ASRResult(text="raw text", segments=[], language="en")
        mock_inner_backend.execute.return_value = inner_result
        
        result = backend.execute(b"audio")
        
        assert result is inner_result
        assert backend.filter is None

    def test_unload_delegates(self, mock_inner_backend, mock_settings):
        """Test that unload is delegated to inner backend."""
        backend = FilteringASRBackend(mock_inner_backend)
        backend.unload()
        mock_inner_backend.unload.assert_called_once()

    def test_get_version_info_delegates(self, mock_inner_backend, mock_settings):
        """Test that get_version_info is delegated to inner backend."""
        backend = FilteringASRBackend(mock_inner_backend)
        info = backend.get_version_info()
        
        mock_inner_backend.get_version_info.assert_called_once()
        assert info == {"name": "mock_backend"}

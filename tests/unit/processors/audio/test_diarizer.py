"""
Unit tests for diarization processor.

Tests cover:
- IoU calculation
- Alignment algorithm (single speaker, no speakers, multiple speakers)
- Proportional splitting without word timestamps
- Segment merging
- Graceful CUDA handling
- Error handling

Follows TDD red-green-refactor principle.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest


@dataclass
class MockDiarizationSpan:
    """Mock diarization span (from pyannote)."""

    start: float
    end: float
    speaker: str

    def __repr__(self) -> str:
        return f"MockDiarizationSpan(start={self.start}, end={self.end}, speaker={self.speaker})"


@dataclass
class MockASRSegment:
    """Mock ASR segment."""

    start: float
    end: float
    text: str
    speaker: str | None = None

    def __repr__(self) -> str:
        return f"MockASRSegment(start={self.start}, end={self.end}, text={self.text!r}, speaker={self.speaker})"


class TestDiarizer:
    """Tests for Diarizer class."""

    @pytest.fixture
    def diarizer_class(self):
        """Lazy import of Diarizer class to allow test discovery."""
        from src.processors.audio.diarizer import Diarizer

        return Diarizer

    def test_diarizer_can_be_instantiated(self, diarizer_class):
        """Test that Diarizer can be instantiated without GPU requirements in unit test."""
        # This should NOT fail if CUDA is unavailable - it's a unit test
        diarizer = diarizer_class(model_path="data/models/speaker-diarization-community-1", require_cuda=False)
        assert diarizer is not None

    def test_iou_single_overlap(self, diarizer_class):
        """Test IoU calculation for overlapping intervals - DEPRECATED."""
        pytest.skip("IoU calculation is deprecated - WhisperX uses temporal overlap")

    def test_iou_no_overlap(self, diarizer_class):
        """Test IoU calculation for non-overlapping intervals - DEPRECATED."""
        pytest.skip("IoU calculation is deprecated - WhisperX uses temporal overlap")

    def test_iou_complete_containment(self, diarizer_class):
        """Test IoU calculation when one interval contains the other - DEPRECATED."""
        pytest.skip("IoU calculation is deprecated - WhisperX uses temporal overlap")

    def test_iou_identical_intervals(self, diarizer_class):
        """Test IoU calculation for identical intervals - DEPRECATED."""
        pytest.skip("IoU calculation is deprecated - WhisperX uses temporal overlap")

    def test_align_single_speaker(self, diarizer_class):
        """Test alignment when ASR segment overlaps exactly one speaker - DEPRECATED."""
        pytest.skip("Legacy segment-level alignment is deprecated - use WhisperX-style word-level assignment")

    def test_align_no_speakers(self, diarizer_class):
        """Test alignment when ASR segment has no overlapping speakers - DEPRECATED."""
        pytest.skip("Legacy segment-level alignment is deprecated - use WhisperX-style word-level assignment")

    def test_align_multiple_speakers_dominant_assignment(self, diarizer_class):
        """Test alignment when one speaker dominates (>=0.7 of ASR segment) - DEPRECATED."""
        pytest.skip("Legacy segment-level alignment is deprecated - use WhisperX-style word-level assignment")

    def test_align_multiple_speakers_proportional_split(self, diarizer_class):
        """Test alignment with proportional split when no speaker dominates - DEPRECATED."""
        pytest.skip("Legacy segment-level alignment is deprecated - use WhisperX-style word-level assignment")

    def test_proportional_split_preserves_all_words(self, diarizer_class):
        """Test that proportional splitting doesn't lose any words - DEPRECATED."""
        pytest.skip("Legacy segment-level alignment is deprecated - use WhisperX-style word-level assignment")

    def test_merge_adjacent_same_speaker(self, diarizer_class):
        """Test merging of adjacent segments with the same speaker."""
        diarizer = diarizer_class(require_cuda=False)
        segments = [
            MockASRSegment(start=0.0, end=3.0, text="hello", speaker="S1"),
            MockASRSegment(start=3.0, end=6.0, text="world", speaker="S1"),
            MockASRSegment(start=6.0, end=9.0, text="test", speaker="S2"),
        ]

        result = diarizer.merge_adjacent_same_speaker(segments)

        assert len(result) == 2
        assert result[0].text == "hello world"  # Merged
        assert result[0].speaker == "S1"
        assert result[1].text == "test"
        assert result[1].speaker == "S2"

    def test_merge_maintains_speaker_and_timestamps(self, diarizer_class):
        """Test that merging preserves speaker info and timestamps."""
        diarizer = diarizer_class(require_cuda=False)
        segments = [
            MockASRSegment(start=1.0, end=2.0, text="first", speaker="S1"),
            MockASRSegment(start=2.0, end=3.0, text="second", speaker="S1"),
        ]

        result = diarizer.merge_adjacent_same_speaker(segments)

        assert len(result) == 1
        assert result[0].start == 1.0
        assert result[0].end == 3.0
        assert result[0].speaker == "S1"

    def test_merge_different_speakers_no_merge(self, diarizer_class):
        """Test that different speakers are not merged even if adjacent."""
        diarizer = diarizer_class(require_cuda=False)
        segments = [
            MockASRSegment(start=0.0, end=2.0, text="hello", speaker="S1"),
            MockASRSegment(start=2.0, end=4.0, text="world", speaker="S2"),
        ]

        result = diarizer.merge_adjacent_same_speaker(segments)

        assert len(result) == 2

    def test_merge_none_speaker_not_merged(self, diarizer_class):
        """Test that None speakers are not merged (they represent uncertainty)."""
        diarizer = diarizer_class(require_cuda=False)
        segments = [
            MockASRSegment(start=0.0, end=2.0, text="hello", speaker=None),
            MockASRSegment(start=2.0, end=4.0, text="world", speaker=None),
        ]

        result = diarizer.merge_adjacent_same_speaker(segments)

        # None speakers should NOT be merged (they represent uncertain speaker attribution)
        assert len(result) == 2

    def test_graceful_cuda_not_required_in_unit_test(self, diarizer_class):
        """Test that CUDA is not required in unit tests (require_cuda=False)."""
        # This should NOT raise even without CUDA
        diarizer = diarizer_class(model_path="data/models/speaker-diarization-community-1", require_cuda=False)
        assert diarizer is not None

    def test_get_diarizer_returns_none_without_cuda_when_required(self, diarizer_class):
        """Test that get_diarizer returns None gracefully when CUDA required but unavailable."""
        # This tests the graceful failure path
        from src.processors.audio.diarizer import get_diarizer

        with patch("src.processors.audio.diarizer.has_cuda", return_value=False):
            # When require_cuda=True and no CUDA, should return None
            result = get_diarizer(require_cuda=True)
            assert result is None

    def test_full_pipeline_single_speaker_multiple_segments(self, diarizer_class):
        """Test full diarization pipeline with multiple segments, single speaker - DEPRECATED."""
        pytest.skip("Legacy segment-level alignment is deprecated - use WhisperX-style word-level assignment")


class TestDiarizationIntegration:
    """Integration-like tests (still mocked, for faster unit test suite)."""

    @pytest.fixture
    def diarizer_class(self):
        from src.processors.audio.diarizer import Diarizer

        return Diarizer

    def test_diarize_returns_none_when_cuda_unavailable(self, diarizer_class):
        """Test that diarize() returns None gracefully when CUDA unavailable."""
        # Create diarizer with require_cuda=False, then manually set it to True
        diarizer = diarizer_class(require_cuda=False)

        with patch("src.processors.audio.diarizer.has_cuda", return_value=False):
            # Now set require_cuda=True for this diarizer instance
            diarizer.require_cuda = True
            # Should raise RuntimeError when trying to load
            with pytest.raises(RuntimeError):
                diarizer.diarize("/fake/audio.wav", num_speakers=None)

    def test_diarize_with_mocked_model(self, diarizer_class):
        """Test diarize method with mocked pyannote model."""
        diarizer = diarizer_class(require_cuda=False)

        # Mock the model with proper pyannote interface
        mock_segment_1 = Mock()
        mock_segment_1.start = 0.0
        mock_segment_1.end = 5.0
        
        mock_segment_2 = Mock()
        mock_segment_2.start = 5.0
        mock_segment_2.end = 10.0
        
        mock_model_output = [
            (mock_segment_1, "_", "S1"),
            (mock_segment_2, "_", "S2"),
        ]
        
        # Create a mock model that returns itself when called, and has itertracks method
        mock_model = Mock()
        mock_model.return_value = mock_model  # So model(audio_path) returns the model itself
        mock_model.itertracks = Mock(return_value=iter(mock_model_output))

        with patch.object(diarizer, "_load_pyannote_model", return_value=mock_model):
            with patch("src.processors.audio.diarizer.has_cuda", return_value=True):
                result = diarizer.diarize("/fake/audio.wav", num_speakers=2)
                # Should return diarization spans
                assert result is not None
                assert len(result) == 2
                assert result[0]["speaker"] == "S1"
                assert result[1]["speaker"] == "S2"

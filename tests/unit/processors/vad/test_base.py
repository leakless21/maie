"""Unit tests for VAD base classes and data structures."""

import pytest
from src.processors.vad.base import VADSegment, VADResult


class TestVADSegment:
    """Test VADSegment dataclass."""

    def test_vad_segment_creation(self):
        """Test basic VADSegment creation."""
        segment = VADSegment(start=0.0, end=1.5, confidence=0.95, is_speech=True)
        assert segment.start == 0.0
        assert segment.end == 1.5
        assert segment.confidence == 0.95
        assert segment.is_speech is True

    def test_vad_segment_duration(self):
        """Test VADSegment duration calculation."""
        segment = VADSegment(start=1.0, end=4.5, confidence=0.9, is_speech=True)
        assert segment.duration() == 3.5

    def test_vad_segment_silence(self):
        """Test VADSegment for silence."""
        silence = VADSegment(start=2.0, end=3.0, confidence=0.0, is_speech=False)
        assert silence.duration() == 1.0
        assert silence.is_speech is False


class TestVADResult:
    """Test VADResult dataclass."""

    def test_vad_result_creation(self):
        """Test basic VADResult creation."""
        segments = [
            VADSegment(0.0, 1.0, 0.0, False),
            VADSegment(1.0, 5.0, 1.0, True),
            VADSegment(5.0, 6.0, 0.0, False),
        ]
        result = VADResult(
            segments=segments,
            total_duration=6.0,
            speech_duration=4.0,
            speech_ratio=4.0 / 6.0,
            processing_time=0.1,
            backend_info={"backend": "test"},
        )
        assert len(result.segments) == 3
        assert result.total_duration == 6.0
        assert result.speech_duration == 4.0

    def test_vad_result_metrics(self):
        """Test VADResult metric calculations."""
        segments = [
            VADSegment(0.0, 1.0, 0.0, False),
            VADSegment(1.0, 5.0, 1.0, True),
            VADSegment(5.0, 6.0, 0.0, False),
        ]
        result = VADResult(
            segments=segments,
            total_duration=6.0,
            speech_duration=4.0,
            speech_ratio=4.0 / 6.0,
            processing_time=0.1,
            backend_info={"backend": "test"},
        )
        assert result.non_speech_duration() == 2.0
        assert result.speech_ratio == pytest.approx(0.667, abs=0.01)

    def test_get_speech_segments(self):
        """Test getting only speech segments."""
        segments = [
            VADSegment(0.0, 1.0, 0.0, False),
            VADSegment(1.0, 3.0, 1.0, True),
            VADSegment(3.0, 4.0, 0.0, False),
            VADSegment(4.0, 5.0, 1.0, True),
        ]
        result = VADResult(
            segments=segments,
            total_duration=5.0,
            speech_duration=3.0,
            speech_ratio=0.6,
            processing_time=0.1,
            backend_info={"backend": "test"},
        )
        speech_segs = result.get_speech_segments()
        assert len(speech_segs) == 2
        assert all(seg.is_speech for seg in speech_segs)

    def test_get_silence_segments(self):
        """Test getting only silence segments."""
        segments = [
            VADSegment(0.0, 1.0, 0.0, False),
            VADSegment(1.0, 3.0, 1.0, True),
            VADSegment(3.0, 4.0, 0.0, False),
        ]
        result = VADResult(
            segments=segments,
            total_duration=4.0,
            speech_duration=2.0,
            speech_ratio=0.5,
            processing_time=0.1,
            backend_info={"backend": "test"},
        )
        silence_segs = result.get_silence_segments()
        assert len(silence_segs) == 2
        assert all(not seg.is_speech for seg in silence_segs)

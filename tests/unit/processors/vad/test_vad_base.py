"""Unit tests for VAD base classes and interfaces."""

from src.processors.vad.base import VADSegment, VADResult, VADBackend


class TestVADSegment:
    """Test VADSegment data class."""

    def test_vad_segment_creation_speech(self):
        """Test creating a speech segment."""
        segment = VADSegment(start=0.0, end=1.5, confidence=0.95, is_speech=True)
        assert segment.start == 0.0
        assert segment.end == 1.5
        assert segment.confidence == 0.95
        assert segment.is_speech is True

    def test_vad_segment_creation_silence(self):
        """Test creating a silence segment."""
        segment = VADSegment(start=1.5, end=2.0, confidence=0.0, is_speech=False)
        assert segment.start == 1.5
        assert segment.end == 2.0
        assert segment.confidence == 0.0
        assert segment.is_speech is False

    def test_vad_segment_duration(self):
        """Test segment duration calculation."""
        segment = VADSegment(start=0.0, end=2.5, confidence=0.9, is_speech=True)
        assert segment.duration() == 2.5

    def test_vad_segment_zero_duration(self):
        """Test segment with zero duration."""
        segment = VADSegment(start=1.0, end=1.0, confidence=0.5, is_speech=False)
        assert segment.duration() == 0.0


class TestVADResult:
    """Test VADResult data class."""

    def test_vad_result_creation(self):
        """Test creating a VAD result."""
        segments = [
            VADSegment(0.0, 1.0, 1.0, True),
            VADSegment(1.0, 1.5, 0.0, False),
            VADSegment(1.5, 3.0, 1.0, True),
        ]
        result = VADResult(
            segments=segments,
            total_duration=3.0,
            speech_duration=2.5,
            speech_ratio=0.833,
            processing_time=0.05,
            backend_info={"backend": "silero"},
        )
        assert len(result.segments) == 3
        assert result.total_duration == 3.0
        assert result.speech_duration == 2.5
        assert result.speech_ratio == 0.833

    def test_vad_result_non_speech_duration(self):
        """Test non-speech duration calculation."""
        segments = [
            VADSegment(0.0, 1.0, 1.0, True),
            VADSegment(1.0, 1.5, 0.0, False),
        ]
        result = VADResult(
            segments=segments,
            total_duration=1.5,
            speech_duration=1.0,
            speech_ratio=2.0 / 3.0,
            processing_time=0.01,
            backend_info={},
        )
        assert result.non_speech_duration() == 0.5

    def test_vad_result_get_speech_segments(self):
        """Test filtering speech segments."""
        segments = [
            VADSegment(0.0, 1.0, 1.0, True),
            VADSegment(1.0, 1.5, 0.0, False),
            VADSegment(1.5, 3.0, 1.0, True),
        ]
        result = VADResult(
            segments=segments,
            total_duration=3.0,
            speech_duration=2.5,
            speech_ratio=0.833,
            processing_time=0.05,
            backend_info={},
        )
        speech_segs = result.get_speech_segments()
        assert len(speech_segs) == 2
        assert all(seg.is_speech for seg in speech_segs)

    def test_vad_result_get_silence_segments(self):
        """Test filtering silence segments."""
        segments = [
            VADSegment(0.0, 1.0, 1.0, True),
            VADSegment(1.0, 1.5, 0.0, False),
            VADSegment(1.5, 3.0, 1.0, True),
        ]
        result = VADResult(
            segments=segments,
            total_duration=3.0,
            speech_duration=2.5,
            speech_ratio=0.833,
            processing_time=0.05,
            backend_info={},
        )
        silence_segs = result.get_silence_segments()
        assert len(silence_segs) == 1
        assert all(not seg.is_speech for seg in silence_segs)


class MockVADBackend(VADBackend):
    """Mock VAD backend for testing."""

    def detect_speech(self, audio_path: str) -> VADResult:
        """Return mock VAD result."""
        return VADResult(
            segments=[
                VADSegment(0.0, 1.0, 1.0, True),
                VADSegment(1.0, 1.5, 0.0, False),
            ],
            total_duration=1.5,
            speech_duration=1.0,
            speech_ratio=2.0 / 3.0,
            processing_time=0.01,
            backend_info={"backend": "mock"},
        )

    def unload(self) -> None:
        """No-op unload for mock."""
        pass

    def get_version_info(self) -> dict:  # type: ignore
        """Return mock version info."""
        return {"name": "mock", "version": "1.0"}


class TestVADBackendInterface:
    """Test VAD backend interface."""

    def test_vad_backend_detect_speech(self):
        """Test VAD backend can detect speech."""
        backend = MockVADBackend()
        result = backend.detect_speech("/fake/audio.wav")
        assert isinstance(result, VADResult)
        assert result.total_duration == 1.5

    def test_vad_backend_unload(self):
        """Test VAD backend can unload."""
        backend = MockVADBackend()
        backend.unload()  # Should not raise

    def test_vad_backend_get_version_info(self):
        """Test VAD backend provides version info."""
        backend = MockVADBackend()
        info = backend.get_version_info()
        assert info["name"] == "mock"
        assert info["version"] == "1.0"

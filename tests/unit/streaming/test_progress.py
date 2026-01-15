"""Unit tests for streaming progress tracking and events."""

from __future__ import annotations

import json
import time

import pytest

from src.streaming import (
    ErrorEvent,
    ProgressEvent,
    ProgressStage,
    ProgressTracker,
    ResultEvent,
    RTF_ESTIMATES,
    SegmentEvent,
    SSEEventType,
    STAGE_RANGES,
)


class TestProgressStage:
    """Tests for ProgressStage enum."""

    def test_stage_values(self) -> None:
        """Test that all stages have expected values."""
        assert ProgressStage.QUEUED.value == "queued"
        assert ProgressStage.UPLOADING.value == "uploading"
        assert ProgressStage.PREPROCESSING.value == "preprocessing"
        assert ProgressStage.TRANSCRIBING.value == "transcribing"
        assert ProgressStage.POSTPROCESSING.value == "postprocessing"
        assert ProgressStage.COMPLETED.value == "completed"
        assert ProgressStage.FAILED.value == "failed"

    def test_stage_is_string_enum(self) -> None:
        """Test that ProgressStage can be used as string."""
        assert str(ProgressStage.QUEUED) == "ProgressStage.QUEUED"
        assert ProgressStage.QUEUED.value == "queued"


class TestSSEEventType:
    """Tests for SSEEventType enum."""

    def test_event_types(self) -> None:
        """Test all event types exist."""
        assert SSEEventType.PROGRESS.value == "progress"
        assert SSEEventType.SEGMENT.value == "segment"
        assert SSEEventType.RESULT.value == "result"
        assert SSEEventType.ERROR.value == "error"


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_basic_event(self) -> None:
        """Test basic progress event creation."""
        event = ProgressEvent(
            stage=ProgressStage.UPLOADING,
            progress=5.0,
        )
        assert event.stage == ProgressStage.UPLOADING
        assert event.progress == 5.0
        assert event.message is None
        assert event.details is None

    def test_event_with_message_and_details(self) -> None:
        """Test event with optional fields."""
        event = ProgressEvent(
            stage=ProgressStage.TRANSCRIBING,
            progress=45.5,
            message="Processing audio...",
            details={"segment_count": 5},
        )
        assert event.message == "Processing audio..."
        assert event.details == {"segment_count": 5}

    def test_to_sse_data_serialization(self) -> None:
        """Test JSON serialization for SSE."""
        event = ProgressEvent(
            stage=ProgressStage.PREPROCESSING,
            progress=15.123456,
            message="Analyzing...",
        )
        data = json.loads(event.to_sse_data())

        assert data["stage"] == "preprocessing"
        assert data["progress"] == 15.1  # Rounded to 1 decimal
        assert data["message"] == "Analyzing..."
        assert data["details"] is None


class TestSegmentEvent:
    """Tests for SegmentEvent dataclass."""

    def test_segment_creation(self) -> None:
        """Test segment event creation."""
        event = SegmentEvent(
            index=1,
            start=0.0,
            end=5.5,
            text="Hello world",
        )
        assert event.index == 1
        assert event.start == 0.0
        assert event.end == 5.5
        assert event.text == "Hello world"
        assert event.speaker is None

    def test_segment_with_speaker(self) -> None:
        """Test segment with speaker info."""
        event = SegmentEvent(
            index=2,
            start=5.5,
            end=10.0,
            text="Speaker two",
            speaker="SPEAKER_01",
        )
        assert event.speaker == "SPEAKER_01"

    def test_to_sse_data(self) -> None:
        """Test segment serialization."""
        event = SegmentEvent(
            index=1,
            start=0.0,
            end=3.5,
            text="Test",
            speaker="SPK_0",
        )
        data = json.loads(event.to_sse_data())

        assert data["index"] == 1
        assert data["start"] == 0.0
        assert data["end"] == 3.5
        assert data["text"] == "Test"
        assert data["speaker"] == "SPK_0"


class TestResultEvent:
    """Tests for ResultEvent dataclass."""

    def test_result_creation(self) -> None:
        """Test result event creation."""
        segments = [{"start": 0, "end": 5, "text": "Hello"}]
        event = ResultEvent(
            task_id="abc123",
            transcript="Hello world",
            segments=segments,
            duration_seconds=10.5,
            processing_time_seconds=3.2,
            asr_backend="whisper",
        )

        assert event.task_id == "abc123"
        assert event.transcript == "Hello world"
        assert event.segments == segments
        assert event.duration_seconds == 10.5
        assert event.processing_time_seconds == 3.2
        assert event.asr_backend == "whisper"
        assert event.language is None

    def test_to_sse_data(self) -> None:
        """Test result serialization."""
        event = ResultEvent(
            task_id="test-123",
            transcript="Test transcript",
            segments=[],
            duration_seconds=5.0,
            processing_time_seconds=1.5,
            asr_backend="chunkformer",
            language="vi",
        )
        data = json.loads(event.to_sse_data())

        assert data["task_id"] == "test-123"
        assert data["language"] == "vi"
        assert data["asr_backend"] == "chunkformer"


class TestErrorEvent:
    """Tests for ErrorEvent dataclass."""

    def test_error_creation(self) -> None:
        """Test error event creation."""
        event = ErrorEvent(error="Something went wrong")
        assert event.error == "Something went wrong"
        assert event.error_code is None
        assert event.stage is None

    def test_error_with_code_and_stage(self) -> None:
        """Test error with code and stage."""
        event = ErrorEvent(
            error="File not found",
            error_code="FILE_NOT_FOUND",
            stage=ProgressStage.UPLOADING,
        )
        assert event.error_code == "FILE_NOT_FOUND"
        assert event.stage == ProgressStage.UPLOADING

    def test_to_sse_data(self) -> None:
        """Test error serialization."""
        event = ErrorEvent(
            error="Test error",
            error_code="TEST_ERR",
            stage=ProgressStage.TRANSCRIBING,
        )
        data = json.loads(event.to_sse_data())

        assert data["error"] == "Test error"
        assert data["error_code"] == "TEST_ERR"
        assert data["stage"] == "transcribing"


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_initial_state(self) -> None:
        """Test tracker initial state."""
        tracker = ProgressTracker(audio_duration=60.0)

        assert tracker.audio_duration == 60.0
        assert tracker.asr_backend == "whisper"
        assert tracker.current_stage == ProgressStage.QUEUED
        assert tracker.current_progress == 0.0
        assert len(tracker.segments) == 0

    def test_set_stage_transition(self) -> None:
        """Test stage transitions."""
        tracker = ProgressTracker(audio_duration=60.0)

        event = tracker.set_stage(ProgressStage.UPLOADING)
        assert event.stage == ProgressStage.UPLOADING
        assert event.progress == 0.0  # Start of uploading range

        event = tracker.set_stage(ProgressStage.PREPROCESSING)
        assert event.stage == ProgressStage.PREPROCESSING
        assert event.progress == 10.0  # Start of preprocessing range

    def test_update_progress_within_stage(self) -> None:
        """Test progress updates within a stage."""
        tracker = ProgressTracker(audio_duration=60.0)
        tracker.set_stage(ProgressStage.TRANSCRIBING)

        # 50% through transcribing (20-95% range)
        event = tracker.update_progress(50.0)

        # Expected: 20 + (50/100) * (95 - 20) = 20 + 37.5 = 57.5
        expected = 20.0 + (50.0 / 100.0) * (95.0 - 20.0)
        assert abs(event.progress - expected) < 0.1

    def test_progress_clamped_to_stage_max(self) -> None:
        """Test that progress is clamped to stage maximum."""
        tracker = ProgressTracker(audio_duration=60.0)
        tracker.set_stage(ProgressStage.PREPROCESSING)

        # Try to exceed stage range
        event = tracker.update_progress(150.0)

        # Should be clamped to end of preprocessing range (20%)
        assert event.progress <= 20.0

    def test_add_segment_updates_progress(self) -> None:
        """Test that adding segments updates progress."""
        tracker = ProgressTracker(audio_duration=60.0)
        tracker.set_stage(ProgressStage.TRANSCRIBING)

        # Add segment at 30s of 60s audio (50% through audio)
        event = tracker.add_segment({"start": 0, "end": 30, "text": "Hello"})

        # Should be ~50% through transcribing stage
        assert 50 < event.progress < 60
        assert len(tracker.segments) == 1

    def test_elapsed_time(self) -> None:
        """Test elapsed time tracking."""
        tracker = ProgressTracker(audio_duration=60.0)

        # Small sleep to ensure measurable time
        time.sleep(0.01)

        assert tracker.elapsed_time > 0

    def test_callback_notification(self) -> None:
        """Test callback notification on progress updates."""
        tracker = ProgressTracker(audio_duration=60.0)
        events_received: list[ProgressEvent] = []

        def callback(event: ProgressEvent) -> None:
            events_received.append(event)

        tracker.add_callback(callback)
        tracker.set_stage(ProgressStage.UPLOADING)
        tracker.update_progress(50.0)

        assert len(events_received) == 2
        assert events_received[0].stage == ProgressStage.UPLOADING
        assert events_received[1].progress > 0


class TestStageRanges:
    """Tests for stage range constants."""

    def test_all_stages_have_ranges(self) -> None:
        """Test that all stages have defined ranges."""
        expected_stages = [
            ProgressStage.QUEUED,
            ProgressStage.UPLOADING,
            ProgressStage.PREPROCESSING,
            ProgressStage.TRANSCRIBING,
            ProgressStage.POSTPROCESSING,
            ProgressStage.COMPLETED,
        ]
        for stage in expected_stages:
            assert stage in STAGE_RANGES

    def test_ranges_are_sequential(self) -> None:
        """Test that stage ranges follow logical order."""
        assert STAGE_RANGES[ProgressStage.QUEUED] == (0.0, 0.0)
        assert STAGE_RANGES[ProgressStage.UPLOADING][0] == 0.0
        assert STAGE_RANGES[ProgressStage.UPLOADING][1] <= STAGE_RANGES[ProgressStage.PREPROCESSING][0]
        assert STAGE_RANGES[ProgressStage.COMPLETED] == (100.0, 100.0)


class TestRTFEstimates:
    """Tests for RTF estimate constants."""

    def test_rtf_estimates_exist(self) -> None:
        """Test that RTF estimates are defined."""
        assert "whisper" in RTF_ESTIMATES
        assert "chunkformer" in RTF_ESTIMATES
        assert "whisper_cpu" in RTF_ESTIMATES

    def test_rtf_values_reasonable(self) -> None:
        """Test that RTF values are within reasonable range."""
        for backend, rtf in RTF_ESTIMATES.items():
            assert 0.0 < rtf < 10.0, f"RTF for {backend} seems unreasonable: {rtf}"

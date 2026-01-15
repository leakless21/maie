"""Integration tests for Edge API streaming endpoint."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest


class TestStreamingEndpoint:
    """Tests for /v1/transcribe/stream endpoint."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for tests."""
        settings = MagicMock()
        settings.paths.audio_dir = MagicMock()
        settings.paths.audio_dir.__truediv__ = MagicMock(
            return_value=MagicMock(
                mkdir=MagicMock(),
                __truediv__=MagicMock(
                    return_value=MagicMock(write_bytes=MagicMock())
                ),
            )
        )
        settings.environment = "test"
        settings.debug = False
        settings.api.max_file_size_mb = 100
        return settings

    @pytest.fixture
    def sample_audio_bytes(self) -> bytes:
        """Create minimal valid WAV file bytes for testing."""
        # Minimal WAV header + some audio data
        wav_header = bytes([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x24, 0x00, 0x00, 0x00,  # ChunkSize
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # Subchunk1Size (16 for PCM)
            0x01, 0x00,              # AudioFormat (1 = PCM)
            0x01, 0x00,              # NumChannels (1 = mono)
            0x44, 0xAC, 0x00, 0x00,  # SampleRate (44100)
            0x88, 0x58, 0x01, 0x00,  # ByteRate
            0x02, 0x00,              # BlockAlign
            0x10, 0x00,              # BitsPerSample (16)
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x00, 0x00, 0x00, 0x00,  # Subchunk2Size
        ])
        return wav_header

    def test_streaming_handler_creation(self):
        """Test that StreamingTranscriptionHandler can be created."""
        from src.streaming import StreamingTranscriptionHandler

        handler = StreamingTranscriptionHandler(
            task_id="test-123",
            asr_backend="whisper",
            audio_duration=60.0,
        )

        assert handler.task_id == "test-123"
        assert handler.asr_backend == "whisper"
        assert handler.tracker.audio_duration == 60.0

    @pytest.mark.asyncio
    async def test_emit_progress(self):
        """Test emitting progress events."""
        from src.streaming import ProgressStage, StreamingTranscriptionHandler

        handler = StreamingTranscriptionHandler(
            task_id="test-123",
            asr_backend="whisper",
        )

        # First call sets stage to UPLOADING (starts at 0%)
        msg = await handler.emit_progress(
            ProgressStage.UPLOADING,
            message="Starting upload...",
        )

        assert msg["event"] == "progress"
        assert "data" in msg
        data = json.loads(msg["data"])
        assert data["stage"] == "uploading"
        assert data["progress"] == 0.0  # Start of uploading range

        # Second call updates progress within stage
        msg2 = await handler.emit_progress(
            ProgressStage.UPLOADING,
            progress_in_stage=50.0,
            message="Uploading file...",
        )
        data2 = json.loads(msg2["data"])
        assert data2["progress"] > 0  # Should be 5.0 (50% of 0-10% range)

    @pytest.mark.asyncio
    async def test_emit_segment(self):
        """Test emitting segment events."""
        from src.streaming import StreamingTranscriptionHandler

        handler = StreamingTranscriptionHandler(
            task_id="test-123",
            asr_backend="whisper",
            audio_duration=60.0,
        )

        segment = {
            "start": 0.0,
            "end": 5.5,
            "text": "Hello world",
            "speaker": "SPEAKER_00",
        }
        msg = await handler.emit_segment(segment)

        assert msg["event"] == "segment"
        data = json.loads(msg["data"])
        assert data["start"] == 0.0
        assert data["end"] == 5.5
        assert data["text"] == "Hello world"
        assert data["speaker"] == "SPEAKER_00"

    @pytest.mark.asyncio
    async def test_emit_result(self):
        """Test emitting result events."""
        from src.streaming import StreamingTranscriptionHandler

        handler = StreamingTranscriptionHandler(
            task_id="test-123",
            asr_backend="whisper",
            audio_duration=60.0,
        )

        msg = await handler.emit_result(
            transcript="Hello world transcript",
            segments=[{"start": 0, "end": 5, "text": "Hello"}],
            processing_time=3.5,
            language="en",
        )

        assert msg["event"] == "result"
        data = json.loads(msg["data"])
        assert data["task_id"] == "test-123"
        assert data["transcript"] == "Hello world transcript"
        assert data["processing_time_seconds"] == 3.5
        assert data["language"] == "en"

    @pytest.mark.asyncio
    async def test_emit_error(self):
        """Test emitting error events."""
        from src.streaming import StreamingTranscriptionHandler

        handler = StreamingTranscriptionHandler(
            task_id="test-123",
            asr_backend="whisper",
        )

        msg = await handler.emit_error(
            error="Something went wrong",
            error_code="TEST_ERROR",
        )

        assert msg["event"] == "error"
        data = json.loads(msg["data"])
        assert data["error"] == "Something went wrong"
        assert data["error_code"] == "TEST_ERROR"

    @pytest.mark.asyncio
    async def test_run_with_progress_pipeline(self):
        """Test full progress pipeline execution."""
        from src.streaming import ProgressStage, StreamingTranscriptionHandler

        handler = StreamingTranscriptionHandler(
            task_id="test-123",
            asr_backend="whisper",
            audio_duration=10.0,
        )

        # Mock functions
        upload_called = False
        preprocess_called = False
        transcribe_called = False

        async def mock_upload():
            nonlocal upload_called
            upload_called = True

        async def mock_preprocess():
            nonlocal preprocess_called
            preprocess_called = True
            return {"duration": 10.0}

        async def mock_transcribe():
            nonlocal transcribe_called
            transcribe_called = True
            return {
                "transcript": "Test transcript",
                "segments": [{"start": 0, "end": 5, "text": "Test"}],
                "language": "en",
            }

        # Collect events
        events: list[dict[str, Any]] = []
        async for msg in handler.run_with_progress(
            upload_func=mock_upload,
            preprocess_func=mock_preprocess,
            transcribe_func=mock_transcribe,
            emit_segments=True,
        ):
            events.append(msg)

        # Verify all stages were processed
        assert upload_called
        assert preprocess_called
        assert transcribe_called

        # Verify events were emitted
        assert len(events) > 0

        # Parse all event data
        stages_seen = set()
        for event in events:
            data = json.loads(event["data"])
            if "stage" in data:
                stages_seen.add(data["stage"])

        # Should see key stages
        assert "queued" in stages_seen
        assert "uploading" in stages_seen
        assert "preprocessing" in stages_seen
        assert "transcribing" in stages_seen
        # completed shows up in result event

        # Last event should be result
        last_event = events[-1]
        assert last_event["event"] == "result"


class TestProgressTrackerIntegration:
    """Integration tests for ProgressTracker with handlers."""

    @pytest.mark.asyncio
    async def test_tracker_callback_integration(self):
        """Test tracker callbacks work with handler."""
        from src.streaming import ProgressStage, ProgressTracker

        tracker = ProgressTracker(audio_duration=30.0, asr_backend="whisper")
        callback_events: list = []

        def on_progress(event):
            callback_events.append(event)

        tracker.add_callback(on_progress)

        # Simulate pipeline
        tracker.set_stage(ProgressStage.UPLOADING)
        tracker.update_progress(100)
        tracker.set_stage(ProgressStage.PREPROCESSING)
        tracker.update_progress(100)
        tracker.set_stage(ProgressStage.TRANSCRIBING)
        tracker.add_segment({"start": 0, "end": 15, "text": "Hello"})
        tracker.set_stage(ProgressStage.COMPLETED)

        # Should have received callbacks for all progress events
        assert len(callback_events) >= 5

    def test_segment_progress_calculation(self):
        """Test that segment progress is calculated correctly."""
        from src.streaming import ProgressStage, ProgressTracker

        tracker = ProgressTracker(audio_duration=100.0, asr_backend="whisper")
        tracker.set_stage(ProgressStage.TRANSCRIBING)

        # Add segment at 25% through audio
        event = tracker.add_segment({"start": 0, "end": 25, "text": "..."})

        # Transcribing is 20-95% range
        # 25% of audio = 25% through stage
        # Expected: 20 + 0.25 * 75 = 38.75
        assert 35 < event.progress < 45

    def test_time_estimation_with_rtf(self):
        """Test time-based progress estimation."""
        from src.streaming import ProgressStage, ProgressTracker, RTF_ESTIMATES
        import time

        tracker = ProgressTracker(audio_duration=60.0, asr_backend="whisper")
        tracker.set_stage(ProgressStage.TRANSCRIBING)

        # Manually set stage start time to simulate elapsed time
        tracker._stage_start_time = time.time() - 10  # 10 seconds elapsed

        event = tracker.estimate_progress_by_time()

        # Should show progress based on estimated RTF
        # 10 seconds elapsed, RTF ~0.3, 60s audio
        # estimated_total = 60 * 0.3 = 18s
        # progress = 10 / 18 * 100 ≈ 55% through stage
        assert event.progress > 20  # Past start of transcribing

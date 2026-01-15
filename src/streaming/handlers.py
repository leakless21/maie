"""Streaming handler utilities for SSE endpoints.

This module provides the StreamingTranscriptionHandler class and utilities
for implementing SSE-based progress streaming endpoints.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, Optional

from src.streaming.events import (
    ErrorEvent,
    ProgressEvent,
    ProgressStage,
    ResultEvent,
    SegmentEvent,
    SSEEventType,
)
from src.streaming.progress import ProgressTracker


async def create_sse_message(
    event_type: SSEEventType,
    data: str,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an SSE message dictionary.

    Compatible with Litestar's ServerSentEventMessage.

    Args:
        event_type: Type of SSE event
        data: JSON data string
        event_id: Optional event ID for client tracking

    Returns:
        Dict compatible with ServerSentEventMessage
    """
    return {
        "event": event_type.value,
        "data": data,
        "id": event_id,
    }


class StreamingTranscriptionHandler:
    """Handler for streaming transcription with progress updates.

    This handler manages the lifecycle of a streaming transcription request,
    emitting progress events, segment events, and final results via SSE.

    Usage:
        handler = StreamingTranscriptionHandler(
            task_id="abc123",
            asr_backend="whisper",
        )

        async for message in handler.process(audio_path, process_func):
            yield ServerSentEventMessage(**message)

    Attributes:
        task_id: Unique task identifier
        asr_backend: ASR backend being used
        tracker: Progress tracker instance
    """

    def __init__(
        self,
        task_id: str,
        asr_backend: str = "whisper",
        audio_duration: float = 0.0,
    ):
        """Initialize the streaming handler.

        Args:
            task_id: Unique identifier for this transcription task
            asr_backend: ASR backend to use ('whisper' or 'chunkformer')
            audio_duration: Duration of audio in seconds (if known)
        """
        self.task_id = task_id
        self.asr_backend = asr_backend
        self.tracker = ProgressTracker(
            audio_duration=audio_duration,
            asr_backend=asr_backend,
        )
        self._event_id = 0

    def _next_event_id(self) -> str:
        """Generate the next sequential event ID."""
        self._event_id += 1
        return f"{self.task_id}-{self._event_id}"

    async def emit_progress(
        self,
        stage: ProgressStage,
        progress_in_stage: float = 0.0,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a progress event.

        Args:
            stage: Processing stage
            progress_in_stage: Progress within stage (0-100)
            message: Optional status message

        Returns:
            SSE message dict
        """
        if self.tracker.current_stage != stage:
            event = self.tracker.set_stage(stage, message)
        else:
            event = self.tracker.update_progress(progress_in_stage, message)

        return await create_sse_message(
            SSEEventType.PROGRESS,
            event.to_sse_data(),
            self._next_event_id(),
        )

    async def emit_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Emit a segment event and update progress.

        Args:
            segment: Segment dict with start, end, text, optional speaker

        Returns:
            SSE message dict
        """
        # Update tracker
        self.tracker.add_segment(segment)

        # Create segment event
        seg_event = SegmentEvent(
            index=len(self.tracker.segments),
            start=segment.get("start", 0.0),
            end=segment.get("end", 0.0),
            text=segment.get("text", ""),
            speaker=segment.get("speaker"),
        )

        return await create_sse_message(
            SSEEventType.SEGMENT,
            seg_event.to_sse_data(),
            self._next_event_id(),
        )

    async def emit_result(
        self,
        transcript: str,
        segments: list,
        processing_time: float,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit the final result event.

        Args:
            transcript: Full transcript text
            segments: List of transcript segments
            processing_time: Total processing time in seconds
            language: Detected or specified language

        Returns:
            SSE message dict
        """
        self.tracker.set_stage(ProgressStage.COMPLETED)

        result_event = ResultEvent(
            task_id=self.task_id,
            transcript=transcript,
            segments=segments,
            duration_seconds=self.tracker.audio_duration,
            processing_time_seconds=processing_time,
            asr_backend=self.asr_backend,
            language=language,
        )

        return await create_sse_message(
            SSEEventType.RESULT,
            result_event.to_sse_data(),
            self._next_event_id(),
        )

    async def emit_error(
        self,
        error: str,
        error_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit an error event.

        Args:
            error: Error message
            error_code: Optional error code

        Returns:
            SSE message dict
        """
        error_event = ErrorEvent(
            error=error,
            error_code=error_code,
            stage=self.tracker.current_stage,
        )

        return await create_sse_message(
            SSEEventType.ERROR,
            error_event.to_sse_data(),
            self._next_event_id(),
        )

    async def run_with_progress(
        self,
        upload_func: Callable[[], Any],
        preprocess_func: Callable[[], Any],
        transcribe_func: Callable[[], Any],
        emit_segments: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the full transcription pipeline with progress streaming.

        This method orchestrates the entire transcription process,
        yielding SSE messages for each stage and progress update.

        Args:
            upload_func: Async function to handle file upload/save
            preprocess_func: Async function to preprocess audio (returns dict with 'duration')
            transcribe_func: Async function to run transcription
            emit_segments: Whether to emit individual segment events

        Yields:
            SSE message dictionaries compatible with ServerSentEventMessage
        """
        start_time = time.time()

        try:
            # Stage 1: Queued
            yield await self.emit_progress(
                ProgressStage.QUEUED, message="Request received"
            )

            # Stage 2: Uploading
            yield await self.emit_progress(
                ProgressStage.UPLOADING, message="Saving audio file"
            )
            await upload_func()
            yield await self.emit_progress(
                ProgressStage.UPLOADING, 100, "File saved"
            )

            # Stage 3: Preprocessing
            yield await self.emit_progress(
                ProgressStage.PREPROCESSING, message="Analyzing audio"
            )
            audio_info = await preprocess_func()
            if audio_info and "duration" in audio_info:
                self.tracker.audio_duration = audio_info["duration"]
            yield await self.emit_progress(
                ProgressStage.PREPROCESSING, 100, "Audio ready"
            )

            # Stage 4: Transcribing
            yield await self.emit_progress(
                ProgressStage.TRANSCRIBING, message="Starting transcription"
            )

            result = await transcribe_func()

            # Handle streaming segments if available (async generator)
            if hasattr(result, "__aiter__"):
                async for item in result:
                    if item.get("type") == "segment" and emit_segments:
                        yield await self.emit_segment(item["segment"])
                    elif item.get("type") == "progress":
                        yield await self.emit_progress(
                            ProgressStage.TRANSCRIBING,
                            item.get("progress", 50),
                        )
                    elif item.get("type") == "complete":
                        result = item["result"]
                        break

            # Stage 5: Completed
            processing_time = time.time() - start_time
            yield await self.emit_result(
                transcript=result.get("transcript", ""),
                segments=result.get("segments", []),
                processing_time=processing_time,
                language=result.get("language"),
            )

        except Exception as e:
            yield await self.emit_error(str(e), error_code="TRANSCRIPTION_ERROR")
            raise

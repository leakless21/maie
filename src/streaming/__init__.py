"""Streaming module for real-time progress updates via SSE.

This module provides backend-agnostic progress tracking and SSE event
handling for audio transcription pipelines.

Example usage:
    from src.streaming import StreamingTranscriptionHandler, ProgressStage

    handler = StreamingTranscriptionHandler(task_id="abc", asr_backend="whisper")

    async for message in handler.run_with_progress(...):
        yield ServerSentEventMessage(**message)
"""

from src.streaming.events import (
    ErrorEvent,
    ProgressEvent,
    ProgressStage,
    ResultEvent,
    SegmentEvent,
    SSEEventType,
)
from src.streaming.handlers import (
    StreamingTranscriptionHandler,
    create_sse_message,
)
from src.streaming.progress import (
    ProgressTracker,
    RTF_ESTIMATES,
    STAGE_RANGES,
)

__all__ = [
    # Events
    "ProgressStage",
    "SSEEventType",
    "ProgressEvent",
    "SegmentEvent",
    "ResultEvent",
    "ErrorEvent",
    # Progress
    "ProgressTracker",
    "RTF_ESTIMATES",
    "STAGE_RANGES",
    # Handlers
    "StreamingTranscriptionHandler",
    "create_sse_message",
]

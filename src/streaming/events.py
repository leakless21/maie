"""SSE Event definitions for streaming progress.

This module defines the event types and data structures used for
Server-Sent Events (SSE) based progress streaming in audio transcription.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ProgressStage(str, Enum):
    """Processing stages with progress ranges.

    Progress ranges:
        - QUEUED: 0%
        - UPLOADING: 0-10%
        - PREPROCESSING: 10-20%
        - TRANSCRIBING: 20-95%
        - POSTPROCESSING: 95-99%
        - COMPLETED: 100%
        - FAILED: N/A
    """

    QUEUED = "queued"
    UPLOADING = "uploading"
    PREPROCESSING = "preprocessing"
    TRANSCRIBING = "transcribing"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"


class SSEEventType(str, Enum):
    """SSE event types for transcription streaming."""

    PROGRESS = "progress"
    SEGMENT = "segment"
    RESULT = "result"
    ERROR = "error"


@dataclass
class ProgressEvent:
    """Progress update event.

    Attributes:
        stage: Current processing stage
        progress: Progress percentage (0-100)
        message: Optional status message
        details: Optional additional details
    """

    stage: ProgressStage
    progress: float
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_sse_data(self) -> str:
        """Serialize to SSE data string."""
        return json.dumps(
            {
                "stage": self.stage.value,
                "progress": round(self.progress, 1),
                "message": self.message,
                "details": self.details,
            }
        )


@dataclass
class SegmentEvent:
    """Transcription segment event.

    Emitted when a segment of the transcript is available.

    Attributes:
        index: Segment index (1-based)
        start: Start timestamp in seconds
        end: End timestamp in seconds
        text: Transcribed text
        speaker: Optional speaker ID
    """

    index: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    def to_sse_data(self) -> str:
        """Serialize to SSE data string."""
        return json.dumps(asdict(self))


@dataclass
class ResultEvent:
    """Final result event.

    Emitted when transcription is complete.

    Attributes:
        task_id: Unique task identifier
        transcript: Full transcript text
        segments: List of transcript segments
        duration_seconds: Audio duration in seconds
        processing_time_seconds: Total processing time
        asr_backend: ASR backend used
        language: Detected or specified language
    """

    task_id: str
    transcript: str
    segments: List[Dict[str, Any]]
    duration_seconds: float
    processing_time_seconds: float
    asr_backend: str
    language: Optional[str] = None

    def to_sse_data(self) -> str:
        """Serialize to SSE data string."""
        return json.dumps(asdict(self))


@dataclass
class ErrorEvent:
    """Error event.

    Emitted when an error occurs during processing.

    Attributes:
        error: Error message
        error_code: Optional error code
        stage: Stage where error occurred
    """

    error: str
    error_code: Optional[str] = None
    stage: Optional[ProgressStage] = None

    def to_sse_data(self) -> str:
        """Serialize to SSE data string."""
        return json.dumps(
            {
                "error": self.error,
                "error_code": self.error_code,
                "stage": self.stage.value if self.stage else None,
            }
        )

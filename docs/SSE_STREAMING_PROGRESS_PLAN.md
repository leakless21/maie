# SSE Streaming Progress API - Implementation Plan

## Overview

This document outlines the plan to implement Server-Sent Events (SSE) based progress streaming for audio transcription. The design is **backend-agnostic** and can be used by both the Edge API (Jetson) and the Main API.

## Goals

1. **Real-time progress updates** - Stream progress to clients without polling
2. **Backend-agnostic** - Works with any ASR backend (Whisper, ChunkFormer, future backends)
3. **Reusable module** - Shared progress tracking logic between Edge and Main APIs
4. **Flexible architecture** - Easy to extend with new stages or backends

## Why Chunk-Based Progress (Option B)?

| Aspect             | Segment-Based (Option A)       | Chunk-Based (Option B) ✓ |
| ------------------ | ------------------------------ | ------------------------ |
| Backend dependency | Requires ASR to yield segments | Works with any backend   |
| Predictability     | Depends on ASR internals       | Consistent stage-based   |
| Maintainability    | Needs per-backend logic        | Single implementation    |
| Extensibility      | Hard to add new backends       | Easy to add new backends |

---

## Architecture

### Module Structure

```
src/
├── streaming/                    # NEW: Reusable streaming module
│   ├── __init__.py
│   ├── progress.py              # Progress tracking & estimation
│   ├── events.py                # SSE event definitions
│   └── handlers.py              # Streaming handler utilities
├── api/
│   ├── edge_main.py             # Edge API (uses streaming module)
│   └── routes.py                # Main API (uses streaming module)
```

### Progress Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐    ┌───────────┐
│   queued    │ -> │  uploading   │ -> │preprocessing│ -> │ transcribing  │ -> │ completed │
│    0%       │    │   0-10%      │    │   10-20%    │    │    20-95%     │    │   100%    │
└─────────────┘    └──────────────┘    └─────────────┘    └───────────────┘    └───────────┘
```

---

## Phase 1: Core Streaming Module

### 1.1 Event Definitions

**File:** `src/streaming/events.py`

```python
"""SSE Event definitions for streaming progress."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class ProgressStage(str, Enum):
    """Processing stages with progress ranges."""
    QUEUED = "queued"              # 0%
    UPLOADING = "uploading"        # 0-10%
    PREPROCESSING = "preprocessing" # 10-20%
    TRANSCRIBING = "transcribing"  # 20-95%
    POSTPROCESSING = "postprocessing"  # 95-99% (optional: diarization, etc.)
    COMPLETED = "completed"        # 100%
    FAILED = "failed"              # N/A


class SSEEventType(str, Enum):
    """SSE event types."""
    PROGRESS = "progress"
    SEGMENT = "segment"
    RESULT = "result"
    ERROR = "error"


@dataclass
class ProgressEvent:
    """Progress update event."""
    stage: ProgressStage
    progress: float  # 0-100
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_sse_data(self) -> str:
        return json.dumps({
            "stage": self.stage.value,
            "progress": round(self.progress, 1),
            "message": self.message,
            "details": self.details,
        })


@dataclass
class SegmentEvent:
    """Transcription segment event (optional real-time segments)."""
    index: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    def to_sse_data(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ResultEvent:
    """Final result event."""
    task_id: str
    transcript: str
    segments: List[Dict[str, Any]]
    duration_seconds: float
    processing_time_seconds: float
    asr_backend: str
    language: Optional[str] = None

    def to_sse_data(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ErrorEvent:
    """Error event."""
    error: str
    error_code: Optional[str] = None
    stage: Optional[ProgressStage] = None

    def to_sse_data(self) -> str:
        return json.dumps({
            "error": self.error,
            "error_code": self.error_code,
            "stage": self.stage.value if self.stage else None,
        })
```

### 1.2 Progress Tracking

**File:** `src/streaming/progress.py`

```python
"""Progress tracking and estimation utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from enum import Enum

from src.streaming.events import ProgressStage, ProgressEvent


# Real-Time Factor estimates per backend (processing_time / audio_duration)
# Lower = faster. These are approximate values for GPU.
RTF_ESTIMATES: Dict[str, float] = {
    "whisper": 0.3,       # ~3x real-time on GPU
    "chunkformer": 0.2,   # ~5x real-time on GPU
    "whisper_cpu": 1.5,   # ~0.7x real-time on CPU
}

# Progress ranges for each stage
STAGE_RANGES: Dict[ProgressStage, tuple[float, float]] = {
    ProgressStage.QUEUED: (0.0, 0.0),
    ProgressStage.UPLOADING: (0.0, 10.0),
    ProgressStage.PREPROCESSING: (10.0, 20.0),
    ProgressStage.TRANSCRIBING: (20.0, 95.0),
    ProgressStage.POSTPROCESSING: (95.0, 99.0),
    ProgressStage.COMPLETED: (100.0, 100.0),
}


@dataclass
class ProgressTracker:
    """
    Tracks progress through processing stages.

    Usage:
        tracker = ProgressTracker(audio_duration=60.0, asr_backend="whisper")

        async for event in tracker.run_with_progress(process_func):
            yield event.to_sse_data()
    """

    audio_duration: float = 0.0
    asr_backend: str = "whisper"

    # Internal state
    _stage: ProgressStage = field(default=ProgressStage.QUEUED, init=False)
    _progress: float = field(default=0.0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)
    _stage_start_time: float = field(default_factory=time.time, init=False)
    _segments: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _callbacks: List[Callable[[ProgressEvent], None]] = field(default_factory=list, init=False)

    def add_callback(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Add a callback to be called on progress updates."""
        self._callbacks.append(callback)

    def _notify(self, event: ProgressEvent) -> None:
        """Notify all callbacks of progress update."""
        for callback in self._callbacks:
            callback(event)

    def set_stage(self, stage: ProgressStage, message: Optional[str] = None) -> ProgressEvent:
        """
        Transition to a new stage.

        Returns the progress event for the new stage.
        """
        self._stage = stage
        self._stage_start_time = time.time()

        # Set progress to start of stage range
        stage_start, _ = STAGE_RANGES.get(stage, (0.0, 100.0))
        self._progress = stage_start

        event = ProgressEvent(
            stage=stage,
            progress=self._progress,
            message=message or f"Stage: {stage.value}",
        )
        self._notify(event)
        return event

    def update_progress(
        self,
        progress_in_stage: float,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ProgressEvent:
        """
        Update progress within current stage.

        Args:
            progress_in_stage: Progress within the stage (0-100)
            message: Optional status message
            details: Optional additional details

        Returns:
            ProgressEvent with absolute progress (0-100)
        """
        stage_start, stage_end = STAGE_RANGES.get(self._stage, (0.0, 100.0))
        stage_range = stage_end - stage_start

        # Calculate absolute progress
        self._progress = stage_start + (progress_in_stage / 100.0) * stage_range
        self._progress = min(self._progress, stage_end)  # Clamp to stage max

        event = ProgressEvent(
            stage=self._stage,
            progress=self._progress,
            message=message,
            details=details,
        )
        self._notify(event)
        return event

    def add_segment(self, segment: Dict[str, Any]) -> ProgressEvent:
        """
        Add a completed segment and update progress.

        Uses segment end time to estimate progress.
        """
        self._segments.append(segment)

        if self.audio_duration > 0:
            segment_end = segment.get("end", 0.0)
            progress_in_stage = (segment_end / self.audio_duration) * 100.0
            return self.update_progress(
                progress_in_stage,
                message=f"Transcribed {segment_end:.1f}s / {self.audio_duration:.1f}s",
                details={"segments_count": len(self._segments)},
            )

        return self.update_progress(50.0, message="Processing...")

    def estimate_progress_by_time(self) -> ProgressEvent:
        """
        Estimate progress based on elapsed time and RTF.

        Useful when segments aren't available (e.g., batch processing).
        """
        if self._stage != ProgressStage.TRANSCRIBING:
            return ProgressEvent(stage=self._stage, progress=self._progress)

        elapsed = time.time() - self._stage_start_time
        rtf = RTF_ESTIMATES.get(self.asr_backend, 0.5)
        estimated_total = self.audio_duration * rtf

        if estimated_total > 0:
            progress_in_stage = min((elapsed / estimated_total) * 100.0, 95.0)
        else:
            progress_in_stage = 50.0

        return self.update_progress(
            progress_in_stage,
            message=f"Transcribing... ({elapsed:.1f}s elapsed)",
        )

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time since tracking started."""
        return time.time() - self._start_time

    @property
    def segments(self) -> List[Dict[str, Any]]:
        """Get collected segments."""
        return self._segments.copy()
```

### 1.3 Streaming Handler Utilities

**File:** `src/streaming/handlers.py`

```python
"""Streaming handler utilities for SSE endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, Optional, TypeVar

from src.streaming.events import (
    ProgressStage,
    SSEEventType,
    ProgressEvent,
    SegmentEvent,
    ResultEvent,
    ErrorEvent,
)
from src.streaming.progress import ProgressTracker


T = TypeVar("T")


async def create_sse_message(
    event_type: SSEEventType,
    data: str,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an SSE message dictionary.

    Compatible with Litestar's ServerSentEventMessage.
    """
    return {
        "event": event_type.value,
        "data": data,
        "id": event_id,
    }


class StreamingTranscriptionHandler:
    """
    Handler for streaming transcription with progress updates.

    Usage:
        handler = StreamingTranscriptionHandler(
            task_id="abc123",
            asr_backend="whisper",
        )

        async for message in handler.process(audio_path, process_func):
            yield ServerSentEventMessage(**message)
    """

    def __init__(
        self,
        task_id: str,
        asr_backend: str = "whisper",
        audio_duration: float = 0.0,
    ):
        self.task_id = task_id
        self.asr_backend = asr_backend
        self.tracker = ProgressTracker(
            audio_duration=audio_duration,
            asr_backend=asr_backend,
        )
        self._event_id = 0

    def _next_event_id(self) -> str:
        self._event_id += 1
        return f"{self.task_id}-{self._event_id}"

    async def emit_progress(
        self,
        stage: ProgressStage,
        progress_in_stage: float = 0.0,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a progress event."""
        if self.tracker._stage != stage:
            event = self.tracker.set_stage(stage, message)
        else:
            event = self.tracker.update_progress(progress_in_stage, message)

        return await create_sse_message(
            SSEEventType.PROGRESS,
            event.to_sse_data(),
            self._next_event_id(),
        )

    async def emit_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Emit a segment event and update progress."""
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
        """Emit the final result event."""
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
        """Emit an error event."""
        error_event = ErrorEvent(
            error=error,
            error_code=error_code,
            stage=self.tracker._stage,
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
        """
        Run the full transcription pipeline with progress streaming.

        Args:
            upload_func: Async function to handle file upload
            preprocess_func: Async function to preprocess audio
            transcribe_func: Async function to run transcription
            emit_segments: Whether to emit individual segment events

        Yields:
            SSE message dictionaries
        """
        import time
        start_time = time.time()

        try:
            # Stage 1: Queued
            yield await self.emit_progress(ProgressStage.QUEUED, message="Request received")

            # Stage 2: Uploading
            yield await self.emit_progress(ProgressStage.UPLOADING, message="Saving audio file")
            await upload_func()
            yield await self.emit_progress(ProgressStage.UPLOADING, 100, "File saved")

            # Stage 3: Preprocessing
            yield await self.emit_progress(ProgressStage.PREPROCESSING, message="Analyzing audio")
            audio_info = await preprocess_func()
            if audio_info and "duration" in audio_info:
                self.tracker.audio_duration = audio_info["duration"]
            yield await self.emit_progress(ProgressStage.PREPROCESSING, 100, "Audio ready")

            # Stage 4: Transcribing
            yield await self.emit_progress(ProgressStage.TRANSCRIBING, message="Starting transcription")

            result = await transcribe_func()

            # Handle streaming segments if available
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
```

### 1.4 Module Init

**File:** `src/streaming/__init__.py`

```python
"""
Streaming module for real-time progress updates via SSE.

This module provides backend-agnostic progress tracking and SSE event
handling for audio transcription pipelines.

Usage:
    from src.streaming import StreamingTranscriptionHandler, ProgressStage

    handler = StreamingTranscriptionHandler(task_id="abc", asr_backend="whisper")

    async for message in handler.run_with_progress(...):
        yield ServerSentEventMessage(**message)
"""

from src.streaming.events import (
    ProgressStage,
    SSEEventType,
    ProgressEvent,
    SegmentEvent,
    ResultEvent,
    ErrorEvent,
)
from src.streaming.progress import (
    ProgressTracker,
    RTF_ESTIMATES,
    STAGE_RANGES,
)
from src.streaming.handlers import (
    StreamingTranscriptionHandler,
    create_sse_message,
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
```

---

## Phase 2: Edge API Integration

### 2.1 Add SSE Endpoint to Edge API

**File:** `src/api/edge_main.py` (additions)

````python
from litestar.response import ServerSentEvent, ServerSentEventMessage
from src.streaming import StreamingTranscriptionHandler, ProgressStage


@post("/v1/transcribe/stream", summary="Transcribe with progress streaming", tags=["Transcription"])
async def transcribe_audio_stream(
    data: UploadFile = Body(media_type=RequestEncodingType.MULTI_PART),
    asr_backend: str = "whisper",
    enable_vad: bool = True,
    vad_threshold: float = 0.5,
    language: Optional[str] = None,
) -> ServerSentEvent:
    """
    Transcribe audio file with real-time progress updates via Server-Sent Events.

    This endpoint streams progress updates as the transcription proceeds:

    **Event Types:**
    - `progress`: Stage and percentage updates
    - `segment`: Individual transcript segments (as they complete)
    - `result`: Final complete result
    - `error`: Error information if processing fails

    **Progress Stages:**
    - `queued` (0%): Request received
    - `uploading` (0-10%): Saving audio file
    - `preprocessing` (10-20%): Analyzing audio
    - `transcribing` (20-95%): ASR processing
    - `completed` (100%): Done

    **Client Usage (JavaScript):**
    ```javascript
    const response = await fetch('/v1/transcribe/stream', {
        method: 'POST',
        body: formData
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value).split('\\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const event = JSON.parse(line.slice(6));
                console.log(`${event.stage}: ${event.progress}%`);
            }
        }
    }
    ```
    """
    global _current_task_id

    task_id = str(uuid4())
    current_settings = get_settings()

    # Validate ASR backend
    if asr_backend not in ("whisper", "chunkformer"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ASR backend: {asr_backend}. Must be 'whisper' or 'chunkformer'.",
        )

    async def event_generator() -> AsyncGenerator[ServerSentEventMessage, None]:
        handler = StreamingTranscriptionHandler(
            task_id=task_id,
            asr_backend=asr_backend,
        )

        audio_path: Optional[Path] = None

        async def upload():
            nonlocal audio_path
            audio_dir = current_settings.paths.audio_dir / task_id
            audio_dir.mkdir(parents=True, exist_ok=True)

            filename = data.filename or "audio"
            ext = Path(filename).suffix or ".wav"
            audio_path = audio_dir / f"input{ext}"

            content = await data.read()
            audio_path.write_bytes(content)

            logger.info("Audio file saved", task_id=task_id, path=str(audio_path), size=len(content))

        async def preprocess():
            duration = _get_audio_duration(audio_path)
            handler.tracker.audio_duration = duration
            return {"duration": duration}

        async def transcribe():
            return await _run_asr(
                task_id=task_id,
                audio_path=audio_path,
                asr_backend=asr_backend,
                enable_vad=enable_vad,
                vad_threshold=vad_threshold,
                language=language,
            )

        # Check if busy
        if _process_lock.locked():
            yield ServerSentEventMessage(
                data=ProgressEvent(
                    stage=ProgressStage.QUEUED,
                    progress=0,
                    message="Waiting for current task to complete",
                ).to_sse_data(),
                event="progress",
            )

        async with _process_lock:
            _current_task_id = task_id

            try:
                async for msg in handler.run_with_progress(
                    upload_func=upload,
                    preprocess_func=preprocess,
                    transcribe_func=transcribe,
                    emit_segments=True,
                ):
                    yield ServerSentEventMessage(**msg)

            except Exception as e:
                logger.error("Streaming transcription failed", task_id=task_id, error=str(e))
                yield ServerSentEventMessage(
                    data=ErrorEvent(error=str(e), stage=handler.tracker._stage).to_sse_data(),
                    event="error",
                )
            finally:
                _current_task_id = None

    return ServerSentEvent(event_generator())
````

---

## Phase 3: Main API Integration (Future)

The streaming module can be integrated with the main API by:

1. **Redis Pub/Sub for distributed progress:**

   ```python
   # Worker publishes progress
   redis.publish(f"task:{task_id}:progress", event.to_sse_data())

   # API subscribes and streams
   async for message in redis.subscribe(f"task:{task_id}:progress"):
       yield ServerSentEventMessage(data=message)
   ```

2. **WebSocket alternative:**
   For bidirectional communication (e.g., cancel requests)

3. **Hybrid approach:**
   SSE for progress, REST for final results

---

## Phase 4: Testing

### 4.1 Unit Tests

**File:** `tests/streaming/test_progress.py`

```python
import pytest
from src.streaming import ProgressTracker, ProgressStage, STAGE_RANGES


class TestProgressTracker:
    def test_stage_transition(self):
        tracker = ProgressTracker(audio_duration=60.0)

        event = tracker.set_stage(ProgressStage.UPLOADING)
        assert event.stage == ProgressStage.UPLOADING
        assert event.progress == 0.0  # Start of uploading range

    def test_progress_within_stage(self):
        tracker = ProgressTracker(audio_duration=60.0)
        tracker.set_stage(ProgressStage.TRANSCRIBING)

        event = tracker.update_progress(50.0)  # 50% through transcribing

        # Transcribing is 20-95%, so 50% should be ~57.5%
        expected = 20.0 + (50.0 / 100.0) * (95.0 - 20.0)
        assert abs(event.progress - expected) < 0.1

    def test_segment_progress(self):
        tracker = ProgressTracker(audio_duration=60.0)
        tracker.set_stage(ProgressStage.TRANSCRIBING)

        # Add segment at 30s of 60s audio
        event = tracker.add_segment({"start": 0, "end": 30, "text": "Hello"})

        # Should be ~50% through transcribing stage
        assert 50 < event.progress < 60


class TestProgressEstimation:
    def test_time_based_estimation(self):
        tracker = ProgressTracker(audio_duration=60.0, asr_backend="whisper")
        tracker.set_stage(ProgressStage.TRANSCRIBING)

        # Simulate some elapsed time
        tracker._stage_start_time -= 10  # 10 seconds elapsed

        event = tracker.estimate_progress_by_time()
        assert event.progress > 20  # Should be past start of transcribing
```

### 4.2 Integration Tests

**File:** `tests/api/test_edge_streaming.py`

```python
import pytest
from litestar.testing import TestClient
from src.api.edge_main import app


class TestStreamingEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_stream_progress_events(self, client, sample_audio_bytes):
        """Test that streaming endpoint emits progress events."""
        response = client.post(
            "/v1/transcribe/stream",
            files={"data": ("test.wav", sample_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200

        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        # Should have progress events
        stages = [e.get("stage") for e in events if "stage" in e]
        assert "uploading" in stages
        assert "preprocessing" in stages
        assert "transcribing" in stages
        assert "completed" in stages

    def test_stream_error_handling(self, client):
        """Test that errors are properly streamed."""
        response = client.post(
            "/v1/transcribe/stream",
            files={"data": ("test.txt", b"not audio", "text/plain")},
        )

        events = list(response.iter_lines())
        # Should contain an error event
        assert any("error" in line for line in events)
```

---

## Phase 5: Client Examples

### JavaScript (Browser)

```javascript
async function transcribeWithProgress(
  file,
  onProgress,
  onSegment,
  onComplete,
  onError
) {
  const formData = new FormData();
  formData.append("data", file);

  try {
    const response = await fetch("/v1/transcribe/stream", {
      method: "POST",
      body: formData,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7);
        } else if (line.startsWith("data: ")) {
          const data = JSON.parse(line.slice(6));

          switch (currentEvent) {
            case "progress":
              onProgress?.(data.stage, data.progress, data.message);
              break;
            case "segment":
              onSegment?.(data);
              break;
            case "result":
              onComplete?.(data);
              break;
            case "error":
              onError?.(data);
              break;
          }
        }
      }
    }
  } catch (error) {
    onError?.({ error: error.message });
  }
}

// Usage
transcribeWithProgress(
  audioFile,
  (stage, progress, message) => {
    progressBar.style.width = `${progress}%`;
    statusText.textContent = message || stage;
  },
  (segment) => {
    transcriptDiv.innerHTML += `<p>[${segment.start.toFixed(1)}s] ${
      segment.text
    }</p>`;
  },
  (result) => {
    console.log("Complete:", result);
  },
  (error) => {
    alert(`Error: ${error.error}`);
  }
);
```

### Python (httpx)

```python
import httpx
import json


async def transcribe_with_progress(audio_path: str, callback=None):
    """
    Transcribe audio with streaming progress updates.

    Args:
        audio_path: Path to audio file
        callback: Optional callback(event_type, data) for progress updates

    Returns:
        Final transcription result
    """
    async with httpx.AsyncClient(timeout=None) as client:
        with open(audio_path, "rb") as f:
            files = {"data": (audio_path, f, "audio/wav")}

            async with client.stream("POST", "/v1/transcribe/stream", files=files) as response:
                current_event = None
                result = None

                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: "):
                        data = json.loads(line[6:])

                        if callback:
                            callback(current_event, data)

                        if current_event == "result":
                            result = data
                        elif current_event == "error":
                            raise Exception(data.get("error", "Unknown error"))

                return result


# Usage
async def main():
    result = await transcribe_with_progress(
        "audio.wav",
        callback=lambda event, data: print(f"{event}: {data}")
    )
    print(f"Transcript: {result['transcript']}")
```

---

## Implementation Checklist

- [ ] **Phase 1: Core Module**

  - [ ] Create `src/streaming/` directory
  - [ ] Implement `events.py` - Event definitions
  - [ ] Implement `progress.py` - Progress tracking
  - [ ] Implement `handlers.py` - Streaming utilities
  - [ ] Create `__init__.py` - Module exports

- [ ] **Phase 2: Edge API**

  - [ ] Add `/v1/transcribe/stream` endpoint
  - [ ] Integrate `StreamingTranscriptionHandler`
  - [ ] Update route handlers list

- [ ] **Phase 3: Testing**

  - [ ] Unit tests for progress tracking
  - [ ] Integration tests for SSE endpoint
  - [ ] Client compatibility testing

- [ ] **Phase 4: Documentation**
  - [ ] Update `CLIENT_DEVELOPER_GUIDE.md`
  - [ ] Add client code examples
  - [ ] API reference documentation

---

## Future Enhancements

1. **WebSocket Support** - For bidirectional communication
2. **Cancellation** - Allow clients to cancel in-progress transcriptions
3. **Resume Support** - Resume interrupted transcriptions
4. **Batch Progress** - Progress for multiple file uploads
5. **Custom Events** - Plugin system for custom progress events

---

## References

- [Litestar SSE Documentation](https://docs.litestar.dev/latest/usage/responses.html#server-sent-events-sse)
- [MDN: Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [faster-whisper Progress Tracking](https://github.com/SYSTRAN/faster-whisper)

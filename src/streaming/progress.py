"""Progress tracking and estimation utilities.

This module provides the ProgressTracker class for tracking progress
through processing stages and estimating remaining time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.streaming.events import ProgressEvent, ProgressStage


# Real-Time Factor estimates per backend (processing_time / audio_duration)
# Lower = faster. These are approximate values for GPU.
RTF_ESTIMATES: Dict[str, float] = {
    "whisper": 0.3,  # ~3x real-time on GPU
    "chunkformer": 0.2,  # ~5x real-time on GPU
    "whisper_cpu": 1.5,  # ~0.7x real-time on CPU
}

# Progress ranges for each stage (start%, end%)
STAGE_RANGES: Dict[ProgressStage, tuple] = {
    ProgressStage.QUEUED: (0.0, 0.0),
    ProgressStage.UPLOADING: (0.0, 10.0),
    ProgressStage.PREPROCESSING: (10.0, 20.0),
    ProgressStage.TRANSCRIBING: (20.0, 95.0),
    ProgressStage.POSTPROCESSING: (95.0, 99.0),
    ProgressStage.COMPLETED: (100.0, 100.0),
}


@dataclass
class ProgressTracker:
    """Tracks progress through processing stages.

    The tracker manages state transitions and calculates absolute progress
    based on the current stage and progress within that stage.

    Usage:
        tracker = ProgressTracker(audio_duration=60.0, asr_backend="whisper")

        # Set stage and get event
        event = tracker.set_stage(ProgressStage.UPLOADING)

        # Update progress within stage
        event = tracker.update_progress(50.0)  # 50% of uploading

        # Add segments for transcription progress
        event = tracker.add_segment({"start": 0, "end": 30, "text": "Hello"})

    Attributes:
        audio_duration: Duration of the audio file in seconds
        asr_backend: ASR backend being used (for RTF estimation)
    """

    audio_duration: float = 0.0
    asr_backend: str = "whisper"

    # Internal state (using field with init=False for runtime state)
    _stage: ProgressStage = field(default=ProgressStage.QUEUED, init=False)
    _progress: float = field(default=0.0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)
    _stage_start_time: float = field(default_factory=time.time, init=False)
    _segments: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _callbacks: List[Callable[[ProgressEvent], None]] = field(
        default_factory=list, init=False
    )

    def add_callback(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Add a callback to be called on progress updates.

        Args:
            callback: Function to call with ProgressEvent when progress updates
        """
        self._callbacks.append(callback)

    def _notify(self, event: ProgressEvent) -> None:
        """Notify all callbacks of progress update.

        Args:
            event: The progress event to broadcast
        """
        for callback in self._callbacks:
            callback(event)

    def set_stage(
        self, stage: ProgressStage, message: Optional[str] = None
    ) -> ProgressEvent:
        """Transition to a new stage.

        Sets the current stage and resets progress to the start of that stage's range.

        Args:
            stage: The new stage to transition to
            message: Optional status message

        Returns:
            ProgressEvent for the new stage
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
        """Update progress within current stage.

        Calculates absolute progress based on stage range and relative progress.

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
        """Add a completed segment and update progress.

        Uses segment end time relative to audio duration to estimate progress.

        Args:
            segment: Segment dict with 'start', 'end', 'text' keys

        Returns:
            ProgressEvent with updated progress
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
        """Estimate progress based on elapsed time and RTF.

        Useful when segments aren't available (e.g., batch processing).
        Uses Real-Time Factor (RTF) estimates to predict progress.

        Returns:
            ProgressEvent with time-based progress estimate
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
        """Get collected segments (copy)."""
        return self._segments.copy()

    @property
    def current_stage(self) -> ProgressStage:
        """Get current processing stage."""
        return self._stage

    @property
    def current_progress(self) -> float:
        """Get current absolute progress (0-100)."""
        return self._progress

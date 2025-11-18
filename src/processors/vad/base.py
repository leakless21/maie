"""
Base module for VAD (Voice Activity Detection) backends.
Provides abstract interface for VAD implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.processors.base import VersionInfo


@dataclass
class VADSegment:
    """Individual speech segment detected by VAD."""

    start: float
    """Start time of segment in seconds"""
    end: float
    """End time of segment in seconds"""
    confidence: float
    """Confidence score (0.0-1.0)"""
    is_speech: bool
    """Whether segment contains speech"""

    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start


@dataclass
class VADResult:
    """Result from VAD processing."""

    segments: List[VADSegment]
    """List of detected speech/non-speech segments"""
    total_duration: float
    """Total audio duration in seconds"""
    speech_duration: float
    """Total duration of speech segments in seconds"""
    speech_ratio: float
    """Ratio of speech to total duration (0.0-1.0)"""
    processing_time: float
    """Time taken for VAD processing in seconds"""
    backend_info: Dict[str, Any]
    """Backend-specific metadata"""

    def non_speech_duration(self) -> float:
        """Get total duration of non-speech segments."""
        return self.total_duration - self.speech_duration

    def get_speech_segments(self) -> List[VADSegment]:
        """Get only speech segments."""
        return [seg for seg in self.segments if seg.is_speech]

    def get_silence_segments(self) -> List[VADSegment]:
        """Get only silence/non-speech segments."""
        return [seg for seg in self.segments if not seg.is_speech]


class VADBackend(ABC):
    """
    Abstract base class for VAD backends.
    Defines the interface all VAD implementations must follow.
    """

    @abstractmethod
    def detect_speech(self, audio_path: str) -> VADResult:
        """
        Detect speech segments in audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            VADResult containing detected speech segments and metadata

        Raises:
            FileNotFoundError: If audio file not found
            ValueError: If audio file is invalid or unsupported
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload resources used by VAD backend.
        Called when backend is no longer needed.
        """
        pass

    @abstractmethod
    def get_version_info(self) -> VersionInfo:
        """
        Get version information for VAD backend.

        Returns:
            VersionInfo dictionary containing version information
        """
        pass

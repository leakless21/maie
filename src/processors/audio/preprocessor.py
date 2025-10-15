"""
AudioPreprocessor — clean, documented, and matching docs/guide.md

This module implements the minimal, well-scoped behaviour required by tests and
the documentation: probe audio metadata with ffprobe, decide if normalization
is required, perform normalization with ffmpeg, and expose a convenience
integration helper that calls the ASR factory.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Public constants aligned with docs
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
MIN_DURATION_SEC = 1.0


class AudioPreprocessor:
    """Normalize and validate audio for ASR backends.

    Responsibilities are single and clear:
    - _probe_audio: extract metadata via ffprobe
    - _needs_normalization: decide if ffmpeg is required
    - _normalize_audio: run ffmpeg to produce WAV 16kHz mono
    - preprocess: high-level flow that returns metadata + optional normalized path
    - process_task_audio: helper that integrates with ASR factory
    """

    def _probe_audio(self, path: Path) -> Dict[str, Any]:
        """Run ffprobe and parse essential metadata.

        Returns a dict with keys: format, duration, sample_rate, channels.

        Raises ValueError when ffprobe fails or no audio stream found.
        """
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-print_format",
            "json",
            str(path),
        ]
        logger.debug("ffprobe cmd: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if getattr(result, "returncode", 0) != 0:
            stderr = getattr(result, "stderr", "")
            logger.error("ffprobe failed: %s", stderr)
            raise ValueError(f"FFprobe failed to get audio info: {stderr}")

        data = json.loads(getattr(result, "stdout", "{}"))
        streams = data.get("streams", [])
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), None
        )
        if audio_stream is None:
            raise ValueError("No audio stream found in file")

        fmt = data.get("format", {})
        return {
            "format": fmt.get("format_name", ""),
            "duration": float(fmt.get("duration", 0.0)),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
        }

    def _needs_normalization(self, metadata: Dict[str, Any]) -> bool:
        """Return True when audio must be converted to WAV 16kHz mono."""
        return (
            int(metadata.get("sample_rate", 0)) != TARGET_SAMPLE_RATE
            or int(metadata.get("channels", 0)) != TARGET_CHANNELS
            or str(metadata.get("format", "")).lower() not in ("pcm_s16le", "wav")
        )

    def _normalize_audio(self, input_path: Path, metadata: Dict[str, Any]) -> Path:
        """Normalize audio using ffmpeg to WAV 16kHz mono (PCM S16LE).

        Returns the Path to the normalized file. Raises ValueError on failure.
        """
        output_path = input_path.parent / "preprocessed.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-ac",
            str(TARGET_CHANNELS),
            "-sample_fmt",
            "s16",
            str(output_path),
        ]
        logger.debug("ffmpeg normalize cmd: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if getattr(result, "returncode", 0) != 0:
            stderr = getattr(result, "stderr", "")
            logger.error("ffmpeg normalization failed: %s", stderr)
            raise ValueError(f"FFmpeg failed to normalize audio: {stderr}")
        return output_path

    def preprocess(self, input_path: Path) -> Dict[str, Any]:
        """High level preprocess flow per docs/guide.md.

        Returns metadata dict with at least: format, duration, sample_rate, channels,
        and normalized_path (Path) when normalization occurred, otherwise None.
        """
        metadata = self._probe_audio(input_path)

        if metadata["duration"] < MIN_DURATION_SEC:
            raise ValueError(
                f"Audio too short: {metadata['duration']}s < {MIN_DURATION_SEC}s"
            )

        if self._needs_normalization(metadata):
            normalized = self._normalize_audio(input_path, metadata)
            metadata["normalized_path"] = normalized
        else:
            metadata["normalized_path"] = None

        return metadata

    def process_task_audio(self, task_id: str, raw_path: str) -> Any:
        """Integration helper: preprocess then delegate to ASR factory.

        Calls ASRFactory.create(backend, config={...}) with the final
        audio path (normalized if produced).
        """
        metadata = self.preprocess(Path(raw_path))
        final_path = str(metadata.get("normalized_path") or Path(raw_path))

        # Import here to avoid import cycles at module import time
        from src.processors.asr.factory import ASRFactory

        return ASRFactory.create(
            "whisper", config={"task_id": task_id, "audio_path": final_path}
        )

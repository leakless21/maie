"""
Silero VAD implementation for Voice Activity Detection.
Uses Silero VAD model for speech/non-speech classification.
"""

import time
import torch
import torchaudio
from pathlib import Path
from typing import Optional

from src.config.logging import get_module_logger
from src.processors.vad.base import VADBackend, VADResult, VADSegment
from src.processors.base import VersionInfo

logger = get_module_logger(__name__)

try:
    import silero_vad
    from silero_vad import get_speech_timestamps
except ImportError:
    silero_vad = None  # type: ignore
    get_speech_timestamps = None  # type: ignore


class SileroVADBackend(VADBackend):
    """
    Silero VAD backend implementation.
    
    Silero VAD is a lightweight voice activity detection model that works
    with various audio formats and sampling rates.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_ms: int = 30000,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize Silero VAD backend.

        Args:
            model_path: Optional explicit ONNX model path; if None, uses silero-vad's built-in loader
            threshold: Speech confidence threshold (0.0-1.0)
            sampling_rate: Audio sampling rate in Hz
            min_speech_duration_ms: Minimum speech segment duration
            max_speech_duration_ms: Maximum continuous speech duration
            min_silence_duration_ms: Minimum silence duration between segments
            window_size_samples: Window size for VAD (in samples)
            device: Device to use ('cuda' or 'cpu')
        """
        if silero_vad is None:
            raise RuntimeError(
                "silero-vad not installed. Install it with: pip install silero-vad"
            )

        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_ms = max_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.device = device
        self.model_path = model_path

        # Load the Silero VAD model
        self.model = silero_vad.load_silero_vad()

        logger.info(
            "Silero VAD backend initialized",
            threshold=threshold,
            sampling_rate=sampling_rate,
            device=device,
        )

    def detect_speech(self, audio_path: str) -> VADResult:
        """
        Detect speech segments in audio file using Silero VAD.

        Uses get_speech_timestamps utility function with streaming chunk processing.
        Official Silero VAD API as per github.com/snakers4/silero-vad.

        Args:
            audio_path: Path to audio file

        Returns:
            VADResult containing detected speech segments

        Raises:
            FileNotFoundError: If audio file not found
            ValueError: If audio file is invalid or unsupported
        """
        start_time = time.time()

        # Validate file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            if get_speech_timestamps is None:
                raise RuntimeError(
                    "silero-vad not properly installed. Install with: pip install silero-vad"
                )

            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))

            # Resample if needed
            if sample_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.sampling_rate
                )
                waveform = resampler(waveform)
                sample_rate = self.sampling_rate

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            wav = waveform.squeeze(0)

            # Get speech timestamps using official Silero VAD API
            # This function handles streaming chunk processing internally
            speech_timestamps = get_speech_timestamps(
                wav,
                self.model,
                sampling_rate=self.sampling_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                return_seconds=True,
            )

            # Convert timestamps to segments
            segments = []
            current_pos = 0.0
            total_duration = wav.shape[-1] / self.sampling_rate

            # Create VAD segments from speech timestamps
            for speech_dict in speech_timestamps:
                start_sec = float(speech_dict.get("start", 0.0))
                end_sec = float(speech_dict.get("end", 0.0))

                # Add silence segment if gap exists
                if current_pos < start_sec:
                    segments.append(
                        VADSegment(
                            start=float(current_pos),
                            end=float(start_sec),
                            confidence=0.0,
                            is_speech=False,
                        )
                    )

                # Add speech segment
                segments.append(
                    VADSegment(
                        start=float(start_sec),
                        end=float(end_sec),
                        confidence=1.0,  # Silero VAD with threshold
                        is_speech=True,
                    )
                )

                current_pos = end_sec

            # Add final silence segment if needed
            if current_pos < total_duration:
                segments.append(
                    VADSegment(
                        start=float(current_pos),
                        end=float(total_duration),
                        confidence=0.0,
                        is_speech=False,
                    )
                )

            # Calculate metrics
            speech_duration = sum(
                seg.duration() for seg in segments if seg.is_speech
            )
            speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0

            processing_time = time.time() - start_time

            result = VADResult(
                segments=segments,
                total_duration=total_duration,
                speech_duration=speech_duration,
                speech_ratio=speech_ratio,
                processing_time=processing_time,
                backend_info={
                    "backend": "silero",
                    "model_path": str(self.model_path) if self.model_path else "built-in",
                    "threshold": self.threshold,
                    "device": self.device,
                },
            )

            logger.info(
                "Speech detection completed",
                total_duration=total_duration,
                speech_duration=speech_duration,
                speech_ratio=speech_ratio,
                num_segments=len(segments),
                processing_time=processing_time,
            )

            return result

        except Exception as exc:
            logger.error(
                "Speech detection failed",
                audio_path=audio_path,
                error=str(exc),
                exc_info=True,
            )
            raise

    def unload(self) -> None:
        """Unload the Silero VAD model and release resources."""
        if hasattr(self, "model"):
            # Delete model reference
            del self.model

            logger.info("Silero VAD backend unloaded")

    def get_version_info(self) -> VersionInfo:
        """Get version information for Silero VAD backend."""
        return VersionInfo(
            name="silero-vad",
            backend="silero_vad",
            version="5.0",  # Silero VAD version
            device=self.device,
        )

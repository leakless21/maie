"""Audio metrics collector module for MAIE.

This module provides audio quality metrics collection including
VAD coverage, confidence scoring, and audio property validation.
"""

from typing import Any, Dict

import numpy as np
from scipy.io import wavfile


class AudioMetricsCollector:
    """Audio metrics collector for measuring audio quality and properties."""

    def __init__(self):
        """Initialize the audio metrics collector."""
        pass

    def calculate_vad_coverage(self, audio_path: str) -> float:
        """Calculate VAD (Voice Activity Detection) coverage.

        Args:
            audio_path: Path to audio file

        Returns:
            Ratio of voice-active segments to total duration (0.0 to 1.0)
        """
        try:
            # Load audio file
            sample_rate, audio_data = wavfile.read(audio_path)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Simple VAD: calculate energy-based voice activity
            # Use a simple energy threshold to detect voice segments
            frame_size = int(0.02 * sample_rate)  # 20ms frames
            energy_threshold = np.std(audio_data) * 0.1  # 10% of std as threshold

            # Calculate frame energies
            num_frames = len(audio_data) // frame_size
            frame_energies = []

            for i in range(num_frames):
                start_idx = i * frame_size
                end_idx = (i + 1) * frame_size
                frame = audio_data[start_idx:end_idx]
                energy = np.mean(frame**2)
                frame_energies.append(energy)

            # Count frames above threshold
            active_frames = sum(
                1 for energy in frame_energies if energy > energy_threshold
            )

            # Return ratio of active frames to total frames
            vad_coverage = active_frames / num_frames if num_frames > 0 else 0.0
            return vad_coverage

        except Exception as e:
            raise ValueError(f"Failed to calculate VAD coverage: {str(e)}")

    def calculate_confidence(self, audio_path: str) -> float:
        """Calculate overall audio confidence score.

        Args:
            audio_path: Path to audio file

        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            # Load audio file
            sample_rate, audio_data = wavfile.read(audio_path)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Calculate various confidence metrics
            signal_power = np.mean(audio_data**2)
            signal_std = np.std(audio_data)

            # Simple confidence calculation based on signal properties
            # This is a basic implementation - can be enhanced with more sophisticated metrics
            confidence = min(
                1.0, float(signal_power) / 100.0
            )  # Normalize based on expected power
            confidence = min(
                confidence, float(signal_std) / 1000.0
            )  # Also consider standard deviation

            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))

            return float(confidence)

        except Exception as e:
            raise ValueError(f"Failed to calculate confidence: {str(e)}")

    def validate_audio_properties(self, audio_path: str) -> Dict[str, Any]:
        """Validate audio properties against expected standards.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing validation results
        """
        try:
            # Load audio file
            sample_rate, audio_data = wavfile.read(audio_path)

            # Get duration
            duration = len(audio_data) / sample_rate if sample_rate > 0 else 0

            # Check if mono
            is_mono = (
                len(audio_data.shape) == 1 or audio_data.shape[1] == 1
                if len(audio_data.shape) > 1
                else True
            )

            # Validation results
            validation_result = {
                "is_valid": True,
                "sample_rate": sample_rate,
                "duration": duration,
                "is_mono": is_mono,
                "num_samples": len(audio_data),
                "bit_depth": audio_data.dtype.itemsize * 8,
                "issues": [],
            }

            # Check sample rate (should be 16kHz for normalized audio)
            if sample_rate != 16000:
                validation_result["issues"].append("sample_rate_not_16khz")

            # Check if mono
            if not is_mono:
                validation_result["issues"].append("not_mono")

            # Check duration (reasonable limits)
            if duration < 0.1:  # Less than 100ms
                validation_result["issues"].append("too_short")
            elif duration > 3600:  # More than 1 hour
                validation_result["issues"].append("too_long")

            # Update validity based on issues
            validation_result["is_valid"] = len(validation_result["issues"]) == 0

            return validation_result

        except Exception as e:
            return {"is_valid": False, "error": str(e), "issues": ["validation_error"]}

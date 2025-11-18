"""
Voice Activity Detection (VAD) module for MAIE.
Provides unified VAD interface across multiple backends.
"""

from src.processors.vad.base import VADBackend, VADResult, VADSegment
from src.processors.vad.silero import SileroVADBackend
from src.processors.vad.factory import VADFactory

__all__ = [
    "VADBackend",
    "VADResult",
    "VADSegment",
    "SileroVADBackend",
    "VADFactory",
]

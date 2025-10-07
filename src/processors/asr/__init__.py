"""
ASR module initialization for MAIE processors.
Exports all ASR-related classes and functions.
"""

from src.processors.asr.factory import ASRFactory
from src.processors.base import ASRBackend
from src.processors.asr.whisper import WhisperBackend
from src.processors.asr.chunkformer import ChunkFormerBackend

__all__ = [
    "ASRFactory",
    "ASRBackend",
    "WhisperBackend",
    "ChunkFormerBackend"
]
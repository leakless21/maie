"""
Processors module initialization for MAIE.
Exports all processor classes and functions.
"""

from src.processors.base import ASRResult, LLMResult, ASRBackend, LLMBackend, Processor
from src.processors.asr.factory import ASRFactory
from src.processors.asr import ASRBackend, WhisperBackend, ChunkFormerBackend

try:
    from src.processors.llm import LLMProcessor
except Exception:  # optional dependency may be missing in test environments
    LLMProcessor = None

# Audio processors
from src.processors.audio.preprocessor import AudioPreprocessor
from src.processors.audio.metrics import AudioMetricsCollector

# Prompt processors
from src.processors.prompt.template_loader import TemplateLoader
from src.processors.prompt.renderer import PromptRenderer

__all__ = [
    "ASRResult",
    "LLMResult",
    "ASRBackend",
    "LLMBackend",
    "Processor",
    "ASRFactory",
    "WhisperBackend",
    "ChunkFormerBackend",
    "LLMProcessor",
    "AudioPreprocessor",
    "AudioMetricsCollector",
    "TemplateLoader",
    "PromptRenderer",
]

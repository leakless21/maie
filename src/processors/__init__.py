"""
Processors module initialization for MAIE.
Exports all processor classes and functions.
"""

from src.processors.asr import ChunkFormerBackend, WhisperBackend
from src.processors.asr.factory import ASRFactory
from src.processors.base import ASRBackend, ASRResult, LLMBackend, LLMResult, Processor

try:
    from src.processors.llm import LLMProcessor
except Exception:  # optional dependency may be missing in test environments
    LLMProcessor = None

# Audio processors
from src.processors.audio.preprocessor import AudioPreprocessor
from src.processors.prompt.renderer import PromptRenderer

# Prompt processors
from src.processors.prompt.template_loader import TemplateLoader

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
    "TemplateLoader",
    "PromptRenderer",
]

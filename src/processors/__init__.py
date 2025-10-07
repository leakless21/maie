"""
Processors module initialization for MAIE.
Exports all processor classes and functions.
"""

from src.processors.base import ASRResult, LLMResult, ASRBackend, LLMBackend, Processor
from src.processors.asr.factory import ASRProcessorFactory
from src.processors.asr import ASRBackend, WhisperBackend, ChunkFormerBackend
from src.processors.llm import LLMProcessor
from src.processors.chat.template_manager import ChatTemplateManager

# Audio processors
from src.processors.audio.preprocessor import AudioPreprocessor
from src.processors.audio.metrics import AudioMetricsCollector

# Prompt processors
from src.processors.prompt.template_loader import PromptTemplateLoader
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
    "ChatTemplateManager",
    "AudioPreprocessor",
    "AudioMetricsCollector",
    "PromptTemplateLoader",
    "PromptRenderer"
]
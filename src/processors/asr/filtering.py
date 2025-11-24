"""
Filtering ASR backend wrapper.

This module implements the Decorator pattern to add hallucination filtering capabilities
to any ASR backend without modifying the backend itself.
"""

from typing import Any, Dict, Optional

from src import config as cfg
from src.config.logging import get_module_logger
from src.processors.base import ASRBackend, ASRResult, VersionInfo
from src.utils.asr_filters import HallucinationFilter, create_filter_from_config

logger = get_module_logger(__name__)


class FilteringASRBackend(ASRBackend):
    """
    Wrapper for ASR backends that applies hallucination filtering to the output.

    This class implements the ASRBackend protocol and delegates execution to an
    inner backend instance. After the inner backend produces a result, this wrapper
    applies the configured HallucinationFilter to clean up the segments and text.
    """

    def __init__(self, inner_backend: ASRBackend):
        """
        Initialize the filtering wrapper.

        Args:
            inner_backend: The actual ASR backend to wrap (e.g., WhisperBackend, ChunkFormerBackend)
        """
        self.inner_backend = inner_backend
        self.filter: Optional[HallucinationFilter] = None
        
        # Initialize filter if enabled in config
        if getattr(cfg.settings.asr.hallucination, "enabled", False):
            try:
                self.filter = create_filter_from_config()
                logger.info("Initialized hallucination filter for ASR backend")
            except Exception as e:
                logger.warning("Failed to initialize hallucination filter: {}", e)

    def execute(self, audio_data: bytes, **kwargs) -> ASRResult:
        """
        Execute ASR processing and apply filtering to the result.

        Args:
            audio_data: Raw audio data in bytes
            **kwargs: Additional backend-specific parameters

        Returns:
            ASRResult with filtered text and segments
        """
        # 1. Delegate to inner backend
        result = self.inner_backend.execute(audio_data, **kwargs)

        # 2. Apply filtering if enabled and filter is initialized
        if self.filter and result.segments:
            original_count = len(result.segments)
            
            # Filter segments
            filtered_segments = self.filter.filter_segments(result.segments)
            
            # Update result if segments were removed
            if len(filtered_segments) < original_count:
                logger.info(
                    "Hallucination filter removed segments",
                    original_count=original_count,
                    filtered_count=len(filtered_segments),
                    removed_count=original_count - len(filtered_segments),
                )
                
                # Rebuild text from filtered segments
                # Note: We assume 'text' field exists in segments as per ASRResult contract
                text_parts = [
                    seg.get("text", "").strip() 
                    for seg in filtered_segments 
                    if seg.get("text", "").strip()
                ]
                filtered_text = " ".join(text_parts)
                
                # Return new result with filtered content
                # Preserve other metadata from original result
                return ASRResult(
                    text=filtered_text,
                    segments=filtered_segments,
                    language=result.language,
                    confidence=result.confidence,
                    duration=result.duration,
                    error=result.error
                )

        return result

    def unload(self) -> None:
        """Unload the inner backend."""
        self.inner_backend.unload()

    def get_version_info(self) -> VersionInfo:
        """Get version info from the inner backend."""
        info = self.inner_backend.get_version_info()
        
        return info

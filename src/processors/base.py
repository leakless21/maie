"""
Base module for MAIE processors containing abstract interfaces and common data structures.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass


@dataclass
class ASRResult:
    """
    Data structure for ASR (Automatic Speech Recognition) results.
    """
    text: str
    segments: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class LLMResult:
    """
    Data structure for LLM (Large Language Model) results.
    """
    text: str
    tokens_used: Optional[int] = None
    model_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class Processor(ABC):
    """
    Abstract base class for all processors in MAIE.
    """
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the processing task.
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Unload resources used by the processor.
        """
        pass


class ASRBackend(Protocol):
    """
    Protocol defining the interface for ASR backends.
    """
    
    def execute(self, audio_data: bytes, **kwargs) -> ASRResult:
        """
        Execute ASR processing on audio data.
        
        Args:
            audio_data: Raw audio data in bytes
            **kwargs: Additional backend-specific parameters
            
        Returns:
            ASRResult containing the transcription and metadata
        """
        ...
    
    def unload(self) -> None:
        """
        Unload the ASR backend and release resources.
        """
        ...
    
    def get_version_info(self) -> Dict[str, Any]:
        """
        Get version information for the backend.
        
        Returns:
            Dictionary containing version information
        """
        ...


class LLMBackend(Protocol):
    """
    Protocol defining the interface for LLM backends.
    """
    
    def execute(self, text: str, **kwargs) -> LLMResult:
        """
        Execute LLM processing on text.
        
        Args:
            text: Input text to process
            **kwargs: Additional backend-specific parameters
            
        Returns:
            LLMResult containing the processed output and metadata
        """
        ...
    
    def unload(self) -> None:
        """
        Unload the LLM backend and release resources.
        """
        ...
    
    def get_version_info(self) -> Dict[str, Any]:
        """
        Get version information for the backend.
        
        Returns:
            Dictionary containing version information
        """
        ...
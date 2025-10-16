"""
Base module for MAIE processors containing abstract interfaces and common data structures.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    runtime_checkable,
)


@dataclass
class ASRResult:
    """
    Data structure for ASR (Automatic Speech Recognition) results.

    Exposes a dict-like interface so callers can use both attribute and
    mapping-style access (e.g., result.text and result["text"]). This keeps
    backward compatibility with tests that expect a plain dict while
    maintaining type safety for newer code.
    """

    text: str
    segments: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    # Structured error information (if execution failed)
    error: Optional[Dict[str, Any]] = None

    # --- dict-like helpers ---
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


@dataclass
class LLMResult:
    """
    Data structure for LLM (Large Language Model) results.
    """

    text: str
    tokens_used: Optional[int] = None
    model_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    # Structured error information (if execution failed)
    error: Optional[Dict[str, Any]] = None


class VersionInfo(TypedDict, total=False):
    """
    Standardized version metadata for processors/backends.
    All fields are optional to support different backend capabilities.
    """

    name: str
    version: str
    checkpoint_hash: str
    model_size: str
    backend: str  # Backend type identifier (e.g., "whisper", "chunkformer")
    model_variant: str  # Model variant identifier (e.g., "openai/whisper-large")
    model_path: str  # Path to model directory or identifier
    device: str  # Device used (cuda, cpu, auto)
    compute_type: str  # Compute/precision type (float16, int8, etc.)
    cpu_threads: Optional[int]  # Number of CPU threads for inference
    beam_size: Optional[int]  # Beam size for decoding
    vad_filter: bool  # Whether VAD filtering is enabled
    condition_on_previous_text: bool  # Whether context conditioning is used
    language: Optional[str]  # Language code for transcription
    library: str  # Underlying library name (e.g., "faster-whisper")


class ProcessorError(Exception):
    """Base exception for processor-related errors."""

    pass


class Processor(ABC):
    """
    Abstract base class for all processors in MAIE.
    Provides a small async shim and leaves concrete execute/unload to subclasses.
    Also supports context-manager usage and optional memory management hooks.
    """

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the processing task synchronously.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload resources used by the processor.
        """
        pass

    def reduce_memory(self) -> None:
        """
        Optional hook for processors to release or trim memory usage.
        Default is a no-op; ML-backed processors may override this to
        free GPU/CPU caches or perform other memory reclamation.
        """
        return None

    async def async_execute(self, *args, **kwargs) -> Any:
        """
        Async-compatible execution helper. By default runs the sync `execute`
        in the default thread pool executor. Backends may override with a true
        async implementation.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, *args, **kwargs)

    # Context manager support (useful for deterministic resource cleanup)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Ensure resources are released deterministically
        try:
            self.unload()
        except Exception:
            # Do not suppress exceptions from context manager exit
            pass
        return False


def safe_execute_sync(
    func: Callable[..., Any], *args, **kwargs
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Execute `func` safely, catching exceptions and returning a tuple of
    (result, error). If execution succeeds, `error` is None. If an exception
    occurs, result is None and `error` contains structured information.

    Returns:
        (result, error)
    """
    try:
        return func(*args, **kwargs), None
    except Exception as exc:  # capture any exception and normalize
        return None, {"type": type(exc).__name__, "message": str(exc)}


@runtime_checkable
class ASRBackend(Protocol):
    """
    Protocol defining the interface for ASR backends.
    Runtime-checkable for convenience in tests and factories.
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

    def get_version_info(self) -> VersionInfo:
        """
        Get version information for the backend.

        Returns:
            VersionInfo dictionary containing version information
        """
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """
    Protocol defining the interface for LLM backends.
    Runtime-checkable for convenience in tests and factories.
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

    def get_version_info(self) -> VersionInfo:
        """
        Get version information for the backend.

        Returns:
            VersionInfo dictionary containing version information
        """
        ...

"""
Base module for MAIE processors containing abstract interfaces and common data structures.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
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

from src.utils.error_handling import safe_execute


@dataclass
class ASRResult:
    """
    Pure dataclass for ASR (Automatic Speech Recognition) results.

    Provides attribute-based access with basic validation for data integrity.
    Use dataclasses.asdict() for serialization when needed.
    """

    text: str
    segments: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    # Structured error information (if execution failed)
    error: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate data integrity after initialization."""
        # Basic validation rules
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise TypeError(
                    f"confidence must be numeric, got {type(self.confidence).__name__}"
                )
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(
                    f"confidence must be between 0.0 and 1.0, got {self.confidence}"
                )

        if self.duration is not None:
            if not isinstance(self.duration, (int, float)):
                raise TypeError(
                    f"duration must be numeric, got {type(self.duration).__name__}"
                )
            if self.duration < 0:
                raise ValueError(f"duration must be non-negative, got {self.duration}")

        if self.language is not None and not isinstance(self.language, str):
            raise TypeError(f"language must be str, got {type(self.language).__name__}")

        if self.segments is not None and not isinstance(self.segments, list):
            raise TypeError(
                f"segments must be list, got {type(self.segments).__name__}"
            )

        if self.error is not None and not isinstance(self.error, dict):
            raise TypeError(f"error must be dict, got {type(self.error).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the result."""
        from dataclasses import asdict

        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.to_dict()

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
    # ChunkFormer-specific parameters
    chunk_size: int
    left_context_size: int
    right_context_size: int
    total_batch_duration: int
    return_timestamps: bool
    library: str  # Library used for model (e.g., "whisper", "chunkformer")
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
        Async-compatible execution helper.

        Preferred behavior:
        - Call the current running loop's run_in_executor and pass through
          positional and keyword arguments (this allows tests that mock
          loop.run_in_executor to observe the same call signature).
        - If the loop's run_in_executor doesn't accept keyword arguments,
          fall back to using functools.partial to bind kwargs.
        - If no running loop is available, fall back to asyncio.to_thread.
        """
        try:
            loop = asyncio.get_running_loop()
            try:
                # Try to call run_in_executor with kwargs (works with mocked loops)
                return await loop.run_in_executor(None, self.execute, *args, **kwargs)
            except TypeError:
                # Real event loop's run_in_executor may not accept kwargs; wrap call.
                from functools import partial

                return await loop.run_in_executor(
                    None, partial(self.execute, *args, **kwargs)
                )
        except RuntimeError:
            # No running loop — run in thread
            return await asyncio.to_thread(self.execute, *args, **kwargs)

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
    return safe_execute(func, *args, **kwargs)


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

"""
ASR Factory module implementing the Factory Pattern for ASR backends with integrated audio processing.
"""

from typing import Any, Dict, Type

from src.processors.asr.chunkformer import ChunkFormerBackend
from src.processors.asr.whisper import WhisperBackend
from src.processors.audio.metrics import AudioMetricsCollector
from src.processors.audio.preprocessor import AudioPreprocessor
from src.processors.base import ASRBackend


class ASRFactory:
    """
    Factory class for creating ASR backend instances with integrated audio processing.
    """

    # Registry of available ASR backends
    BACKENDS: Dict[str, Type[ASRBackend]] = {}

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[ASRBackend]) -> None:
        """
        Register a new ASR backend.

        Args:
            name: Name of the backend
            backend_class: Class of the backend to register
        """
        cls.BACKENDS[name] = backend_class

    @classmethod
    def create(cls, backend_type: str, **kwargs) -> ASRBackend:
        """
        Create an instance of the specified ASR backend.

        Args:
            backend_type: Type of ASR backend to create
            **kwargs: Additional parameters for backend initialization

        Returns:
            Instance of the requested ASRBackend

        Raises:
            ValueError: If the requested backend type is not registered
        """
        if backend_type not in cls.BACKENDS:
            available_backends = ", ".join(cls.BACKENDS.keys())
            raise ValueError(
                f"Unknown ASR backend type: {backend_type}. "
                f"Available backends: {available_backends}"
            )

        backend_class = cls.BACKENDS[backend_type]
        return backend_class(**kwargs)

    @classmethod
    def create_with_audio_processing(
        cls, backend_type: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Create an ASR processor with integrated audio processing components.

        Args:
            backend_type: Type of ASR backend to create
            **kwargs: Additional parameters for backend initialization

        Returns:
            Dictionary containing ASR processor and audio processing components
        """
        asr_processor = cls.create(backend_type, **kwargs)

        return {
            "asr_processor": asr_processor,
            "audio_preprocessor": AudioPreprocessor(),
            "audio_metrics_collector": AudioMetricsCollector(),
        }


# Register the supported backends
ASRFactory.register_backend("whisper", WhisperBackend)
ASRFactory.register_backend("chunkformer", ChunkFormerBackend)

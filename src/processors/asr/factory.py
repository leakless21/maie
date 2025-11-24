"""
ASR Factory module implementing the Factory Pattern for ASR backends with integrated audio processing.
"""

from typing import Any, Dict, Type

from src.processors.asr.chunkformer import ChunkFormerBackend
from src.processors.asr.whisper import WhisperBackend
from src.processors.audio.preprocessor import AudioPreprocessor
from src.processors.base import ASRBackend
from src.processors.vad.factory import VADFactory
from src.config.model import VADSettings
from src import config as cfg


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
        backend = backend_class(**kwargs)

        # Apply hallucination filtering if enabled
        # Note: We check the config directly here.
        if getattr(cfg.settings.asr.hallucination, "enabled", False):
            from src.processors.asr.filtering import FilteringASRBackend
            return FilteringASRBackend(backend)
            
        return backend

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
        }

    @classmethod
    def create_vad_backend(cls, vad_config: VADSettings) -> Any:
        """
        Create a VAD backend from configuration.

        Args:
            vad_config: VAD configuration settings

        Returns:
            VAD backend instance

        Raises:
            ValueError: If VAD configuration is invalid
        """
        if not vad_config.enabled:
            return None

        return VADFactory.create(
            vad_config.backend,
            model_path=vad_config.silero_model_path,
            threshold=vad_config.silero_threshold,
            sampling_rate=vad_config.silero_sampling_rate,
            min_speech_duration_ms=vad_config.min_speech_duration_ms,
            max_speech_duration_ms=vad_config.max_speech_duration_ms,
            min_silence_duration_ms=vad_config.min_silence_duration_ms,
            window_size_samples=vad_config.window_size_samples,
            device=vad_config.device,
        )


# Register the supported backends
ASRFactory.register_backend("whisper", WhisperBackend)
ASRFactory.register_backend("chunkformer", ChunkFormerBackend)

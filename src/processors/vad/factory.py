"""
VAD Factory module implementing the Factory Pattern for VAD backends.
"""

from typing import Type

from src.processors.vad.base import VADBackend
from src.processors.vad.silero import SileroVADBackend


class VADFactory:
    """
    Factory class for creating VAD backend instances.
    Supports pluggable VAD backends with configuration-driven creation.
    """

    # Registry of available VAD backends
    BACKENDS: dict[str, Type[VADBackend]] = {}

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[VADBackend]) -> None:
        """
        Register a new VAD backend.

        Args:
            name: Name of the backend
            backend_class: Class of the backend to register
        """
        cls.BACKENDS[name] = backend_class

    @classmethod
    def create(cls, backend_type: str, **kwargs) -> VADBackend:
        """
        Create an instance of the specified VAD backend.

        Args:
            backend_type: Type of VAD backend to create
            **kwargs: Additional parameters for backend initialization

        Returns:
            Instance of the requested VADBackend

        Raises:
            ValueError: If the requested backend type is not registered
        """
        if backend_type not in cls.BACKENDS:
            available_backends = ", ".join(cls.BACKENDS.keys())
            raise ValueError(
                f"Unknown VAD backend type: {backend_type}. "
                f"Available backends: {available_backends}"
            )

        # Filter out VADSettings-specific fields that backends don't accept
        backend_kwargs = {}
        for key, value in kwargs.items():
            # Skip config-only fields
            if key in ("enabled", "backend"):
                continue
            # Map silero_ prefixed fields to backend parameters (remove prefix)
            if key.startswith("silero_"):
                backend_key = key[7:]  # Remove 'silero_' prefix
                backend_kwargs[backend_key] = value
            else:
                # Pass through other parameters
                backend_kwargs[key] = value

        backend_class = cls.BACKENDS[backend_type]
        return backend_class(**backend_kwargs)


# Register supported backends
VADFactory.register_backend("silero", SileroVADBackend)

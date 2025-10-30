"""Device detection and selection helpers for MAIE."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:  # pragma: no cover - import guard
    import torch  # type: ignore
except ImportError:  # pragma: no cover - handled by helpers
    torch = None  # type: ignore

DEVICE_ENV_VAR = "MAIE_DEVICE"


@lru_cache(maxsize=None)
def has_cuda() -> bool:
    """Return True when CUDA is available for the current interpreter."""
    if torch is None:
        return False

    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


@lru_cache(maxsize=None)
def _mps_available() -> bool:
    """Return True when Apple's Metal Performance Shaders backend is available."""
    if torch is None:
        return False

    try:
        backends = getattr(torch, "backends", None)
        mps_backend = getattr(backends, "mps", None) if backends else None
        if mps_backend is None:
            return False
        return bool(mps_backend.is_available())
    except (AttributeError, RuntimeError):
        return False


@lru_cache(maxsize=None)
def select_device(prefer: Optional[str] = None, allow_mps: bool = True) -> str:
    """
    Select the preferred device for PyTorch operations.

    Args:
        prefer: Optional explicit device string (e.g. \"cuda:0\", \"cpu\").
        allow_mps: When True, allow MPS fallback before CPU.

    Returns:
        Device string suitable for torch.device / component configuration.
    """
    forced = prefer or os.getenv(DEVICE_ENV_VAR)
    if forced:
        normalized = forced.strip()
        if normalized and normalized.lower() != "auto":
            return normalized

    if has_cuda():
        return "cuda"

    if allow_mps and _mps_available():
        return "mps"

    return "cpu"


def ensure_cuda_available(message: Optional[str] = None) -> None:
    """
    Raise RuntimeError when CUDA is required but not available.

    Args:
        message: Optional custom error message.
    """
    if not has_cuda():
        raise RuntimeError(message or "CUDA is required but not available.")


def reset_device_cache() -> None:
    """Clear cached device detection results (useful for tests)."""
    has_cuda.cache_clear()
    _mps_available.cache_clear()
    select_device.cache_clear()


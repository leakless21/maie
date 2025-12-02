"""
ChunkFormer ASR backend implementation for MAIE.
Supports chunkformer-rnnt-large-vie model.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Any

from src import config as cfg
from src.processors.base import ASRBackend, ASRResult, VersionInfo
from src.utils.device import ensure_cuda_available, select_device
from src.config.logging import get_module_logger

logger = get_module_logger(__name__)


# -----------------------------------------------------------------------------
# Local helpers: normalize timestamps to float seconds
# Supports formats: float/int seconds, "HH:MM:SS.mm" (centiseconds),
# "HH:MM:SS.mmm" (milliseconds), optionally wrapped in brackets [ ... ].
# -----------------------------------------------------------------------------
def _normalize_timestamp(value: Any) -> Optional[float]:
    """Normalize ChunkFormer timestamps to float seconds.

    Supported formats:
    - float/int seconds
    - strings like "HH:MM:SS.mmm" (optionally wrapped in brackets)
    - strings like "HH:MM:SS:mmm" (ChunkFormer format with colon)
    Returns None when parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().strip("[]")
        import re

        # Try HH:MM:SS.mmm format (dot separator)
        m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})", s)
        if m:
            hh, mm, ss, frac = m.groups()
            base = int(hh) * 3600 + int(mm) * 60 + int(ss)
            return base + int(frac) / 1000.0

        # Try HH:MM:SS:mmm format (colon separator - ChunkFormer format)
        m = re.match(r"(\d{1,2}):(\d{2}):(\d{2}):(\d{3})", s)
        if m:
            hh, mm, ss, frac = m.groups()
            base = int(hh) * 3600 + int(mm) * 60 + int(ss)
            return base + int(frac) / 1000.0

        try:
            return float(s)
        except Exception:
            return None
    return None


def _normalize_segment(seg: Any) -> dict:
    """Return a normalized segment dict with float start/end and stripped text.

    Unknown or unparsable timestamps are returned as None.
    """
    # Extract text field (decode preferred, else text)
    if isinstance(seg, dict):
        text = seg.get("decode", seg.get("text", ""))
        start_raw = seg.get("start")
        end_raw = seg.get("end")
    else:
        text = getattr(seg, "decode", None) or getattr(seg, "text", "")
        start_raw = getattr(seg, "start", None)
        end_raw = getattr(seg, "end", None)

    return {
        "start": _normalize_timestamp(start_raw),
        "end": _normalize_timestamp(end_raw),
        "text": (text or "").strip(),
    }


class ChunkFormerBackend(ASRBackend):
    """
    ChunkFormer ASR backend implementation.

    Minimal, test-focused implementation that:
    - Lazily imports `chunkformer` so unit tests can inject a fake module.
    - Supports two model APIs:
      - model.decode(audio_path, left_context=..., right_context=..., **kwargs) -> dict
      - model.transcribe(audio_path, **kwargs) -> (segments, info)
    - Produces an ASRResult from the returned structure.
    """

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        self.model_path = model_path
        self.model = None
        self._explicit_model_arg = model_path is not None

        # Resolve from env or config if not explicitly provided
        env_model = os.getenv("CHUNKFORMER_MODEL_PATH")
        # preserve config_model for later decision logic
        config_model = (
            getattr(cfg.settings.chunkformer, "chunkformer_model_path", None)
            if hasattr(cfg, "settings")
            else None
        )
        if self.model_path is None:
            if env_model:
                self.model_path = env_model
            elif config_model is not None:
                self.model_path = str(config_model)

        # Decide whether to attempt loading:
        should_load = False
        if self.model_path is not None:
            # If explicit constructor arg -> always try
            if self._explicit_model_arg:
                should_load = True
            else:
                # Heuristic to decide if string looks like a filesystem path
                def _looks_like_path(p: str) -> bool:
                    if not isinstance(p, str):
                        return False
                    if p.startswith(("/", "./", "../", "~")):
                        return True
                    if os.sep in p:
                        return True
                    return False

                if (
                    not _looks_like_path(self.model_path)
                    or Path(self.model_path).exists()
                ):
                    should_load = True

        if should_load:
            self._load_model(**kwargs)
        else:
            # Defensive: if the configuration object contains a chunkformer path,
            # ensure we attempt to load it even when heuristics above decided not to.
            # Tests rely on being able to set cfg.settings.chunkformer_model_path via monkeypatch.
            try:
                cfg_model_check = getattr(
                    cfg.settings.chunkformer, "chunkformer_model_path", None
                )
            except Exception:
                cfg_model_check = None
            if not self._explicit_model_arg and cfg_model_check is not None:
                self._load_model(**kwargs)

    def _load_model(self, **kwargs) -> None:
        """
        Load the ChunkFormer model.

        - Lazy-imports the chunkformer module to allow tests to inject a fake module.
        - Prefer high-level `ChunkFormerModel.from_pretrained(...)` if available,
          otherwise fall back to `cf.load_model(...)` or `ChunkFormerModel(...)`.
        - Resolve device via env/cfg with auto-detection fallback (torch.cuda).
        - Annotate self.model.device when possible and raise informative errors.
        """
        try:
            import chunkformer as cf  # type: ignore
        except Exception as exc:
            if self._explicit_model_arg or self.model_path is not None:
                raise RuntimeError(
                    "chunkformer library is not installed or could not be imported"
                ) from exc
            return

        # Device resolution: backend-specific env > config > universal device utility
        # Check backend-specific env var first (CHUNKFORMER_DEVICE)
        device_env = os.getenv("CHUNKFORMER_DEVICE")
        cfg_device = (
            getattr(cfg.settings.chunkformer, "chunkformer_device", None)
            if hasattr(cfg, "settings")
            else None
        )

        # Use select_device() utility for consistent device selection
        # It handles "auto", CUDA detection, and fallbacks
        prefer_device = device_env or cfg_device or "auto"
        requires_cuda = prefer_device in ("auto", "cuda") or prefer_device.startswith(
            "cuda"
        )
        device = select_device(prefer=prefer_device, allow_mps=False)

        # If CUDA is required but not available, raise error
        if requires_cuda:
            ensure_cuda_available(
                "CUDA is not available. GPU is required for offline deployment."
            )
            # Ensure device is set to cuda if we required it
            if device != "cuda" and not device.startswith("cuda"):
                device = "cuda"

        try:
            # Prefer Class.from_pretrained if present
            ModelCls = getattr(cf, "ChunkFormerModel", None)
            if (
                ModelCls is not None
                and hasattr(ModelCls, "from_pretrained")
                and callable(getattr(ModelCls, "from_pretrained"))
            ):
                # from_pretrained doesn't accept device parameter, load first then move to device
                self.model = ModelCls.from_pretrained(self.model_path, **kwargs)
            elif hasattr(cf, "load_model") and callable(getattr(cf, "load_model")):
                # Try load_model with device, but may not be supported
                try:
                    self.model = cf.load_model(self.model_path, device=device, **kwargs)
                except TypeError:
                    # If load_model doesn't accept device, load then move
                    self.model = cf.load_model(self.model_path, **kwargs)
            elif ModelCls is not None:
                # Try constructor with device, but may not be supported
                try:
                    self.model = ModelCls(self.model_path, device=device, **kwargs)
                except TypeError:
                    # If constructor doesn't accept device, create then move
                    self.model = ModelCls(self.model_path, **kwargs)
            else:
                raise RuntimeError("chunkformer module has no known model entrypoints")

            # Explicitly move model to the target device
            # ChunkFormer models are PyTorch models and support .to() method
            if device and device != "cpu":
                try:
                    import torch

                    if hasattr(self.model, "to") and callable(
                        getattr(self.model, "to")
                    ):
                        self.model = self.model.to(device)
                        logger.debug("Moved ChunkFormer model to device: {}", device)
                        # Verify device placement
                        try:
                            if hasattr(self.model, "parameters") and callable(
                                self.model.parameters
                            ):
                                first_param = next(self.model.parameters(), None)
                                if first_param is not None:
                                    actual_device = str(first_param.device)
                                    logger.debug(
                                        "ChunkFormer model device verification: requested={}, actual={}",
                                        device,
                                        actual_device,
                                    )
                        except Exception:
                            pass  # Verification failed, but model was moved
                except Exception as move_exc:
                    logger.warning(
                        "Failed to move ChunkFormer model to device {}: {}",
                        device,
                        move_exc,
                    )

            try:
                setattr(self.model, "device", device)
            except Exception:
                pass
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ChunkFormer model from '{self.model_path}': {exc}"
            ) from exc

    def execute(self, audio_data: bytes, **kwargs) -> ASRResult:
        """
        Execute ASR processing.

        Accepts raw bytes or a file path (str/Path). Bytes are written to a temp file.
        Calls model.decode(...) preferred, otherwise model.transcribe(...).

        Raises RuntimeError when model is not loaded or API incompatible.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        cleanup_path: Optional[str] = None
        audio_path = audio_data

        # If bytes provided, write to temp file
        if isinstance(audio_data, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(audio_data)
                cleanup_path = tf.name
                audio_path = cleanup_path

        # Prepare a no_grad context if torch is available to reduce memory during inference
        try:
            import torch as _torch  # type: ignore

            no_grad_ctx = _torch.no_grad
        except Exception:
            # fallback to a no-op context manager
            from contextlib import nullcontext  # type: ignore

            no_grad_ctx = nullcontext

        try:
            # DEBUG: Log ASR input metadata
            import os

            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                logger.debug(
                    "ChunkFormer ASR input metadata",
                    audio_path=audio_path,
                    file_size_bytes=file_size,
                    kwargs=kwargs,
                )

            with no_grad_ctx():
                # Prefer endless_decode (long-form) if available
                if hasattr(self.model, "endless_decode") and callable(
                    getattr(self.model, "endless_decode")
                ):
                    # Build parameter dict with all ChunkFormer settings
                    # User kwargs override config defaults
                    params = {
                        "chunk_size": kwargs.pop(
                            "chunk_size",
                            cfg.settings.chunkformer.chunkformer_chunk_size,
                        ),
                        "left_context_size": kwargs.pop(
                            "left_context",
                            cfg.settings.chunkformer.chunkformer_left_context_size,
                        ),
                        "right_context_size": kwargs.pop(
                            "right_context",
                            cfg.settings.chunkformer.chunkformer_right_context_size,
                        ),
                        "total_batch_duration": kwargs.pop(
                            "total_batch_duration",
                            cfg.settings.chunkformer.chunkformer_total_batch_duration,
                        ),
                        "return_timestamps": kwargs.pop(
                            "return_timestamps",
                            cfg.settings.chunkformer.chunkformer_return_timestamps,
                        ),
                    }
                    # Add any additional kwargs
                    params.update(kwargs)

                    result = self.model.endless_decode(audio_path, **params)

                    # Debug log: See what ChunkFormer actually returns
                    logger.debug(
                        "ChunkFormer endless_decode result type: {}",
                        type(result).__name__,
                    )

                    # ChunkFormer's endless_decode returns a list of segments directly
                    # Each segment may have: start/end as HH:MM:SS.mm/.mmm or float; and "decode" for text
                    if isinstance(result, list):
                        # Direct list of segments from ChunkFormer
                        segments = result
                        language = (
                            None  # ChunkFormer doesn't provide language detection
                        )
                        confidence = (
                            None  # ChunkFormer doesn't provide confidence scores
                        )

                        logger.debug("ChunkFormer returned {} segments", len(segments))

                        # Normalize segments: ChunkFormer uses "decode" field, we need "text" for API consistency
                        normalized_segments = []
                        for seg in segments:
                            normalized_segments.append(_normalize_segment(seg))

                        segments = normalized_segments

                        if segments:
                            logger.debug(
                                "First segment sample - start: {} end: {} text: {}",
                                segments[0].get("start"),
                                segments[0].get("end"),
                                segments[0].get("text", "")[:50],
                            )
                    elif isinstance(result, dict):
                        # Fallback for dict response (shouldn't happen with endless_decode)
                        segments = result.get("segments", [])
                        language = result.get("language", None)
                        confidence = result.get("confidence", None)

                        # Normalize dict segments as well
                        normalized_segments = []
                        for seg in segments:
                            normalized_segments.append(_normalize_segment(seg))
                        segments = normalized_segments
                    else:
                        # Treat as plain text (fallback for unexpected return type)
                        text_str = str(result).strip()
                        segments = (
                            [{"start": None, "end": None, "text": text_str}]
                            if text_str
                            else []
                        )
                        language = None
                        confidence = None

                        logger.warning(
                            "ChunkFormer returned unexpected type: {}, treating as plain text",
                            type(result).__name__,
                        )
                elif hasattr(self.model, "decode") and callable(
                    getattr(self.model, "decode")
                ):
                    # Build parameter dict for decode method
                    # User kwargs override config defaults
                    params = {
                        "left_context": kwargs.pop(
                            "left_context",
                            cfg.settings.chunkformer.chunkformer_left_context_size,
                        ),
                        "right_context": kwargs.pop(
                            "right_context",
                            cfg.settings.chunkformer.chunkformer_right_context_size,
                        ),
                    }
                    # Add any additional kwargs
                    params.update(kwargs)

                    # Model.decode may accept an iterator/list of chunks or a file path;
                    # pass the path and let the model handle chunking if it supports it.
                    result = self.model.decode(audio_path, **params)

                    logger.debug(
                        "ChunkFormer decode result type: {}", type(result).__name__
                    )

                    if not isinstance(result, dict):
                        raise RuntimeError(
                            "Unexpected return type from ChunkFormerModel.decode"
                        )

                    segments = result.get("segments", [])
                    language = result.get("language", None)
                    confidence = result.get("confidence", None)

                    # Normalize segments: timestamps to float seconds, map decode→text
                    normalized_segments = []
                    for seg in segments:
                        normalized_segments.append(_normalize_segment(seg))

                    segments = normalized_segments

                    if segments:
                        logger.debug(
                            "First segment from decode - start: {} end: {} text: {}",
                            segments[0].get("start"),
                            segments[0].get("end"),
                            segments[0].get("text", "")[:50],
                        )
                else:
                    # Fallback to transcribe-like API
                    if hasattr(self.model, "transcribe") and callable(
                        getattr(self.model, "transcribe")
                    ):
                        segs, info = self.model.transcribe(audio_path, **kwargs)

                        # Normalize segments to list of dicts
                        normalized_segments = []
                        for s in segs:
                            normalized_segments.append(_normalize_segment(s))

                        segments = normalized_segments

                        language = (
                            getattr(info, "language", None)
                            if info is not None
                            else None
                        )
                        confidence = (
                            getattr(info, "confidence", None)
                            if info is not None
                            else None
                        )

                        if segments:
                            logger.debug(
                                "Transcribe returned {} segments", len(segments)
                            )
                    else:
                        raise RuntimeError(
                            "Loaded ChunkFormer model has no compatible decode/transcribe API"
                        )

            # Extract full text by joining all segment texts
            # All segments should now have normalized "text" field
            text_parts = [
                seg.get("text", "") for seg in segments if seg.get("text", "").strip()
            ]
            text = " ".join(text_parts).strip()

            # DEBUG: Log ASR output preview
            text_preview = text[:200] + "..." if len(text) > 200 else text
            logger.debug(
                "ChunkFormer ASR output preview",
                text_preview=text_preview,
                segment_count=len(segments),
                language=language,
                confidence=confidence,
                char_count=len(text),
                word_count=len(text.split()) if text else 0,
            )

            return ASRResult(
                text=text, segments=segments, language=language, confidence=confidence
            )
        finally:
            if cleanup_path:
                try:
                    os.remove(cleanup_path)
                except Exception:
                    pass

    def unload(self) -> None:
        """Unload model and release resources."""
        if self.model is not None:
            try:
                close_fn = getattr(self.model, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
        self.model = None

    def get_version_info(self) -> VersionInfo:
        """Return version metadata for the backend (NFR-1 compliance)."""
        info: VersionInfo = {}
        info["backend"] = "chunkformer"
        info["model_variant"] = cfg.settings.chunkformer.chunkformer_model_variant
        info["model_path"] = self.model_path
        info["library"] = "chunkformer"

        # Get library version if available
        try:
            import chunkformer

            info["version"] = getattr(chunkformer, "__version__", "unknown")
        except Exception:
            info["version"] = "unknown"

        # Add architecture parameters for reproducibility (NFR-1 requirement)
        # Only include valid fields in VersionInfo TypedDict
        info["model_variant"] = cfg.settings.chunkformer.chunkformer_model_variant
        info["model_path"] = str(self.model_path) if self.model_path else "unknown"

        # Device info
        try:
            device = getattr(self.model, "device", None)
            # Convert torch.device to string if needed
            if device is not None:
                info["device"] = str(device)
        except Exception:
            pass

        # Checkpoint hash (if available from model)
        try:
            checkpoint_hash = getattr(self.model, "checkpoint_hash", None)
            info["checkpoint_hash"] = checkpoint_hash if checkpoint_hash else ""
        except Exception:
            info["checkpoint_hash"] = ""

        # Add ChunkFormer-specific architecture parameters for reproducibility
        info["chunk_size"] = cfg.settings.chunkformer.chunkformer_chunk_size
        info["left_context_size"] = (
            cfg.settings.chunkformer.chunkformer_left_context_size
        )
        info["right_context_size"] = (
            cfg.settings.chunkformer.chunkformer_right_context_size
        )
        info["total_batch_duration"] = (
            cfg.settings.chunkformer.chunkformer_total_batch_duration
        )
        info["return_timestamps"] = (
            cfg.settings.chunkformer.chunkformer_return_timestamps
        )

        return info

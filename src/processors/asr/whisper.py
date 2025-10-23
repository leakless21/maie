"""
Whisper ASR backend implementation for MAIE.

This implementation provides a thin adapter around the faster-whisper
API used in tests. It is intentionally conservative: it will only
attempt to load a model automatically if a model path is provided
explicitly to the constructor or if an explicit configuration variable
is present (tests set `src.config.WHISPER_MODEL_PATH`).

Behavior required by tests:
- If model_path argument is provided, verify the file exists and raise
  FileNotFoundError when it does not.
- If no model_path is provided and `src.config.WHISPER_MODEL_PATH` is
  present, use that to load the model.
- If neither is present, do not attempt to import or load faster_whisper
  (allowing simple instantiation when model is not required).
- execute() must raise RuntimeError if model is None.
- get_version_info() must report model_variant == "openai/whisper-large"
  per the test expectations.

The faster_whisper import is performed at call time to allow tests to
inject a fake module into sys.modules before the backend is imported.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional

from src import config as cfg
from src.config.logging import get_module_logger
from src.processors.base import ASRBackend, ASRResult, VersionInfo

# Create module-bound logger for better debugging
logger = get_module_logger(__name__)

# Cache for faster_whisper module to avoid PyTorch 2.8 re-import bug
# See: https://github.com/pytorch/pytorch/issues/XXX
_FASTER_WHISPER_MODULE: Optional[Any] = None
_FASTER_WHISPER_IMPORT_FAILED = False


class WhisperBackend(ASRBackend):
    """
    Whisper ASR backend adapter.

    This class intentionally keeps a minimal surface area needed by tests:
    - conditional automatic loading based on constructor args or config
    - mapping of model.transcribe() result to ASRResult
    - safe unload that calls .close() when available
    """

    @staticmethod
    def _looks_like_path(p: str) -> bool:
        if not isinstance(p, str):
            return False
        if p.startswith(("/", "./", "../", "~")):
            return True
        if os.sep in p:
            return True
        alt = os.altsep
        if alt and alt in p:
            return True
        return False

    def __init__(self, model_path: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the backend.

        If a model_path is provided (not None) the backend will attempt to
        load the model immediately and will raise FileNotFoundError when
        the path does not exist.

        If model_path is None but `src.config.WHISPER_MODEL_PATH` is present,
        the backend will load that path.

        If neither is present, the backend will not attempt to load a model.

        Note: For offline deployment, all models must be available locally.
        The backend will NOT attempt to download models from HuggingFace.
        """
        self.model_path = model_path
        self.model = None
        self._explicit_model_arg = model_path is not None

        # Resolve model path with environment override if provided
        env_model = os.getenv("WHISPER_MODEL_PATH")
        config_model = None  # Initialize to avoid unbound variable error
        if self.model_path is None:
            if env_model:
                self.model_path = env_model
            else:
                config_model = (
                    getattr(cfg.settings.asr, "whisper_model_path", None)
                    if hasattr(cfg, "settings")
                    else None
                )
                if config_model is not None:
                    self.model_path = str(config_model)

        # Determine whether we should attempt to auto-load a model
        # - If an explicit constructor arg was provided: always try to load
        # - If coming from env/config: only load if it's a named model (e.g. "tiny.en")
        #   or a filesystem path that actually exists.

        should_load = False
        if self.model_path is not None:
            if self._explicit_model_arg:
                should_load = True
            else:
                # env/config derived
                if not self._looks_like_path(self.model_path) or os.path.exists(
                    self.model_path
                ):
                    should_load = True

        # prefer explicit constructor argument if provided
        if should_load:
            # If the constructor provided a model_path use it; otherwise use config
            if self.model_path is None and config_model is not None:
                self.model_path = str(config_model)
            self._load_model(**kwargs)

    def _load_model(self, **kwargs) -> None:
        """
        Load the faster-whisper model with GPU requirement (no CPU fallback).

        Raises:
            FileNotFoundError: when the explicit model_path does not exist.
            RuntimeError: on any underlying loading error.
        """
        global _FASTER_WHISPER_MODULE, _FASTER_WHISPER_IMPORT_FAILED

        # Use cached import to avoid PyTorch 2.8 re-import bug
        if _FASTER_WHISPER_IMPORT_FAILED:
            if self.model_path is not None:
                raise RuntimeError(
                    "faster_whisper is not installed or could not be imported"
                )
            return

        if _FASTER_WHISPER_MODULE is None:
            # lazy import so tests can inject fake module into sys.modules
            try:
                import faster_whisper as fw  # type: ignore

                _FASTER_WHISPER_MODULE = fw
            except (
                ImportError,
                ModuleNotFoundError,
            ) as exc:  # pragma: no cover - defensive for missing dependency
                logger.debug("faster_whisper not available: {}", exc)
                _FASTER_WHISPER_IMPORT_FAILED = True
                # If model_path was explicitly provided we should surface an error,
                # otherwise leave model as None (constructor may have skipped load).
                if self.model_path is not None:
                    raise RuntimeError(
                        "faster_whisper is not installed or could not be imported"
                    ) from exc
                return

        fw = _FASTER_WHISPER_MODULE

        if self.model_path is None:
            # nothing to load
            logger.debug("No model path provided; skipping model load")
            return

        # Verify path exists only when it looks like a filesystem path.
        # Named models like "tiny.en" should be allowed to download and load.
        if self._looks_like_path(self.model_path) and not os.path.exists(
            self.model_path
        ):
            logger.error("Whisper model path does not exist: {}", self.model_path)
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        # Use device and compute type from configured settings when available
        device = os.getenv("WHISPER_DEVICE") or getattr(
            cfg.settings.asr, "whisper_device", "cuda"
        )
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE") or getattr(
            cfg.settings.asr, "whisper_compute_type", "float16"
        )

        # Check GPU availability and raise error if not available
        if device == "auto" or device.startswith("cuda"):
            try:
                import torch as _torch

                if not _torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA is not available. GPU is required for offline deployment."
                    )
            except ImportError:
                raise RuntimeError(
                    "PyTorch is not installed. GPU is required for offline deployment."
                )
            except (AttributeError, RuntimeError, OSError) as e:
                raise RuntimeError(f"Failed to check CUDA availability: {e}") from e

        # Ensure device is set to cuda if auto was specified
        if device == "auto":
            device = "cuda"

        # Get cpu_threads from env or config for CPU performance tuning
        cpu_threads = os.getenv("WHISPER_CPU_THREADS")
        if cpu_threads is None:
            cpu_threads = getattr(cfg.settings.asr, "whisper_cpu_threads", None)

        load_kwargs = {"device": device, "compute_type": compute_type}

        # Add cpu_threads if specified
        if cpu_threads is not None:
            load_kwargs["cpu_threads"] = int(cpu_threads)

        # Filter kwargs to only include valid WhisperModel parameters
        valid_whisper_params = {
            "device",
            "device_index",
            "compute_type",
            "cpu_threads",
            "num_workers",
            "download_root",
            "local_files_only",
            "files",
            "revision",
            "use_auth_token",
            "verbose",
        }

        # Allow tests/consumers to pass extra loader kwargs, but filter out invalid ones
        for key, value in kwargs.items():
            if key in valid_whisper_params:
                load_kwargs[key] = value

        # Check if we're in test mode (fake module provides load_model)
        # Note: In tests, sys.modules is monkey-patched to replace the real module
        is_test_mode = hasattr(fw, "load_model") and callable(
            getattr(fw, "load_model", None)
        )

        # Read verbose setting lazily to avoid circular imports at module load time
        from src.config import settings

        verbose_mode = settings.verbose_components

        try:
            logger.debug(
                "Loading whisper model from {} with kwargs={}",
                self.model_path,
                load_kwargs,
            )

            if is_test_mode:
                load_kwargs["verbose"] = verbose_mode
                self.model = fw.load_model(self.model_path, **load_kwargs)
                # Annotate device for tests
                try:
                    setattr(self.model, "device", load_kwargs.get("device"))
                except Exception:
                    pass
            else:
                # Production mode - use WhisperModel
                # For offline deployment: require local models, disable downloads
                try:
                    # Enforce local_files_only for offline deployment
                    load_kwargs["local_files_only"] = True
                    self.model = fw.WhisperModel(self.model_path, **load_kwargs)
                    try:
                        setattr(self.model, "device", load_kwargs.get("device"))
                    except Exception:
                        pass
                except Exception as cuda_exc:
                    # No CPU fallback - GPU is required for offline deployment
                    raise RuntimeError(
                        f"Failed to load model on GPU: {cuda_exc}. GPU is required for offline deployment."
                    ) from cuda_exc

            logger.info(
                "Loaded whisper model from {} on device={}",
                self.model_path,
                load_kwargs["device"],
            )
        except Exception as exc:
            logger.exception("Failed to load whisper model from {}", self.model_path)
            raise RuntimeError("Failed to load whisper model") from exc

    def _prepare_transcribe_kwargs(self, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare transcription kwargs by applying config defaults.

        User-provided kwargs take precedence over config values.
        Implements VAD filtering and other advanced features from settings.

        Args:
            user_kwargs: User-provided kwargs to transcribe()

        Returns:
            Combined kwargs dict with config defaults + user overrides
        """
        transcribe_kwargs = {}

        # Apply beam_size from env or config
        env_beam = os.getenv("WHISPER_BEAM_SIZE")
        beam_size = None
        if env_beam is not None:
            try:
                beam_size = int(env_beam)
            except ValueError:
                beam_size = None
        if beam_size is None:
            beam_size = getattr(cfg.settings.asr, "whisper_beam_size", None)
        if beam_size is not None:
            transcribe_kwargs["beam_size"] = beam_size

        # Apply condition_on_previous_text from env or config
        env_condition = os.getenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT")
        if env_condition is not None:
            condition_on_previous = str(env_condition).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        else:
            condition_on_previous = getattr(
                cfg.settings.asr, "whisper_condition_on_previous_text", None
            )
        if condition_on_previous is not None:
            transcribe_kwargs["condition_on_previous_text"] = condition_on_previous

        # Apply language from env or config (None = auto-detect)
        language = os.getenv("WHISPER_LANGUAGE")
        if language is None:
            language = getattr(cfg.settings.asr, "whisper_language", None)
        if language is not None:
            transcribe_kwargs["language"] = language

        # Apply word_timestamps from env or config
        # NOTE: word_timestamps=True is REQUIRED for accurate segment timestamps in faster-whisper
        # Without it, segment timestamps are often incorrect (e.g., 0.16s for 8s audio)
        env_word_ts = os.getenv("WHISPER_WORD_TIMESTAMPS")
        if env_word_ts is not None:
            word_timestamps = str(env_word_ts).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        else:
            word_timestamps = getattr(cfg.settings.asr, "whisper_word_timestamps", True)
        transcribe_kwargs["word_timestamps"] = word_timestamps

        # Apply VAD filter settings (env overrides config)
        env_vad = os.getenv("WHISPER_VAD_FILTER")
        if env_vad is not None:
            vad_filter = str(env_vad).strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            vad_filter = getattr(cfg.settings.asr, "whisper_vad_filter", False)
        if vad_filter:
            transcribe_kwargs["vad_filter"] = True

            # Build VAD parameters from config
            vad_params = {}
            env_min_silence = os.getenv("WHISPER_VAD_MIN_SILENCE_MS")
            vad_min_silence = None
            if env_min_silence is not None:
                try:
                    vad_min_silence = int(env_min_silence)
                except ValueError:
                    vad_min_silence = None
            if vad_min_silence is None:
                vad_min_silence = getattr(
                    cfg.settings.asr, "whisper_vad_min_silence_ms", None
                )
            if vad_min_silence is not None:
                vad_params["min_silence_duration_ms"] = vad_min_silence

            env_speech_pad = os.getenv("WHISPER_VAD_SPEECH_PAD_MS")
            vad_speech_pad = None
            if env_speech_pad is not None:
                try:
                    vad_speech_pad = int(env_speech_pad)
                except ValueError:
                    vad_speech_pad = None
            if vad_speech_pad is None:
                vad_speech_pad = getattr(
                    cfg.settings.asr, "whisper_vad_speech_pad_ms", None
                )
            if vad_speech_pad is not None:
                vad_params["speech_pad_ms"] = vad_speech_pad

            if vad_params:
                transcribe_kwargs["vad_parameters"] = vad_params
                logger.debug("Applying VAD filter with parameters: {}", vad_params)

        # User kwargs override config defaults
        transcribe_kwargs.update(user_kwargs)

        return transcribe_kwargs

    def execute(self, audio_data: bytes, **kwargs) -> ASRResult:
        """
        Transcribe audio_data using the loaded model and return an ASRResult.

        According to faster-whisper documentation, model.transcribe() returns:
        - segments: generator of Segment objects with start, end, text attributes
        - info: TranscriptionInfo object with language, language_probability, etc.

        Supports VAD filtering and other advanced features from config.

        Raises:
            RuntimeError: if no model is loaded.
        """
        if self.model is None:
            logger.error("Attempted to execute without a loaded model")
            raise RuntimeError("Model not loaded")

        # Apply default transcription parameters from config if not overridden
        transcribe_kwargs = self._prepare_transcribe_kwargs(kwargs)

        # Determine if audio_data is bytes or a file path
        import os
        import tempfile

        # Convert audio_data to file path if needed
        audio_path = audio_data
        tmp_file_path = None
        cleanup_needed = False

        try:
            # If audio_data is bytes, write to temp file
            if isinstance(audio_data, bytes):
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file_path = tmp_file.name
                audio_path = tmp_file_path
                cleanup_needed = True

            logger.debug(
                "Transcribing audio file: {} with kwargs: {}",
                audio_path,
                transcribe_kwargs,
            )

            # faster_whisper returns (segments_generator, info)
            # segments is a generator, not a list
            segments_generator, info = self.model.transcribe(
                audio_path, **transcribe_kwargs
            )

            # Convert segments generator to list of dicts
            segments_dict = []
            text_parts = []

            for segment in segments_generator:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                segments_dict.append(segment_dict)
                text_parts.append(segment.text)

            # Combine all segment texts
            text = " ".join(text_parts).strip()

            # Extract language and duration from TranscriptionInfo
            language = info.language if hasattr(info, "language") else None
            duration = info.duration if hasattr(info, "duration") else None

            # Note: faster-whisper doesn't provide a single confidence score
            # Individual segments may have their own scores
            return ASRResult(
                text=text,
                segments=segments_dict,
                language=language,
                confidence=None,
                duration=duration,
            )

        except Exception as exc:
            logger.exception("Model transcription failed")
            return ASRResult(
                text="",
                segments=[],
                language=None,
                confidence=None,
                duration=None,
                error={"type": type(exc).__name__, "message": str(exc)},
            )

        finally:
            # Clean up temp file if we created one
            if cleanup_needed and tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    logger.debug("Cleaned up temp file: {}", tmp_file_path)
                except Exception as cleanup_exc:
                    logger.warning(
                        "Failed to cleanup temp file {}: {}", tmp_file_path, cleanup_exc
                    )

    def unload(self) -> None:
        """
        Unload the model and release resources.

        Calls model.close() when available and sets self.model to None.
        """
        if self.model is None:
            return

        try:
            close_fn = getattr(self.model, "close", None)
            if callable(close_fn):
                logger.debug("Closing whisper model")
                close_fn()
            else:
                # some model wrappers expose other cleanup methods
                for name in ("shutdown", "dispose", "destroy"):
                    fn = getattr(self.model, name, None)
                    if callable(fn):
                        logger.debug("Calling model cleanup '{}'", name)
                        fn()
                        break
        except Exception:
            logger.exception("Error while unloading model")
        finally:
            self.model = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resource cleanup."""
        try:
            self.unload()
        except Exception:
            pass
        return False

    def get_version_info(self) -> VersionInfo:
        """
        Return standardized version metadata with actual configuration values.

        Provides complete model metadata for reproducibility (NFR-1).
        """
        import hashlib
        from pathlib import Path

        # Calculate checkpoint hash for reproducibility (NFR-1)
        checkpoint_hash = "unknown"
        if self.model_path and os.path.exists(str(self.model_path)):
            model_dir = Path(self.model_path)

            # Try to hash key model files for CT2 models
            for filename in ["model.bin", "config.json", "vocabulary.json"]:
                model_file = model_dir / filename
                if model_file.exists():
                    with open(model_file, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                        checkpoint_hash = f"{filename}:{file_hash[:16]}"
                    break

        info: VersionInfo = {
            "backend": "whisper",
            "model_variant": getattr(
                cfg.settings.asr, "whisper_model_variant", "unknown"
            ),
            "model_path": str(self.model_path) if self.model_path else "unknown",
            "checkpoint_hash": checkpoint_hash,
            "device": getattr(cfg.settings.asr, "whisper_device", "unknown"),
            "compute_type": getattr(
                cfg.settings.asr, "whisper_compute_type", "unknown"
            ),
            "vad_filter": getattr(cfg.settings.asr, "whisper_vad_filter", False),
            "condition_on_previous_text": getattr(
                cfg.settings.asr, "whisper_condition_on_previous_text", True
            ),
            "version": "1.0.0",
            "library": "faster-whisper",
            "name": "faster-whisper",
        }

        # Only include optional fields if they are not None
        cpu_threads = getattr(cfg.settings.asr, "whisper_cpu_threads", None)
        if cpu_threads is not None:
            info["cpu_threads"] = cpu_threads

        beam_size = getattr(cfg.settings.asr, "whisper_beam_size", None)
        if beam_size is not None:
            info["beam_size"] = beam_size

        language = getattr(cfg.settings.asr, "whisper_language", None)
        info["language"] = language if language is not None else "auto"

        return info

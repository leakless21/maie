"""
Sequential processing pipeline for the Modular Audio Intelligence Engine (MAIE).

This module contains the main processing logic that executes:
1. Audio preprocessing (normalization, validation)
2. ASR (Automatic Speech Recognition) processing
3. LLM (Large Language Model) processing (enhancement + summarization)
With proper state management, GPU memory management, and error handling.
"""

import json
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Optional torch import (may not be available in test environment)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from loguru import logger
from redis import Redis
from rq import get_current_job

from src.config.logging import get_module_logger

# Create module-bound logger for better debugging
logger = get_module_logger(__name__)

from src.api.errors import (
    ASRProcessingError,
    AudioPreprocessingError,
    AudioValidationError,
    FileSystemError,
    LLMProcessingError,
    ModelLoadError,
    ModelMemoryError,
    NetworkError,
    ProcessingError,
)
from src.core.error_handler import leverage_native_error
from src.api.schemas import TaskStatus
from src.config import settings

# =============================================================================
# Helper Functions
# =============================================================================


def _sanitize_metadata(value: Any, _seen: Optional[set[int]] = None) -> Any:
    """
    Convert metadata into JSON-serializable primitives.

    Handles nested dicts/lists and falls back to string representation for
    complex objects (e.g., MagicMock instances in tests).

    Args:
        value: The value to sanitize
        _seen: Internal set to track object IDs and prevent recursion

    Returns:
        JSON-serializable representation of the value
    """
    # Initialize recursion tracking
    if _seen is None:
        _seen = set()

    # Handle MagicMock and other test objects early
    if hasattr(value, "_mock_name") or hasattr(value, "_mock_parent"):
        return f"[Mock object: {type(value).__name__}]"

    # Check for recursion using object identity
    obj_id = id(value)
    if obj_id in _seen:
        return f"[Recursive reference to {type(value).__name__}]"

    # Add current object to seen set
    _seen.add(obj_id)

    try:
        if isinstance(value, dict):
            return {str(k): _sanitize_metadata(v, _seen) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_sanitize_metadata(v, _seen) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "dict") and callable(value.dict):
            try:
                return _sanitize_metadata(value.dict(), _seen)
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(
                    "Failed to serialize object with dict() method",
                    error=str(e),
                    object_type=type(value).__name__,
                )
                return str(value)
        if hasattr(value, "__iter__") and not isinstance(
            value, (bytes, bytearray, str)
        ):
            try:
                return [_sanitize_metadata(v, _seen) for v in list(value)]
            except (TypeError, ValueError) as e:
                logger.debug(
                    "Failed to iterate over object",
                    error=str(e),
                    object_type=type(value).__name__,
                )
                return str(value)
        return str(value)
    finally:
        # Clean up tracking for this branch
        _seen.discard(obj_id)


def _update_status(
    client: Redis,
    task_key: str,
    status: TaskStatus,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Update task status in Redis with proper JSON serialization.

    Args:
        client: Redis client instance
        task_key: Redis key for the task (format: "task:{task_id}")
        status: New task status
        details: Optional additional details to store with status
    """
    # Build update data
    update_data = {
        "status": status.value,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Merge in additional details if provided
    if details:
        # Serialize complex objects (dicts, lists) to JSON strings
        for key, value in details.items():
            if isinstance(value, (dict, list)):
                update_data[key] = json.dumps(_sanitize_metadata(value))
            else:
                update_data[key] = value

    # Update in Redis
    client.hset(task_key, mapping=update_data)

    # Log status change
    logger.info(
        "Task status updated",
        task_key=task_key,
        status=status.value,
        has_details=bool(details),
    )


def _calculate_edit_rate(original: str, enhanced: str) -> float:
    """
    Calculate Levenshtein distance ratio between two strings.

    Uses dynamic programming to compute the minimum edit distance,
    then normalizes by the maximum string length.

    Args:
        original: Original string
        enhanced: Enhanced/modified string

    Returns:
        Float between 0.0 (identical) and 1.0 (completely different)

    Algorithm:
        - Levenshtein distance via DP matrix
        - Normalized by max(len(original), len(enhanced))
        - 0.0 = no changes, 1.0 = complete replacement
    """
    # Handle empty strings
    if not original and not enhanced:
        return 0.0
    if not original or not enhanced:
        return 1.0

    len1, len2 = len(original), len(enhanced)

    # Create DP matrix
    # dp[i][j] = minimum edits to transform original[:i] to enhanced[:j]
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize base cases
    # Converting from empty string requires insertions
    for j in range(len2 + 1):
        dp[0][j] = j

    # Converting to empty string requires deletions
    for i in range(len1 + 1):
        dp[i][0] = i

    # Fill DP matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # Cost of substitution (0 if characters match, 1 otherwise)
            cost = 0 if original[i - 1] == enhanced[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution or match
            )

    # The edit distance is in the bottom-right corner
    distance = dp[len1][len2]

    # Normalize by the maximum length
    max_len = max(len1, len2)
    edit_rate = distance / max_len if max_len > 0 else 0.0

    return edit_rate


# =============================================================================
# Legacy Functions (to be refactored)
# =============================================================================


@dataclass
class ProcessingResult:
    """Data class to hold the result of audio processing."""

    transcription: Optional[str] = None
    summary: Optional[Dict[str, Any]] = (
        None  # Changed to Dict[str, Any] to match structured summary
    )
    rtf: Optional[float] = None  # Real-time factor
    confidence: Optional[float] = None
    version_metadata: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def load_asr_model(asr_backend: str = "whisper", **config) -> Any:
    """
    Load ASR model into memory based on the specified backend.

    Args:
        asr_backend: ASR backend type ("whisper" or "chunkformer")
        **config: Additional backend-specific configuration

    Returns:
        ASR backend instance

    Raises:
        ModelLoadError: If model loading fails
    """
    try:
        logger.info("Loading ASR model", backend=asr_backend, config=config)
        from src.processors.asr.factory import ASRFactory

        asr_model = ASRFactory.create(backend_type=asr_backend, **config)
        logger.info("ASR model loaded successfully", backend=asr_backend)
        return asr_model
    except ValueError as e:
        logger.error("Invalid ASR backend", backend=asr_backend, error=str(e))
        raise ModelLoadError(
            message=f"Invalid ASR backend: {asr_backend}",
            details={
                "backend": asr_backend,
                "available_backends": ["whisper", "chunkformer"],
            },
        )
    except (ImportError, RuntimeError, FileNotFoundError, OSError) as e:
        logger.error("Failed to load ASR model", backend=asr_backend, error=str(e))
        raise ModelLoadError(
            message=f"Failed to load ASR model: {str(e)}",
            details={"backend": asr_backend, "error_type": type(e).__name__},
        ) from e

    except Exception as e:
        logger.error(
            "Unexpected error loading ASR model", backend=asr_backend, error=str(e)
        )
        raise ModelLoadError(
            message=f"Unexpected error loading ASR model: {str(e)}",
            details={"backend": asr_backend, "error_type": type(e).__name__},
        ) from e


def execute_asr_transcription(
    asr_model: Any, audio_path: str, audio_duration: float
) -> Tuple[str, float, float, Dict[str, Any]]:
    """
    Execute ASR transcription on the provided audio.

    Args:
        asr_model: Loaded ASR backend instance
        audio_path: Path to audio file (normalized or raw)
        audio_duration: Actual audio duration in seconds from preprocessing

    Returns:
        Tuple of (transcription, rtf, confidence, asr_metadata)

    Raises:
        ASRProcessingError: If transcription fails
    """
    try:
        logger.info("Executing ASR transcription", audio_path=audio_path)

        start_time = time.time()

        # Execute transcription - read file as bytes for backend
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        result = asr_model.execute(audio_bytes)

        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0

        # Extract transcription text
        if hasattr(result, "text"):
            transcription = result.text
        elif hasattr(result, "transcript"):
            transcription = result.transcript
        else:
            transcription = str(result)

        # Extract confidence
        confidence = getattr(result, "confidence", None) or getattr(
            result, "confidence_avg", 0.0
        )

        # Collect ASR-specific metadata for versioning (NFR-1)
        asr_metadata = {}
        if hasattr(result, "model_name"):
            asr_metadata["model_name"] = result.model_name
        if hasattr(result, "checkpoint_hash"):
            asr_metadata["checkpoint_hash"] = result.checkpoint_hash
        if hasattr(result, "duration_ms"):
            asr_metadata["duration_ms"] = result.duration_ms
        if hasattr(result, "language"):
            asr_metadata["language"] = result.language
        if hasattr(result, "language_probability"):
            asr_metadata["language_probability"] = result.language_probability

        try:
            transcript_length = len(transcription)  # type: ignore[arg-type]
        except TypeError:
            transcript_length = len(str(transcription))
        except Exception:
            transcript_length = 0

        logger.info(
            "ASR transcription complete",
            transcript_length=transcript_length,
            rtf=rtf,
            confidence=confidence,
            processing_time=processing_time,
        )

        return transcription, rtf, confidence, asr_metadata

    except FileNotFoundError as e:
        logger.error("Audio file not found", audio_path=audio_path, error=str(e))
        raise ASRProcessingError(
            message=f"Audio file not found: {audio_path}",
            details={"audio_path": audio_path},
        ) from e
    except (RuntimeError, OSError, ValueError, TypeError) as e:
        logger.error(
            "ASR transcription failed", error=str(e), traceback=traceback.format_exc()
        )
        raise ASRProcessingError(
            message=f"ASR transcription failed: {str(e)}",
            details={"error_type": type(e).__name__, "audio_path": audio_path},
        ) from e

    except Exception as e:
        logger.error(
            "ASR transcription failed",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        raise ASRProcessingError(
            message=f"ASR transcription failed: {str(e)}",
            details={"error_type": type(e).__name__, "audio_path": audio_path},
        ) from e


def unload_asr_model(asr_model: Any) -> None:
    """
    Unload ASR model from memory to free GPU resources.

    Critical for sequential processing: Always called in finally block
    to guarantee GPU memory is freed even if ASR execution fails.

    Args:
        asr_model: ASR backend instance to unload
    """
    try:
        logger.info("Unloading ASR model")
        if asr_model is not None and hasattr(asr_model, "unload"):
            asr_model.unload()

        # Critical: Clear CUDA cache to free GPU memory
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("ASR model unloaded and GPU cache cleared")
        else:
            logger.info("ASR model unloaded")
    except (AttributeError, RuntimeError, OSError) as e:
        logger.warning("Error during ASR model unload", error=str(e))
    except Exception as e:
        logger.warning("Unexpected error during ASR model unload", error=str(e))


def load_llm_model() -> Any:
    """
    Load LLM model into memory.

    Returns:
        LLM processor instance

    Raises:
        ModelLoadError: If model loading fails
    """
    try:
        logger.info("Loading LLM model")
        from src.processors.llm import LLMProcessor

        llm_model = LLMProcessor()
        # Trigger lazy loading
        llm_model._load_model()
        logger.info("LLM model loaded successfully")
        return llm_model
    except (ImportError, RuntimeError, FileNotFoundError, OSError) as e:
        logger.error("Failed to load LLM model", error=str(e))
        raise ModelLoadError(
            message=f"Failed to load LLM model: {str(e)}",
            details={"error_type": type(e).__name__},
        ) from e

    except Exception as e:
        logger.error("Failed to load LLM model", error=str(e))
        raise ModelLoadError(
            message=f"Failed to load LLM model: {str(e)}",
            details={"error_type": type(e).__name__},
        ) from e


def execute_llm_processing(
    llm_model: Any,
    transcription: str,
    features: list,
    template_id: Optional[str] = None,
    asr_backend: str = "whisper",
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Execute LLM processing with proper error handling.

    Args:
        llm_model: LLM processor instance
        transcription: Raw transcription text
        features: List of requested features
        template_id: Template ID for structured summarization
        asr_backend: ASR backend used (affects enhancement decision)

    Returns:
        Tuple of (enhanced_transcript, structured_summary)

    Raises:
        LLMProcessingError: If LLM processing fails critically
    """
    try:
        # Check for empty transcript before any processing
        safe_text = (transcription or "").strip()
        if not safe_text:
            raise LLMProcessingError(
                message="Empty transcript after ASR; no content to process",
                details={"transcription_length": 0},
            )

        logger.info(
            "Executing LLM processing", features=features, template_id=template_id
        )

        results = {}

        # Text enhancement (conditional based on ASR backend capabilities)
        # Skip enhancement if ASR backend provides native punctuation (e.g., Whisper with erax-wow-turbo)
        needs_enhancement = (
            "clean_transcript" in features
            and hasattr(llm_model, "needs_enhancement")
            and llm_model.needs_enhancement(asr_backend)
        )

        if needs_enhancement:
            try:
                logger.info("Applying text enhancement")
                enhanced_result = llm_model.enhance_text(transcription)
                if enhanced_result.get("enhancement_applied", False):
                    results["clean_transcript"] = enhanced_result["enhanced_text"]
                    results["enhancement_metrics"] = {
                        "edit_rate_cleaning": enhanced_result.get("edit_rate", 0),
                        "edit_distance": enhanced_result.get("edit_distance", 0),
                    }
                    logger.info(
                        "Text enhancement applied",
                        edit_rate=enhanced_result.get("edit_rate", 0),
                    )
                else:
                    results["clean_transcript"] = transcription
                    logger.info("Text enhancement skipped - not needed")
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning(
                    "Text enhancement failed, using raw transcript", error=str(e)
                )
                results["clean_transcript"] = transcription
            except Exception as e:
                logger.warning(
                    "Unexpected error in text enhancement, using raw transcript",
                    error=str(e),
                )
                results["clean_transcript"] = transcription
        else:
            # Skip enhancement - ASR backend provides native punctuation
            results["clean_transcript"] = transcription
            logger.info(
                "Text enhancement skipped - ASR backend provides native punctuation"
            )

        # Structured summarization
        if "summary" in features:
            if not template_id:
                error_msg = "template_id required when summary feature requested"
                logger.error(error_msg)
                raise LLMProcessingError(
                    message=error_msg,
                    details={"features": features, "template_id": template_id},
                )

            try:
                logger.info("Generating structured summary", template_id=template_id)
                summary_result = llm_model.generate_summary(
                    transcript=results["clean_transcript"], template_id=template_id
                )
                if summary_result.get("summary"):
                    results["summary"] = summary_result["summary"]
                    results["summary_metadata"] = {
                        "retry_count": summary_result.get("retry_count", 0),
                        "model_info": summary_result.get("model_info", {}),
                    }
                    logger.info(
                        "Summary generated successfully",
                        retry_count=summary_result.get("retry_count", 0),
                    )
                else:
                    error_msg = summary_result.get("error", "Unknown error")
                    logger.error("Summary generation failed", error=error_msg)
                    raise LLMProcessingError(
                        message=f"Summary generation failed: {error_msg}",
                        details={"template_id": template_id},
                    )
            except LLMProcessingError:
                raise
            except (RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.error("Summary generation failed", error=str(e))
                raise LLMProcessingError(
                    message=f"Summary generation failed: {str(e)}",
                    details={
                        "template_id": template_id,
                        "error_type": type(e).__name__,
                    },
                )
            except Exception as e:
                logger.error("Unexpected summary generation exception", error=str(e))
                raise LLMProcessingError(
                    message=f"Unexpected summary generation exception: {str(e)}",
                    details={
                        "template_id": template_id,
                        "error_type": type(e).__name__,
                    },
                )

        return results.get("clean_transcript"), results.get("summary")

    except LLMProcessingError:
        raise
    except Exception as e:
        logger.error(
            "LLM processing failed", error=str(e), traceback=traceback.format_exc()
        )
        raise LLMProcessingError(
            message=f"LLM processing failed: {str(e)}",
            details={"error_type": type(e).__name__},
        )


def unload_llm_model(llm_model: Any) -> None:
    """
    Unload LLM model from memory to free GPU resources.

    Critical for sequential processing: Always called in finally block
    to guarantee GPU memory is freed even if LLM execution fails.

    Args:
        llm_model: LLM processor instance to unload
    """
    try:
        logger.info("Unloading LLM model")
        if llm_model is not None and hasattr(llm_model, "unload"):
            llm_model.unload()

        # Critical: Clear CUDA cache to free GPU memory
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("LLM model unloaded and GPU cache cleared")
        else:
            logger.info("LLM model unloaded")
    except (AttributeError, RuntimeError, OSError) as e:
        logger.warning("Error during LLM model unload", error=str(e))
    except Exception as e:
        logger.warning("Unexpected error during LLM model unload", error=str(e))


def get_version_metadata(
    asr_result_metadata: Dict[str, Any], llm_model: Any = None
) -> Dict[str, Any]:
    """
    Collect version metadata for the models and processing pipeline per VersionsSchema.

    Args:
        asr_result_metadata: Metadata from ASR backend execution
        llm_model: LLM processor instance (optional)

    Returns:
        Complete version metadata dict matching VersionsSchema
    """
    # Build ASR backend info per ASRBackendSchema
    asr_backend = {
        "name": asr_result_metadata.get("name", "whisper") if asr_result_metadata else "whisper",
        "model_variant": asr_result_metadata.get("model_variant", "unknown") if asr_result_metadata else "unknown",
        "model_path": asr_result_metadata.get("model_path", "") if asr_result_metadata else "",
        "checkpoint_hash": asr_result_metadata.get("checkpoint_hash", "") if asr_result_metadata else "",
        "compute_type": asr_result_metadata.get("compute_type", "float16") if asr_result_metadata else "float16",
        "decoding_params": asr_result_metadata.get("decoding_params", {}) if asr_result_metadata else {},
    }

    # Build LLM info per LLMSchema
    llm_info = None
    if llm_model and hasattr(llm_model, "get_version_info"):
        try:
            llm_info = llm_model.get_version_info()
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("Failed to get LLM version info", error=str(e))
            llm_info = None
        except Exception as e:
            logger.error("Unexpected error getting LLM version info", error=str(e))
            llm_info = None

    if llm_info is None:
        llm = {
            "name": "not_loaded",
            "checkpoint_hash": "",
            "quantization": "",
            "thinking": False,
            "reasoning_parser": None,
            "structured_output": {},
            "decoding_params": {},
        }
    else:
        llm = {
            "name": llm_info.get("name", "unknown"),
            "checkpoint_hash": llm_info.get("checkpoint_hash", ""),
            "quantization": llm_info.get("quantization", ""),
            "thinking": llm_info.get("thinking", False),
            "reasoning_parser": llm_info.get("reasoning_parser"),
            "structured_output": llm_info.get("structured_output", {}),
            "decoding_params": llm_info.get("decoding_params", {}),
        }

    version_metadata = {
        "pipeline_version": settings.pipeline_version,
        "asr_backend": asr_backend,
        "llm": llm,
    }

    return version_metadata


def calculate_metrics(
    transcription: str,
    clean_transcript: Optional[str],
    start_time: float,
    audio_duration: float,
    asr_rtf: float,
) -> Dict[str, Any]:
    """Calculate runtime metrics for the processing per MetricsSchema."""
    total_processing_time = time.time() - start_time
    total_rtf = total_processing_time / audio_duration if audio_duration > 0 else 0

    # Map to MetricsSchema fields
    metrics = {
        "input_duration_seconds": audio_duration,
        "processing_time_seconds": total_processing_time,
        "rtf": total_rtf,
        "vad_coverage": 0.0,  # Default if not available from ASR
        "asr_confidence_avg": 0.0,  # Default if not available from ASR
    }

    # Add enhancement metrics if text enhancement was performed
    if clean_transcript and clean_transcript != transcription:
        # Calculate proper edit distance using Levenshtein algorithm
        edit_rate = _calculate_edit_rate(transcription, clean_transcript)
        metrics["edit_rate_cleaning"] = edit_rate

    return metrics


def update_task_status(
    job_id: str,
    status: TaskStatus,
    redis_conn: Redis,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Update the task status in Redis."""
    if redis_conn:
        status_data = {"status": status.value, "updated_at": time.time()}
        if details:
            status_data.update(details)

        redis_conn.hset(
            f"task:{job_id}", mapping=status_data
        )  # Fixed key format per TDD


def process_audio_task(task_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main pipeline function that processes audio through ASR and LLM sequentially.

    Follows the sequential processing pattern per TDD.md section 3.2:
    1. Redis Connection (DB 1 for results)
    2. Status: PREPROCESSING - Audio validation & normalization
    3. Status: PROCESSING_ASR - Load → Transcribe → Unload → Clear GPU
    4. Status: PROCESSING_LLM - Load → Enhance/Summarize → Unload → Clear GPU
    5. Collect versions and metrics
    6. Status: COMPLETE - Store final results

    Args:
        task_params: Dictionary containing task parameters including:
            - audio_path: Path to the audio file to process
            - asr_backend: ASR backend to use (default: "whisper")
            - features: List of features to process (default: ["clean_transcript", "summary"])
            - template_id: Template ID for summarization (required if "summary" in features)
            - config: Additional configuration for the ASR backend
            - redis_host, redis_port, redis_db: Redis connection parameters

    Returns:
        Dictionary containing the processing result with versions, metrics, and results
    """
    job = get_current_job()
    job_id = job.id if job else "unknown"
    task_key = f"task:{job_id}"

    # Extract parameters
    audio_path = task_params.get("audio_path")
    asr_backend = task_params.get("asr_backend", "whisper")
    features = task_params.get("features", ["clean_transcript", "summary"])
    template_id = task_params.get("template_id")
    config = task_params.get("config", {})

    # Connect to Redis results database (DB 1 per TDD.md section 3.6)
    redis_host = task_params.get("redis_host", "localhost")
    redis_port = task_params.get("redis_port", 6379)
    redis_db = task_params.get("redis_db", 1)  # Use DB 1 for results per TDD
    redis_conn = (
        Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)
        if job
        else None
    )

    start_time = time.time()
    asr_model = None  # Track loaded ASR model for cleanup
    llm_model = None  # Track loaded LLM model for cleanup
    audio_duration = 10.0  # Default fallback, will be updated from preprocessing
    processing_audio_path = audio_path  # Path to use for ASR (normalized or raw)
    version_metadata: Optional[Dict[str, Any]] = None

    try:
        # =====================================================================
        # Stage 1: PREPROCESSING - Audio validation and normalization
        # =====================================================================
        logger.info(
            "Starting audio processing",
            task_id=job_id,
            audio_path=audio_path,
            asr_backend=asr_backend,
            features=features,
        )

        if redis_conn:
            _update_status(redis_conn, task_key, TaskStatus.PREPROCESSING)

        # Validate audio_path before preprocessing
        if not audio_path or not isinstance(audio_path, str):
            error = AudioValidationError(
                message=f"Invalid audio_path provided: {audio_path}",
                details={
                    "audio_path": audio_path,
                    "expected_type": "str",
                    "task_id": job_id,
                },
            )
            logger.error(
                "Invalid audio path",
                task_id=job_id,
                error_code=error.error_code,
                audio_path=audio_path,
            )

            if redis_conn:
                handle_processing_error(
                    redis_conn,
                    task_key,
                    error,
                    stage="preprocessing",
                    error_code=error.error_code,
                )

            raise error

        # Audio preprocessing - validate and normalize to 16kHz mono WAV
        try:
            from src.processors.audio import AudioPreprocessor

            preprocessor = AudioPreprocessor()
            metadata = preprocessor.preprocess(Path(audio_path))

            # Update audio_path if normalization was performed
            if metadata.get("normalized_path"):
                processing_audio_path = str(metadata["normalized_path"])
                logger.info(
                    "Audio normalized",
                    task_id=job_id,
                    original_format=metadata.get("format"),
                    duration=metadata.get("duration"),
                )
            else:
                processing_audio_path = audio_path

            # Store audio duration for metrics calculation
            audio_duration = metadata.get("duration", audio_duration)

            logger.info(
                "Audio preprocessing complete",
                task_id=job_id,
                duration=audio_duration,
                sample_rate=metadata.get("sample_rate"),
                channels=metadata.get("channels"),
                normalized=metadata.get("normalized_path") is not None,
            )
        except ValueError as e:
            # Audio validation errors (too short, etc.)
            error = AudioValidationError(
                message=f"Audio validation failed: {str(e)}",
                details={
                    "task_id": job_id,
                    "audio_path": audio_path,
                    "validation_error": str(e),
                },
            )
            logger.error(
                "Audio validation failed",
                task_id=job_id,
                error_code=error.error_code,
                error=str(e),
            )
            if redis_conn:
                handle_processing_error(
                    redis_conn,
                    task_key,
                    error,
                    stage="preprocessing",
                    error_code=error.error_code,
                )
            raise error
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            subprocess.CalledProcessError,
        ) as e:
            # Preprocessing errors (ffmpeg failures, etc.)
            error = AudioPreprocessingError(
                message=f"Audio preprocessing failed: {str(e)}",
                details={
                    "task_id": job_id,
                    "audio_path": audio_path,
                    "original_error": str(e),
                },
            )
            logger.error(
                "Audio preprocessing failed",
                task_id=job_id,
                error_code=error.error_code,
                error=str(e),
            )
            if redis_conn:
                handle_processing_error(
                    redis_conn,
                    task_key,
                    error,
                    stage="preprocessing",
                    error_code=error.error_code,
                )
            raise error
        except Exception as e:
            # Unexpected preprocessing errors
            error = AudioPreprocessingError(
                message=f"Unexpected audio preprocessing error: {str(e)}",
                details={
                    "task_id": job_id,
                    "audio_path": audio_path,
                    "original_error": str(e),
                },
            )
            logger.error(
                "Audio preprocessing failed",
                task_id=job_id,
                error_code=error.error_code,
                error=str(e),
            )
            if redis_conn:
                handle_processing_error(
                    redis_conn,
                    task_key,
                    error,
                    stage="preprocessing",
                    error_code=error.error_code,
                )
            raise error

        # =====================================================================
        # Stage 2: PROCESSING_ASR - ASR transcription with sequential GPU usage
        # =====================================================================
        try:
            if redis_conn:
                _update_status(redis_conn, task_key, TaskStatus.PROCESSING_ASR)

            logger.info("Loading ASR model", task_id=job_id, backend=asr_backend)

            # Step 2a: Load ASR model based on backend
            asr_model = load_asr_model(asr_backend, **config)

            # Step 2b: Execute ASR transcription with actual duration
            transcription, asr_rtf, confidence, asr_metadata = (
                execute_asr_transcription(
                    asr_model, processing_audio_path, audio_duration
                )
            )

            try:
                process_transcript_length = len(transcription)  # type: ignore[arg-type]
            except TypeError:
                process_transcript_length = len(str(transcription))
            except Exception:
                process_transcript_length = 0

            logger.info(
                "ASR transcription complete",
                task_id=job_id,
                transcript_length=process_transcript_length,
                rtf=asr_rtf,
                confidence=confidence,
            )
        finally:
            # Step 2c: CRITICAL - Always unload ASR model to free GPU memory
            if asr_model is not None:
                unload_asr_model(asr_model)
                asr_model = None

        # =====================================================================
        # Stage 3: PROCESSING_LLM - LLM processing (enhancement + summarization)
        # =====================================================================
        try:
            if redis_conn:
                safe_transcript_len = 0
                if transcription:
                    try:
                        safe_transcript_len = len(transcription)  # type: ignore[arg-type]
                    except TypeError:
                        safe_transcript_len = len(str(transcription))
                    except Exception:
                        safe_transcript_len = 0

                _update_status(
                    redis_conn,
                    task_key,
                    TaskStatus.PROCESSING_LLM,
                    {"transcription_length": safe_transcript_len},
                )

            # Check for empty transcript before loading LLM
            safe_text = (transcription or "").strip()
            if not safe_text:
                raise LLMProcessingError(
                    message="Empty transcript after ASR; no content to process",
                    details={"transcription_length": 0},
                )

            logger.info("Loading LLM model", task_id=job_id)

            # Step 3a: Load LLM model
            llm_model = load_llm_model()

            # Step 3b: Execute LLM processing (enhancement and/or summarization)
            clean_transcript, structured_summary = execute_llm_processing(
                llm_model, transcription, features, template_id, asr_backend
            )

            # Step 3c: Collect version metadata while models are loaded
            try:
                version_metadata = get_version_metadata(asr_metadata, llm_model)
            except Exception as metadata_error:
                logger.error(
                    "Failed to collect version metadata",
                    error=str(metadata_error),
                )
                version_metadata = None

            logger.info(
                "LLM processing complete",
                task_id=job_id,
                enhanced=(
                    clean_transcript != transcription if clean_transcript else False
                ),
                has_summary=structured_summary is not None,
            )
        finally:
            # Step 3d: CRITICAL - Always unload LLM model to free GPU memory
            if llm_model is not None:
                unload_llm_model(llm_model)
                llm_model = None

        # =====================================================================
        # Stage 4: COMPLETE - Collect metadata and store results
        # =====================================================================

        # Calculate metrics (FR-5)
        metrics = calculate_metrics(
            transcription, clean_transcript, start_time, audio_duration, asr_rtf
        )

        # Collect version metadata if not already resolved
        if version_metadata is None:
            version_metadata = get_version_metadata(asr_metadata, None)

        # Prepare result - only include requested features
        result = {"versions": version_metadata, "metrics": metrics, "results": {}}

        # Include requested features in results per ResultsSchema
        if "raw_transcript" in features:
            result["results"]["raw_transcript"] = transcription

        if "clean_transcript" in features:
            result["results"]["clean_transcript"] = clean_transcript or transcription

        if "summary" in features and structured_summary:
            result["results"]["summary"] = structured_summary

        # Update status to COMPLETE and store final results
        if redis_conn:
            _update_status(
                redis_conn,
                task_key,
                TaskStatus.COMPLETE,
                {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "versions": version_metadata,
                    "metrics": metrics,
                    "results": result["results"],
                },
            )

        logger.info(
            "Audio processing completed successfully",
            task_id=job_id,
            processing_time=time.time() - start_time,
            rtf=metrics.get("total_rtf"),
        )

        # Return result as dictionary
        return result

    except (
        AudioValidationError,
        AudioPreprocessingError,
        ASRProcessingError,
        LLMProcessingError,
        ModelLoadError,
    ) as e:
        # Known MAIE errors - already logged with proper context
        logger.error(
            "Audio processing failed with known error",
            task_id=job_id,
            error_code=e.error_code,
            error=str(e),
        )

        # Update status to FAILED using error handler
        if redis_conn:
            handle_processing_error(
                redis_conn,
                task_key,
                e,
                stage="pipeline_execution",
                error_code=e.error_code,
            )

        # Re-raise the exception so RQ marks the job as failed
        raise e
    except (ConnectionError, TimeoutError, OSError) as e:
        # Network and I/O related errors - leverage native errors
        if isinstance(e, (ConnectionError, TimeoutError)):
            leverage_native_error(
                e,
                f"Network error during audio processing: {str(e)}",
                NetworkError,
                task_id=job_id,
            )
        else:  # OSError
            leverage_native_error(
                e,
                f"File system error during audio processing: {str(e)}",
                FileSystemError,
                task_id=job_id,
            )

    except (MemoryError, RuntimeError) as e:
        # System resource errors - leverage native errors
        if isinstance(e, MemoryError):
            leverage_native_error(
                e,
                f"Insufficient memory during audio processing: {str(e)}",
                ModelMemoryError,
                task_id=job_id,
            )
        else:  # RuntimeError
            leverage_native_error(
                e,
                f"Runtime error during audio processing: {str(e)}",
                ProcessingError,
                task_id=job_id,
            )

    except Exception as e:
        # Unknown/unexpected errors - leverage native errors
        leverage_native_error(
            e,
            f"Unexpected error during audio processing: {str(e)}",
            ProcessingError,
            task_id=job_id,
        )

    # This should never be reached due to exception handling above
    return {
        "versions": None,
        "metrics": None,
        "results": None,
        "status": "error",
        "error": {"message": "Unexpected error path taken"},
    }


def handle_processing_error(
    client: Redis,
    task_key: str,
    error: Exception,
    stage: str,
    error_code: Optional[str] = None,
) -> None:
    """
    Handle processing errors by updating task status to FAILED.

    DEPRECATED: Use handle_maie_error from src.api.errors instead.
    This function is kept for backward compatibility.

    Args:
        client: Redis client for updating task status
        task_key: Redis key for the task
        error: The exception that occurred
        stage: Processing stage where error occurred (preprocessing, asr, llm, etc.)
        error_code: Optional error code (e.g., AUDIO_DECODE_ERROR, MODEL_LOAD_ERROR)
    """
    from datetime import datetime, timezone

    from loguru import logger

    try:
        # Prepare error details
        error_message = str(error)
        error_details = {
            "status": TaskStatus.FAILED.value,
            "error": error_message,
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add error code if provided
        if error_code:
            error_details["error_code"] = error_code

        # Update task status in Redis
        client.hset(task_key, mapping=error_details)

        # Log the error
        logger.error(
            f"Task {task_key} failed at stage '{stage}': {error_message}",
            extra={
                "task_key": task_key,
                "stage": stage,
                "error_code": error_code,
                "error_type": type(error).__name__,
            },
        )

    except Exception as redis_error:
        # If Redis update fails, log but don't crash
        logger.error(
            f"Failed to update error status for task {task_key}: {redis_error}",
            extra={
                "task_key": task_key,
                "original_error": str(error),
                "redis_error": str(redis_error),
            },
        )

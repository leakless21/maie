"""
Sequential processing pipeline for the Modular Audio Intelligence Engine (MAIE).

This module contains the main processing logic that executes:
1. Audio preprocessing (normalization, validation)
2. ASR (Automatic Speech Recognition) processing
3. LLM (Large Language Model) processing (enhancement + summary)
With proper state management, GPU memory management, and error handling.
"""

import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Optional torch import (may not be available in test environment)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from redis import Redis
from rq import get_current_job

from src.config.logging import (
    get_module_logger,
    bind_correlation_id,
    clear_correlation_id,
)
from src.api.errors import (
    ASRProcessingError,
    AudioPreprocessingError,
    AudioValidationError,
    LLMProcessingError,
    ModelLoadError,
)
from src.api.schemas import TaskStatus
from src.config import settings

# Create module-bound logger for better debugging
logger = get_module_logger(__name__)

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
    # Build ASR metadata variants - ensure all required fields are strings (not None)
    asr_backend = {
        "name": (asr_result_metadata.get("name") or "whisper")
        if asr_result_metadata
        else "unknown",
        "model_variant": (asr_result_metadata.get("model_variant") or "unknown")
        if asr_result_metadata
        else "unknown",
        "model_path": str(asr_result_metadata.get("model_path") or "")
        if asr_result_metadata
        else "",
        "checkpoint_hash": str(asr_result_metadata.get("checkpoint_hash") or "")
        if asr_result_metadata
        else "",
        "compute_type": (asr_result_metadata.get("compute_type") or "float16")
        if asr_result_metadata
        else "float16",
        "decoding_params": asr_result_metadata.get("decoding_params") or {}
        if asr_result_metadata
        else {},
    }

    # Build LLM info; support multiple expectations across tests
    llm_info_raw = None
    if llm_model and hasattr(llm_model, "get_version_info"):
        try:
            llm_info_raw = llm_model.get_version_info()
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("Failed to get LLM version info", error=str(e))
            llm_info_raw = {"model_name": "unavailable", "error": str(e)}
        except Exception as e:
            logger.error("Unexpected error getting LLM version info", error=str(e))
            llm_info_raw = {"model_name": "unavailable", "error": str(e)}

    if llm_model and hasattr(llm_model, "get_version_info") and llm_info_raw is None:
        # Explicit None indicates missing version info
        llm_block: Any = None
    elif not llm_model:
        llm_block = {
            "model_name": "not_loaded",
            "reason": "no_model_provided",
            # API schema compatible fields
            "name": "not_loaded",
            "checkpoint_hash": "",
            "quantization": "",
            "thinking": False,
            "reasoning_parser": None,
            "structured_output": {},
            "decoding_params": {},
        }
    else:
        # Normalize llm info block containing both API and legacy keys
        _llm = llm_info_raw or {}
        llm_block = {
            "model_name": _llm.get("model_name") or _llm.get("name", "unknown"),
            "checkpoint_hash": _llm.get("checkpoint_hash") or "",
            "backend": _llm.get("backend") or _llm.get("provider"),
            # API schema compatible
            "name": _llm.get("name") or _llm.get("model_name", "unknown"),
            "quantization": _llm.get("quantization") or "",
            "thinking": _llm.get("thinking", False),
            "reasoning_parser": _llm.get("reasoning_parser"),
            "structured_output": _llm.get("structured_output", {}),
            "decoding_params": _llm.get("decoding_params", {}),
            # When prior error, also surface error for tests
            **({"error": _llm.get("error")} if "error" in _llm else {}),
        }

    # If an LLM object is provided but does NOT have get_version_info, mark as not_loaded/method_missing
    if llm_model and not hasattr(llm_model, "get_version_info"):
        llm_block = {
            "model_name": "not_loaded",
            "reason": "method_missing",
            "name": "not_loaded",
            "checkpoint_hash": "",
            "quantization": "",
            "thinking": False,
            "reasoning_parser": None,
            "structured_output": {},
            "decoding_params": {},
        }

    version_metadata: Dict[str, Any] = {
        # API schema key
        "pipeline_version": settings.pipeline_version,
        # Standardized ASR backend information
        "asr_backend": asr_backend,
        # LLM block may be None per tests
        "llm": llm_block,
    }

    return version_metadata


def calculate_metrics(
    transcription: str,
    clean_transcript: Optional[str],
    start_time: float,
    audio_duration: float,
    asr_rtf: float,
) -> Dict[str, Any]:
    """Calculate runtime metrics for the processing per MetricsSchema and FR-5."""
    total_processing_time = time.time() - start_time
    total_rtf = total_processing_time / audio_duration if audio_duration > 0 else 0

    # Coerce inputs to strings for robust metrics
    original_text = (
        transcription if isinstance(transcription, str) else str(transcription)
    )
    enhanced_text = (
        clean_transcript
        if (isinstance(clean_transcript, str) or clean_transcript is None)
        else str(clean_transcript)
    )

    try:
        transcription_length = len(original_text)
    except Exception:
        transcription_length = 0

    # API schema compatible fields
    metrics: Dict[str, Any] = {
        "input_duration_seconds": audio_duration,
        "processing_time_seconds": total_processing_time,
        "rtf": total_rtf,
        "vad_coverage": 0.0,
        "asr_confidence_avg": 0.0,
        "asr_rtf": asr_rtf,
        "transcription_length": transcription_length,
        "audio_duration": audio_duration,  # Kept for backward compatibility with tests
    }

    # Add enhancement metrics if text enhancement was performed
    if enhanced_text and enhanced_text != original_text:
        edit_rate = _calculate_edit_rate(original_text, enhanced_text)
        metrics["edit_rate_cleaning"] = edit_rate

    return metrics


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
            - template_id: Template ID for summary (required if "summary" in features)
            - config: Additional configuration for the ASR backend
            - redis_host, redis_port, redis_db: Redis connection parameters

    Returns:
        Dictionary containing the processing result with versions, metrics, and results
    """
    job = get_current_job()
    job_id = job.id if job else "unknown"
    task_key = f"task:{job_id}"

    # Bind correlation ID for end-to-end traceability across worker logs
    _cid = task_params.get("correlation_id")
    if _cid:
        try:
            bind_correlation_id(str(_cid))
        except Exception:
            # Non-fatal; proceed without a bound ID
            pass

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
                # Update task status to FAILED using direct Redis update
                error_details = {
                    "status": TaskStatus.FAILED.value,
                    "error": str(error),
                    "stage": "preprocessing",
                    "error_code": error.error_code,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                redis_conn.hset(task_key, mapping=error_details)

                # Log the error
                logger.error(
                    f"Task {task_key} failed at stage 'preprocessing': {error}",
                    extra={
                        "task_key": task_key,
                        "stage": "preprocessing",
                        "error_code": error.error_code,
                        "error_type": type(error).__name__,
                    },
                )

            try:
                clear_correlation_id()
            except Exception:
                pass
            try:
                clear_correlation_id()
            except Exception:
                pass
            return {
                "status": "error",
                "error": {
                    "code": error.error_code,
                    "message": str(error),
                    "type": error.__class__.__name__,
                },
                "task_id": job_id,
            }

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
                # Update task status to FAILED using direct Redis update
                error_details = {
                    "status": TaskStatus.FAILED.value,
                    "error": str(error),
                    "stage": "preprocessing",
                    "error_code": error.error_code,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                redis_conn.hset(task_key, mapping=error_details)

                # Log the error
                logger.error(
                    f"Task {task_key} failed at stage 'preprocessing': {error}",
                    extra={
                        "task_key": task_key,
                        "stage": "preprocessing",
                        "error_code": error.error_code,
                        "error_type": type(error).__name__,
                    },
                )
            try:
                clear_correlation_id()
            except Exception:
                pass
            return {
                "status": "error",
                "error": {
                    "code": error.error_code,
                    "message": str(error),
                    "type": error.__class__.__name__,
                },
                "task_id": job_id,
            }
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
                # Update task status to FAILED using direct Redis update
                error_details = {
                    "status": TaskStatus.FAILED.value,
                    "error": str(error),
                    "stage": "preprocessing",
                    "error_code": error.error_code,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                redis_conn.hset(task_key, mapping=error_details)

                # Log the error
                logger.error(
                    f"Task {task_key} failed at stage 'preprocessing': {error}",
                    extra={
                        "task_key": task_key,
                        "stage": "preprocessing",
                        "error_code": error.error_code,
                        "error_type": type(error).__name__,
                    },
                )
            return {
                "status": "error",
                "error": {
                    "code": error.error_code,
                    "message": str(error),
                    "type": error.__class__.__name__,
                },
                "task_id": job_id,
            }
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
                # Update task status to FAILED using direct Redis update
                error_details = {
                    "status": TaskStatus.FAILED.value,
                    "error": str(error),
                    "stage": "preprocessing",
                    "error_code": error.error_code,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                redis_conn.hset(task_key, mapping=error_details)

                # Log the error
                logger.error(
                    f"Task {task_key} failed at stage 'preprocessing': {error}",
                    extra={
                        "task_key": task_key,
                        "stage": "preprocessing",
                        "error_code": error.error_code,
                        "error_type": type(error).__name__,
                    },
                )
            return {
                "status": "error",
                "error": {
                    "code": error.error_code,
                    "message": str(error),
                    "type": error.__class__.__name__,
                },
                "task_id": job_id,
            }

        # =====================================================================
        # Stage 2: PROCESSING_ASR - ASR transcription with sequential GPU usage
        # =====================================================================
        try:
            if redis_conn:
                _update_status(redis_conn, task_key, TaskStatus.PROCESSING_ASR)

            logger.info("Loading ASR model", task_id=job_id, backend=asr_backend)

            # Step 2a: Load ASR model based on backend using ASRFactory directly
            from src.processors.asr.factory import ASRFactory

            asr_model = ASRFactory.create(backend_type=asr_backend, **config)

            # Step 2b: Execute ASR transcription with actual duration
            # Phase 1: Now returns full ASResult object with segments/timestamps
            try:
                logger.info(
                    "Executing ASR transcription", audio_path=processing_audio_path
                )

                start_time = time.time()

                # Execute transcription - read file as bytes for backend
                with open(processing_audio_path, "rb") as f:
                    audio_bytes = f.read()

                result = asr_model.execute(audio_bytes)

                processing_time = time.time() - start_time
                asr_rtf = processing_time / audio_duration if audio_duration > 0 else 0

                # Collect ASR-specific metadata for versioning (NFR-1)
                asr_metadata: Dict[str, Any] = {}
                if hasattr(asr_model, "get_version_info"):
                    try:
                        backend_info = asr_model.get_version_info()
                        if isinstance(backend_info, dict):
                            asr_metadata.update(backend_info)
                    except Exception as exc:
                        logger.warning(
                            "Failed to collect ASR version info", error=str(exc)
                        )

                asr_metadata["language"] = result.language

                model_name = asr_metadata.get("model_name") or getattr(
                    result, "model_name", None
                )
                if not model_name:
                    model_name = asr_metadata.get("model_variant") or asr_metadata.get(
                        "backend"
                    )
                asr_metadata["model_name"] = model_name or "unknown"

                # Ensure checkpoint_hash is always a string, never None
                if not asr_metadata.get("checkpoint_hash"):
                    checkpoint_hash = getattr(result, "checkpoint_hash", None)
                    asr_metadata["checkpoint_hash"] = (
                        checkpoint_hash if checkpoint_hash else ""
                    )

                # Calculate character and word counts for visibility
                char_count = len(result.text)
                word_count = len(result.text.split()) if result.text else 0
                segment_count = len(result.segments) if result.segments else 0

                logger.info(
                    f"ASR transcription complete | {char_count:,} chars | {word_count:,} words | {segment_count} segment(s) | RTF: {asr_rtf:.3f}",
                    transcript_length=char_count,
                    word_count=word_count,
                    segment_count=segment_count,
                    rtf=asr_rtf,
                    confidence=result.confidence,
                    processing_time=processing_time,
                )

                # Phase 1: Return full ASRResult object directly from backend
                # Backends (ChunkFormer, Whisper) already return proper ASRResult with segments
                asr_result = result
            except FileNotFoundError as e:
                logger.error(
                    "Audio file not found",
                    audio_path=processing_audio_path,
                    error=str(e),
                )
                raise ASRProcessingError(
                    message=f"Audio file not found: {processing_audio_path}",
                    details={"audio_path": processing_audio_path},
                ) from e
            except (RuntimeError, OSError, ValueError, TypeError) as e:
                logger.error(
                    "ASR transcription failed",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
                raise ASRProcessingError(
                    message=f"ASR transcription failed: {str(e)}",
                    details={
                        "error_type": type(e).__name__,
                        "audio_path": processing_audio_path,
                    },
                ) from e

            except Exception as e:
                logger.error(
                    "ASR transcription failed",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
                raise ASRProcessingError(
                    message=f"ASR transcription failed: {str(e)}",
                    details={
                        "error_type": type(e).__name__,
                        "audio_path": processing_audio_path,
                    },
                ) from e

            # Extract text and confidence directly from ASRResult
            transcription = asr_result.text
            confidence = asr_result.confidence

            logger.info(
                "ASR transcription complete",
                task_id=job_id,
                transcript_length=len(transcription),
                rtf=asr_rtf,
                confidence=confidence,
            )

        finally:
            # Step 2c: CRITICAL - Always unload ASR model to free GPU memory
            if asr_model is not None:
                try:
                    logger.info("Unloading ASR model")
                    if asr_model is not None and hasattr(asr_model, "unload"):
                        asr_model.unload()

                    # Critical: Clear CUDA cache to free GPU memory
                    if (
                        TORCH_AVAILABLE
                        and torch is not None
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()
                        logger.info("ASR model unloaded and GPU cache cleared")
                    else:
                        logger.info("ASR model unloaded")
                except (AttributeError, RuntimeError, OSError) as e:
                    logger.warning("Error during ASR model unload", error=str(e))
                except Exception as e:
                    logger.warning(
                        "Unexpected error during ASR model unload", error=str(e)
                    )
                asr_model = None

        # =====================================================================
        # Stage 3: PROCESSING_LLM - LLM processing (enhancement + summary)
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

            # Step 3a: Load LLM model using LLMProcessor directly
            from src.processors.llm import LLMProcessor

            llm_model = LLMProcessor()
            # Trigger lazy loading
            llm_model._load_model()

            # Step 3b: Execute LLM processing (enhancement and/or summary)
            # Text enhancement (conditional based on ASR backend capabilities)
            # Skip enhancement if ASR backend provides native punctuation (e.g., Whisper with erax-wow-turbo)
            needs_enhancement = (
                "clean_transcript" in features
                and hasattr(llm_model, "needs_enhancement")
                and llm_model.needs_enhancement(asr_backend)
            )

            if needs_enhancement:
                try:
                    input_char_count = len(transcription)
                    input_word_count = (
                        len(transcription.split()) if transcription else 0
                    )

                    logger.info(
                        f"=== LLM INPUT: Text enhancement | {input_char_count:,} chars | {input_word_count:,} words ===",
                        char_count=input_char_count,
                        word_count=input_word_count,
                    )

                    enhanced_result = llm_model.enhance_text(transcription)
                    if enhanced_result.get("enhancement_applied", False):
                        clean_transcript = enhanced_result["enhanced_text"]
                        logger.info(
                            f"=== LLM OUTPUT: Text enhanced | {len(clean_transcript):,} chars | {len(clean_transcript.split()) if clean_transcript else 0:,} words | Edit rate: {enhanced_result.get('edit_rate', 0):.2%} ===",
                            char_count=len(clean_transcript),
                            word_count=len(clean_transcript.split())
                            if clean_transcript
                            else 0,
                            edit_rate=enhanced_result.get("edit_rate", 0),
                            edit_distance=enhanced_result.get("edit_distance", 0),
                        )
                    else:
                        clean_transcript = transcription
                        logger.info("Text enhancement skipped - not needed")
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.warning(
                        "Text enhancement failed, using raw transcript", error=str(e)
                    )
                    clean_transcript = transcription
                except Exception as e:
                    logger.warning(
                        "Unexpected error in text enhancement, using raw transcript",
                        error=str(e),
                    )
                    clean_transcript = transcription
            else:
                # Skip enhancement - ASR backend provides native punctuation
                clean_transcript = transcription
                logger.info(
                    "Text enhancement skipped - ASR backend provides native punctuation"
                )

            # Structured summary
            if "summary" in features:
                if not template_id:
                    error_msg = "template_id required when summary feature requested"
                    logger.error(error_msg)
                    raise LLMProcessingError(
                        message=error_msg,
                        details={"features": features, "template_id": template_id},
                    )

                try:
                    logger.info(
                        "Generating structured summary", template_id=template_id
                    )

                    # Log the ASR transcript being passed to LLM
                    transcript_char_count = len(clean_transcript)
                    transcript_word_count = (
                        len(clean_transcript.split()) if clean_transcript else 0
                    )

                    logger.info(
                        f"=== LLM INPUT: ASR transcript for summary | {transcript_char_count:,} chars | {transcript_word_count:,} words ===",
                        template_id=template_id,
                        char_count=transcript_char_count,
                        word_count=transcript_word_count,
                        preview=clean_transcript[:200]
                        + ("..." if len(clean_transcript) > 200 else ""),
                    )

                    summary_result = llm_model.generate_summary(
                        transcript=clean_transcript, template_id=template_id
                    )
                    if summary_result.get("summary"):
                        structured_summary = summary_result["summary"]
                        logger.info(
                            f"=== LLM OUTPUT: Summary generated | {len(str(structured_summary)):,} chars | {len(str(structured_summary).split()) if str(structured_summary) else 0:,} words ===",
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
                    logger.error(
                        "Unexpected summary generation exception", error=str(e)
                    )
                    raise LLMProcessingError(
                        message=f"Unexpected summary generation exception: {str(e)}",
                        details={
                            "template_id": template_id,
                            "error_type": type(e).__name__,
                        },
                    )
            else:
                structured_summary = None

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
                try:
                    logger.info("Unloading LLM model")
                    if llm_model is not None and hasattr(llm_model, "unload"):
                        llm_model.unload()

                    # Critical: Clear CUDA cache to free GPU memory
                    if (
                        TORCH_AVAILABLE
                        and torch is not None
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()
                        logger.info("LLM model unloaded and GPU cache cleared")
                    else:
                        logger.info("LLM model unloaded")
                except (AttributeError, RuntimeError, OSError) as e:
                    logger.warning("Error during LLM model unload", error=str(e))
                except Exception as e:
                    logger.warning(
                        "Unexpected error during LLM model unload", error=str(e)
                    )
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
        # Always include legacy 'transcript' key for compatibility
        result["results"]["transcript"] = transcription
        if "raw_transcript" in features:
            result["results"]["raw_transcript"] = transcription

        if "clean_transcript" in features:
            result["results"]["clean_transcript"] = clean_transcript or transcription

        if "summary" in features and structured_summary:
            result["results"]["summary"] = structured_summary

        # Phase 1: Include ASR segments for future use (timestamps, confidence per segment)
        # This preserves rich ASR data without breaking existing API consumers
        if asr_result.segments:
            result["results"]["segments"] = asr_result.segments
            logger.debug(
                "ASR segments preserved in results",
                task_id=job_id,
                segment_count=len(asr_result.segments),
            )

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
        try:
            clear_correlation_id()
        except Exception:
            pass
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

        # Update status to FAILED using direct Redis update
        if redis_conn:
            error_details = {
                "status": TaskStatus.FAILED.value,
                "error": str(e),
                "stage": "pipeline_execution",
                "error_code": e.error_code,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            redis_conn.hset(task_key, mapping=error_details)

            # Log the error
            logger.error(
                f"Task {task_key} failed at stage 'pipeline_execution': {e}",
                extra={
                    "task_key": task_key,
                    "stage": "pipeline_execution",
                    "error_code": e.error_code,
                    "error_type": type(e).__name__,
                },
            )

        # Return structured error response
        try:
            clear_correlation_id()
        except Exception:
            pass
        return {
            "status": "error",
            "error": {
                "code": e.error_code,
                "message": str(e),
                "type": e.__class__.__name__,
            },
            "task_id": job_id,
        }
    except (ConnectionError, TimeoutError, OSError) as e:
        # Network and I/O related errors - return structured response
        error_code = (
            "NETWORK_ERROR"
            if isinstance(e, (ConnectionError, TimeoutError))
            else "FILE_SYSTEM_ERROR"
        )
        error_type = (
            "NetworkError"
            if isinstance(e, (ConnectionError, TimeoutError))
            else "FileSystemError"
        )

        logger.error(
            f"Network/IO error during audio processing: {str(e)}",
            task_id=job_id,
            error_code=error_code,
            error=str(e),
        )

        try:
            clear_correlation_id()
        except Exception:
            pass
        return {
            "status": "error",
            "error": {
                "code": error_code,
                "message": str(e),
                "type": error_type,
            },
            "task_id": job_id,
        }

    except (MemoryError, RuntimeError) as e:
        # System resource errors - return structured response
        error_code = (
            "MODEL_MEMORY_ERROR" if isinstance(e, MemoryError) else "PROCESSING_ERROR"
        )
        error_type = (
            "ModelMemoryError" if isinstance(e, MemoryError) else "ProcessingError"
        )

        logger.error(
            f"System resource error during audio processing: {str(e)}",
            task_id=job_id,
            error_code=error_code,
            error=str(e),
        )

        try:
            clear_correlation_id()
        except Exception:
            pass
        return {
            "status": "error",
            "error": {
                "code": error_code,
                "message": str(e),
                "type": error_type,
            },
            "task_id": job_id,
        }

    except Exception as e:
        # Unknown/unexpected errors - return structured response
        logger.error(
            f"Unexpected error during audio processing: {str(e)}",
            task_id=job_id,
            error_code="PROCESSING_ERROR",
            error=str(e),
        )

        try:
            clear_correlation_id()
        except Exception:
            pass
        return {
            "status": "error",
            "error": {
                "code": "PROCESSING_ERROR",
                "message": str(e),
                "type": "ProcessingError",
            },
            "task_id": job_id,
        }

    # This should never be reached due to exception handling above
    try:
        clear_correlation_id()
    except Exception:
        pass
    return {
        "versions": None,
        "metrics": None,
        "results": None,
        "status": "error",
        "error": {"message": "Unexpected error path taken"},
    }

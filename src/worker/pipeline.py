"""
Sequential processing pipeline for the Modular Audio Intelligence Engine (MAIE).

This module contains the main processing logic that executes:
1. Audio preprocessing (normalization, validation)
2. ASR (Automatic Speech Recognition) processing
3. LLM (Large Language Model) processing (enhancement + summarization)
With proper state management, GPU memory management, and error handling.
"""

import time
import traceback
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

# Optional torch import (may not be available in test environment)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from redis import Redis
from rq import get_current_job
from loguru import logger

from src.api.schemas import TaskStatus
from src.config import settings


# =============================================================================
# Helper Functions
# =============================================================================


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
    update_data = {"status": status.value, "updated_at": datetime.now(timezone.utc).isoformat()}

    # Merge in additional details if provided
    if details:
        # Serialize complex objects (dicts, lists) to JSON strings
        for key, value in details.items():
            if isinstance(value, (dict, list)):
                update_data[key] = json.dumps(value)
            else:
                update_data[key] = value

    # Update in Redis
    client.hset(task_key, mapping=update_data)

    # Log status change
    logger.info(
        f"Task status updated",
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
    """Load ASR model into memory based on the specified backend."""
    print(f"Loading ASR model with backend: {asr_backend}")
    # In a real implementation, this would load the ASR model from the processor module
    from src.processors.asr.factory import ASRFactory

    asr_model = ASRFactory.create(backend_id=asr_backend, **config)
    return asr_model


def execute_asr_transcription(
    asr_model: Any, audio_path: str
) -> Tuple[str, float, float, Dict[str, Any]]:
    """Execute ASR transcription on the provided audio."""
    print(f"Executing ASR transcription on {audio_path}")

    start_time = time.time()
    # In a real implementation, this would call the ASR model to transcribe the audio
    result = asr_model.execute(audio_path)  # This is the actual method per TDD

    processing_time = time.time() - start_time
    audio_duration = 10.0  # This would be calculated from the actual audio file
    rtf = processing_time / audio_duration if audio_duration > 0 else 0

    transcription = result.transcript if hasattr(result, "transcript") else str(result)
    confidence = result.confidence_avg if hasattr(result, "confidence_avg") else 0.0

    # Collect ASR-specific metadata
    asr_metadata = {}
    if hasattr(result, "model_name"):
        asr_metadata["model_name"] = result.model_name
    if hasattr(result, "checkpoint_hash"):
        asr_metadata["checkpoint_hash"] = result.checkpoint_hash
    if hasattr(result, "duration_ms"):
        asr_metadata["duration_ms"] = result.duration_ms

    return transcription, rtf, confidence, asr_metadata


def unload_asr_model(asr_model: Any) -> None:
    """Unload ASR model from memory to free GPU resources."""
    print("Unloading ASR model...")
    # In a real implementation, this would properly clean up the ASR model resources
    asr_model.unload()  # Use the unload method per TDD
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_llm_model() -> Any:
    """Load LLM model into memory."""
    print("Loading LLM model...")
    # In a real implementation, this would load the LLM model from the processor module
    from src.processors.llm import LLMProcessor

    llm_model = LLMProcessor()
    return llm_model


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
    """
    print("Executing LLM processing...")

    results = {}

    # Text enhancement (conditional)
    if "clean_transcript" in features and llm_model.needs_enhancement(asr_backend):
        try:
            enhanced_result = llm_model.enhance_text(transcription)
            if enhanced_result["enhancement_applied"]:
                results["clean_transcript"] = enhanced_result["enhanced_text"]
                results["enhancement_metrics"] = {
                    "edit_rate_cleaning": enhanced_result.get("edit_rate", 0),
                    "edit_distance": enhanced_result.get("edit_distance", 0),
                }
            else:
                results["clean_transcript"] = transcription
        except Exception as e:
            print(f"Text enhancement failed: {e}")
            results["clean_transcript"] = transcription
    else:
        results["clean_transcript"] = transcription

    # Structured summarization
    if "summary" in features:
        if not template_id:
            print("Warning: template_id required when summary requested")
            results["summary"] = None
        else:
            try:
                summary_result = llm_model.generate_summary(
                    transcript=results["clean_transcript"], template_id=template_id
                )
                if summary_result.get("summary"):
                    results["summary"] = summary_result["summary"]
                    results["summary_metadata"] = {
                        "retry_count": summary_result.get("retry_count", 0),
                        "model_info": summary_result.get("model_info", {}),
                    }
                else:
                    print(
                        f"Summary generation failed: {summary_result.get('error', 'Unknown error')}"
                    )
                    results["summary"] = None
            except Exception as e:
                print(f"Summary generation failed: {e}")
                results["summary"] = None

    return results.get("clean_transcript"), results.get("summary")


def unload_llm_model(llm_model: Any) -> None:
    """Unload LLM model from memory to free GPU resources."""
    print("Unloading LLM model...")
    # In a real implementation, this would properly clean up the LLM model resources
    if hasattr(llm_model, "unload"):
        llm_model.unload()
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_version_metadata(
    asr_result_metadata: Dict[str, Any], llm_model: Any = None
) -> Dict[str, Any]:
    """
    Collect version metadata for the models and processing pipeline.
    
    Per TDD NFR-1, collects:
    - ASR backend metadata (model_name, checkpoint_hash, config)
    - LLM version info (if model loaded)
    - Pipeline version from settings
    
    Args:
        asr_result_metadata: Metadata from ASR backend execution
        llm_model: LLM processor instance (optional)
    
    Returns:
        Complete version metadata dict
    """
    version_metadata = {
        "asr": asr_result_metadata,  # Include ASR-specific metadata
        "maie_worker": "1.0.0",
        "processing_pipeline": settings.pipeline_version,  # Use settings per TDD
    }

    # Add LLM version info if model is available
    if llm_model and hasattr(llm_model, "get_version_info"):
        try:
            version_metadata["llm"] = llm_model.get_version_info()
        except Exception as e:
            logger.error(
                "Failed to get LLM version info",
                error=str(e),
            )
            version_metadata["llm"] = {"model_name": "unavailable", "error": str(e)}
    else:
        version_metadata["llm"] = {
            "model_name": "not_loaded",
            "reason": "Model not available",
        }

    return version_metadata


def calculate_metrics(
    transcription: str,
    clean_transcript: Optional[str],
    start_time: float,
    audio_duration: float,
    asr_rtf: float,
) -> Dict[str, Any]:
    """Calculate runtime metrics for the processing."""
    total_processing_time = time.time() - start_time
    total_rtf = total_processing_time / audio_duration if audio_duration > 0 else 0

    metrics = {
        "total_processing_time": total_processing_time,
        "total_rtf": total_rtf,
        "asr_rtf": asr_rtf,
        "transcription_length": len(transcription) if transcription else 0,
        "audio_duration": audio_duration,
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

    try:
        # =====================================================================
        # Stage 1: PREPROCESSING - Audio validation and normalization
        # =====================================================================
        logger.info(
            f"Starting audio processing for task {job_id}",
            task_id=job_id,
            audio_path=audio_path,
        )

        if redis_conn:
            _update_status(redis_conn, task_key, TaskStatus.PREPROCESSING)

        # Validate audio_path before preprocessing
        if not audio_path or not isinstance(audio_path, str):
            error_msg = f"Invalid audio_path provided: {audio_path}"
            logger.error(error_msg, task_id=job_id)
            
            if redis_conn:
                _update_status(
                    redis_conn,
                    task_key,
                    TaskStatus.FAILED,
                    details={"error": error_msg},
                )
            
            raise ValueError(error_msg)

        # Audio preprocessing - validate and normalize to 16kHz mono WAV
        try:
            from src.processors.audio import AudioPreprocessor
            
            preprocessor = AudioPreprocessor()
            metadata = preprocessor.preprocess(Path(audio_path))
            
            # Update audio_path if normalization was performed
            if "normalized_path" in metadata:
                audio_path = str(metadata["normalized_path"])
                logger.info(
                    "Audio normalized",
                    task_id=job_id,
                    original_format=metadata.get("format"),
                    duration=metadata.get("duration"),
                )
            
            # Store audio duration for metrics calculation
            audio_duration = metadata.get("duration", audio_duration)
            
            logger.info(
                "Audio preprocessing complete",
                task_id=job_id,
                duration=audio_duration,
                sample_rate=metadata.get("sample_rate"),
                channels=metadata.get("channels"),
            )
        except Exception as e:
            logger.error(
                "Audio preprocessing failed",
                task_id=job_id,
                error=str(e),
            )
            if redis_conn:
                handle_processing_error(
                    redis_conn, task_key, e,
                    stage="preprocessing",
                    error_code="AUDIO_PREPROCESSING_ERROR"
                )
            raise

        # =====================================================================
        # Stage 2: PROCESSING_ASR - ASR transcription with sequential GPU usage
        # =====================================================================
        try:
            if redis_conn:
                _update_status(redis_conn, task_key, TaskStatus.PROCESSING_ASR)

            logger.info("Loading ASR model", task_id=job_id, backend=asr_backend)

            # Step 2a: Load ASR model based on backend
            asr_model = load_asr_model(asr_backend, **config)

            # Step 2b: Execute ASR transcription
            transcription, asr_rtf, confidence, asr_metadata = execute_asr_transcription(
                asr_model, audio_path
            )

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
                unload_asr_model(asr_model)
                asr_model = None

        # =====================================================================
        # Stage 3: PROCESSING_LLM - LLM processing (enhancement + summarization)
        # =====================================================================
        try:
            if redis_conn:
                _update_status(
                    redis_conn,
                    task_key,
                    TaskStatus.PROCESSING_LLM,
                    {"transcription_length": len(transcription) if transcription else 0},
                )

            logger.info("Loading LLM model", task_id=job_id)

            # Step 3a: Load LLM model
            llm_model = load_llm_model()

            # Step 3b: Execute LLM processing (enhancement and/or summarization)
            clean_transcript, structured_summary = execute_llm_processing(
                llm_model, transcription, features, template_id, asr_backend
            )
            
            # Step 3c: Collect LLM version info BEFORE unloading
            llm_version_info = None
            if llm_model and hasattr(llm_model, "get_version_info"):
                try:
                    llm_version_info = llm_model.get_version_info()
                except Exception as e:
                    logger.error("Failed to get LLM version info", error=str(e))

            logger.info(
                "LLM processing complete",
                task_id=job_id,
                enhanced=clean_transcript != transcription if clean_transcript else False,
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
        # Note: audio_duration was already extracted from preprocessing metadata
        metrics = calculate_metrics(
            transcription, clean_transcript, start_time, audio_duration, asr_rtf
        )

        # Collect version metadata (NFR-1 - full reproducibility)
        # Build version metadata manually since we already collected LLM info before unload
        version_metadata = {
            "asr": asr_metadata,
            "maie_worker": "1.0.0",
            "processing_pipeline": settings.pipeline_version,
        }
        
        # Add LLM version info if it was collected
        if llm_version_info:
            version_metadata["llm"] = llm_version_info
        else:
            version_metadata["llm"] = {
                "model_name": "not_loaded",
                "reason": "Model not available or not used",
            }

        # Prepare result - only include requested features
        result = {"versions": version_metadata, "metrics": metrics, "results": {}}

        # Include requested features in results
        if "raw_transcript" in features or "clean_transcript" in features:
            result["results"]["transcript"] = clean_transcript or transcription

        if "summary" in features and structured_summary:
            result["results"]["summary"] = structured_summary

        # Update status to COMPLETE and store final results
        if redis_conn:
            _update_status(
                redis_conn,
                task_key,
                TaskStatus.COMPLETE,
                {
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

    except Exception as e:
        error_msg = f"Error in audio processing: {str(e)}\n{traceback.format_exc()}"

        logger.error(
            "Audio processing failed",
            task_id=job_id,
            error=str(e),
            traceback=traceback.format_exc(),
        )

        # Update status to FAILED using error handler
        if redis_conn:
            handle_processing_error(
                redis_conn, task_key, e, stage="pipeline_execution", error_code="PROCESSING_ERROR"
            )

        # Return error result
        return {
            "versions": None,
            "metrics": None,
            "results": None,
            "status": "error",
            "error": error_msg,
        }

    finally:
        if redis_conn:
            redis_conn.close()


def handle_processing_error(
    client: Redis,
    task_key: str,
    error: Exception,
    stage: str,
    error_code: Optional[str] = None,
) -> None:
    """
    Handle processing errors by updating task status to FAILED.

    Args:
        client: Redis client for updating task status
        task_key: Redis key for the task
        error: The exception that occurred
        stage: Processing stage where error occurred (preprocessing, asr, llm, etc.)
        error_code: Optional error code (e.g., AUDIO_DECODE_ERROR, MODEL_LOAD_ERROR)
    """
    import logging
    from datetime import datetime, timezone

    logger = logging.getLogger(__name__)

    try:
        # Prepare error details
        error_message = str(error)
        error_details = {
            "status": TaskStatus.FAILED.value,
            "error": error_message,
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
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

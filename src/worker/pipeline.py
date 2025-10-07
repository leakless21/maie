"""
Sequential processing pipeline for the Modular Audio Intelligence Engine (MAIE).

This module contains the main processing logic that executes:
1. ASR (Automatic Speech Recognition) processing
2. LLM (Large Language Model) processing
With proper state management, GPU memory management, and error handling.
"""

import asyncio
import time
import traceback
import torch
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from redis import Redis
from rq import get_current_job


class TaskStatus(Enum):
    """Enumeration of possible task statuses."""
    PENDING = "PENDING"
    PROCESSING_ASR = "PROCESSING_ASR"
    PROCESSING_LLM = "PROCESSING_LLM"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class ProcessingResult:
    """Data class to hold the result of audio processing."""
    transcription: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None  # Changed to Dict[str, Any] to match structured summary
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


def execute_asr_transcription(asr_model: Any, audio_path: str) -> Tuple[str, float, float, Dict[str, Any]]:
    """Execute ASR transcription on the provided audio."""
    print(f"Executing ASR transcription on {audio_path}")
    
    start_time = time.time()
    # In a real implementation, this would call the ASR model to transcribe the audio
    result = asr_model.execute(audio_path)  # This is the actual method per TDD
    
    processing_time = time.time() - start_time
    audio_duration = 10.0  # This would be calculated from the actual audio file
    rtf = processing_time / audio_duration if audio_duration > 0 else 0
    
    transcription = result.transcript if hasattr(result, 'transcript') else str(result)
    confidence = result.confidence_avg if hasattr(result, 'confidence_avg') else 0.0
    
    # Collect ASR-specific metadata
    asr_metadata = {}
    if hasattr(result, 'model_name'):
        asr_metadata['model_name'] = result.model_name
    if hasattr(result, 'checkpoint_hash'):
        asr_metadata['checkpoint_hash'] = result.checkpoint_hash
    if hasattr(result, 'duration_ms'):
        asr_metadata['duration_ms'] = result.duration_ms
    
    return transcription, rtf, confidence, asr_metadata


def unload_asr_model(asr_model: Any) -> None:
    """Unload ASR model from memory to free GPU resources."""
    print("Unloading ASR model...")
    # In a real implementation, this would properly clean up the ASR model resources
    asr_model.unload()  # Use the unload method per TDD
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_llm_model() -> Any:
    """Load LLM model into memory."""
    print("Loading LLM model...")
    # In a real implementation, this would load the LLM model from the processor module
    from src.processors.llm import LLMProcessor
    
    llm_model = LLMProcessor()
    return llm_model


def execute_llm_processing(llm_model: Any, transcription: str, features: list, template_id: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Execute LLM processing (e.g., summarization) on the transcription."""
    print("Executing LLM processing...")
    
    # Determine what features to process based on the request
    clean_transcript = None
    structured_summary = None
    
    # Check if text enhancement is needed
    # According to TDD, skip enhancement when Whisper with erax-wow-turbo is used (has native punctuation)
    needs_enhancement = False  # Placeholder - would be determined based on ASR backend used
    
    if "clean_transcript" in features and needs_enhancement:
        clean_transcript = llm_model.enhance_text(transcription)
    
    if "summary" in features and template_id:
        structured_summary = llm_model.generate_summary_with_template(transcription, template_id)
    
    return clean_transcript, structured_summary


def unload_llm_model(llm_model: Any) -> None:
    """Unload LLM model from memory to free GPU resources."""
    print("Unloading LLM model...")
    # In a real implementation, this would properly clean up the LLM model resources
    if hasattr(llm_model, 'unload'):
        llm_model.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_version_metadata(asr_result_metadata: Dict[str, Any]) -> Dict[str, str]:
    """Collect version metadata for the models and processing pipeline."""
    # This would collect actual version information from the models
    # For now, returning placeholder values with ASR metadata
    return {
        "asr": asr_result_metadata,  # Include ASR-specific metadata
        "llm": {
            "model_name": "cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
            "quantization": "awq-4bit",
            "model_type": "Qwen3-4B-Instruct"
        },
        "maie_worker": "1.0.0",
        "processing_pipeline": "1.0.0"
    }


def calculate_metrics(transcription: str, clean_transcript: Optional[str], start_time: float,
                     audio_duration: float, asr_rtf: float) -> Dict[str, Any]:
    """Calculate runtime metrics for the processing."""
    total_processing_time = time.time() - start_time
    total_rtf = total_processing_time / audio_duration if audio_duration > 0 else 0
    
    metrics = {
        "total_processing_time": total_processing_time,
        "total_rtf": total_rtf,
        "asr_rtf": asr_rtf,
        "transcription_length": len(transcription) if transcription else 0,
        "audio_duration": audio_duration
    }
    
    # Add enhancement metrics if text enhancement was performed
    if clean_transcript and clean_transcript != transcription:
        # Calculate edit distance for edit_rate_cleaning metric
        # This is a simplified version - in practice, you'd use a proper edit distance algorithm
        original_length = len(transcription)
        enhanced_length = len(clean_transcript)
        max_len = max(original_length, enhanced_length)
        if max_len > 0:
            edit_rate = abs(original_length - enhanced_length) / max_len
            metrics["edit_rate_cleaning"] = edit_rate
    
    return metrics


def update_task_status(job_id: str, status: TaskStatus, redis_conn: Redis, details: Optional[Dict[str, Any]] = None) -> None:
    """Update the task status in Redis."""
    if redis_conn:
        status_data = {
            "status": status.value,
            "updated_at": time.time()
        }
        if details:
            status_data.update(details)
        
        redis_conn.hset(f"task:{job_id}", mapping=status_data)  # Fixed key format per TDD


def process_audio_task(task_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main pipeline function that processes audio through ASR and LLM sequentially.
    
    Args:
        task_params: Dictionary containing task parameters including:
            - audio_path: Path to the audio file to process
            - asr_backend: ASR backend to use (default: "whisper")
            - features: List of features to process (default: ["clean_transcript", "summary"])
            - template_id: Template ID for summarization (required if "summary" in features)
            - config: Additional configuration for the ASR backend
            
    Returns:
        Dictionary containing the processing result
    """
    job = get_current_job()
    job_id = job.id if job else "unknown"
    
    # Extract parameters
    audio_path = task_params.get("audio_path")
    asr_backend = task_params.get("asr_backend", "whisper")
    features = task_params.get("features", ["clean_transcript", "summary"])
    template_id = task_params.get("template_id")
    config = task_params.get("config", {})
    
    # Get Redis connection
    redis_host = task_params.get("redis_host", "localhost")
    redis_port = task_params.get("redis_port", 6379)
    redis_db = task_params.get("redis_db", 1)  # Use DB 1 for results per TDD
    redis_conn = Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=False
    ) if job else None
    
    start_time = time.time()
    
    try:
        # Update status to PROCESSING_ASR
        if redis_conn:
            update_task_status(job_id, TaskStatus.PROCESSING_ASR, redis_conn)
        
        print(f"Starting audio processing for: {audio_path}")
        
        # Step 1: Load ASR model based on backend
        asr_model = load_asr_model(asr_backend, **config)
        
        # Step 2: Execute ASR transcription
        transcription, asr_rtf, confidence, asr_metadata = execute_asr_transcription(asr_model, audio_path)
        
        # Step 3: Unload ASR model to free GPU memory
        unload_asr_model(asr_model)
        
        # Update status to PROCESSING_LLM
        if redis_conn:
            update_task_status(
                job_id,
                TaskStatus.PROCESSING_LLM,
                redis_conn,
                {"transcription_length": len(transcription) if transcription else 0}
            )
        
        # Step 4: Load LLM model
        llm_model = load_llm_model()
        
        # Step 5: Execute LLM processing
        clean_transcript, structured_summary = execute_llm_processing(
            llm_model, transcription, features, template_id
        )
        
        # Step 6: Unload LLM model to free GPU memory
        unload_llm_model(llm_model)
        
        # Calculate audio duration from the file (placeholder implementation)
        audio_duration = 10.0  # Would be calculated from the actual audio file
        
        # Calculate metrics
        metrics = calculate_metrics(
            transcription, clean_transcript, start_time, audio_duration, asr_rtf
        )
        
        # Collect version metadata
        version_metadata = get_version_metadata(asr_metadata)
        
        # Prepare result - only include requested features
        result = {
            "versions": version_metadata,
            "metrics": metrics,
            "results": {}
        }
        
        if "raw_transcript" in features or "clean_transcript" in features:
            result["results"]["transcript"] = clean_transcript or transcription
        
        if "summary" in features and structured_summary:
            result["results"]["summary"] = structured_summary
        
        # Update status to COMPLETE
        if redis_conn:
            update_task_status(
                job_id,
                TaskStatus.COMPLETE,
                redis_conn,
                {
                    "versions": version_metadata,
                    "metrics": metrics,
                    "results": result["results"]
                }
            )
        
        print(f"Audio processing completed for: {audio_path}")
        
        # Return result as dictionary
        return result
        
    except Exception as e:
        error_msg = f"Error in audio processing: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # Update status to FAILED
        if redis_conn:
            update_task_status(
                job_id,
                TaskStatus.FAILED,
                redis_conn,
                {"error_message": error_msg}
            )
        
        # Return error result
        return {
            "versions": None,
            "metrics": None,
            "results": None,
            "status": "error",
            "error": error_msg
        }
    
    finally:
        if redis_conn:
            redis_conn.close()


# Placeholder function for error handling and retry logic
def handle_processing_error(error: Exception, task_data: Dict[str, Any], attempt: int) -> bool:
    """
    Handle processing errors and determine if task should be retried.
    
    Args:
        error: The exception that occurred
        task_data: Data about the current task
        attempt: Current attempt number
        
    Returns:
        True if the task should be retried, False otherwise
    """
    # Define max retry attempts
    max_retries = 3
    
    # Some errors might be non-retryable
    non_retryable_errors = [
        FileNotFoundError,
        PermissionError
    ]
    
    if any(isinstance(error, err_type) for err_type in non_retryable_errors):
        return False
    
    # For other errors, retry up to max_retries times
    return attempt < max_retries
"""
Simplified error tracking utilities for MAIE project.

This module provides clean, straightforward error tracking that focuses on
making error pinpointing and debugging easier without adding complexity.
"""

import traceback
from typing import Any, Dict, Optional
from loguru import logger
from contextvars import ContextVar

# Context variable to track the current processing stage
current_stage: ContextVar[Optional[str]] = ContextVar("current_stage", default=None)

def set_processing_stage(stage: str) -> None:
    """Set the current processing stage for error context."""
    current_stage.set(stage)

def get_processing_stage() -> Optional[str]:
    """Get the current processing stage."""
    return current_stage.get()

def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    stage: Optional[str] = None,
    task_id: Optional[str] = None
) -> None:
    """
    Log an error with comprehensive context for easy debugging.
    
    Args:
        error: The exception that occurred
        context: Additional context information about the error
        stage: Current processing stage (e.g., 'preprocessing', 'asr', 'llm')
        task_id: Task identifier for correlation
    """
    # Set stage if provided
    if stage:
        set_processing_stage(stage)
    
    # Build error context
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stage": stage or get_processing_stage() or "unknown",
        "traceback": traceback.format_exc()
    }
    
    if context:
        error_context.update(context)
    
    if task_id:
        error_context["task_id"] = task_id
    
    # Log the error with all context
    logger.error("Error occurred | Context: {context}", context=error_context)

def log_stage_entry(stage_name: str, **kwargs) -> None:
    """Log entry to a processing stage with context."""
    set_processing_stage(stage_name)
    context = {"stage": stage_name}
    context.update(kwargs)
    logger.info("Entering stage: {stage} | Context: {context}", stage=stage_name, context=kwargs)

def log_stage_exit(stage_name: str, success: bool = True, **kwargs) -> None:
    """Log exit from a processing stage."""
    context = {"stage": stage_name, "success": success}
    context.update(kwargs)
    if success:
        logger.info("Exiting stage: {stage} | Success: {success} | Context: {context}", 
                   stage=stage_name, success=success, context=kwargs)
    else:
        logger.warning("Exiting stage: {stage} | Success: {success} | Context: {context}", 
                      stage=stage_name, success=success, context=kwargs)

def log_error_with_context(
    message: str,
    error: Optional[Exception] = None,
    **context
) -> None:
    """
    Log an error message with structured context.
    
    Args:
        message: Error message to log
        error: Optional exception object
        **context: Additional context as keyword arguments
    """
    if error:
        log_error(error, context=context)
    else:
        logger.error("Error: {message} | Context: {context}", message=message, context=context)

# Convenience functions for common error scenarios
def log_asr_error(error: Exception, audio_file: str = "unknown", **context) -> None:
    """Log ASR-specific errors."""
    log_error(error, context={"audio_file": audio_file, **context}, stage="asr")

def log_llm_error(error: Exception, model: str = "unknown", **context) -> None:
    """Log LLM-specific errors."""
    log_error(error, context={"model": model, **context}, stage="llm")

def log_preprocessing_error(error: Exception, input_file: str = "unknown", **context) -> None:
    """Log preprocessing-specific errors."""
    log_error(error, context={"input_file": input_file, **context}, stage="preprocessing")

def log_pipeline_error(error: Exception, task_id: str = "unknown", **context) -> None:
    """Log pipeline-specific errors."""
    log_error(error, context={"task_id": task_id, **context}, stage="pipeline")
"""
Worker module initialization for the Modular Audio Intelligence Engine (MAIE).

This module exports the main worker functions and pipeline processing functions
for easy import and use throughout the application.
"""

# Export main worker functions
from .main import start_worker, setup_redis_connection, verify_models

# Export pipeline processing functions
from .pipeline import (
    process_audio_task,
    TaskStatus,
    ProcessingResult,
    handle_processing_error,
)

__all__ = [
    # Main worker functions
    "start_worker",
    "setup_redis_connection",
    "verify_models",
    # Pipeline processing functions
    "process_audio_task",
    "TaskStatus",
    "ProcessingResult",
    "handle_processing_error",
]

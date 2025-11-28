"""
Worker module initialization for the Modular Audio Intelligence Engine (MAIE).

This module exports the main worker functions and pipeline processing functions
for easy import and use throughout the application.
"""

# Export main worker functions
from .main import setup_redis_connection, start_worker

# Export pipeline processing functions
from .pipeline import (
    TaskStatus,
    process_audio_task,
)

__all__ = [
    # Main worker functions
    "start_worker",
    "setup_redis_connection",

    # Pipeline processing functions
    "process_audio_task",
    "TaskStatus",
]

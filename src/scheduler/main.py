"""
RQ Scheduler setup for MAIE application cleanup tasks.

This module provides automated scheduling for cleanup tasks using RQ Scheduler,
eliminating the need for external cron job configuration.
"""

from __future__ import annotations

import os
import signal
import sys
import warnings
from typing import NoReturn
from datetime import datetime, timezone

# Suppress torchaudio deprecation warnings
warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.utils.info has been deprecated",
    category=UserWarning,
    module="pyannote.audio.core.io"
)
warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.common.AudioMetaData has been deprecated",
    category=UserWarning,
    module="torchaudio._backend.soundfile_backend"
)
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
    category=UserWarning,
    module="torchaudio._backend.utils"
)
warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.list_audio_backends has been deprecated",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="TensorFloat-32.*has been disabled",
    category=UserWarning,
    module="pyannote.audio.utils.reproducibility"
)
warnings.filterwarnings(
    "ignore",
    message="std.*degrees of freedom is <= 0",
    category=UserWarning,
    module="pyannote.audio.models.blocks.pooling"
)

# Make CLI runnable from source tree with src layout
if not __package__:
    # Prepend the project root to sys.path so absolute imports work
    # __file__ is /home/cetech/maie/src/scheduler/main.py
    # We need to go up two levels to get to /home/cetech/maie
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

import redis
from rq import Queue
from rq_scheduler.scheduler import Scheduler

from src.config.logging import get_module_logger
from src.config.loader import settings
from src.cleanup.tasks import (
    cleanup_audio_files,
    cleanup_logs,
    cleanup_cache,
    disk_monitor,
)


logger = get_module_logger(__name__)


def get_redis_connection():
    """Get Redis connection using application settings."""
    try:
        connection = redis.from_url(settings.redis.url, db=settings.redis.results_db)
        connection.ping()  # Test connection
        return connection
    except redis.ConnectionError as e:
        logger.error("Failed to connect to Redis", extra={"error": str(e)})
        raise


def create_scheduler() -> Scheduler:
    """
    Create and configure RQ Scheduler instance.

    :return: Configured scheduler instance
    """
    logger.info("Creating RQ Scheduler instance")

    # Use queue DB (0 by default) for scheduler
    queue_db = 0 if settings.redis.results_db == 1 else settings.redis.results_db
    redis_conn = redis.from_url(settings.redis.url, db=queue_db)

    # Create scheduler with configuration
    # Note: RQ Scheduler doesn't have a logging_level parameter
    scheduler = Scheduler(
        connection=redis_conn,
        queue_class=Queue,
        interval=60,  # Check for scheduled jobs every 60 seconds
    )

    logger.info("RQ Scheduler created", extra={"interval": 60, "queue_db": queue_db})

    return scheduler


def schedule_cleanup_jobs(scheduler: Scheduler) -> None:
    """
    Schedule all cleanup jobs with appropriate intervals.

    :param scheduler: RQ Scheduler instance
    """
    logger.info("Scheduling cleanup jobs")

    # Cleanup intervals - configurable via settings
    from src.config.loader import settings

    audio_cleanup_interval = settings.cleanup.audio_cleanup_interval
    log_cleanup_interval = settings.cleanup.log_cleanup_interval
    cache_cleanup_interval = settings.cleanup.cache_cleanup_interval
    disk_monitor_interval = settings.cleanup.disk_monitor_interval

    # Schedule audio cleanup - hourly
    try:
        scheduler.schedule(
            scheduled_time=datetime.now(timezone.utc),  # schedule starting now (timezone-aware)
            func=cleanup_audio_files,
            args=[False],  # Not dry run
            interval=audio_cleanup_interval,
            id="cleanup_audio_files",
            queue_name="cleanup",
        )
        logger.info(
            "Scheduled audio cleanup",
            extra={"interval_seconds": audio_cleanup_interval, "queue": "cleanup"},
        )
    except Exception as e:
        logger.error("Failed to schedule audio cleanup", extra={"error": str(e)})

    # Schedule log cleanup - daily
    try:
        scheduler.schedule(
            scheduled_time=datetime.now(timezone.utc),  # schedule starting now (timezone-aware)
            func=cleanup_logs,
            args=[False],  # Not dry run
            interval=log_cleanup_interval,
            id="cleanup_logs",
            queue_name="cleanup",
        )
        logger.info(
            "Scheduled log cleanup",
            extra={"interval_seconds": log_cleanup_interval, "queue": "cleanup"},
        )
    except Exception as e:
        logger.error("Failed to schedule log cleanup", extra={"error": str(e)})

    # Schedule cache cleanup - every 30 minutes
    try:
        scheduler.schedule(
            scheduled_time=datetime.now(timezone.utc),  # schedule starting now (timezone-aware)
            func=cleanup_cache,
            args=[False],  # Not dry run
            interval=cache_cleanup_interval,
            id="cleanup_cache",
            queue_name="cleanup",
        )
        logger.info(
            "Scheduled cache cleanup",
            extra={"interval_seconds": cache_cleanup_interval, "queue": "cleanup"},
        )
    except Exception as e:
        logger.error("Failed to schedule cache cleanup", extra={"error": str(e)})

    # Schedule disk monitoring - every 5 minutes
    try:
        scheduler.schedule(
            scheduled_time=datetime.now(timezone.utc),  # schedule starting now (timezone-aware)
            func=disk_monitor,
            args=[],
            interval=disk_monitor_interval,
            id="disk_monitor",
            queue_name="cleanup",
        )
        logger.info(
            "Scheduled disk monitor",
            extra={"interval_seconds": disk_monitor_interval, "queue": "cleanup"},
        )
    except Exception as e:
        logger.error("Failed to schedule disk monitor", extra={"error": str(e)})


def run_scheduler() -> None:
    """
    Run the scheduler main loop.

    This function blocks until interrupted by a signal.
    """
    logger.info("Starting RQ Scheduler main loop")

    try:
        scheduler = create_scheduler()
        schedule_cleanup_jobs(scheduler)

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal", extra={"signal": signum})
            # rq_scheduler doesn't expose a stop() API; register death and remove lock
            try:
                scheduler.register_death()
            except Exception:
                logger.debug("Error registering scheduler death during shutdown")
            try:
                scheduler.remove_lock()
            except Exception:
                logger.debug("Error removing scheduler lock during shutdown")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the scheduler (blocking)
        # rq-scheduler exposes `run()` (not `start()`), call run() to begin the loop
        scheduler.run()

    except Exception as e:
        logger.exception("Scheduler failed to start")
        sys.exit(1)


def get_scheduler_status(scheduler: Scheduler) -> dict:
    """
    Get current scheduler status and job information.

    :param scheduler: Scheduler instance
    :return: Status information
    """
    try:
        # Get scheduled jobs
        scheduled_jobs = []
        for job_id, job_data in scheduler.connection.hgetall(
            "rq:scheduler:scheduled_jobs"
        ).items():
            if isinstance(job_id, bytes):
                job_id = job_id.decode()
            if isinstance(job_data, bytes):
                job_data = job_data.decode()
            scheduled_jobs.append(
                {
                    "id": job_id,
                    "data": job_data[:100] + "..." if len(job_data) > 100 else job_data,
                }
            )

        # Determine status from scheduler's redis presence
        try:
            is_alive = (
                scheduler.connection.exists(scheduler.key)
                and not scheduler.connection.hexists(scheduler.key, "death")
            )
        except Exception:
            is_alive = False

        return {
            "status": "running" if is_alive else "stopped",
            "scheduled_jobs_count": len(scheduled_jobs),
            "scheduled_jobs": scheduled_jobs[:5],  # First 5 for brevity
            "interval": getattr(scheduler, "_interval", None),
        }

    except Exception as e:
        logger.warning("Failed to get scheduler status", extra={"error": str(e)})
        return {"error": str(e)}


def cleanup_scheduler(scheduler: Scheduler) -> None:
    """
    Clean up scheduler resources.

    :param scheduler: Scheduler instance to clean up
    """
    try:
        try:
            scheduler.register_death()
        except Exception:
            logger.debug("Error registering death during cleanup")
        try:
            scheduler.remove_lock()
        except Exception:
            logger.debug("Error removing lock during cleanup")

        logger.info("Scheduler stopped gracefully")
    except Exception as e:
        logger.warning("Error stopping scheduler", extra={"error": str(e)})


if __name__ == "__main__":
    # Run as standalone process
    run_scheduler()

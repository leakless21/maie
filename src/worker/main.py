"""
RQ Worker entry point for the Modular Audio Intelligence Engine (MAIE).

This module sets up and configures the RQ worker with Redis connection,
model verification on startup, and proper worker configuration.
"""

import sys
from pathlib import Path

import warnings

# Suppress torchaudio deprecation warnings
warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.utils.info has been deprecated",
    category=UserWarning,
    module="pyannote.audio.core.io",
)
warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.common.AudioMetaData has been deprecated",
    category=UserWarning,
    module="torchaudio._backend.soundfile_backend",
)
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
    category=UserWarning,
    module="torchaudio._backend.utils",
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
    module="pyannote.audio.utils.reproducibility",
)
warnings.filterwarnings(
    "ignore",
    message="std.*degrees of freedom is <= 0",
    category=UserWarning,
    module="pyannote.audio.models.blocks.pooling",
)

from redis import Redis
from rq import SimpleWorker

# Opt-in: import logging configuration helpers defensively.
from src.config import configure_logging, get_logger, settings
from src.config.logging import get_module_logger


def setup_redis_connection() -> Redis:
    """Initialize Redis connection with configuration from centralized settings."""
    # Create Redis connection from URL
    redis_conn = Redis.from_url(settings.redis.url, decode_responses=False)

    # Verify Redis connection
    try:
        redis_conn.ping()
        get_logger().info("Successfully connected to Redis")
    except Exception:
        get_logger().exception("Failed to connect to Redis")
        sys.exit(1)

    return redis_conn





def start_worker() -> None:
    """Start the RQ worker with proper configuration."""


    # Set up Redis connection
    redis_conn = setup_redis_connection()

    # Define the queues this worker will listen to
    listen = ["default", "audio_processing"]

    # Create RQ SimpleWorker to avoid forking issues with CUDA
    # SimpleWorker executes jobs in the main process without os.fork()
    # This preserves CUDA context and avoids "CUDA initialization error"
    # Reference: https://github.com/rq/rq/blob/master/docs/docs/workers.md
    worker = SimpleWorker(
        listen,
        connection=redis_conn,
        name=settings.worker.worker_name,
        exception_handlers=[],
    )

    get_logger().info(
        "Starting worker {} (SimpleWorker - no forking, max_jobs=1 for GPU reset)",
        worker.name,
    )
    worker.work(max_jobs=1)


if __name__ == "__main__":
    # CRITICAL: Disable vLLM V1 multiprocessing to avoid hanging in RQ worker context
    # vLLM's V1 engine spawns a separate EngineCore process, but this causes issues
    # when vLLM is loaded from within an RQ worker process (not __main__).
    # See: https://docs.vllm.ai/en/latest/design/multiprocessing.html
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Enforce spawn start method for CUDA compatibility with vLLM
    import multiprocessing as mp

    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass

    # Apply opt-in Loguru configuration at worker startup.
    # Always configure Loguru at startup.
    logger = configure_logging()
    logger = logger if logger is not None else get_logger()
    logger = get_module_logger(__name__)
    logger.info("Loguru configuration active (phase1) - worker")
    
    # DEBUG: Log current logging configuration state
    import logging
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    logger.info("Root logger handlers count: {}", len(handlers))
    for i, handler in enumerate(handlers):
        logger.info("Handler {}: {} (level: {})", i, type(handler).__name__, handler.level)
    
    try:
        from src.config import settings as _settings_for_log

        logger.info(
            "verbose_components={} debug={}",
            _settings_for_log.verbose_components,
            _settings_for_log.debug,
        )
    except Exception:
        pass

    start_worker()

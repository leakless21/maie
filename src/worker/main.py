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

from redis import Redis
from rq import Worker

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


def verify_models() -> bool:
    """Verify that required models are available before starting worker."""
    # This function will check if the required models are accessible
    # Implementation will depend on the specific model loading mechanism
    get_logger().info("Verifying models are available...")

    # Check if required model directories exist using configured paths from settings
    required_paths = [
        Path(settings.asr.whisper_model_path),
        Path(settings.chunkformer.chunkformer_model_path),
        Path(settings.llm_enhance.model),
    ]

    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        get_logger().error(
            "Missing models: {}. Run scripts/download_models.sh", missing
        )
        return False

    # Verify processor modules are available without importing
    try:
        import importlib.util

        has_asr_factory = (
            importlib.util.find_spec("src.processors.asr.factory") is not None
        )
        has_llm = importlib.util.find_spec("src.processors.llm") is not None

        if has_asr_factory and has_llm:
            get_logger().info("All required models and modules are available")
            return True
        get_logger().error(
            f"Missing required modules: ASRFactory={has_asr_factory}, LLM={has_llm}"
        )
        return False
    except Exception:
        get_logger().exception("Model verification failed")
        return False


def start_worker() -> None:
    """Start the RQ worker with proper configuration."""
    # Verify models are available before connecting to Redis
    if not verify_models():
        get_logger().error("Model verification failed. Exiting.")
        sys.exit(1)

    # Set up Redis connection
    redis_conn = setup_redis_connection()

    # Define the queues this worker will listen to
    listen = ["default", "audio_processing"]

    # Create RQ worker with connection - use settings for worker name
    worker = Worker(
        listen,
        connection=redis_conn,
        name=settings.worker.worker_name,
        exception_handlers=[],
    )

    get_logger().info("Starting worker {}", worker.name)
    worker.work()


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

"""
RQ Worker entry point for the Modular Audio Intelligence Engine (MAIE).

This module sets up and configures the RQ worker with Redis connection,
model verification on startup, and proper worker configuration.
"""

import sys

from redis import Redis
from rq import Worker

# Opt-in: import logging configuration helpers defensively.
from src.config import configure_logging, get_logger, settings


def setup_redis_connection() -> Redis:
    """Initialize Redis connection with configuration from centralized settings."""
    # Create Redis connection from URL
    redis_conn = Redis.from_url(settings.redis_url, decode_responses=False)

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

    # Check if required model directories exist (per TDD section 3.7)
    # Use the new config with Path objects
    required_paths = [
        settings.get_model_path("whisper/erax-wow-turbo"),
        settings.get_model_path("chunkformer/large-vie"),
        settings.get_model_path("llm/qwen3-4b-awq"),
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
            "Missing required modules: ASRFactory=%s, LLM=%s",
            has_asr_factory,
            has_llm,
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
        listen, connection=redis_conn, name=settings.worker_name, exception_handlers=[]
    )

    get_logger().info("Starting worker {}", worker.name)
    worker.work()


if __name__ == "__main__":
    # Apply opt-in Loguru configuration at worker startup.
    # Always configure Loguru at startup.
    logger = configure_logging()
    logger = logger if logger is not None else get_logger()
    logger.info("Loguru configuration active (phase1) - worker")

    start_worker()

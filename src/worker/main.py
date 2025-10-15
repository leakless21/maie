"""
RQ Worker entry point for the Modular Audio Intelligence Engine (MAIE).

This module sets up and configures the RQ worker with Redis connection,
model verification on startup, and proper worker configuration.
"""

import sys
from redis import Redis
from rq import Worker

from src.config import settings


def setup_redis_connection() -> Redis:
    """Initialize Redis connection with configuration from centralized settings."""
    # Create Redis connection from URL
    redis_conn = Redis.from_url(settings.redis_url, decode_responses=False)

    # Verify Redis connection
    try:
        redis_conn.ping()
        print("Successfully connected to Redis")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        sys.exit(1)

    return redis_conn


def verify_models() -> bool:
    """Verify that required models are available before starting worker."""
    # This function will check if the required models are accessible
    # Implementation will depend on the specific model loading mechanism
    print("Verifying models are available...")

    # Check if required model directories exist (per TDD section 3.7)
    # Use the new config with Path objects
    required_paths = [
        settings.get_model_path("whisper/erax-wow-turbo"),
        settings.get_model_path("chunkformer/large-vie"),
        settings.get_model_path("llm/qwen3-4b-awq"),
    ]

    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(f"Missing models: {missing}. Run scripts/download_models.sh")
        return False

    # Verify processor modules are available
    try:
        from src.processors.asr.factory import ASRFactory
        from src.processors.llm import LLMProcessor

        print("All required models and modules are available")
        return True
    except ImportError as e:
        print(f"Missing required modules: {e}")
        return False
    except Exception as e:
        print(f"Model verification failed: {e}")
        return False


def start_worker() -> None:
    """Start the RQ worker with proper configuration."""
    # Verify models are available before connecting to Redis
    if not verify_models():
        print("Model verification failed. Exiting.")
        sys.exit(1)

    # Set up Redis connection
    redis_conn = setup_redis_connection()

    # Define the queues this worker will listen to
    listen = ["default", "audio_processing"]

    # Create RQ worker with connection - use settings for worker name
    worker = Worker(
        listen, connection=redis_conn, name=settings.worker_name, exception_handlers=[]
    )

    print(f"Starting worker {worker.name}...")
    worker.work()


if __name__ == "__main__":
    start_worker()

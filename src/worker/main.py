"""
RQ Worker entry point for the Modular Audio Intelligence Engine (MAIE).

This module sets up and configures the RQ worker with Redis connection,
model verification on startup, and proper worker configuration.
"""

import os
import sys
import os.path
from redis import Redis
from rq import Worker, Queue, Connection
from typing import Optional


def setup_redis_connection() -> Redis:
    """Initialize Redis connection with configuration from environment."""
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))  # Use DB 0 for queues per TDD
    redis_password = os.getenv("REDIS_PASSWORD", None)
    
    redis_conn = Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=False
    )
    
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
    required_model_paths = [
        "/data/models/whisper/erax-wow-turbo",
        "/data/models/chunkformer/large-vie",
        "/data/models/llm/qwen3-4b-awq"
    ]
    
    # Allow for relative paths during development
    model_dir = os.getenv("MODEL_DIR", "./data/models")
    required_paths = [
        f"{model_dir}/whisper/erax-wow-turbo",
        f"{model_dir}/chunkformer/large-vie",
        f"{model_dir}/llm/qwen3-4b-awq"
    ]
    
    missing = [p for p in required_paths if not os.path.exists(p)]
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
    listen = ['default', 'audio_processing']
    
    # Create RQ worker with connection
    with Connection(redis_conn):
        worker = Worker(
            map(Queue, listen),
            name=os.getenv("WORKER_NAME", "maie_worker"),
            exception_handlers=[]
        )
        
        print(f"Starting worker {worker.name}...")
        worker.work()


if __name__ == "__main__":
    start_worker()
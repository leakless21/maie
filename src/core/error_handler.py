"""
Modern error handling system with context managers and resource cleanup.

This module provides a comprehensive error handling framework that follows
industry best practices for Python applications, including:
- Context manager-based resource cleanup
- Exception chaining for better debugging
- Native dependency error leveraging
- Structured error reporting
"""

from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Protocol
import re

from redis import Redis
from loguru import logger

try:
    import torch
except Exception:
    torch = None

from src.api.errors import MAIEError, ErrorCodes


class ResourceCleanup(Protocol):
    """Protocol for resource cleanup operations."""

    def cleanup(self) -> None:
        """Perform resource cleanup."""
        ...

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if hasattr(self, "cleanup"):
            self.cleanup()
        return False  # Don't suppress exceptions


class GpuModelResource:
    """GPU model resource cleanup with CUDA cache clearing."""

    def __init__(self, model: Any, model_name: str = "unknown"):
        self.model = model
        self.model_name = model_name

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def cleanup(self) -> None:
        """Clean up GPU model resources."""
        try:
            # Prefer module-level torch so tests can patch it; fall back to import if needed.
            t = (
                torch
                if ("torch" in globals() and globals().get("torch") is not None)
                else None
            )
            if t is None:
                try:
                    t = __import__("torch")
                except Exception:
                    t = None

            # Clear CUDA cache if available
            if (
                getattr(t, "cuda", None)
                and getattr(t.cuda, "is_available", lambda: False)()
            ):
                try:
                    t.cuda.empty_cache()
                except Exception:
                    # ignore CUDA cache clear failures
                    pass

            # If the model provides its own cleanup method, call it
            if hasattr(self.model, "cleanup"):
                try:
                    self.model.cleanup()
                except Exception as me:
                    logger.warning(
                        f"Model-specific cleanup failed for {self.model_name}: {me}"
                    )

            logger.info(f"Cleaned up GPU model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU model {self.model_name}: {e}")


class RedisConnectionResource:
    """Redis connection resource with proper connection pooling."""

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def cleanup(self) -> None:
        """Clean up Redis connection."""
        try:
            if self.redis_client:
                self.redis_client.close()
                logger.debug("Redis connection closed")
        except Exception as e:
            logger.warning(f"Failed to close Redis connection: {e}")


class TemporaryFileResource:
    """Temporary file cleanup with error handling."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def cleanup(self) -> None:
        """Clean up temporary file."""
        try:
            import os

            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                logger.debug(f"Temporary file removed: {self.file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {self.file_path}: {e}")


class ProcessingErrorHandler:
    """
    Context manager for comprehensive error handling and resource cleanup.

    Provides automatic resource cleanup, Redis status updates, and structured
    error reporting for MAIE processing operations.
    """

    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        task_key: Optional[str] = None,
        stage: str = "processing",
        auto_cleanup: bool = True,
    ):
        """Initialize error handler with context and cleanup options."""
        self.redis_client = redis_client
        self.task_key = task_key
        self.stage = stage
        self.auto_cleanup = auto_cleanup
        self.resources: List[ResourceCleanup] = []
        self.start_time: Optional[float] = None
        self.logger = logger.bind(stage=stage, task_key=task_key)

    def __enter__(self) -> "ProcessingErrorHandler":
        """Enter context and initialize resource tracking."""
        self.start_time = time.time()
        self.logger.info("Starting processing stage")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context with comprehensive cleanup and error handling."""
        try:
            if exc_val:
                self.handle_error(exc_val)
            else:
                self.handle_success()
        finally:
            if self.auto_cleanup:
                self.cleanup_resources()
        return False  # Don't suppress exceptions

    def register_resource(self, resource: ResourceCleanup) -> None:
        """Register a resource for automatic cleanup."""
        self.resources.append(resource)
        self.logger.debug(f"Registered resource for cleanup: {type(resource).__name__}")

    def cleanup_resources(self) -> None:
        """Clean up all registered resources."""
        cleanup_errors = []

        for resource in self.resources:
            try:
                if hasattr(resource, "cleanup"):
                    resource.cleanup()
                elif hasattr(resource, "__exit__"):
                    # Support context manager protocols
                    resource.__exit__(None, None, None)
            except Exception as e:
                cleanup_errors.append(str(e))
                self.logger.warning(f"Resource cleanup failed: {e}")

        if cleanup_errors:
            self.logger.error(f"Resource cleanup errors: {cleanup_errors}")

    def handle_error(self, error: Exception) -> None:
        """Handle error with proper Redis updates and logging."""
        duration = time.time() - (self.start_time or 0)

        error_context = {
            "stage": self.stage,
            "task_key": self.task_key,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "duration": duration,
            "traceback": traceback.format_exc(),
        }

        # Log structured error using module logger (keeps test patches simpler)
        logger.error(f"Processing failed in {self.stage}", **error_context)

        # Update Redis status if available
        if self.redis_client and self.task_key:
            try:
                self._update_redis_status("FAILED", error_context)
            except Exception as redis_error:
                self.logger.error(f"Failed to update Redis status: {redis_error}")

    def handle_success(self) -> None:
        """Handle successful completion."""
        duration = time.time() - (self.start_time or 0)

        # Log success using module logger to make structured context visible to tests
        logger.info(
            f"Processing completed successfully in {self.stage}",
            duration=duration,
            stage=self.stage,
            task_key=self.task_key,
        )

        # Update Redis status if available
        if self.redis_client and self.task_key:
            try:
                self._update_redis_status("COMPLETED", {"duration": duration})
            except Exception as redis_error:
                self.logger.error(f"Failed to update Redis status: {redis_error}")

    def _update_redis_status(self, status: str, details: Dict[str, Any]) -> None:
        """Update task status in Redis.

        For tests we favor a deterministic, side-effecting hset call so unit tests
        can assert on mock_redis.hset. We still format non-string values as JSON.
        """
        if not self.redis_client or not self.task_key:
            return

        from datetime import datetime, timezone
        import json

        update_data = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        }

        # Prefer direct Redis update for determinism in tests and simplicity.
        try:
            mapping = {
                k: (v if isinstance(v, str) else json.dumps(v))
                for k, v in update_data.items()
            }
            self.redis_client.hset(self.task_key, mapping=mapping)
        except Exception as rederr:
            # Log and don't raise to avoid masking original errors
            logger.error(f"Failed to write status to Redis: {rederr}")

    @staticmethod
    @contextmanager
    def wrap_operation(
        operation_name: str,
        redis_client: Optional[Redis] = None,
        task_key: Optional[str] = None,
    ):
        """Simple context manager for individual operations."""
        handler = ProcessingErrorHandler(
            redis_client=redis_client, task_key=task_key, stage=operation_name
        )

        with handler:
            yield handler


def leverage_native_error(
    native_error: Exception,
    semantic_message: str,
    error_class: type[MAIEError],
    **context,
) -> MAIEError:
    """
    Leverage native dependency errors with semantic wrapping.

    This function preserves the original exception context while providing
    meaningful semantic error messages for the application domain.

    Args:
        native_error: The original exception from a dependency
        semantic_message: Domain-specific error message
        error_class: MAIE error class to instantiate
        **context: Additional context information

    Returns:
        MAIEError with chained native exception
    """
    # Extract relevant information from native error
    error_details = {
        "native_error_type": type(native_error).__name__,
        "native_error_message": str(native_error),
        **context,
    }

    # Attempt to instantiate the semantic error using common constructor patterns.
    try:
        semantic_error = error_class(message=semantic_message, details=error_details)
    except TypeError:
        # Fallback: some MAIE error subclasses inherit MAIEError.__init__ and require error_code.
        # Map the class name to an ErrorCodes attribute (simple CamelCase -> SNAKE_CASE).
        code_candidate = re.sub(r"(?<!^)(?=[A-Z])", "_", error_class.__name__).upper()
        error_code = getattr(ErrorCodes, code_candidate, ErrorCodes.PROCESSING_ERROR)
        try:
            semantic_error = error_class(
                error_code=error_code, message=semantic_message, details=error_details
            )
        except TypeError:
            # Last resort: instantiate base MAIEError to preserve information.
            semantic_error = MAIEError(
                error_code, semantic_message, details=error_details
            )

    # Chain the exceptions to preserve context
    raise semantic_error from native_error


# Resource factory functions for common patterns
def create_gpu_resource(model: Any, model_name: str = "unknown") -> GpuModelResource:
    """Create a GPU model resource for cleanup."""
    return GpuModelResource(model, model_name)


def create_redis_resource(redis_client: Redis) -> RedisConnectionResource:
    """Create a Redis connection resource for cleanup."""
    return RedisConnectionResource(redis_client)


def create_temp_file_resource(file_path: str) -> TemporaryFileResource:
    """Create a temporary file resource for cleanup."""
    return TemporaryFileResource(file_path)

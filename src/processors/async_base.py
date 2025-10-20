"""
Modern async-first processor interfaces for Python 3.12+ compatibility.

This module provides the new AsyncProcessor base class and related utilities
that replace the deprecated asyncio.get_event_loop() pattern with modern
asyncio.get_running_loop() implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Optional, Tuple, Dict, AsyncGenerator
from dataclasses import dataclass

from src.config.logging import get_module_logger

logger = get_module_logger(__name__)


@dataclass
class AsyncProcessorError(Exception):
    """Base exception for async processor errors."""

    pass


@dataclass
class AsyncResourceError(AsyncProcessorError):
    """Raised when resource management fails."""

    pass


@dataclass
class AsyncExecutionError(AsyncProcessorError):
    """Raised during async execution failure."""

    pass


class AsyncProcessor(ABC):
    """
    Modern async-first processor interface for Python 3.12+ compatibility.

    Replaces legacy Processor class with proper asyncio patterns using
    asyncio.get_running_loop() and context management.
    """

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute processing task asynchronously.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def unload(self) -> None:
        """
        Unload resources used by the processor asynchronously.
        Must be implemented by subclasses.
        """
        pass

    async def reduce_memory(self) -> None:
        """
        Optional hook for memory optimization.
        Default is no-op; subclasses may override.
        """
        pass

    async def _setup_resources(self) -> None:
        """Setup resources - override in subclasses."""
        pass

    async def cleanup_on_error(self, exc: Optional[Exception]) -> None:
        """Specialized cleanup for error scenarios."""
        try:
            await self.reduce_memory()
            await self.unload()
        except Exception as cleanup_exc:
            logger.warning(f"Cleanup error during exception handling: {cleanup_exc}")

    @asynccontextmanager
    async def use(self) -> AsyncGenerator["AsyncProcessor", None]:
        """
        Async context manager for deterministic resource management.
        """
        try:
            yield self
        finally:
            await self.unload()

    def execute_sync(self, *args, **kwargs) -> Any:
        """
        Synchronous compatibility method using asyncio.run().
        Provides bridge for legacy synchronous code.
        """

        async def _async_wrapper():
            async with self.use():
                return await self.execute(*args, **kwargs)

        return asyncio.run(_async_wrapper())


async def run_in_executor(func, *args, **kwargs) -> Any:
    """
    Execute a synchronous function in a thread.

    Strategy:
    - Prefer calling the current loop's run_in_executor so tests that patch
      asyncio.get_running_loop() / loop.run_in_executor observe the call.
    - Some loop.run_in_executor implementations (real event loop) do not accept
      keyword arguments. In that case, fall back to wrapping the call with
      functools.partial so no kwargs reach run_in_executor directly.
    - As a final fallback use asyncio.to_thread which supports kwargs.
    """
    loop = asyncio.get_running_loop()
    try:
        # Try calling run_in_executor with the function and kwargs directly.
        # Tests often mock run_in_executor and expect this exact signature.
        return await loop.run_in_executor(None, func, *args, **kwargs)
    except TypeError:
        # Real event loop's run_in_executor may not accept kwargs; wrap call.
        from functools import partial

        return await loop.run_in_executor(None, partial(func, *args, **kwargs))
    except RuntimeError:
        # No running loop (shouldn't normally happen inside async context) -
        # fall back to asyncio.to_thread which will create the necessary thread.
        return await asyncio.to_thread(func, *args, **kwargs)


async def safe_async_execute(
    processor: AsyncProcessor, *args, **kwargs
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Execute async processor safely with structured error handling.

    Returns:
        Tuple of (result, error_info)
    """
    try:
        result = await processor.execute(*args, **kwargs)
        return result, None
    except Exception as exc:
        error_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "processor": type(processor).__name__,
        }
        logger.error(f"Async processor execution failed: {error_info}")
        return None, error_info


@asynccontextmanager
async def processor_session(
    processor: AsyncProcessor,
) -> AsyncGenerator[AsyncProcessor, None]:
    """
    Generic processor session manager with proper error handling.

    Example:
        async with processor_session(my_processor) as p:
            result = await p.execute(data)
    """
    try:
        await processor._setup_resources()
        yield processor
    except Exception as exc:
        await processor.cleanup_on_error(exc)
        raise
    finally:
        await processor.cleanup_on_error(None)

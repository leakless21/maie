"""
Simplified Edge API for Jetson Nano/Orin.

This module provides a minimal, synchronous HTTP server optimized for edge deployments:
- Single-process, synchronous execution (no Redis/RQ)
- ASR-only processing (no LLM summarization)
- Application-level lock for single-task semantics
- Minimal dependencies and memory footprint

Usage:
    # Start server
    uvicorn src.api.edge_main:app --host 0.0.0.0 --port 8000 --workers 1

    # Or with pixi
    ENVIRONMENT=jetson pixi run serve

    # Test endpoint
    curl -X POST http://localhost:8000/v1/transcribe -F "file=@audio.mp3"
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from litestar import Litestar, MediaType, Request, Response, delete, get, post, put
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.exceptions import HTTPException, NotFoundException, ValidationException
from litestar.params import Body
from litestar.response import ServerSentEvent, ServerSentEventMessage
from pydantic import BaseModel, Field

from src.api.schemas import (
    TemplateCreateSchema,
    TemplateDetailSchema,
    TemplateUpdateSchema,
    TemplatesResponseSchema,
)
from src.api.template_utils import (
    load_template_detail,
    load_template_schema,
    scan_templates_directory,
)
from src.config import configure_logging, get_settings, settings
from src.config.logging import get_module_logger
from src.streaming import (
    ErrorEvent,
    ProgressEvent,
    ProgressStage,
    StreamingTranscriptionHandler,
)
from src.utils.template_manager import TemplateManager

# Configure logging
_logger = configure_logging()
logger = get_module_logger(__name__)

# Global lock for single-task processing
_process_lock = asyncio.Lock()
_current_task_id: Optional[str] = None


# =============================================================================
# Request/Response Schemas
# =============================================================================


class EdgeHealthResponse(BaseModel):
    """Health check response for edge API."""

    status: Literal["healthy", "busy", "unhealthy"]
    version: str = "jetson-1.0"
    environment: str
    current_task: Optional[str] = None
    features: Dict[str, bool] = Field(default_factory=dict)


class EdgeTranscribeRequest(BaseModel):
    """Request schema for transcription endpoint."""

    asr_backend: Literal["whisper", "chunkformer"] = Field(
        default="whisper",
        description="ASR backend to use for transcription",
    )
    enable_vad: bool = Field(
        default=True,
        description="Enable Voice Activity Detection preprocessing",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="VAD speech confidence threshold",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'vi', 'en'). Auto-detect if None.",
    )


class TranscriptSegment(BaseModel):
    """A single transcript segment with timestamps."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class EdgeTranscribeResponse(BaseModel):
    """Response schema for transcription endpoint."""

    task_id: str
    status: Literal["completed", "failed"]
    transcript: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    duration_seconds: float
    processing_time_seconds: float
    asr_backend: str
    language: Optional[str] = None
    error: Optional[str] = None


class EdgeErrorResponse(BaseModel):
    """Error response schema."""

    detail: str
    task_id: Optional[str] = None


# =============================================================================
# Route Handlers
# =============================================================================


@get("/health", summary="Health check", tags=["Health"])
async def health_check() -> EdgeHealthResponse:
    """
    Health check endpoint for edge API.

    Returns current status, environment info, and available features.
    """
    current_settings = get_settings()

    status: Literal["healthy", "busy", "unhealthy"] = "healthy"
    if _process_lock.locked():
        status = "busy"

    return EdgeHealthResponse(
        status=status,
        version="jetson-1.0",
        environment=current_settings.environment,
        current_task=_current_task_id,
        features={
            "asr": True,
            "vad": current_settings.vad.enabled,
            "llm": current_settings.features.enable_llm,
            "diarization": current_settings.features.enable_diarization,
            "templates": True,
        },
    )


@get("/", summary="API info", tags=["Info"])
async def root() -> Dict[str, Any]:
    """Return API information and available endpoints."""
    return {
        "name": "MAIE Edge API (Jetson)",
        "version": "1.0.0-jetson",
        "status": "running",
        "description": "Simplified ASR API for Jetson Nano/Orin edge devices",
        "endpoints": {
            "health": "/health",
            "transcribe": "/v1/transcribe",
            "transcribe_stream": "/v1/transcribe/stream",
            "models": "/v1/models",
            "templates": "/v1/templates",
        },
        "features": {
            "asr": ["whisper", "chunkformer"],
            "vad": True,
            "llm": False,
            "diarization": False,
            "sse_streaming": True,
            "templates": True,
        },
    }


@get("/v1/models", summary="List available models", tags=["Models"])
async def list_models() -> Dict[str, Any]:
    """List available ASR models for edge deployment."""
    current_settings = get_settings()

    models = {
        "asr_backends": ["whisper", "chunkformer"],
        "default_backend": current_settings.api.default_asr_backend,
        "models": {
            "whisper": {
                "path": current_settings.asr.whisper_model_path,
                "device": current_settings.asr.whisper_device,
                "compute_type": current_settings.asr.whisper_compute_type,
            },
            "chunkformer": {
                "path": current_settings.chunkformer.chunkformer_model_path,
                "device": current_settings.chunkformer.chunkformer_device,
            },
        },
    }
    return models


@get("/v1/templates", summary="List available templates", tags=["Templates"])
async def list_templates_endpoint() -> TemplatesResponseSchema:
    """List available templates on disk for parity with the main API."""
    return scan_templates_directory()


@get(
    "/v1/templates/{template_id:str}",
    summary="Get template detail",
    tags=["Templates"],
)
async def get_template_detail_endpoint(template_id: str) -> TemplateDetailSchema:
    """Return full template details including prompt content."""
    manager = TemplateManager()
    return await load_template_detail(template_id, manager)


@get(
    "/v1/templates/{template_id:str}/schema",
    summary="Get template schema",
    tags=["Templates"],
)
async def get_template_schema_endpoint(template_id: str) -> Dict[str, Any]:
    """Return the JSON schema for a template."""
    return load_template_schema(template_id)


@post(
    "/v1/templates",
    summary="Create template",
    tags=["Templates"],
    status_code=201,
)
async def create_template_endpoint(data: TemplateCreateSchema) -> TemplateDetailSchema:
    """Create a new template bundle on the edge device."""
    manager = TemplateManager()
    try:
        await manager.create_template(
            template_id=data.id,
            schema=data.schema_data,
            prompt=data.prompt_template,
            example=data.example,
        )
    except FileExistsError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Template {data.id} already exists",
        ) from exc
    except Exception as exc:
        logger.error("Failed to create template {}", data.id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return await load_template_detail(data.id, manager)


@put("/v1/templates/{template_id:str}", summary="Update template", tags=["Templates"])
async def update_template_endpoint(
    template_id: str, data: TemplateUpdateSchema
) -> TemplateDetailSchema:
    """Update an existing template bundle."""
    manager = TemplateManager()
    if not manager.exists(template_id):
        raise NotFoundException(f"Template {template_id} not found")

    try:
        await manager.update_template(
            template_id=template_id,
            schema=data.schema_data,
            prompt=data.prompt_template,
            example=data.example,
        )
    except Exception as exc:
        logger.error("Failed to update template {}", template_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return await load_template_detail(template_id, manager)


@delete(
    "/v1/templates/{template_id:str}",
    summary="Delete template",
    tags=["Templates"],
    status_code=204,
)
async def delete_template_endpoint(template_id: str) -> None:
    """Delete a template bundle from disk."""
    manager = TemplateManager()
    if not manager.exists(template_id):
        raise NotFoundException(f"Template {template_id} not found")

    try:
        await manager.delete_template(template_id)
    except Exception as exc:
        logger.error("Failed to delete template {}", template_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return None


@post("/v1/transcribe", summary="Transcribe audio", tags=["Transcription"])
async def transcribe_audio(
    data: UploadFile = Body(media_type=RequestEncodingType.MULTI_PART),
    asr_backend: str = "whisper",
    enable_vad: bool = True,
    vad_threshold: float = 0.5,
    language: Optional[str] = None,
) -> EdgeTranscribeResponse:
    """
    Transcribe audio file using ASR.

    This is a synchronous endpoint - only one request is processed at a time.
    If a request is already being processed, this will wait for the lock.

    Args:
        data: Audio file (WAV, MP3, M4A, FLAC, etc.)
        asr_backend: ASR backend to use ('whisper' or 'chunkformer')
        enable_vad: Enable Voice Activity Detection preprocessing
        vad_threshold: VAD confidence threshold (0.0-1.0)
        language: Language code for transcription (auto-detect if None)

    Returns:
        Transcription result with segments and metadata
    """
    global _current_task_id

    task_id = str(uuid4())
    start_time = time.time()
    current_settings = get_settings()

    # Validate ASR backend
    if asr_backend not in ("whisper", "chunkformer"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ASR backend: {asr_backend}. Must be 'whisper' or 'chunkformer'.",
        )

    # Check if we can accept the request
    if _process_lock.locked():
        logger.warning(
            "Request received while processing another task", task_id=task_id
        )

    async with _process_lock:
        _current_task_id = task_id
        logger.info(
            "Starting transcription",
            task_id=task_id,
            asr_backend=asr_backend,
            filename=data.filename,
        )

        try:
            # Create task directory
            audio_dir = current_settings.paths.audio_dir / task_id
            audio_dir.mkdir(parents=True, exist_ok=True)

            # Determine file extension
            filename = data.filename or "audio"
            ext = Path(filename).suffix or ".wav"
            audio_path = audio_dir / f"input{ext}"

            # Write uploaded file
            content = await data.read()
            audio_path.write_bytes(content)

            logger.info(
                "Audio file saved",
                task_id=task_id,
                path=str(audio_path),
                size_bytes=len(content),
            )

            # Run ASR processing
            result = await _run_asr(
                task_id=task_id,
                audio_path=audio_path,
                asr_backend=asr_backend,
                enable_vad=enable_vad,
                vad_threshold=vad_threshold,
                language=language,
            )

            processing_time = time.time() - start_time

            logger.info(
                "Transcription completed",
                task_id=task_id,
                processing_time=processing_time,
                transcript_length=len(result.get("transcript", "")),
            )

            # Build response
            segments = [
                TranscriptSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", ""),
                    speaker=seg.get("speaker"),
                )
                for seg in result.get("segments", [])
            ]

            return EdgeTranscribeResponse(
                task_id=task_id,
                status="completed",
                transcript=result.get("transcript", ""),
                segments=segments,
                duration_seconds=result.get("duration", 0.0),
                processing_time_seconds=processing_time,
                asr_backend=asr_backend,
                language=result.get("language"),
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Transcription failed",
                task_id=task_id,
                error=str(e),
                processing_time=processing_time,
            )
            return EdgeTranscribeResponse(
                task_id=task_id,
                status="failed",
                transcript="",
                duration_seconds=0.0,
                processing_time_seconds=processing_time,
                asr_backend=asr_backend,
                error=str(e),
            )
        finally:
            _current_task_id = None


@post("/v1/transcribe/stream", summary="Transcribe with progress streaming", tags=["Transcription"])
async def transcribe_audio_stream(
    data: UploadFile = Body(media_type=RequestEncodingType.MULTI_PART),
    asr_backend: str = "whisper",
    enable_vad: bool = True,
    vad_threshold: float = 0.5,
    language: Optional[str] = None,
) -> ServerSentEvent:
    """
    Transcribe audio file with real-time progress updates via Server-Sent Events.

    This endpoint streams progress updates as the transcription proceeds:

    **Event Types:**
    - `progress`: Stage and percentage updates
    - `segment`: Individual transcript segments (as they complete)
    - `result`: Final complete result
    - `error`: Error information if processing fails

    **Progress Stages:**
    - `queued` (0%): Request received
    - `uploading` (0-10%): Saving audio file
    - `preprocessing` (10-20%): Analyzing audio
    - `transcribing` (20-95%): ASR processing
    - `completed` (100%): Done

    **Client Usage (JavaScript):**
    ```javascript
    const response = await fetch('/v1/transcribe/stream', {
        method: 'POST',
        body: formData
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value).split('\\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const event = JSON.parse(line.slice(6));
                console.log(`${event.stage}: ${event.progress}%`);
            }
        }
    }
    ```

    Args:
        data: Audio file (WAV, MP3, M4A, FLAC, etc.)
        asr_backend: ASR backend to use ('whisper' or 'chunkformer')
        enable_vad: Enable Voice Activity Detection preprocessing
        vad_threshold: VAD confidence threshold (0.0-1.0)
        language: Language code for transcription (auto-detect if None)

    Returns:
        Server-Sent Events stream with progress updates and final result
    """
    global _current_task_id

    task_id = str(uuid4())
    current_settings = get_settings()

    # Validate ASR backend
    if asr_backend not in ("whisper", "chunkformer"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ASR backend: {asr_backend}. Must be 'whisper' or 'chunkformer'.",
        )

    async def event_generator() -> AsyncGenerator[ServerSentEventMessage, None]:
        global _current_task_id

        handler = StreamingTranscriptionHandler(
            task_id=task_id,
            asr_backend=asr_backend,
        )

        audio_path: Optional[Path] = None

        async def upload() -> None:
            nonlocal audio_path
            audio_dir = current_settings.paths.audio_dir / task_id
            audio_dir.mkdir(parents=True, exist_ok=True)

            filename = data.filename or "audio"
            ext = Path(filename).suffix or ".wav"
            audio_path = audio_dir / f"input{ext}"

            content = await data.read()
            audio_path.write_bytes(content)

            logger.info(
                "Audio file saved",
                task_id=task_id,
                path=str(audio_path),
                size=len(content),
            )

        async def preprocess() -> Dict[str, Any]:
            duration = _get_audio_duration(audio_path)
            handler.tracker.audio_duration = duration
            return {"duration": duration}

        async def transcribe() -> Dict[str, Any]:
            return await _run_asr(
                task_id=task_id,
                audio_path=audio_path,
                asr_backend=asr_backend,
                enable_vad=enable_vad,
                vad_threshold=vad_threshold,
                language=language,
            )

        # Check if busy and emit waiting message
        if _process_lock.locked():
            yield ServerSentEventMessage(
                data=ProgressEvent(
                    stage=ProgressStage.QUEUED,
                    progress=0,
                    message="Waiting for current task to complete",
                ).to_sse_data(),
                event="progress",
            )

        async with _process_lock:
            _current_task_id = task_id

            try:
                async for msg in handler.run_with_progress(
                    upload_func=upload,
                    preprocess_func=preprocess,
                    transcribe_func=transcribe,
                    emit_segments=True,
                ):
                    yield ServerSentEventMessage(**msg)

            except Exception as e:
                logger.error(
                    "Streaming transcription failed",
                    task_id=task_id,
                    error=str(e),
                )
                yield ServerSentEventMessage(
                    data=ErrorEvent(
                        error=str(e),
                        stage=handler.tracker.current_stage,
                    ).to_sse_data(),
                    event="error",
                )
            finally:
                _current_task_id = None

    return ServerSentEvent(event_generator())


# =============================================================================
# ASR Processing
# =============================================================================


async def _run_asr(
    task_id: str,
    audio_path: Path,
    asr_backend: str,
    enable_vad: bool,
    vad_threshold: float,
    language: Optional[str],
) -> Dict[str, Any]:
    """
    Run ASR processing on audio file.

    This wraps the blocking ASR call in a thread to avoid blocking the event loop.
    """
    import anyio

    def _sync_asr() -> Dict[str, Any]:
        return _run_asr_sync(
            task_id=task_id,
            audio_path=audio_path,
            asr_backend=asr_backend,
            enable_vad=enable_vad,
            vad_threshold=vad_threshold,
            language=language,
        )

    return await anyio.to_thread.run_sync(_sync_asr)


def _run_asr_sync(
    task_id: str,
    audio_path: Path,
    asr_backend: str,
    enable_vad: bool,
    vad_threshold: float,
    language: Optional[str],
) -> Dict[str, Any]:
    """
    Synchronous ASR processing.

    Loads the appropriate ASR backend and processes the audio file.
    """
    from src.processors.asr.factory import ASRFactory
    from src.processors.base import ASRResult

    # Create ASR processor
    logger.info(f"Creating ASR processor: {asr_backend}", task_id=task_id)
    processor = ASRFactory.create(asr_backend)

    # Get audio duration
    duration = _get_audio_duration(audio_path)

    # Read audio file
    audio_data = audio_path.read_bytes()

    # Process audio
    logger.info("Running ASR inference", task_id=task_id, duration=duration)
    result: ASRResult = processor.execute(audio_data, language=language)

    # Extract transcript and segments from ASRResult dataclass
    transcript = result.text
    segments = []

    # Handle segments if available
    if result.segments:
        for seg in result.segments:
            if isinstance(seg, dict):
                segments.append(
                    {
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", ""),
                    }
                )
            else:
                segments.append(
                    {
                        "start": getattr(seg, "start", 0.0),
                        "end": getattr(seg, "end", 0.0),
                        "text": getattr(seg, "text", ""),
                    }
                )

    # Get detected language
    detected_language = result.language

    return {
        "transcript": transcript,
        "segments": segments,
        "duration": result.duration or duration,
        "language": detected_language or language,
    }


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return 0.0


# =============================================================================
# Exception Handlers
# =============================================================================


def _handle_not_found(request: Request, exc: NotFoundException) -> Response:
    """Handle 404 Not Found exceptions gracefully without verbose logging."""
    path = request.url.path
    # Only log non-static file 404s at debug level to reduce noise
    if not path.endswith((".ico", ".png", ".jpg", ".css", ".js", ".map")):
        logger.debug("Resource not found: {}", path)
    return Response(
        media_type=MediaType.JSON,
        content={"detail": "Not Found", "path": path},
        status_code=404,
    )


def _handle_validation_exception(
    request: Request, exc: ValidationException
) -> Response:
    """Handle validation errors with custom format."""
    logger.warning("Validation error on {}: {}", request.url.path, exc.detail)
    return Response(
        media_type=MediaType.JSON,
        content={
            "detail": "Validation Error",
            "message": str(exc.detail) if exc.detail else "Request validation failed",
            "path": request.url.path,
        },
        status_code=400,
    )


def _handle_generic_exception(request: Request, exc: Exception) -> Response:
    """Handle unexpected exceptions."""
    logger.opt(exception=exc).error(
        "Unhandled exception on {}: {}", request.url.path, str(exc)
    )
    return Response(
        media_type=MediaType.JSON,
        content={"detail": "Internal Server Error"},
        status_code=500,
    )


def _handle_http_exception(request: Request, exc: HTTPException) -> Response:
    """Handle HTTP exceptions."""
    logger.warning(
        "HTTP exception on {}: {} - {}", request.url.path, exc.status_code, exc.detail
    )
    return Response(
        media_type=MediaType.JSON,
        content={"detail": getattr(exc, "detail", "Error")},
        status_code=getattr(exc, "status_code", 500),
    )


# =============================================================================
# Application Factory
# =============================================================================


def create_edge_app() -> Litestar:
    """Create and configure the edge Litestar application."""

    logger.info("Creating MAIE Edge API application")

    app = Litestar(
        route_handlers=[
            root,
            health_check,
            list_models,
            list_templates_endpoint,
            get_template_detail_endpoint,
            get_template_schema_endpoint,
            create_template_endpoint,
            update_template_endpoint,
            delete_template_endpoint,
            transcribe_audio,
            transcribe_audio_stream,
        ],
        exception_handlers={
            NotFoundException: _handle_not_found,
            ValidationException: _handle_validation_exception,
            HTTPException: _handle_http_exception,
            Exception: _handle_generic_exception,
        },
        debug=settings.debug,
        request_max_body_size=settings.api.max_file_size_mb * 1024 * 1024,
    )

    logger.info(
        "Edge API initialized",
        environment=settings.environment,
        debug=settings.debug,
    )

    return app


# Create the application instance
app = create_edge_app()


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    current_settings = get_settings()

    uvicorn.run(
        "src.api.edge_main:app",
        host=current_settings.api.host,
        port=current_settings.api.port,
        workers=1,  # Single worker for edge
        log_level="info",
    )

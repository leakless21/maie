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
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from litestar import Litestar, Response, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.exceptions import HTTPException
from litestar.params import Body
from pydantic import BaseModel, Field

from src.config import configure_logging, get_settings, settings
from src.config.logging import get_module_logger

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
        default="chunkformer",
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
            "models": "/v1/models",
        },
        "features": {
            "asr": ["whisper", "chunkformer"],
            "vad": True,
            "llm": False,
            "diarization": False,
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


@post("/v1/transcribe", summary="Transcribe audio", tags=["Transcription"])
async def transcribe_audio(
    data: UploadFile = Body(media_type=RequestEncodingType.MULTI_PART),
    asr_backend: str = "chunkformer",
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
        logger.warning("Request received while processing another task", task_id=task_id)
    
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
                segments.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg.get("text", ""),
                })
            else:
                segments.append({
                    "start": getattr(seg, "start", 0.0),
                    "end": getattr(seg, "end", 0.0),
                    "text": getattr(seg, "text", ""),
                })
    
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
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
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


def _handle_generic_exception(_: Any, exc: Exception) -> Response:
    """Handle unexpected exceptions."""
    logger.opt(exception=exc).error("Unhandled exception: {}", str(exc))
    return Response(
        {"detail": "Internal Server Error"},
        status_code=500,
    )


def _handle_http_exception(_: Any, exc: HTTPException) -> Response:
    """Handle HTTP exceptions."""
    return Response(
        {"detail": getattr(exc, "detail", "Error")},
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
            transcribe_audio,
        ],
        exception_handlers={
            Exception: _handle_generic_exception,
            HTTPException: _handle_http_exception,
        },
        debug=settings.debug,
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

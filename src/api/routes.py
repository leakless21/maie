"""Route definitions for the MAIE API."""

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from litestar import Controller, Request, get, post
from litestar.datastructures import UploadFile
from litestar.exceptions import HTTPException, NotFoundException
from litestar.params import Parameter
from litestar.status_codes import (
    HTTP_202_ACCEPTED,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_429_TOO_MANY_REQUESTS,
)

from src.api.schemas import (
    Feature,
    ModelInfoSchema,
    ModelsResponseSchema,
    ProcessResponse,
    StatusResponseSchema,
    TemplateInfoSchema,
    TemplatesResponseSchema,
)
from src.config import settings


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.

    Removes directory separators, parent directory references, and other
    potentially malicious characters.

    Args:
        filename: Original filename from user input

    Returns:
        Sanitized filename safe for file system operations
    """
    if not filename:
        return "unnamed"

    # Remove path components - just keep the base filename
    filename = Path(filename).name

    # Remove any remaining path traversal attempts
    filename = filename.replace("..", "").replace("/", "").replace("\\", "")

    # Remove potentially dangerous characters but keep extension
    # Allow: letters, numbers, dots, dashes, underscores
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

    # Ensure filename isn't empty after sanitization
    if not filename or filename == ".":
        return "unnamed"

    return filename


def check_queue_depth() -> bool:
    """
    Check if the queue has capacity for more jobs.

    Note: Currently not implemented. Will be wired up when Redis/RQ integration is complete.

    Returns:
        True if queue has capacity, False if full
    """
    # Placeholder - returns True until Redis integration implemented
    return True


async def save_audio_file_streaming(file: UploadFile, task_id: uuid.UUID) -> Path:
    """
    Save uploaded audio file to disk using streaming to avoid loading entire file into memory.

    Args:
        file: Uploaded audio file
        task_id: Task identifier for naming (UUID for security)

    Returns:
        Path to saved file

    Raises:
        HTTPException: If file size exceeds limit during streaming
    """
    import aiofiles

    # SECURITY: Use UUID for filename, not user input (prevents path traversal)
    # Extract extension from user filename for convenience
    if file.filename:
        sanitized = sanitize_filename(file.filename)
        ext = Path(sanitized).suffix
    else:
        ext = ".wav"  # Default extension

    # Ensure extension is in allowed list
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac"}
    if ext.lower() not in allowed_extensions:
        ext = ".wav"  # Fallback to safe default

    # Use UUID-based filename
    file_path = settings.audio_dir / f"{task_id}{ext}"

    # Ensure directory exists
    settings.audio_dir.mkdir(parents=True, exist_ok=True)

    # Stream file content to disk using aiofiles with size validation
    total_size = 0
    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    chunk_size = 64 * 1024  # 64KB chunks

    async with aiofiles.open(file_path, "wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break

            # Validate size during streaming
            total_size += len(chunk)
            if total_size > max_size_bytes:
                # Clean up partial file
                await f.close()
                try:
                    file_path.unlink()
                except OSError:
                    pass  # Ignore cleanup errors

                file_size_mb = total_size / (1024 * 1024)
                raise HTTPException(
                    status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large: {file_size_mb:.2f}MB (max {settings.max_file_size_mb}MB)",
                )

            await f.write(chunk)

    return file_path


def create_task_in_redis(task_id: uuid.UUID, request_params: Dict[str, Any]) -> None:
    """
    Create initial task record in Redis.

    Note: Placeholder stub. Will be implemented when Redis integration is complete.

    Args:
        task_id: Task identifier
        request_params: Processing parameters
    """
    pass


def enqueue_job(
    task_id: uuid.UUID, file_path: Path, request_params: Dict[str, Any]
) -> None:
    """
    Enqueue processing job to Redis queue.

    Note: Placeholder stub. Will be implemented when RQ integration is complete.

    Args:
        task_id: Task identifier
        file_path: Path to audio file
        request_params: Processing parameters
    """
    pass


class ProcessController(Controller):
    """Controller for processing endpoints."""

    @post(
        "/v1/process",
        summary="Process audio file",
        description="Submit an audio file for processing using specified parameters",
        tags=["Processing"],
        status_code=HTTP_202_ACCEPTED,
    )
    async def process_audio(
        self,
        request: Request,
    ) -> ProcessResponse:
        """
        Process an audio file with the specified parameters.

        Args:
            request: Litestar request object

        Returns:
            Response with task ID
        """
        # Parse multipart form data
        form_data = await request.form()

        # Extract and validate file
        file = form_data.get("file")
        if not file or not isinstance(file, UploadFile):
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing or invalid 'file' field",
            )

        # SECURITY: Validate BOTH file extension AND MIME type
        allowed_extensions = {".wav", ".mp3", ".m4a", ".flac"}
        allowed_mime_types = {
            "audio/wav",
            "audio/wave",
            "audio/x-wav",
            "audio/mpeg",
            "audio/mp4",
            "audio/x-m4a",
            "audio/flac",
        }

        file_ext = Path(file.filename).suffix.lower() if file.filename else ""
        content_type = (file.content_type or "").lower()

        # Check extension
        extension_valid = file_ext in allowed_extensions

        # Check MIME type
        mime_valid = content_type.startswith("audio/") or any(
            mime in content_type for mime in allowed_mime_types
        )

        # Both should be valid (or at least one with the other being empty/unknown)
        if not extension_valid:
            if not mime_valid:
                raise HTTPException(
                    status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported file type: {file.filename} (extension: {file_ext}, mime: {content_type})",
                )

        # Additional check: if MIME type is explicitly NOT audio, reject even if extension is valid
        if (
            content_type
            and not content_type.startswith("audio/")
            and content_type not in allowed_mime_types
        ):
            raise HTTPException(
                status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File MIME type must be audio/*, got: {content_type}",
            )

        # Parse other form fields FIRST (before streaming file)
        features_raw = form_data.get("features", '["clean_transcript", "summary"]')
        if isinstance(features_raw, str):
            features = json.loads(features_raw)
        else:
            features = features_raw

        template_id = form_data.get("template_id")

        # Validate template_id if summary is requested
        if "summary" in features or Feature.SUMMARY.value in features:
            if not template_id:
                raise HTTPException(
                    status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="template_id is required when summary is in features",
                )

        # Check queue capacity (backpressure)
        if not check_queue_depth():
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Queue is full. Please try again later.",
            )

        # Generate task ID
        task_id = uuid.uuid4()

        # Save audio file using streaming (validates size during streaming)
        file_path = await save_audio_file_streaming(file, task_id)

        # Create task record in Redis
        request_params = {
            "features": features,
            "template_id": template_id,
            "file_path": str(file_path),
        }
        create_task_in_redis(task_id, request_params)

        # Enqueue job
        enqueue_job(task_id, file_path, request_params)

        return ProcessResponse(task_id=task_id)


def get_task_from_redis(task_id: uuid.UUID) -> Dict[str, Any] | None:
    """
    Retrieve task data from Redis.

    Note: Placeholder stub. Will be implemented when Redis integration is complete.

    Args:
        task_id: Task identifier

    Returns:
        Task data dictionary or None if not found
    """
    return None


def get_available_models() -> ModelsResponseSchema:
    """
    Get list of available ASR models/backends.

    Note: Currently returns hardcoded whisper backend. Will be implemented with ASRFactory
    when processor modules are complete.

    Returns:
        ModelsResponseSchema with models list
    """
    # Placeholder - hardcoded default whisper backend
    models = [
        ModelInfoSchema(
            id="whisper",
            name="Whisper (CTranslate2)",
            description="Default ASR backend with native punctuation",
            type="ASR",
            version="erax-wow-turbo-v1.1",
            supported_languages=["en", "vi", "zh", "ja", "ko"],
        )
    ]
    return ModelsResponseSchema(models=models)


def scan_templates_directory() -> TemplatesResponseSchema:
    """
    Scan templates directory for available templates.

    Returns:
        TemplatesResponseSchema with templates list
    """
    templates = []

    # Scan templates directory
    if settings.templates_dir.exists():
        for template_file in settings.templates_dir.glob("*.json"):
            template_id = template_file.stem

            # Try to load template for metadata
            try:
                with open(template_file, "r") as f:
                    template_data = json.load(f)

                templates.append(
                    TemplateInfoSchema(
                        id=template_id,
                        name=template_data.get("title", template_id),
                        description=template_data.get("description", ""),
                        schema_url=f"/v1/templates/{template_id}/schema",
                        parameters={},
                    )
                )
            except Exception:
                # Skip malformed templates
                continue

    return TemplatesResponseSchema(templates=templates)


class StatusController(Controller):
    """Controller for status checking endpoints."""

    @get(
        "/v1/status/{task_id:uuid}",
        summary="Get processing status",
        description="Check the status of a processing task by its ID",
        tags=["Status"],
    )
    async def get_status(
        self,
        task_id: uuid.UUID = Parameter(..., description="UUID of the processing task"),
    ) -> StatusResponseSchema:
        """
        Get the current status of a processing task.

        Args:
            task_id: Unique identifier of the processing task

        Returns:
            Status information for the specified task
        """
        # Retrieve task from Redis
        task_data = get_task_from_redis(task_id)

        if not task_data:
            raise NotFoundException(f"Task {task_id} not found")

        if isinstance(task_data, StatusResponseSchema):
            return task_data
        if isinstance(task_data, dict):
            return StatusResponseSchema(**task_data)
        # Fallback: attempt to coerce from pydantic-compatible object
        return StatusResponseSchema.model_validate(task_data)


class ModelsController(Controller):
    """Controller for models endpoints."""

    @get(
        "/v1/models",
        summary="Get available models",
        description="Retrieve a list of available audio processing models",
        tags=["Models"],
    )
    async def get_models(self) -> ModelsResponseSchema | Dict[str, Any]:
        """
        Get a list of available audio processing models.

        Returns:
            List of available models with their information
        """
        models = get_available_models()
        if isinstance(models, ModelsResponseSchema):
            return models
        if isinstance(models, dict):
            return models
        return ModelsResponseSchema.model_validate(models)


class TemplatesController(Controller):
    """Controller for templates endpoints."""

    @get(
        "/v1/templates",
        summary="Get available templates",
        description="Retrieve a list of available processing templates",
        tags=["Templates"],
    )
    async def get_templates(self) -> TemplatesResponseSchema | Dict[str, Any]:
        """
        Get a list of available processing templates.

        Returns:
            List of available templates with their information
        """
        templates = scan_templates_directory()
        if isinstance(templates, TemplatesResponseSchema):
            return templates
        if isinstance(templates, dict):
            return templates
        return TemplatesResponseSchema.model_validate(templates)


# Define route handlers for the app
route_handlers: List = [
    ProcessController,
    StatusController,
    ModelsController,
    TemplatesController,
]

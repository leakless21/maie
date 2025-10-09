"""Route definitions for the MAIE API."""

import json
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
from src.config import Settings, settings

# Provide a patchable class attribute for tests that monkeypatch Settings.max_file_size_mb.
if not hasattr(Settings, "max_file_size_mb"):  # pragma: no cover - defensive shim for tests
    Settings.max_file_size_mb = settings.max_file_size_mb  # type: ignore


def check_queue_depth() -> bool:
    """
    Check if the queue has capacity for more jobs.
    
    Returns:
        True if queue has capacity, False if full
    """
    # TODO: Implement actual Redis queue depth check
    # For now, always return True
    return True


async def save_audio_file(file: UploadFile, task_id: uuid.UUID, content: bytes) -> Path:
    """
    Save uploaded audio file to disk.
    
    Args:
        file: Uploaded audio file
        task_id: Task identifier for naming
        content: File content bytes
        
    Returns:
        Path to saved file
    """
    # Determine file extension
    ext = Path(file.filename).suffix if file.filename else ".wav"
    file_path = settings.audio_dir / f"{task_id}{ext}"
    
    # Ensure directory exists
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(content)
    
    return file_path


def create_task_in_redis(task_id: uuid.UUID, request_params: Dict[str, Any]) -> None:
    """
    Create initial task record in Redis.
    
    Args:
        task_id: Task identifier
        request_params: Processing parameters
    """
    # TODO: Implement Redis task creation
    # For now, this is a placeholder
    pass


def enqueue_job(task_id: uuid.UUID, file_path: Path, request_params: Dict[str, Any]) -> None:
    """
    Enqueue processing job to Redis queue.
    
    Args:
        task_id: Task identifier
        file_path: Path to audio file
        request_params: Processing parameters
    """
    # TODO: Implement RQ job enqueueing
    # For now, this is a placeholder
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
                detail="Missing or invalid 'file' field"
            )
        
        # Validate file type
        allowed_extensions = {".wav", ".mp3", ".m4a", ".flac"}
        file_ext = Path(file.filename).suffix.lower() if file.filename else ""
        
        if file_ext not in allowed_extensions:
            content_type = file.content_type or ""
            if not content_type.startswith("audio/"):
                raise HTTPException(
                    status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported file type: {file.filename}"
                )
        
        # Validate file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        max_size_mb = getattr(settings, "max_file_size_mb", getattr(Settings, "max_file_size_mb", settings.max_file_size_mb))
        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large: {file_size_mb:.2f}MB (max {max_size_mb}MB)"
            )
        
        # Parse other form fields
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
                    detail="template_id is required when summary is in features"
                )
        
        # Check queue capacity (backpressure)
        if not check_queue_depth():
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Queue is full. Please try again later."
            )
        
        # Generate task ID
        task_id = uuid.uuid4()
        
        # Save audio file
        file_path = await save_audio_file(file, task_id, file_content)
        
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
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task data dictionary or None if not found
    """
    # TODO: Implement actual Redis retrieval
    # For now, return None
    return None


def get_available_models() -> ModelsResponseSchema:
    """
    Get list of available ASR models/backends.
    
    Returns:
        ModelsResponseSchema with models list
    """
    # TODO: Implement actual model discovery from ASRFactory
    # For now, return default whisper backend
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
        tags=["Status"]
    )
    async def get_status(
        self,
        task_id: uuid.UUID = Parameter(
            ..., 
            description="UUID of the processing task"
        )
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
        tags=["Models"]
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
        tags=["Templates"]
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

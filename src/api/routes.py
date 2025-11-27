"""Route definitions for the MAIE API."""

import json
import logging.config
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List

from litestar import Controller, Request, get, post, put, delete
from litestar.datastructures import UploadFile
from litestar.di import Provide
from litestar.enums import RequestEncodingType
from litestar.exceptions import HTTPException, NotFoundException
from litestar.params import Body
from litestar.params import Parameter
from litestar.status_codes import (
    HTTP_202_ACCEPTED,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_429_TOO_MANY_REQUESTS,
)
from pydantic import ValidationError

from src.api.errors import (
    APIValidationError,
    AudioValidationError,
)

from src.api.schemas import (
    Feature,
    ModelInfoSchema,
    ModelsResponseSchema,
    ProcessRequestSchema,
    ProcessResponse,
    StatusResponseSchema,
    TaskStatus,
    TemplateInfoSchema,
    TemplatesResponseSchema,
    TemplateCreateSchema,
    TemplateUpdateSchema,
    TemplateDetailSchema,
)
from src.api.dependencies import api_key_guard, get_template_manager
from src.config import settings
from src.config.logging import get_module_logger, correlation_id as _cid_var
from src.utils.sanitization import sanitize_filename
from src.utils.sanitization import sanitize_filename
from src.utils.json_utils import safe_json_loads
from src.utils.template_manager import TemplateManager

# Create module-bound logger for better debugging
logger = get_module_logger(__name__)


def _patch_logging_queue_listener() -> None:
    """Ensure Litestar logging config uses ext:// notation for queue listeners."""

    if getattr(logging.config, "_maie_queue_listener_patched", False):
        return

    original_dict_config = logging.config.dictConfig

    def dict_config_patched(config, *args, **kwargs):
        handlers = config.get("handlers", {}) if isinstance(config, dict) else {}
        for handler in handlers.values():
            listener = handler.get("listener")
            if listener == "litestar.logging.standard.LoggingQueueListener":
                handler["listener"] = (
                    "ext://litestar.logging.standard.LoggingQueueListener"
                )
        return original_dict_config(config, *args, **kwargs)

    logging.config.dictConfig = dict_config_patched  # type: ignore[assignment]
    setattr(logging.config, "_maie_queue_listener_patched", True)


_patch_logging_queue_listener()


def check_queue_depth() -> bool:
    """Check if the queue has capacity for more jobs."""
    from src.api.dependencies import get_rq_queue

    try:
        queue = get_rq_queue()
        current_depth = queue.count
        return current_depth < settings.redis.max_queue_depth
    except Exception:
        # Fail open for availability
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

    # SECURITY: Use per-task UUID directory and fixed base name, not user input
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

    # Build per-task directory and raw file path
    task_dir = settings.paths.audio_dir / str(task_id)
    file_path = task_dir / f"raw{ext}"

    # Ensure directories exist
    task_dir.mkdir(parents=True, exist_ok=True)

    # Stream file content to disk using aiofiles with size validation
    total_size = 0
    max_size_bytes = settings.api.max_file_size_mb * 1024 * 1024
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
                error = AudioValidationError(
                    message=f"File too large: {file_size_mb:.2f}MB (max {settings.api.max_file_size_mb}MB)",
                    details={
                        "file_size_mb": file_size_mb,
                        "max_size_mb": settings.api.max_file_size_mb,
                        "filename": file.filename,
                    },
                )
                raise HTTPException(
                    status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=error.message,
                )

            await f.write(chunk)

    return file_path


async def create_task_in_redis(
    task_id: uuid.UUID, request_params: Dict[str, Any]
) -> None:
    """Create initial task record in Redis results DB."""
    from src.api.dependencies import get_results_redis

    redis_client = await get_results_redis()
    try:
        task_key = f"task:{task_id}"
        template_value = request_params.get("template_id") or "generic_summary_v2"
        file_path_value = request_params.get("file_path") or ""
        if isinstance(file_path_value, Path):
            file_path_value = str(file_path_value)
        asr_backend_value = request_params.get("asr_backend") or "whisper"
        features_value = request_params.get("features", ["summary"])
        features_json = json.dumps(features_value)
        task_data = {
            "task_id": str(task_id),
            "status": TaskStatus.PENDING.value,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "features": features_json,
            "template_id": template_value,
            "file_path": file_path_value,
            "asr_backend": asr_backend_value,
            # Store correlation ID for traceability in results DB
            "correlation_id": _cid_var.get() or "",
            # VAD parameters
            "enable_vad": request_params.get("enable_vad"),
            "vad_threshold": request_params.get("vad_threshold"),
        }
        # Redis 7.x is stricter - filter out None values before hset
        task_data = {k: v for k, v in task_data.items() if v is not None}
        await redis_client.hset(task_key, mapping=task_data)
        await redis_client.expire(task_key, settings.worker.result_ttl)
    finally:
        await redis_client.aclose()


def enqueue_job(
    task_id: uuid.UUID, file_path: Path, request_params: Dict[str, Any]
) -> None:
    """Enqueue processing job to Redis queue (DB 0)."""
    from src.api.dependencies import get_rq_queue

    queue = get_rq_queue()

    job_func = "src.worker.pipeline.process_audio_task"

    task_params = {
        "task_id": str(task_id),
        "audio_path": str(file_path),
        "features": request_params.get("features", ["summary"]),
        "template_id": request_params.get("template_id"),
        "asr_backend": request_params.get("asr_backend", "whisper"),
        "enable_diarization": request_params.get("enable_diarization", False),
        "enable_vad": request_params.get("enable_vad"),
        "vad_threshold": request_params.get("vad_threshold"),
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": settings.redis.results_db,
        # Propagate correlation ID to worker for consistent logs
        "correlation_id": _cid_var.get() or None,
    }

    queue.enqueue(
        job_func,
        task_params,
        job_id=str(task_id),
        job_timeout=settings.worker.job_timeout,
        result_ttl=settings.worker.result_ttl,
    )


class ProcessController(Controller):
    """Controller for processing endpoints."""

    @post(
        "/v1/process",
        guards=[api_key_guard],
        summary="Process audio file",
        description=(
            "Submit an audio file for processing using specified parameters.\n\n"
            "This endpoint accepts multipart/form-data with an audio file and optional processing parameters. "
            "The request is processed asynchronously, and a task ID is returned immediately.\n\n"
            "**Request Body Parameters (multipart/form-data):**\n\n"
            "- **file** (required, binary): Audio file to process\n"
            "  - Supported formats: WAV, MP3, M4A, FLAC\n"
            "  - Maximum size: 100MB\n"
            "  - Content-Type: audio/* or application/octet-stream\n\n"
            "- **features** (optional, array of strings): List of desired outputs\n"
            "  - Available: `raw_transcript`, `clean_transcript`, `summary`, `enhancement_metrics`\n"
            "  - Default: `['clean_transcript', 'summary']`\n"
            "  - Send multiple values as separate form fields\n\n"
            "- **template_id** (optional, string): Summary format template ID\n"
            "  - Required if `summary` is included in features\n"
            "  - Use `/v1/templates` to get available template IDs\n"
            "  - Example: `meeting_notes_v2`\n\n"
            "- **asr_backend** (optional, string): ASR backend selection\n"
            "  - Available: `whisper` (default), `chunkformer`\n"
            "  - Use `/v1/models` to get available backends\n\n"
            "- **enable_diarization** (optional, boolean): Enable speaker diarization\n"
            "  - Default: `false`\n"
            "  - When enabled, identifies and labels different speakers in the audio\n"
            "  - Requires word-level timestamps from ASR (automatically enabled with Whisper)\n\n"
            "**Examples:**\n\n"
            "Basic processing (default settings):\n"
            "```bash\n"
            "curl -X POST 'http://localhost:8000/v1/process' \\\n"
            "  -H 'X-API-Key: <your_api_key>' \\\n"
            "  -F 'file=@/path/to/audio.mp3' \\\n"
            "  -F 'features=clean_transcript' \\\n"
            "  -F 'features=summary' \\\n"
            "  -F 'template_id=meeting_notes_v2'\n"
            "```\n\n"
            "Raw transcript only:\n"
            "```bash\n"
            "curl -X POST 'http://localhost:8000/v1/process' \\\n"
            "  -H 'X-API-Key: <your_api_key>' \\\n"
            "  -F 'file=@/path/to/audio.wav' \\\n"
            "  -F 'features=raw_transcript'\n"
            "```\n\n"
            "All features with ChunkFormer backend:\n"
            "```bash\n"
            "curl -X POST 'http://localhost:8000/v1/process' \\\n"
            "  -H 'X-API-Key: <your_api_key>' \\\n"
            "  -F 'file=@/path/to/audio.m4a' \\\n"
            "  -F 'features=raw_transcript' \\\n"
            "  -F 'features=clean_transcript' \\\n"
            "  -F 'features=summary' \\\n"
            "  -F 'features=enhancement_metrics' \\\n"
            "  -F 'template_id=meeting_notes_v2' \\\n"
            "  -F 'asr_backend=chunkformer'\n"
            "```\n\n"
            "With speaker diarization enabled:\n"
            "```bash\n"
            "curl -X POST 'http://localhost:8000/v1/process' \\\n"
            "  -H 'X-API-Key: <your_api_key>' \\\n"
            "  -F 'file=@/path/to/meeting.wav' \\\n"
            "  -F 'features=clean_transcript' \\\n"
            "  -F 'features=summary' \\\n"
            "  -F 'template_id=meeting_notes_v2' \\\n"
            "  -F 'asr_backend=whisper' \\\n"
            "  -F 'enable_diarization=true'\n"
            "```\n\n"
            "**Response:**\n\n"
            "Returns HTTP 202 Accepted with a task ID for tracking the processing status.\n\n"
            "**Error Responses:**\n\n"
            "- 413: File too large (exceeds 100MB)\n"
            "- 415: Unsupported media type\n"
            "- 422: Validation error (missing required fields, invalid parameters)\n"
            "- 429: Queue is full, try again later"
        ),
        tags=["Processing"],
        status_code=HTTP_202_ACCEPTED,
    )
    async def process_audio(
        self,
        request: Request,
        data: Annotated[
            Dict[str, Any],
            Body(media_type=RequestEncodingType.MULTI_PART),
        ],
    ) -> ProcessResponse:
        """
        Process an audio file with the specified parameters.

        Args:
            request: Litestar request object
            data: Parsed multipart request payload containing file and parameters

        Returns:
            Response with task ID
        """
        prepared_data = dict(data)
        features_value = prepared_data.get("features")
        # Use consolidated JSON utility for safe JSON parsing with fallback handling
        if isinstance(features_value, str):
            # Handle JSON string format
            if features_value.startswith("[") and features_value.endswith("]"):
                parsed = safe_json_loads(features_value, default=[features_value])
                if isinstance(parsed, list):
                    prepared_data["features"] = parsed
                else:
                    prepared_data["features"] = [features_value]
            else:
                prepared_data["features"] = [features_value]
        elif (
            isinstance(features_value, list)
            and len(features_value) == 1
            and isinstance(features_value[0], str)
            and features_value[0].startswith("[")
        ):
            # Handle case where multipart parser wraps JSON string in a list
            parsed = safe_json_loads(features_value[0], default=None)
            if parsed is not None and isinstance(parsed, list):
                prepared_data["features"] = parsed
        try:
            payload = ProcessRequestSchema.model_validate(prepared_data)
        except ValidationError as exc:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        # Extract fields from schema
        file = payload.file
        features_raw = payload.features
        template_id = payload.template_id
        asr_backend = payload.asr_backend or "whisper"

        # Normalize and validate asr_backend
        from src.processors.asr.factory import ASRFactory

        if isinstance(asr_backend, str):
            asr_backend = asr_backend.strip().lower()
        else:
            asr_backend = "whisper"

        allowed = set(ASRFactory.BACKENDS.keys())
        if asr_backend not in allowed:
            error = APIValidationError(
                message="Invalid 'asr_backend'",
                details={"received": asr_backend, "allowed": sorted(allowed)},
            )
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error.message,
            )

        # Normalize features to list - support both JSON array and repeated fields
        if (
            isinstance(features_raw, str)
            and features_raw.startswith("[")
            and features_raw.endswith("]")
        ):
            # Use consolidated JSON utility for safe parsing with a sensible fallback
            features_raw = safe_json_loads(features_raw, default=[features_raw])
        elif not isinstance(features_raw, list):
            features_raw = [features_raw] if features_raw else []

        # Parse features into Feature enum
        if features_raw:
            features = [Feature(f) if isinstance(f, str) else f for f in features_raw]
        else:
            features = [Feature.CLEAN_TRANSCRIPT, Feature.SUMMARY]

        # Validate file present and type
        if not file or not isinstance(file, UploadFile):
            error = APIValidationError(
                message="Missing or invalid 'file' field",
                details={"field": "file", "expected_type": "UploadFile"},
            )
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error.message,
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
                error = AudioValidationError(
                    message=f"Unsupported file type: {file.filename}",
                    details={
                        "filename": file.filename,
                        "extension": file_ext,
                        "mime_type": content_type,
                        "allowed_extensions": list(allowed_extensions),
                        "allowed_mime_types": list(allowed_mime_types),
                    },
                )
                raise HTTPException(
                    status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=error.message,
                )

        # Additional check: if MIME type is explicitly NOT audio, reject even if extension is valid
        # Exception: allow application/octet-stream and text/plain if extension is valid (common for multipart uploads)
        if (
            content_type
            and not content_type.startswith("audio/")
            and content_type not in allowed_mime_types
            and content_type not in ["application/octet-stream", "text/plain"]
        ):
            error = AudioValidationError(
                message=f"File MIME type must be audio/*, got: {content_type}",
                details={
                    "filename": file.filename,
                    "mime_type": content_type,
                    "expected_mime_prefix": "audio/",
                },
            )
            raise HTTPException(
                status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=error.message,
            )

        # Normalize features to string values for downstream checks
        normalized_features = [
            f.value if isinstance(f, Feature) else f for f in features
        ]

        # Validate template_id if summary is requested
        if (
            "summary" in normalized_features
            or Feature.SUMMARY.value in normalized_features
        ):
            if not template_id:
                error = APIValidationError(
                    message="template_id is required when summary is in features",
                    details={
                        "features": normalized_features,
                        "required_field": "template_id",
                        "condition": "summary feature requested",
                    },
                )
                raise HTTPException(
                    status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=error.message,
                )

        # Check queue capacity (backpressure)
        if not check_queue_depth():
            error = APIValidationError(
                message="Queue is full. Please try again later.",
                details={
                    "queue_status": "full",
                    "retry_after": "Please wait and retry",
                },
            )
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail=error.message,
            )

        # Generate task ID
        task_id = uuid.uuid4()

        # Save audio file using streaming (validates size during streaming)
        file_path = await save_audio_file_streaming(file, task_id)

        # Create task record in Redis
        request_params = {
            "features": normalized_features,
            "template_id": template_id,
            "file_path": str(file_path),
            "asr_backend": asr_backend,
            "enable_diarization": payload.enable_diarization,
        }
        await create_task_in_redis(task_id, request_params)

        # Enqueue job
        enqueue_job(task_id, file_path, request_params)

        return ProcessResponse(task_id=task_id, status="PENDING")


async def get_task_from_redis(task_id: uuid.UUID) -> Dict[str, Any] | None:
    """Retrieve task data from Redis results DB."""
    from src.api.dependencies import get_results_redis

    redis_client = await get_results_redis()
    try:
        task_key = f"task:{task_id}"
        task_data = await redis_client.hgetall(task_key)

        if not task_data:
            return None

        # Deserialize JSON fields using consolidated utils
        for field in ["features", "results", "metrics", "versions"]:
            if field in task_data and task_data[field]:
                task_data[field] = safe_json_loads(task_data[field], default=None)

        return task_data
    finally:
        await redis_client.aclose()


def get_available_models() -> ModelsResponseSchema:
    """Get list of available ASR models/backends from ASRFactory."""
    from src.processors.asr.factory import ASRFactory

    models = []

    for backend_name, backend_class in ASRFactory.BACKENDS.items():
        model_info = ModelInfoSchema(
            id=backend_name,
            name=f"{backend_name.title()} Backend",
            description=f"ASR backend using {backend_name}",
            type="ASR",
            version="1.0",
            supported_languages=["en", "vi", "zh", "ja", "ko"],
        )
        models.append(model_info)

    return ModelsResponseSchema(models=models)


def scan_templates_directory() -> TemplatesResponseSchema:
    """
    Discover templates by scanning the configured templates directory.

    Rules:
    - Any subdirectory under `<templates_dir>` is considered a template bundle if it contains `schema.json`
    - Template ID = directory name
    - `name` comes from schema.title if present, otherwise prettified ID
    - `description` comes from schema.description if present, otherwise a default
    - `example` is loaded from `<templates_dir>/{id}/example.json` if it exists

    Returns:
        TemplatesResponseSchema: List of discovered templates.
    """
    templates: List[TemplateInfoSchema] = []
    templates_dir = settings.paths.templates_dir

    try:
        # Scan for subdirectories
        template_dirs = sorted(p for p in templates_dir.iterdir() if p.is_dir())
    except Exception as e:
        logger.error(f"Failed to scan templates directory {templates_dir}: {e}")
        return TemplatesResponseSchema(templates=[])

    for bundle_dir in template_dirs:
        template_id = bundle_dir.name
        
        # Skip hidden directories or non-template dirs (e.g. schemas/prompts/examples if they still exist)
        if template_id.startswith(".") or template_id in ["schemas", "prompts", "examples"]:
            continue
            
        schema_path = bundle_dir / "schema.json"
        if not schema_path.exists():
            continue

        try:
            with schema_path.open("r", encoding="utf-8") as f:
                schema_data = json.load(f)
        except Exception as e:
            logger.error(
                "Failed to load schema",
                extra={
                    "template_id": template_id,
                    "path": str(schema_path),
                    "error": str(e),
                },
            )
            continue

        # Derive name/description
        raw_name = schema_data.get("title") or template_id.replace("_", " ").title()
        description = schema_data.get(
            "description",
            "Auto-discovered template based on JSON schema.",
        )

        # Load example if available
        example: Dict[str, Any] | None = None
        example_path = bundle_dir / "example.json"
        if example_path.exists():
            try:
                with example_path.open("r", encoding="utf-8") as ef:
                    example = json.load(ef)
            except Exception as e:
                logger.warning(
                    "Failed to load example JSON",
                    extra={"template_id": template_id, "error": str(e)},
                )

        templates.append(
            TemplateInfoSchema(
                id=template_id,
                name=raw_name,
                description=description,
                schema_url=f"/v1/templates/{template_id}/schema",
                parameters=schema_data.get("properties", {}),
                example=example,
            )
        )

    return TemplatesResponseSchema(templates=templates)


class StatusController(Controller):
    """Controller for status checking endpoints."""

    @get(
        "/v1/status/{task_id:uuid}",
        guards=[api_key_guard],
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
        task_data = await get_task_from_redis(task_id)

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
    path = "/v1/templates"
    dependencies = {"manager": Provide(get_template_manager, sync_to_thread=True)}

    @get(
        summary="Get available templates",
        description="Retrieve a list of available processing templates",
        tags=["Templates"],
    )
    async def get_templates(self, manager: TemplateManager) -> TemplatesResponseSchema | Dict[str, Any]:
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

    @get(
        "/{template_id:str}/schema",
        summary="Get template schema",
        description="Return the JSON Schema for a given template ID",
        tags=["Templates"],
    )
    async def get_template_schema(self, template_id: str) -> Dict[str, Any]:
        """
        Serve the raw JSON schema for a template.

        Args:
            template_id: Template identifier (filename stem under templates/schemas)

        Returns:
            The JSON schema as a dictionary.
        """
        # Prevent path traversal by allowing only safe characters in ID
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", template_id):
            raise NotFoundException("Invalid template ID")

        schema_path = settings.paths.templates_dir / template_id / "schema.json"
        if not schema_path.exists() or not schema_path.is_file():
            raise NotFoundException(f"Schema not found for template: {template_id}")

        try:
            with schema_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid schema JSON for template {template_id}: {e}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Failed to load schema for template {template_id}: {e}",
            )

    async def _get_template_detail_logic(self, template_id: str, manager: TemplateManager) -> TemplateDetailSchema:
        """Helper to get template details."""
        try:
            content = await manager.get_template_content(template_id)
        except FileNotFoundError:
            raise NotFoundException(f"Template {template_id} not found")

        # Map content to schema
        schema_data = content["schema"]
        raw_name = schema_data.get("title") or template_id.replace("_", " ").title()
        description = schema_data.get("description", "Template")

        return TemplateDetailSchema(
            id=template_id,
            name=raw_name,
            description=description,
            schema_url=f"/v1/templates/{template_id}/schema",
            parameters={},
            example=content.get("example"),
            prompt_template=content["prompt"],
            schema_data=schema_data,
        )

    @get(
        "/{template_id:str}",
        summary="Get template details",
        description="Get full details of a template including prompt and schema",
        tags=["Templates"],
    )
    async def get_template_detail(self, template_id: str, manager: TemplateManager) -> TemplateDetailSchema:
        """
        Get full details of a template.
        """
        return await self._get_template_detail_logic(template_id, manager)

    @post(
        "/",
        guards=[api_key_guard],
        summary="Create template",
        description="Create a new processing template",
        tags=["Templates"],
    )
    async def create_template(self, data: TemplateCreateSchema, manager: TemplateManager) -> TemplateDetailSchema:
        """
        Create a new template.
        """
        try:
            await manager.create_template(
                template_id=data.id,
                schema=data.schema_data,
                prompt=data.prompt_template,
                example=data.example,
            )
        except FileExistsError:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Template {data.id} already exists",
            )
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return await self._get_template_detail_logic(data.id, manager)

    @put(
        "/{template_id:str}",
        guards=[api_key_guard],
        summary="Update template",
        description="Update an existing processing template",
        tags=["Templates"],
    )
    async def update_template(
        self, template_id: str, data: TemplateUpdateSchema, manager: TemplateManager
    ) -> TemplateDetailSchema:
        """
        Update an existing template.
        """
        if not manager.exists(template_id):
            raise NotFoundException(f"Template {template_id} not found")

        try:
            await manager.update_template(
                template_id=template_id,
                schema=data.schema_data,
                prompt=data.prompt_template,
                example=data.example,
            )
        except Exception as e:
            logger.error(f"Failed to update template: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return await self._get_template_detail_logic(template_id, manager)

    @delete(
        "/{template_id:str}",
        guards=[api_key_guard],
        summary="Delete template",
        description="Delete a processing template",
        tags=["Templates"],
    )
    async def delete_template(self, template_id: str, manager: TemplateManager) -> None:
        """
        Delete a template.
        """
        if not manager.exists(template_id):
            raise NotFoundException(f"Template {template_id} not found")

        try:
            await manager.delete_template(template_id)
        except Exception as e:
            logger.error(f"Failed to delete template: {e}")
            raise HTTPException(status_code=500, detail=str(e))
# Define route handlers for the app
route_handlers: List = [
    ProcessController,
    StatusController,
    ModelsController,
    TemplatesController,
]

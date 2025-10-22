"""Route definitions for the MAIE API."""

import json
import logging.config
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List

from litestar import Controller, Request, get, post
from litestar.datastructures import UploadFile
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
)
from src.api.dependencies import api_key_guard
from src.config import settings
from src.config.logging import get_module_logger

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
                handler["listener"] = "ext://litestar.logging.standard.LoggingQueueListener"
        return original_dict_config(config, *args, **kwargs)

    logging.config.dictConfig = dict_config_patched  # type: ignore[assignment]
    setattr(logging.config, "_maie_queue_listener_patched", True)


_patch_logging_queue_listener()


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
    file_path = settings.paths.audio_dir / f"{task_id}{ext}"

    # Ensure directory exists
    settings.paths.audio_dir.mkdir(parents=True, exist_ok=True)

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
        task_data = {
            "task_id": str(task_id),
            "status": TaskStatus.PENDING.value,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "features": json.dumps(request_params.get("features", ["summary"])),
            "template_id": request_params.get("template_id", "generic_summary_v1"),
            "file_path": request_params.get("file_path", ""),
            "asr_backend": request_params.get("asr_backend", "chunkformer"),
        }
        await redis_client.hset(task_key, mapping=task_data)
        await redis_client.expire(task_key, settings.worker.result_ttl)
    finally:
        await redis_client.aclose()


def enqueue_job(
    task_id: uuid.UUID, file_path: Path, request_params: Dict[str, Any]
) -> None:
    """Enqueue processing job to Redis queue (DB 0)."""
    from src.api.dependencies import get_rq_queue
    from src.worker.pipeline import process_audio_task

    queue = get_rq_queue()

    task_params = {
        "task_id": str(task_id),
        "audio_path": str(file_path),
        "features": request_params.get("features", ["summary"]),
        "template_id": request_params.get("template_id"),
        "asr_backend": request_params.get("asr_backend", "chunkformer"),
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": settings.redis.results_db,
    }

    queue.enqueue(
        process_audio_task,
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
            "  - Example: `meeting_notes_v1`\n\n"
            "- **asr_backend** (optional, string): ASR backend selection\n"
            "  - Available: `whisper` (default), `chunkformer`\n"
            "  - Use `/v1/models` to get available backends\n\n"
            "**Examples:**\n\n"
            "Basic processing (default settings):\n"
            "```bash\n"
            "curl -X POST 'http://localhost:8000/v1/process' \\\n"
            "  -H 'X-API-Key: <your_api_key>' \\\n"
            "  -F 'file=@/path/to/audio.mp3' \\\n"
            "  -F 'features=clean_transcript' \\\n"
            "  -F 'features=summary' \\\n"
            "  -F 'template_id=meeting_notes_v1'\n"
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
            "  -F 'template_id=meeting_notes_v1' \\\n"
            "  -F 'asr_backend=chunkformer'\n"
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
        if isinstance(features_value, str):
            # Handle JSON string format
            if features_value.startswith('[') and features_value.endswith(']'):
                try:
                    import json
                    parsed = json.loads(features_value)
                    if isinstance(parsed, list):
                        prepared_data["features"] = parsed
                    else:
                        prepared_data["features"] = [features_value]
                except json.JSONDecodeError:
                    prepared_data["features"] = [features_value]
            else:
                prepared_data["features"] = [features_value]
        elif isinstance(features_value, list) and len(features_value) == 1 and isinstance(features_value[0], str) and features_value[0].startswith('['):
            # Handle case where multipart parser wraps JSON string in a list
            try:
                import json
                parsed = json.loads(features_value[0])
                if isinstance(parsed, list):
                    prepared_data["features"] = parsed
            except json.JSONDecodeError:
                pass
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
        asr_backend = payload.asr_backend or "chunkformer"

        # Normalize and validate asr_backend
        from src.processors.asr.factory import ASRFactory
        if isinstance(asr_backend, str):
            asr_backend = asr_backend.strip().lower()
        else:
            asr_backend = "chunkformer"

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
        if isinstance(features_raw, str) and features_raw.startswith('[') and features_raw.endswith(']'):
            try:
                features_raw = json.loads(features_raw)
            except json.JSONDecodeError:
                features_raw = [features_raw]
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
        normalized_features = [f.value if isinstance(f, Feature) else f for f in features]

        # Validate template_id if summary is requested
        if "summary" in normalized_features or Feature.SUMMARY.value in normalized_features:
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

        # Deserialize JSON fields
        for field in ["features", "results", "metrics", "versions"]:
            if field in task_data and task_data[field]:
                try:
                    task_data[field] = json.loads(task_data[field])
                except (json.JSONDecodeError, TypeError):
                    pass

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
    Scans the templates directory and returns a list of available templates with Vietnamese descriptions and examples.

    Returns:
        TemplatesResponseSchema: A list of available templates.
    """
    templates_data = [
        {
            "id": "generic_summary_v1",
            "name": "Bản tóm tắt chung (v1)",
            "description": "Mẫu này cung cấp một cái nhìn tổng quan ngắn gọn về một văn bản. Nó bao gồm một bản tóm tắt ngắn, một tiêu đề và danh sách các điểm chính.",
            "example": {
              "title": "Tác động của AI lên năng suất làm việc",
              "summary": "Bài nói trình bày cách các công cụ trí tuệ nhân tạo hỗ trợ tự động hóa tác vụ lặp lại, gợi ý nội dung và cải thiện tốc độ xử lý công việc. Diễn giả nhấn mạnh tầm quan trọng của việc thiết lập quy trình kiểm duyệt để đảm bảo chất lượng và đạo đức khi áp dụng AI vào môi trường doanh nghiệp.",
              "key_topics": [
                "Tự động hóa tác vụ",
                "Gợi ý nội dung",
                "Quy trình kiểm duyệt",
                "Đạo đức trong AI"
              ],
              "tags": ["ai", "năng suất", "doanh nghiệp"]
            }
        },
        {
            "id": "interview_transcript_v1",
            "name": "Bản ghi phỏng vấn (v1)",
            "description": "Mẫu này được thiết kế để tóm tắt các cuộc phỏng vấn. Nó bao gồm một tiêu đề, một bản tóm tắt ngắn gọn của cuộc trò chuyện và danh sách các câu hỏi và câu trả lời.",
            "example": {
              "interview_summary": "Cuộc phỏng vấn với chị An, trưởng nhóm sản phẩm, xoay quanh kinh nghiệm triển khai quy trình phát hành nhanh. Chị chia sẻ về việc rút ngắn chu kỳ phát hành bằng cách tăng tự động hóa kiểm thử, chuẩn hóa tiêu chí chấp nhận và cải thiện giao tiếp giữa nhóm phát triển và vận hành.",
              "key_insights": [
                "Tự động hóa kiểm thử giúp rút ngắn chu kỳ phát hành",
                "Tiêu chí chấp nhận rõ ràng hạn chế lỗi phát sinh",
                "Tăng cường giao tiếp giữa Dev và Ops"
              ],
              "participant_sentiment": "positive",
              "tags": ["sản phẩm", "quy trình", "phỏng vấn"]
            }
        },
        {
            "id": "meeting_notes_v1",
            "name": "Ghi chú cuộc họp (v1)",
            "description": "Mẫu này dùng để tóm tắt các cuộc họp. Nó bao gồm tiêu đề cuộc họp, bản tóm tắt cuộc thảo luận, danh sách các mục hành động và các quyết định chính đã được đưa ra.",
            "example": {
              "title": "Họp dự án X – rà soát tiến độ và phân công",
              "participants": ["Minh", "Lan", "Hùng"],
              "summary": "Cuộc họp tập trung rà soát tiến độ sprint hiện tại, thống nhất phạm vi đợt phát hành sắp tới và phân công các đầu việc tiếp theo. Nhóm quyết định ưu tiên xử lý lỗi còn tồn đọng trước khi mở rộng phạm vi tính năng.",
              "agenda": [
                "Cập nhật tiến độ sprint",
                "Phạm vi phát hành",
                "Phân công công việc"
              ],
              "decisions": [
                "Ưu tiên xử lý lỗi còn tồn đọng",
                "Giữ nguyên phạm vi tính năng hiện tại"
              ],
              "action_items": [
                {"description": "Lan tổng hợp danh sách lỗi ưu tiên và chia sẻ với nhóm", "assignee": "Lan", "due_date": "2025-10-23"},
                {"description": "Hùng cập nhật test cases cho tính năng thanh toán", "assignee": "Hùng"}
              ],
              "tags": ["họp nhóm", "dự án", "kế hoạch"]
            }
        }
    ]

    templates = []
    for t_data in templates_data:
        templates.append(
            TemplateInfoSchema(
                id=t_data["id"],
                name=t_data["name"],
                description=t_data["description"],
                schema_url=f"/v1/templates/{t_data['id']}/schema",
                parameters={},
                example=t_data["example"],
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

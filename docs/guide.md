# MAIE Implementation Plan - Detailed Architecture Guide

## Project Overview

You're building a **production-grade audio intelligence system** that:

1. Accepts audio uploads via REST API
2. Transcribes audio using Whisper (ASR)
3. Enhances text using LLM (optional)
4. Generates structured summaries using LLM
5. Returns versioned, reproducible results

**Core Architecture Pattern**: Asynchronous task queue with sequential GPU processing

---

## Directory Structure

```
maie/
├── src/
│   ├── __init__.py                          # Empty, marks as package
│   ├── config.py                            # Centralized configuration
│   ├── api/
│   │   ├── __init__.py                      # Exports: create_app
│   │   ├── main.py                          # API server and routes
│   │   ├── schemas.py                       # Pydantic request/response models
│   │   └── dependencies.py                  # Dependency injection (auth, etc.)
│   ├── processors/
│   │   ├── __init__.py                      # Empty
│   │   ├── base.py                          # Abstract base classes
│   │   ├── audio/
│   │   │   ├── __init__.py                  # Exports: AudioPreprocessor
│   │   │   └── preprocessor.py              # Audio normalization logic
│   │   ├── asr/
│   │   │   ├── __init__.py                  # Exports: ASRFactory, WhisperBackend
│   │   │   ├── factory.py                   # Factory pattern for ASR backends
│   │   │   └── whisper.py                   # Whisper implementation
│   │   ├── llm/
│   │   │   ├── __init__.py                  # Exports: Enhancement/Summary processors
│   │   │   ├── base.py                      # Abstract LLM processor
│   │   │   ├── config.py                    # Generation config hierarchy
│   │   │   ├── enhancement_processor.py     # Text enhancement LLM
│   │   │   └── summary_processor.py         # Summarization LLM
│   │   └── prompt/
│   │       ├── __init__.py
│   │       ├── renderer.py
│   │       └── template_loader.py
│   └── worker/
│       ├── __init__.py                      # Empty
│       ├── main.py                          # RQ worker entrypoint
│       └── pipeline.py                      # Sequential processing pipeline
├── tests/
│   ├── conftest.py                          # Shared fixtures and mocks
│   ├── builders.py                          # Test data builders
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_schemas.py
│   │   ├── test_generation_config.py
│   │   ├── test_audio_preprocessor.py
│   │   ├── test_asr_factory.py
│   │   ├── test_enhancement_processor.py
│   │   └── test_summary_processor.py
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   ├── test_redis_operations.py
│   │   └── test_file_handling.py
│   └── e2e/
│       └── test_full_pipeline.py
├── templates/                               # JSON Schema files
│   ├── meeting_notes_v1.json
│   └── prompts/                             # Jinja2 prompt templates
│       ├── text_enhancement_v1.jinja
│       └── meeting_notes_v1.jinja
├── assets/
│   └── chat-templates/                      # Optional Jinja templates
│       └── qwen3_nonthinking.jinja
├── data/                                    # Runtime data (gitignored)
│   ├── audio/
│   ├── models/
│   └── redis/
├── scripts/
│   ├── download_models.sh
│   └── load_test.sh
├── docs/
│   ├── CONFIGURATION_HIERARCHY.md
│   ├── LLM_CONFIGURATION.md
│   ├── DEPLOYMENT.md
│   ├── OPERATIONS.md
│   └── TDD_PROGRESS.md
├── .env.template
├── .env                                     # Your actual config (gitignored)
├── .gitignore
├── .python-version                          # "3.13"
├── pyproject.toml                           # UV dependencies
├── uv.lock                                  # Locked dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Core Configuration Layer

### File: `src/config.py`

**Purpose**: Single source of truth for all configuration using Pydantic Settings v2

**Key Classes**:

1. **`Settings(BaseSettings)`**
   - **Inheritance**: Uses `pydantic_settings.BaseSettings` (NOT `pydantic.BaseSettings`)
   - **Configuration Source**: Loads from `.env` file and environment variables
   - **Case Sensitivity**: `case_sensitive=False` to match env vars like `SECRET_API_KEY`

**Core Settings Groups**:

1. **System Settings**:

   - `pipeline_version: str` - Version for NFR-1 reproducibility (default: "1.0.0")
   - `environment: Literal["development", "production"]` - Runtime environment
   - `debug: bool` - Enable debug logging
   - `secret_api_key: str` - **Required field** (no default), used for X-API-Key header

2. **API Server Settings**:

   - `api_host: str` - Bind address (default: "0.0.0.0")
   - `api_port: int` - Port number (default: 8000)
   - `max_file_size_mb: int` - Upload limit (default: 500)

3. **Redis Settings**:

   - `redis_url: str` - Connection string for queue DB (default: "redis://localhost:6379/0")
   - `redis_results_db: int` - Separate DB for results (default: 1)
   - `max_queue_depth: int` - Backpressure threshold (default: 50)

4. **ASR Settings (defaults are explicit and consistent)**:

   - Common
     - `asr_backend: Literal["whisper", "chunkformer"]` (default: "whisper")
     - `asr_word_timestamps: bool` (default: False) — if True and supported, include word-level timestamps
   - Whisper (faster-whisper)
     - `whisper_model_variant: str` (default: "erax-wow-turbo") — org default CT2 model; use an official model like "large-v3" or "distil-large-v3" if preferred
     - `whisper_beam_size: int` (default: 5)
     - `whisper_vad_filter: bool` (default: True)
     - `whisper_vad_parameters: dict | None` (default: None)
     - `whisper_compute_type: Literal["float16", "int8_float16", "int8"]` (default: "int8_float16")
     - `whisper_device: Literal["cuda", "cpu"]` (default: "cuda")
     - `whisper_condition_on_previous_text: bool` (default: True)
   - ChunkFormer
     - `chunkformer_model_name: str` (default: "khanhld/chunkformer-large-vie")
     - `chunkformer_chunk_size: int` (default: 64)
     - `chunkformer_left_context: int` (default: 128)
     - `chunkformer_right_context: int` (default: 128)
     - `chunkformer_total_batch_duration: int` seconds (default: 14400)
     - `chunkformer_return_timestamps: bool` (default: True)

5. **LLM Settings - Enhancement Task**:

   - `llm_enhancement_model: str` - Model identifier
   - `llm_enhancement_temperature: float | None` - **None = use model default**
   - `llm_enhancement_top_p: float | None`
   - `llm_enhancement_top_k: int | None`
   - `llm_enhancement_max_tokens: int | None`
   - `llm_enhancement_repetition_penalty: float | None`

6. **LLM Settings - Summarization Task**:

   - `llm_summary_model: str` - Model identifier
   - `llm_summary_temperature: float | None` - **None = use model default**
   - `llm_summary_top_p: float | None`
   - `llm_summary_top_k: int | None`
   - `llm_summary_max_tokens: int | None`
   - `llm_summary_repetition_penalty: float | None`

7. **LLM Settings - Shared**:

   - `llm_gpu_memory_utilization: float` - vLLM GPU memory fraction (default: 0.9)
   - `llm_max_model_len: int` - Context window size (default: 32768)

8. **File Paths**:

   - `audio_dir: Path` - Upload directory (default: Path("data/audio"))
   - `models_dir: Path` - Model weights (default: Path("data/models"))
   - `templates_dir: Path` - JSON schemas (default: Path("templates"))
   - `chat_templates_dir: Path` - Jinja templates (default: Path("assets/chat-templates"))

9. **Worker Settings**:
   - `worker_name: str` - Worker identifier (default: "maie-worker")
   - `job_timeout: int` - Max job execution time (default: 600)
   - `result_ttl: int` - Result retention period (default: 86400)

**Important Methods**:

1. **`__init__` override or validator**:

   - Auto-creates directories specified in path fields
   - Uses `Path.mkdir(parents=True, exist_ok=True)`

2. **`get_model_path(model_type: str) -> Path`**:

   - Returns full path: `self.models_dir / model_type`
   - Used for: "whisper", "llm", etc.

3. **`get_template_path(template_id: str) -> Path`**:
   - Returns: `self.templates_dir / f"{template_id}.json"`

**Global Instance**:

- Create singleton: `settings = Settings()`
- Import everywhere: `from src.config import settings`

**Pydantic Settings Config**:

```python
model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore"  # Ignore unknown env vars
)
```

**Field Validators**:

- Use `@field_validator` decorator for path fields to create directories
- Validate ranges (e.g., `ge=0.0, le=2.0` for temperature)

---

## API Layer

### File: `src/api/schemas.py`

**Purpose**: Define all request/response models using Pydantic v2

**Enums**:

1. **`TaskStatus(str, Enum)`**:

   - Values: PENDING, PREPROCESSING, PROCESSING_ASR, PROCESSING_LLM, COMPLETE, FAILED
   - Used for status tracking throughout pipeline

2. **`Feature(str, Enum)`**:
   - Values: `RAW_TRANSCRIPT = "raw_transcript"`, `CLEAN_TRANSCRIPT = "clean_transcript"`, `SUMMARY = "summary"`, `ENHANCEMENT_METRICS = "enhancement_metrics"`
   - User selects which outputs they want
   - **Important**: Enum members must have explicit string values (not bare identifiers)

**Request Models**:

1. **`ProcessRequest(BaseModel)`**:
   - **Fields**:
     - `features: list[Feature]` - Default: `[Feature.CLEAN_TRANSCRIPT, Feature.SUMMARY]`
     - `template_id: str | None` - Required if "summary" in features
   - **Validation**: Custom validator ensures template_id present when summary requested
   - **Config**: Include example in `model_config` for OpenAPI docs

**Response Models**:

1. **`ProcessResponse(BaseModel)`**:

   - **Fields**: `task_id: UUID`
   - Simple acknowledgment of job submission

2. **`ASRVersionInfo(BaseModel)`**:

   - **Fields**: name, model_variant, model_path, checkpoint_hash, compute_type, decoding_params (dict)
   - Used in version tracking

3. **`EnhancementLLMVersionInfo(BaseModel)`**:

   - **Fields**: name, checkpoint_hash, quantization, task (literal "text_enhancement"), chat_template, thinking, reasoning_parser, decoding_params
   - Specific to enhancement task

4. **`SummarizationLLMVersionInfo(BaseModel)`**:

   - **Fields**: Same as enhancement but task="summarization", includes structured_output field
   - Specific to summarization task

5. **`VersionInfo(BaseModel)`**:

   - **Fields**:
     - `pipeline_version: str`
     - `asr_backend: ASRVersionInfo`
     - `enhancement_llm: EnhancementLLMVersionInfo | None`
     - `summarization_llm: SummarizationLLMVersionInfo | None`
   - Complete versioning for NFR-1

6. **`Metrics(BaseModel)`**:

   - **Fields**: input_duration_seconds, processing_time_seconds, rtf, vad_coverage, asr_confidence_avg, edit_rate_cleaning
   - All runtime metrics for FR-5

7. **`TaskResult(BaseModel)`**:

   - **Fields**: raw_transcript, clean_transcript, summary (all Optional)
   - Conditional based on requested features

8. **`StatusResponse(BaseModel)`**:
   - **Fields**:
     - `task_id: UUID`
     - `status: TaskStatus`
     - `submitted_at: datetime | None`
     - `completed_at: datetime | None`
     - `versions: VersionInfo | None`
     - `metrics: Metrics | None`
     - `results: TaskResult | None`
     - `error: str | None`
   - Complete response for GET /v1/status/{task_id}

**Discovery Models**:

1. **`ModelInfo(BaseModel)`**:

   - Backend metadata: backend_id, name, default_variant, model_path, description, capabilities, compute_type, vram_gb

2. **`ModelsResponse(BaseModel)`**:

   - **Fields**: `models: list[ModelInfo]`

3. **`TemplateInfo(BaseModel)`**:

   - Template metadata: template_id, name, description, schema_url

4. **`TemplatesResponse(BaseModel)`**:

   - **Fields**: `templates: list[TemplateInfo]`

5. **`HealthResponse(BaseModel)`**:
   - **Fields**: status, version, redis_connected, queue_depth, worker_active

---

### File: `src/api/dependencies.py`

**Purpose**: Litestar dependency injection utilities

**Functions**:

1. **`async def api_key_guard(connection: ASGIConnection, route_handler: BaseRouteHandler) -> None`**:

   - **Purpose**: Verify X-API-Key header
   - **Signature**: Must accept `ASGIConnection` and `BaseRouteHandler` (not Any)
   - **Logic**:
     - Extract: `connection.headers.get("X-API-Key")`
     - Compare: `api_key != settings.secret_api_key`
     - Raise: `NotAuthorizedException` if invalid/missing
   - **Usage**: Applied to controller classes via `guards=[api_key_guard]`

2. **`async def get_redis_client() -> Redis`**:

   - **Purpose**: Provide async Redis client for queue DB (DB 0)
   - **Import**: `import redis.asyncio as redis` and `from redis.asyncio import Redis`
   - **Returns**:
     ```python
     redis.from_url(
         settings.redis_url,
         encoding="utf-8",
         decode_responses=True
     )
     ```
   - **Usage**: Injected into controller methods for async operations
   - **Production Note**: For production, use connection pooling (see "Redis Connection Pooling" section below)

3. **`async def get_results_redis() -> Redis`**:

   - **Purpose**: Provide async Redis client for results DB (DB 1)
   - **Returns**:
     ```python
     redis.from_url(
         settings.redis_url.replace("/0", f"/{settings.redis_results_db}"),
         encoding="utf-8",
         decode_responses=True,
         socket_timeout=10.0,      # Longer timeout for large results
         socket_connect_timeout=5.0
     )
     ```
   - **Usage**: Injected for status queries and result retrieval
   - **Rationale**: Separate client allows different timeout configuration for result operations

4. **`def get_sync_redis() -> SyncRedis`**:

   - **Purpose**: Provide synchronous Redis client for RQ
   - **Import**: `from redis import Redis as SyncRedis`
   - **Returns**:
     ```python
     SyncRedis.from_url(
         settings.redis_url,
         encoding="utf-8",
         decode_responses=True
     )
     ```
   - **Usage**: Used internally by `get_rq_queue()`
   - **Critical Note**: RQ requires synchronous Redis client; do NOT use async client with RQ

5. **`def get_rq_queue() -> Queue`**:
   - **Purpose**: Provide RQ Queue instance with synchronous Redis client
   - **Import**: `from rq import Queue`
   - **Returns**: `Queue(connection=get_sync_redis(), is_async=False)`
   - **Critical Note**: Queue operations must be wrapped with `asyncio.to_thread()` when called from async contexts

**Complete Implementation Example**:

```python
"""
src/api/dependencies.py - Complete implementation with correct imports and patterns
"""
import hmac
import redis.asyncio as redis
from redis.asyncio import Redis
from redis import Redis as SyncRedis
from rq import Queue

from litestar import ASGIConnection
from litestar.exceptions import NotAuthorizedException
from litestar.handlers.base import BaseRouteHandler

from src.config import settings


# === Guards ===

async def api_key_guard(
    connection: ASGIConnection,
    route_handler: BaseRouteHandler
) -> None:
    """
    Validate X-API-Key header with timing-safe comparison.

    Args:
        connection: ASGI connection (provides headers, state)
        route_handler: Route handler being guarded

    Raises:
        NotAuthorizedException: If API key missing or invalid
    """
    # HTTP headers are case-insensitive, check both variants
    api_key = connection.headers.get("X-API-Key") or connection.headers.get("x-api-key")

    if not api_key:
        raise NotAuthorizedException("Missing X-API-Key header")

    # Use timing-safe comparison to prevent timing attacks
    if not hmac.compare_digest(str(api_key), str(settings.secret_api_key)):
        raise NotAuthorizedException("Invalid API key")


# === Async Redis Dependencies ===

async def get_redis_client() -> Redis:
    """
    Provide async Redis client for queue DB (DB 0).

    Note: For production, consider using connection pooling.
    See "Redis Connection Pooling" section below.
    """
    return redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )


async def get_results_redis() -> Redis:
    """
    Provide async Redis client for results DB (DB 1).

    Separate client allows different timeout configuration for result operations,
    which may involve larger payloads.
    """
    url = settings.redis_url.replace("/0", f"/{settings.redis_results_db}")
    return redis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=10.0,       # Longer timeout for large results
        socket_connect_timeout=5.0
    )


# === Sync Redis + RQ Dependencies ===

def get_sync_redis() -> SyncRedis:
    """
    Provide synchronous Redis client for RQ.

    CRITICAL: RQ requires a synchronous Redis client.
    Do NOT use async Redis client with RQ.
    """
    return SyncRedis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )


def get_rq_queue() -> Queue:
    """
    Provide RQ Queue instance.

    CRITICAL: Queue operations must be wrapped with asyncio.to_thread()
    when called from async contexts:

        queue = get_rq_queue()
        job = await asyncio.to_thread(queue.enqueue, task_func, *args)
    """
    sync_redis = get_sync_redis()
    return Queue(connection=sync_redis, is_async=False)
```

---

### Redis Connection Pooling for Production

For production deployments, use connection pooling to efficiently manage Redis connections:

```python
"""
src/api/dependencies.py - Production-grade Redis with connection pooling
"""
import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis

# Initialize connection pool at module level
_redis_pool: ConnectionPool | None = None
_results_pool: ConnectionPool | None = None


def init_redis_pools():
    """
    Initialize Redis connection pools at application startup.

    Call this in Litestar's lifespan context or on_startup hook.
    """
    global _redis_pool, _results_pool

    _redis_pool = ConnectionPool.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,        # Tune based on expected concurrency
        socket_timeout=5.0,
        socket_connect_timeout=5.0
    )

    results_url = settings.redis_url.replace("/0", f"/{settings.redis_results_db}")
    _results_pool = ConnectionPool.from_url(
        results_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=10,        # Fewer connections for results operations
        socket_timeout=10.0,       # Longer timeout for large results
        socket_connect_timeout=5.0
    )


async def close_redis_pools():
    """
    Close Redis connection pools at application shutdown.

    Call this in Litestar's lifespan context or on_shutdown hook.
    """
    global _redis_pool, _results_pool

    if _redis_pool:
        await _redis_pool.aclose()
    if _results_pool:
        await _results_pool.aclose()


async def get_redis_client() -> Redis:
    """Provide async Redis client from connection pool."""
    if _redis_pool is None:
        raise RuntimeError("Redis pool not initialized. Call init_redis_pools() first.")
    return Redis(connection_pool=_redis_pool)


async def get_results_redis() -> Redis:
    """Provide async Redis client for results from connection pool."""
    if _results_pool is None:
        raise RuntimeError("Results pool not initialized. Call init_redis_pools() first.")
    return Redis(connection_pool=_results_pool)
```

**Usage in main.py**:

```python
from contextlib import asynccontextmanager
from litestar import Litestar

@asynccontextmanager
async def lifespan(app: Litestar):
    """Manage application lifespan with proper Redis pool management."""
    # Startup
    from src.api.dependencies import init_redis_pools
    init_redis_pools()

    yield  # Application runs

    # Shutdown
    from src.api.dependencies import close_redis_pools
    await close_redis_pools()


app = Litestar(
    route_handlers=[...],
    lifespan=[lifespan]
)
```

---

### File: `src/api/main.py`

**Purpose**: Litestar application with HTTP endpoints

**Controllers**:

1. **`ProcessingController(Controller)`**:

   - **Path**: `/v1`
   - **Guards**: `[api_key_guard]` - All endpoints require authentication
   - **Dependencies**:
     ```python
     {
         "queue_redis": Provide(get_redis_client),      # Async Redis for queue operations
         "results_redis": Provide(get_results_redis),   # Async Redis for results DB
         "rq_queue": Provide(get_rq_queue)              # Sync RQ Queue
     }
     ```

   **Endpoints**:

   a. **`@post("/process", status_code=HTTP_202_ACCEPTED)`**:

   - **Parameters**:

     - `data: Annotated[ProcessRequest, Body(media_type=RequestEncodingType.MULTI_PART)]` - Structured request data
     - `file: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)]` - Audio file
     - `queue_redis: Redis` - Injected async client for DB 0
     - `rq_queue: Queue` - Injected RQ queue instance

   - **Logic Flow**:

     1. **Validate file size BEFORE reading entire file**:
        ```python
        # Check file.size attribute first to avoid loading large files into memory
        if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large: {file.size} bytes (max {settings.max_file_size_mb}MB)"
            )
        ```
     2. **Validate MIME type and format**:

        ```python
        # Validate MIME type
        allowed_types = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/x-m4a", "audio/flac"}
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {file.content_type}"
            )

        # Validate file extension
        allowed_extensions = {".wav", ".mp3", ".m4a", ".flac"}
        file_ext = Path(file.filename or "audio.wav").suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file extension: {file_ext}"
            )
        ```

     3. **Sanitize filename**:
        ```python
        # Prevent path traversal attacks
        safe_filename = Path(file.filename or "audio.wav").name
        if ".." in safe_filename or "/" in safe_filename or "\\" in safe_filename:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )
        ```
     4. **Validate template_id**: If `Feature.SUMMARY in data.features`, require `data.template_id`
     5. **Check backpressure**: `depth = await queue_redis.llen("rq:queue:default")`, if `>= settings.max_queue_depth` → return 429
     6. **Generate task_id**: `uuid4()`
     7. **Save file in task directory with streaming** (do NOT load entire file into memory):

        ```python
        import aiofiles

        # Create task-specific directory
        task_dir = settings.audio_dir / str(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        file_ext = Path(file.filename or "audio.wav").suffix

        # Save as raw.{ext} in task directory
        raw_path = task_dir / f"raw{file_ext}"

        # Stream file to disk in chunks to avoid memory exhaustion
        async with aiofiles.open(raw_path, 'wb') as f:
            while chunk := await file.read(8192):  # Read 8KB at a time
                await f.write(chunk)
        ```

     8. **Initialize task in Redis DB 1**:
        - Use `results_redis` injected dependency
        - Key: `f"task:{task_id}"`
        - Fields: status="PENDING", submitted_at, request_params (JSON)
     9. **Enqueue job to Redis DB 0**:
        - **Critical**: Wrap sync RQ operation with `asyncio.to_thread()`
        ```python
        job = await asyncio.to_thread(
            rq_queue.enqueue,
            "src.worker.pipeline.process_audio_task",
            kwargs={
                "task_id": str(task_id),
                "audio_path": str(audio_path),
                "template_id": data.template_id,
                "features": data.features
            },
            job_id=str(task_id),
            timeout=settings.job_timeout,
            result_ttl=settings.result_ttl
        )
        ```
     10. **Return**: `ProcessResponse(task_id=task_id)`

   - **Error Handling**:
     - 400: Invalid filename (path traversal attempt)
     - 413: File too large
     - 415: Unsupported format or MIME type
     - 422: Missing template_id
     - 429: Queue full

   **Security Considerations**:

   - Always validate file size BEFORE reading content to prevent memory exhaustion DoS
   - Validate both MIME type and file extension to prevent type confusion attacks
   - Sanitize filenames to prevent path traversal (check for `..`, `/`, `\`)
   - Stream large files to disk instead of loading into memory
   - Use generated UUIDs for task directories, not user-supplied names
   - Store files in isolated task directories: `data/audio/{task-id}/raw.{ext}`

   b. **`@get("/status/{task_id:uuid}")`**:

   - **Parameters**: `task_id: UUID`, `results_redis: Redis`
   - **Logic**:

     1. Use injected `results_redis` (already connected to DB 1)
     2. `task_key = f"task:{task_id}"`
     3. `task_data = await results_redis.hgetall(task_key)`
     4. If empty → raise `NotFoundException`
     5. Parse status
     6. Build StatusResponse:
        - If FAILED: include error
        - If COMPLETE: parse and include versions, metrics, results (JSON)
     7. Return StatusResponse

   - **Error Handling**:
     - 404: Task not found
     - 401: No API key

2. **`DiscoveryController(Controller)`**:

   - **Path**: `/v1`
   - **Guards**: `[api_key_guard]`

   **Endpoints**:

   a. **`@get("/models")`**:

   - **Logic**:
     1. `from src.processors.asr.factory import ASRFactory`
     2. `models = ASRFactory.list_available()`
     3. Return `ModelsResponse(models=models)`

   b. **`@get("/templates")`**:

   - **Logic**:
     1. Scan `settings.templates_dir.glob("*.json")`
     2. For each file:
        - Extract template_id (stem)
        - Load JSON to get title, description
        - Build TemplateInfo with schema_url
     3. Return `TemplatesResponse(templates=templates)`

3. **`HealthController(Controller)`**:

   - **Path**: `/`
   - **No Guards**: Public endpoint

   **Endpoints**:

   a. **`@get("/health")`**:

   - **Logic**:

     1. Try to connect and ping Redis
     2. Get queue depth: `queue.count`
     3. Check worker heartbeat:
        - `worker_heartbeat = await redis_client.get("worker:heartbeat")`
        - Parse ISO timestamp
        - Check if < 60 seconds old
     4. Determine status: "healthy" if redis_connected AND worker_active
     5. Return HealthResponse

   - **Error Handling**: Catch all exceptions, return "unhealthy"

**Application Factory**:

1. **`def create_app() -> Litestar`**:

   - **Setup Logging**:

     - Remove default loguru handler
     - Add custom handler with format and colorization
     - Set level based on `settings.debug`
     - Include request IDs for tracing (recommended)

   - **Configure Exception Handlers**:

     ```python
     from litestar.exceptions import NotAuthorizedException, HTTPException
     import traceback

     def handle_not_authorized(request: Request, exc: NotAuthorizedException) -> Response:
         """Handle authentication failures."""
         logger.warning(
             f"Authentication failed: {exc}",
             extra={
                 "path": request.url.path,
                 "method": request.method,
                 "client": request.client.host if request.client else None
             }
         )
         return Response({"detail": str(exc)}, status_code=401)

     def handle_generic_exception(request: Request, exc: Exception) -> Response:
         """Handle unexpected errors with proper logging."""
         # Log full context for debugging
         logger.error(
             f"Unhandled exception: {type(exc).__name__}: {exc}",
             exc_info=exc,
             extra={
                 "path": request.url.path,
                 "method": request.method,
                 "client": request.client.host if request.client else None,
                 "headers": dict(request.headers)
             }
         )

         # Return safe response to client
         if settings.debug:
             # In debug mode, return detailed error information
             return Response({
                 "detail": str(exc),
                 "type": type(exc).__name__,
                 "traceback": traceback.format_exc().split("\n")
             }, status_code=500)
         else:
             # In production, hide internal details
             return Response({"detail": "Internal Server Error"}, status_code=500)
     ```

   - **Create App**:

     ```python
     from litestar.di import Provide

     app = Litestar(
         route_handlers=[ProcessingController, DiscoveryController, HealthController],
         dependencies={
             "queue_redis": Provide(get_redis_client),
             "results_redis": Provide(get_results_redis),
             "rq_queue": Provide(get_rq_queue)
         },
         exception_handlers={
             NotAuthorizedException: handle_not_authorized,
             Exception: handle_generic_exception
         },
         debug=settings.debug,
         openapi_config=OpenAPIConfig(...) if settings.debug else None,
         lifespan=[lifespan]  # For Redis pool management (see dependencies section)
     )
     ```

   - **Log Startup**: Version, environment, configuration summary
   - **Return**: Litestar app instance

**Structured Logging Best Practices**:

```python
from loguru import logger
import sys

def setup_logging():
    """Configure structured logging with proper formatting."""
    # Remove default handler
    logger.remove()

    # Add handler with structured format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    if settings.environment == "production":
        # JSON logging for production (easier to parse)
        logger.add(
            sys.stdout,
            format="{time} {level} {message}",
            serialize=True,  # Output as JSON
            level="INFO"
        )
    else:
        # Human-readable format for development
        logger.add(
            sys.stdout,
            format=log_format,
            colorize=True,
            level="DEBUG" if settings.debug else "INFO"
        )

    return logger
```

**Global Instance**:

- `app = create_app()`

**Entry Point** (if `__name__ == "__main__"`):

- Use `uvicorn.run()` with reload in debug mode

---

## Audio Processing Layer

### File: `src/processors/audio/preprocessor.py`

**Purpose**: Validate and normalize audio files for consistent ASR input

**Class: `AudioPreprocessor`**:

**Class Constants**:

- `TARGET_SAMPLE_RATE = 16000` - Optimal for Whisper
- `TARGET_CHANNELS = 1` - Mono
- `MIN_DURATION_SEC = 1.0` - Reject shorter audio

**Methods**:

1. **`def preprocess(self, input_path: Path) -> dict`**:

   - **Returns**: AudioMetadata dict with keys: format, duration, sample_rate, channels, normalized_path (or None)
   - **Flow**:
     1. Call `self._probe_audio(input_path)` → get metadata
     2. Validate `metadata["duration"] >= MIN_DURATION_SEC`, raise ValueError if too short
     3. Call `self._needs_normalization(metadata)`
     4. If True:
        - Call `self._normalize_audio(input_path, metadata)`
        - Set `metadata["normalized_path"] = normalized_path`
        - Log normalization occurred
     5. Else:
        - Set `metadata["normalized_path"] = None`
     6. Return metadata

2. **`def _probe_audio(self, path: Path) -> dict`**:

   - **Purpose**: Extract audio metadata using ffprobe
   - **Logic**:
     1. Run subprocess:
        ```
        ffprobe -v error
                -show_entries format=duration:stream=sample_rate,channels,codec_name
                -of json <path>
        ```
     2. Parse JSON stdout
     3. Extract: format (codec_name), duration, sample_rate, channels
     4. Return dict
   - **Error Handling**: Raise subprocess.CalledProcessError if ffprobe fails

3. **`def _needs_normalization(self, metadata: dict) -> bool`**:

   - **Logic**: Return True if:
     - `sample_rate != TARGET_SAMPLE_RATE` OR
     - `channels != TARGET_CHANNELS` OR
     - `format not in ["pcm_s16le", "wav"]`

4. **`def _normalize_audio(self, input_path: Path, metadata: dict) -> Path`**:
   - **Purpose**: Convert to WAV 16kHz mono using ffmpeg
   - **Logic**:
     1. Get task directory: `task_dir = input_path.parent`
     2. Create output path: `task_dir / "preprocessed.wav"`
     3. Run subprocess:
        ```
        ffmpeg -y -i <input_path>
               -ar 16000
               -ac 1
               -sample_fmt s16
               <output_path>
        ```
     4. Return output_path
   - **Logging**: Log input/output paths, target parameters
   - **Note**: Saves preprocessed file in same task directory as raw file

**TypedDict** (for type hints):

```python
class AudioMetadata(TypedDict):
    format: str
    duration: float
    sample_rate: int
    channels: int
    normalized_path: Path | None
```

---

## ASR Processing Layer

### File: `src/processors/base.py`

**Purpose**: Define protocols and base classes for processors

**Class: `ASRResult`** (dataclass):

- **Fields**:
  - `text: str` - Full transcript text
  - `segments: Optional[List[Dict[str, Any]]]` - Word/phrase segments with timestamps
  - `language: Optional[str]` - Detected language
  - `confidence: Optional[float]` - Average confidence score

**Additional fields for extended implementation**:

- `confidence_avg: float` - Average confidence across segments
- `vad_coverage: float` - Voice Activity Detection coverage ratio
- `duration_ms: int` - Audio duration in milliseconds
- `model_name: str` - Model identifier
- `checkpoint_hash: str` - Model checkpoint hash for versioning
- `decoding_params: dict[str, Any]` - Decoding parameters used

**Protocol: `ASRBackend`**:

```python
class ASRBackend(Protocol):
    def execute(self, audio_data: bytes, **kwargs) -> ASRResult: ...
    def unload(self) -> None: ...
    def get_version_info(self) -> dict[str, Any]: ...
```

**Protocol: `BaseLLMProcessor`** (abstract base class):

- Abstract methods: `load_model()`, `unload()`, `get_version_info()`, `is_loaded()`

---

### File: `src/processors/asr/factory.py`

**Purpose**: Factory pattern for ASR backend instantiation with integrated audio processing

**Class: `ASRProcessorFactory`**:

**Class Attributes**:

- `BACKENDS: dict[str, type[ASRBackend]] = {}` - Registry of available backends

**Class Methods**:

1. **`@classmethod def register_backend(cls, name: str, backend_class: type[ASRBackend]) -> None`**:

   - Add backend to registry: `cls.BACKENDS[name] = backend_class`
   - Log registration

2. **`@classmethod def create(cls, backend_type: str, **kwargs) -> ASRBackend`\*\*:

   - **Logic**:
     1. Check `backend_type in cls.BACKENDS`, raise ValueError if not
     2. Get backend_class from registry
     3. Instantiate: `return backend_class(**kwargs)`
   - **Error**: Raise ValueError with available backends list

3. **`@classmethod def create_with_audio_processing(cls, backend_type: str, **kwargs) -> dict[str, Any]`\*\*:

   - **Purpose**: Create ASR processor with integrated audio preprocessing components
   - **Returns**: Dictionary containing:
     - `asr_processor`: ASR backend instance
     - `audio_preprocessor`: AudioPreprocessor instance
     - `audio_metrics_collector`: AudioMetricsCollector instance
   - Used for complete audio processing pipeline

4. **`@classmethod def list_available(cls) -> list[dict[str, Any]]`**:
   - **Logic**:
     1. Iterate `cls.BACKENDS.items()`
     2. For each backend, get METADATA class attribute
     3. Build list of dicts with backend_id + metadata
     4. Return list
   - Used by `/v1/models` endpoint

---

### ASR Processing Layer

This system supports two backends in v1.0: faster-whisper and ChunkFormer. Keep examples concise and refer to upstream docs for details.

#### Whisper (faster-whisper)

- Library: `faster-whisper` (CTranslate2-optimized Whisper)
- Highlights: up to ~4x faster than original Whisper, supports GPU/CPU, quantization.
- Minimal usage pattern (aligns with official README):

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    model_size_or_path=settings.whisper_model_variant,  # e.g., "large-v3" or org CT2 checkpoint
    device=settings.whisper_device,
    compute_type=settings.whisper_compute_type,
    download_root=str(settings.models_dir / "whisper"),
)

segments, info = model.transcribe(
    audio_path,  # file path, file-like, numpy array, or tensor
    beam_size=settings.whisper_beam_size,
    vad_filter=settings.whisper_vad_filter,
    vad_parameters=settings.whisper_vad_parameters,
    language=None,
    condition_on_previous_text=settings.whisper_condition_on_previous_text,
)

segments = list(segments)  # force completion; segments is a generator
```

**VAD Parameter Mapping:**

The configuration fields map to faster-whisper API parameters as follows:

```python
# Config fields (in src/config.py):
whisper_vad_filter: bool = True
whisper_vad_min_silence_ms: int = 500
whisper_vad_speech_pad_ms: int = 400

# Maps to API call:
segments, info = model.transcribe(
    audio_path,
    vad_filter=True,  # From whisper_vad_filter
    vad_parameters={
        "min_silence_duration_ms": 500,  # From whisper_vad_min_silence_ms
        "speech_pad_ms": 400              # From whisper_vad_speech_pad_ms
    }
)
```

**VAD Parameter Guide:**

| Config Field                 | API Parameter                               | Default | Description                    |
| ---------------------------- | ------------------------------------------- | ------- | ------------------------------ |
| `whisper_vad_filter`         | `vad_filter`                                | `True`  | Enable/disable VAD             |
| `whisper_vad_min_silence_ms` | `vad_parameters["min_silence_duration_ms"]` | `500`   | Minimum silence to remove (ms) |
| `whisper_vad_speech_pad_ms`  | `vad_parameters["speech_pad_ms"]`           | `400`   | Padding around speech (ms)     |

**Note:** Our default `min_silence_duration_ms=500` is less aggressive than faster-whisper's default of 2000ms, preventing accidental speech cutoff.

- Word timestamps: set `word_timestamps=True` when needed. Capture words from `segment.words`.
- Links: official docs and examples — https://github.com/SYSTRAN/faster-whisper

Configuration tips:

- Compute types: `float16` (best quality, GPU), `int8_float16` (balanced), `int8` (CPU).
- CPU consistency: set `OMP_NUM_THREADS`.
- Distil-Whisper: `distil-large-v3` with `condition_on_previous_text=False` often performs well.

Versioning: capture your own `checkpoint_hash` for local model dirs for reproducibility (not provided by faster-whisper).

**Common faster-whisper Gotchas:**

1. **Segments are generators** - They're lazy-evaluated. You must iterate or call `list(segments)` to actually run transcription. Just calling `model.transcribe()` doesn't process audio!

   ```python
   # Wrong - transcription hasn't run yet
   segments, info = model.transcribe("audio.mp3")

   # Correct - force completion
   segments, info = model.transcribe("audio.mp3")
   segments = list(segments)  # NOW transcription actually runs
   ```

2. **Audio input format** - Expects file path (string), not raw bytes. Convert bytes to temp file first.

   ```python
   # Wrong - will fail
   result = model.transcribe(audio_bytes)

   # Correct - save to temp file
   with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
       tmp.write(audio_bytes)
       tmp_path = tmp.name
   result = model.transcribe(tmp_path)
   os.unlink(tmp_path)  # cleanup
   ```

3. **CPU performance varies** - Without `OMP_NUM_THREADS`, performance is inconsistent across runs.

   ```bash
   # Always set for CPU mode
   export OMP_NUM_THREADS=4
   python your_script.py
   ```

4. **Distil-Whisper requires special setting** - Must disable context conditioning or accuracy suffers.

   ```python
   # Required for distil models
   model.transcribe("audio.mp3", condition_on_previous_text=False)
   ```

5. **CUDA libraries must be in path** - On Linux, set `LD_LIBRARY_PATH` or CUDA won't initialize.

   ```bash
   export LD_LIBRARY_PATH=$(python3 -c 'import nvidia.cublas.lib; import nvidia.cudnn.lib; ...')
   ```

#### ChunkFormer

- Models: e.g., `khanhld/chunkformer-large-vie` (HF Hub). Research: https://arxiv.org/abs/2502.14673
- Purpose: long-form ASR with chunking/context techniques to keep memory steady for very long audio.
- Installation: `pip install chunkformer` (provides Python API and `chunkformer-decode` CLI)
- Integration approach:
  - Python API: `from chunkformer import ChunkFormerModel`; `ChunkFormerModel.from_pretrained(...)`; `endless_decode(...)` for single long-form; `batch_decode(...)` for multiple files
  - CLI: `chunkformer-decode` supporting `--model_checkpoint`, `--long_form_audio` or `--audio_list`, `--chunk_size`, `--left_context_size`, `--right_context_size`, `--total_batch_duration`
  - Config maps directly to API args: `chunk_size`, `left_context`, `right_context`, `total_batch_duration`, `return_timestamps`

Recommended defaults (see ASR Settings above):

- `chunkformer_model_name = "khanhld/chunkformer-large-vie"`
- `chunk_size=64`, `left_context=128`, `right_context=128`, `total_batch_duration=14400`, `return_timestamps=True`

Notes:

- Provide a clear mapping from configuration to the underlying decode call in your implementation.
- Link to the upstream model card for usage updates and examples.

**Purpose**: ChunkFormer ASR implementation for long-form speech transcription

**Library**: `chunkformer` - ASR model optimized for long-form audio (up to 16 hours)

**Key Features**:

- **Masked chunking** - Processes audio in chunks with overlap
- **Masked batch technique** - Minimizes padding waste for efficient GPU usage
- **Low memory** - Can process very long audio on consumer GPUs
- **Vietnamese language support** - Specialized for Vietnamese ASR

BEGIN CHUNKFORMER DETAILS (official PyPI API and CLI)
**Class: `ChunkFormerBackend`**:

**Class Attribute**:

- `METADATA: dict` - Contains: name, model_variant, description, capabilities, max_audio_hours

**Instance Attributes**:

- `model_name: str` - Model identifier (e.g., "khanhld/chunkformer-large-vie")
- `chunk_size: int` - Size of each processing chunk in frames (default: 64)
- `left_context_size: int` - Left context frames for each chunk (default: 128)
- `right_context_size: int` - Right context frames for each chunk (default: 128)
- `total_batch_duration: int` - Total batch duration in seconds (default: 14400 = 4 hours)
- `return_timestamps: bool` - Include word/segment timestamps (default: True)
- `device: str` - Device to run on: "cuda" or "cpu" (default: "cuda")
- `model: ChunkFormerModel | None` - Lazy loaded model
- `model_path: Path | None` - Resolved path to model
- `checkpoint_hash: str | None` - Model checkpoint hash

**Methods**:

1. **`def **init**(self, model_name=None, chunk_size=None, left_context_size=None, right_context_size=None, total_batch_duration=None, **kwargs)`\*\*:

   - Set attributes from parameters or fall back to settings
   - Initialize model to None (lazy loading)
   - Validate chunk and context sizes
   - Log initialization with all parameters

2. **`def _load_model(self) -> None`**:

   - **Guard**: If `self.model is not None`, return immediately
   - **Logic**:

     ```python
     from chunkformer import ChunkFormerModel

     # Load from Hugging Face Hub
     self.model = ChunkFormerModel.from_pretrained(
         self.model_name  # e.g., "khanhld/chunkformer-large-vie"
     )

     # Or load from local directory
     self.model = ChunkFormerModel.from_pretrained(
         "/path/to/model_checkpoint_avg_5"
     )
     ```

   - Calculate checkpoint hash if local path
   - Log load time and model info

3. **`def execute(self, audio_data: str, **kwargs) -> ASRResult`\*\*:

   - **Flow**:
     1. Call `self._load_model()` (lazy loading)
     2. Start timer
     3. Single long-form transcription:
        ```python
        transcription = self.model.endless_decode(
            audio_path=audio_path,  # Path to audio file
            chunk_size=self.chunk_size,
            left_context_size=self.left_context_size,
            right_context_size=self.right_context_size,
            total_batch_duration=self.total_batch_duration,
            return_timestamps=self.return_timestamps
        )
        ```
     4. Parse transcription result:
        - Extract text
        - Extract segments with timestamps if enabled
        - Calculate confidence (if available)
     5. End timer
     6. Build ASRResult with all data
     7. Log completion with metrics
     8. Return ASRResult

4. **`def execute_batch(self, audio_files: list[str], **kwargs) -> list[ASRResult]`\*\*:

   - **Purpose**: Batch processing for multiple audio files
   - **Logic**:

     ```python
     transcriptions = self.model.batch_decode(
         audio_paths=audio_files,
         chunk_size=self.chunk_size,
         left_context_size=self.left_context_size,
         right_context_size=self.right_context_size,
         total_batch_duration=self.total_batch_duration
     )

     results = []
     for i, transcription in enumerate(transcriptions):
         results.append(self._parse_transcription(transcription, audio_files[i]))

     return results
     ```

   - Processes multiple files efficiently with batching

5. **`def execute_from_tsv(self, tsv_path: str, **kwargs) -> list[ASRResult]`\*\*:

   - **Purpose**: Batch transcription from TSV file with optional WER calculation
   - **TSV Format**:
     ```
     audio_path    txt
     audio1.wav    Optional reference text
     audio2.wav    Optional reference text
     ```
   - **Logic**:
     - Reads audio paths from TSV
     - Performs batch transcription
     - If `txt` column provided, calculates Word Error Rate (WER)
     - Saves output back to TSV file
   - **Usage**:
     ```bash
     # Command-line interface
     chunkformer-decode \
         --model_checkpoint khanhld/chunkformer-large-vie \
         --audio_list path/to/data.tsv \
         --total_batch_duration 14400 \
         --chunk_size 64 \
         --left_context_size 128 \
         --right_context_size 128
     ```

6. **`def unload(self) -> None`**:

   - **Logic**:
     1. If `self.model is not None`:
        - `del self.model`
        - `self.model = None`
        - `torch.cuda.empty_cache()` (if using GPU)
        - Log unload with memory freed

7. **`def get_version_info(self) -> dict`**:

   - **Returns**: Dict with comprehensive version info:
     ```python
     {
         "backend": "chunkformer",
         "model_variant": self.model_name,
         "model_path": str(self.model_path),
         "checkpoint_hash": self.checkpoint_hash,
         "decoding_params": {
             "chunk_size": self.chunk_size,
             "left_context_size": self.left_context_size,
             "right_context_size": self.right_context_size,
             "total_batch_duration": self.total_batch_duration,
             "return_timestamps": self.return_timestamps
         }
     }
     ```

**Configuration Best Practices**:

1. **Chunk Size Selection**:

   - Smaller chunks (32-64): Lower memory, slightly slower
   - Larger chunks (128-256): Higher memory, faster processing
   - Trade-off between memory usage and speed

2. **Context Size Selection**:

   - Larger context improves accuracy at word boundaries
   - Default (128/128) works well for most cases
   - Adjust based on audio characteristics

3. **Total Batch Duration**:

   - Set based on available GPU memory
   - Default 14400s (4 hours) for 16GB GPU
   - Reduce if encountering OOM errors

4. **Long Audio Processing**:
   - ChunkFormer excels at audio > 1 hour
   - Can process up to 16 hours on consumer GPUs
   - Memory usage stays consistent regardless of audio length

**CLI Usage Examples**:

1. **Single Long-Form Audio**:

   ```bash
   chunkformer-decode \
       --model_checkpoint khanhld/chunkformer-large-vie \
       --long_form_audio path/to/long_audio.wav \
       --total_batch_duration 14400 \
       --chunk_size 64 \
       --left_context_size 128 \
       --right_context_size 128
   ```

2. **Batch Processing with TSV**:
   ```bash
   chunkformer-decode \
       --model_checkpoint path/to/local/checkpoint \
       --audio_list path/to/data.tsv \
       --total_batch_duration 14400
   ```

**Python API Examples**:

```python
from chunkformer import ChunkFormerModel

# Load model
model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-large-vie")

# Single file transcription
transcription = model.endless_decode(
    audio_path="long_podcast.wav",
    chunk_size=64,
    left_context_size=128,
    right_context_size=128,
    total_batch_duration=14400,
    return_timestamps=True
)
print(transcription)

# Batch processing
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
transcriptions = model.batch_decode(
    audio_paths=audio_files,
    chunk_size=64,
    left_context_size=128,
    right_context_size=128,
    total_batch_duration=1800  # 30 minutes total batch
)

for i, transcription in enumerate(transcriptions):
    print(f"Audio {i+1}: {transcription}")
```

Both backends are registered with the `ASRProcessorFactory` and discoverable via the `/v1/models` endpoint.

**Use Cases**:

- **ChunkFormer**: Long-form audio (>1 hour), podcasts, lectures, low-memory environments
- **Faster-Whisper**: General-purpose ASR, multilingual, high accuracy, moderate duration (<1 hour)

---

## Future Enhancements (Post-V1.0 Roadmap)

The following advanced features are available in the underlying libraries but **deferred to future releases** to maintain V1.0 simplicity and focus on sequential processing architecture:

### 1. Word-Level Timestamps (V1.1+)

**Capability**: Extract precise start/end times for each word in transcription

**Use Cases**:

- Video subtitle generation with frame-accurate timing
- Interactive transcript navigation
- Speaker diarization with word-level alignment
- Accessibility features (highlight-as-spoken)

**Implementation (when enabled)**:

```python
# Not in V1.0 - requires additional storage and schema changes
segments, _ = model.transcribe("audio.mp3", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}")
```

**Why Deferred**:

- V1.0 focus: transcript text only (per FR-5 metrics)
- Adds storage overhead (timestamp arrays)
- Requires API schema changes
- Not needed for summarization pipeline

**Target Release**: V1.1 (when timeline analysis features added)

---

### 2. Batched Inference Pipeline (V1.2+)

**Capability**: Process multiple audio files in parallel batches for higher throughput

**Use Cases**:

- Bulk processing of audio archives
- High-volume production environments
- Multi-tenant systems with job batching
- Reduced per-file overhead

**Implementation (when enabled)**:

```python
# Not in V1.0 - contradicts sequential processing architecture
from faster_whisper import BatchedInferencePipeline

model = WhisperModel("turbo", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

# Process multiple files efficiently
segments, info = batched_model.transcribe("audio.mp3", batch_size=16)
```

**Why Deferred**:

- V1.0 architecture: **sequential processing only** (one job at a time)
- Requires model preloading (contradicts load → process → unload pattern)
- Single GPU (16-24GB VRAM) focused on stability over throughput
- Batching needs multi-GPU or larger VRAM for simultaneous processing

**Requirements for Future Implementation**:

- Model preloading strategy (keep models in VRAM between jobs)
- Multi-GPU support or larger single GPU (>24GB VRAM)
- Queue batching logic (group similar-length audio files)
- Revised memory management (shared model instances)

**Target Release**: V1.2 (after multi-GPU support and preloading)

---

### 3. Advanced Batching Strategies (V1.2+)

- For ChunkFormer, use the `batch_decode(...)` API documented above.

**Why Deferred**: Same reasons as batched inference above

---

### 4. Streaming Transcription (V1.3+)

**Capability**: Real-time transcription as audio streams in

**Use Cases**:

- Live meeting transcription
- Real-time closed captioning
- Voice command systems
- Low-latency applications

**Why Deferred**:

- V1.0 focus: batch file processing
- Requires WebSocket/SSE API changes
- Different architecture (stateful connections)
- Complexity in error handling and recovery

**Target Release**: V1.3 (major architectural change)

---

### 5. Multi-GPU Scaling (V1.2+)

**Capability**: Distribute workload across multiple GPUs

**Strategies**:

- **Model Parallelism**: Split large models across GPUs
- **Data Parallelism**: Process multiple jobs simultaneously
- **Pipeline Parallelism**: ASR on GPU 0, LLM on GPU 1

**Why Deferred**:

- V1.0 target: single GPU (16-24GB) deployment
- Adds deployment complexity
- Requires orchestration logic

**Target Release**: V1.2 (for high-throughput deployments)

---

### Roadmap Summary

| Feature                 | V1.0 | V1.1 | V1.2 | V1.3 |
| ----------------------- | ---- | ---- | ---- | ---- |
| Basic Transcription     | ✅   | ✅   | ✅   | ✅   |
| VAD Filtering           | ✅   | ✅   | ✅   | ✅   |
| Sequential Processing   | ✅   | ✅   | ✅   | ✅   |
| Word Timestamps         | ❌   | ✅   | ✅   | ✅   |
| Batched Inference       | ❌   | ❌   | ✅   | ✅   |
| Model Preloading        | ❌   | ❌   | ✅   | ✅   |
| Multi-GPU Support       | ❌   | ❌   | ✅   | ✅   |
| Streaming Transcription | ❌   | ❌   | ❌   | ✅   |

**V1.0 Philosophy**: Simple, stable, sequential - one job, one GPU, load → process → unload

**Post-V1.0 Evolution**: Gradual addition of performance optimizations as deployment patterns emerge

---

## LLM Processing Layer

### File: `src/processors/prompt/template_loader.py`

**Purpose**: Load and cache Jinja2 templates from the filesystem.

```python
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from functools import lru_cache

class TemplateLoader:
    def __init__(self, template_dir: Path):
        self.env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

    @lru_cache(maxsize=128)
    def get_template(self, template_name: str):
        return self.env.get_template(f"{template_name}.jinja")
```

### File: `src/processors/prompt/renderer.py`

**Purpose**: Render Jinja2 templates with a given context.

```python
from .template_loader import TemplateLoader

class PromptRenderer:
    def __init__(self, template_loader: TemplateLoader):
        self.template_loader = template_loader

    def render(self, template_name: str, **context) -> str:
        template = self.template_loader.get_template(template_name)
        return template.render(**context)
```

### File: `src/processors/llm/enhancement_processor.py`

**Purpose**: LLM for text enhancement, using the new prompt rendering system.

```python
from src.processors.prompt.renderer import PromptRenderer

class EnhancementLLMProcessor:
    def __init__(self, prompt_renderer: PromptRenderer, ...):
        self.prompt_renderer = prompt_renderer
        # ... other initializations

    def enhance_text(self, transcript: str) -> str:
        # ...
        prompt = self.prompt_renderer.render(
            'text_enhancement_v1',
            transcript=transcript
        )
        # ... LLM generation logic
```

### File: `src/processors/llm/summary_processor.py`

**Purpose**: LLM for structured summarization, using the new prompt rendering system.

```python
import json
from src.processors.prompt.renderer import PromptRenderer

class SummaryLLMProcessor:
    def __init__(self, prompt_renderer: PromptRenderer, ...):
        self.prompt_renderer = prompt_renderer
        # ... other initializations

    def generate_summary(self, transcript: str, template_id: str, max_retries: int = 2) -> dict[str, Any]:
        # ...
        schema_path = settings.get_template_path(template_id)
        with open(schema_path) as f:
            schema = json.load(f)

        prompt = self.prompt_renderer.render(
            template_id,
            transcript=transcript,
            schema=json.dumps(schema, indent=2)
        )
        # ... LLM generation and validation logic
```

---

## Worker Layer

### File: `src/worker/pipeline.py`

**Purpose**: Sequential processing pipeline executing ASR → Enhancement → Summarization

**Main Function**:

**`def process_audio_task(task_id: str, audio_path: str, features: list[str], template_id: str | None = None) -> dict[str, Any]`**:

This is the **RQ job function** that gets enqueued and executed by the worker.

**Flow**:

**Setup Phase**:

1. Connect to Redis results DB:
   ```python
   results_client = redis.from_url(
       settings.redis_url.replace("/0", f"/{settings.redis_results_db}"),
       decode_responses=True
   )
   task_key = f"task:{task_id}"
   ```
2. Log task start
3. Start timer

**Stage 1: PREPROCESSING**:

1. Update status: `_update_status(results_client, task_key, TaskStatus.PREPROCESSING)`
2. Create AudioPreprocessor instance
3. Call `preprocessor.preprocess(Path(audio_path))`
4. Get metadata with optional normalized_path
5. Determine processing_path: use normalized if exists, else original
6. Log preprocessing complete

**Stage 2: PROCESSING_ASR**:

1. Update status: `_update_status(results_client, task_key, TaskStatus.PROCESSING_ASR)`
2. Create ASR backend: `asr = ASRFactory.create("whisper")`
3. Try:
   - Execute: `asr_result = asr.execute(str(processing_path))`
   - Log completion with metrics
4. Finally:
   - **CRITICAL**: `asr.unload()` - Always called, even on error

**Conditional Logic**: Determine if enhancement needed:

```python
needs_enhancement = (
    Feature.CLEAN_TRANSCRIPT in features
    and "erax-wow-turbo" not in settings.whisper_model_variant
)
```

Note: EraX-WoW-Turbo has native punctuation, so skip enhancement

**Stage 3: PROCESSING_LLM** (if needed):

Update status: `_update_status(results_client, task_key, TaskStatus.PROCESSING_LLM)`

**Sub-stage 3a: Text Enhancement** (if needs_enhancement):

1. Create processor: `enhancement_llm = EnhancementLLMProcessor()`
2. Try:
   - Load: `enhancement_llm.load_model()`
   - Enhance: `clean_transcript = enhancement_llm.enhance_text(asr_result.transcript)`
   - Calculate edit rate: `edit_rate_cleaning = _calculate_edit_rate(original, enhanced)`
   - Get version info: `enhancement_llm_version = enhancement_llm.get_version_info()`
3. Finally:
   - **CRITICAL**: `enhancement_llm.unload()`
   - Log unload

If not needs_enhancement:

- Set `clean_transcript = asr_result.transcript`
- Set `edit_rate_cleaning = 0.0`

**Sub-stage 3b: Structured Summarization** (if Feature.SUMMARY in features):

1. Validate: If no template_id, raise ValueError
2. Create processor: `summary_llm = SummaryLLMProcessor()`
3. Try:
   - Load: `summary_llm.load_model()`
   - Generate: `summary = summary_llm.generate_summary(clean_transcript, template_id)`
   - Get version info: `summary_llm_version = summary_llm.get_version_info()`
4. Finally:
   - **CRITICAL**: `summary_llm.unload()`
   - Log unload

**Stage 4: COMPLETE - Assemble Results**:

1. Calculate total processing time
2. Build versions block:

   ```python
   versions = {
       "pipeline_version": settings.pipeline_version,
       "asr_backend": asr.get_version_info()
   }
   if enhancement_llm_version:
       versions["enhancement_llm"] = enhancement_llm_version
   if summary_llm_version:
       versions["summarization_llm"] = summary_llm_version
   ```

3. Build metrics block:

   ```python
   metrics = {
       "input_duration_seconds": audio_metadata["duration"],
       "processing_time_seconds": pipeline_duration,
       "rtf": pipeline_duration / audio_metadata["duration"],
       "vad_coverage": asr_result.vad_coverage,
       "asr_confidence_avg": asr_result.confidence_avg,
       "edit_rate_cleaning": edit_rate_cleaning
   }
   ```

4. Build results block (conditional):

   ```python
   results = {}
   if Feature.RAW_TRANSCRIPT in features:
       results["raw_transcript"] = asr_result.transcript
   if Feature.CLEAN_TRANSCRIPT in features:
       results["clean_transcript"] = clean_transcript
   if Feature.SUMMARY in features:
       results["summary"] = summary
   if Feature.ENHANCEMENT_METRICS in features and edit_rate_cleaning is not None:
       results["enhancement_metrics"] = {"edit_rate": edit_rate_cleaning}
   ```

5. Store in Redis:

   ```python
   results_client.hset(
       task_key,
       mapping={
           "status": TaskStatus.COMPLETE.value,
           "completed_at": datetime.utcnow().isoformat(),
           "versions": json.dumps(versions),
           "metrics": json.dumps(metrics),
           "results": json.dumps(results)
       }
   )
   ```

6. Log success
7. Return complete result dict

**Error Handling** (outer try/except):

1. Catch all exceptions
2. Log error with full traceback
3. Update task in Redis:
   ```python
   results_client.hset(
       task_key,
       mapping={
           "status": TaskStatus.FAILED.value,
           "error_message": str(e),
           "completed_at": datetime.utcnow().isoformat()
       }
   )
   ```
4. Re-raise exception (for RQ retry mechanism)

**Helper Functions**:

1. **`def _update_status(client: redis.Redis, task_key: str, status: TaskStatus) -> None`**:

   - Simple: `client.hset(task_key, "status", status.value)`
   - Log status change

2. **`def _calculate_edit_rate(original: str, enhanced: str) -> float`**:
   - **Purpose**: Calculate Levenshtein distance ratio
   - **Algorithm**:
     1. Create 2D DP matrix: `dp[len(original)+1][len(enhanced)+1]`
     2. Initialize: first row/column with indices
     3. Fill matrix:
        ```python
        for i in range(1, len1+1):
            for j in range(1, len2+1):
                cost = 0 if original[i-1] == enhanced[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        ```
     4. Distance: `dp[len1][len2]`
     5. Return: `distance / max(len1, len2)`
   - **Returns**: Float between 0.0 (identical) and 1.0 (completely different)

---

### File: `src/worker/main.py`

**Purpose**: RQ worker entrypoint with model verification and heartbeat

**Functions**:

1. **`def verify_models() -> None`**:

   - **Purpose**: Ensure all required models exist before starting
   - **Logic**:
     1. Build list of required paths:
        ```python
        required_models = [
            settings.models_dir / "whisper" / settings.whisper_model_variant,
            settings.models_dir / "llm" / settings.llm_enhancement_model.split("/")[-1],
            settings.models_dir / "llm" / settings.llm_summary_model.split("/")[-1]
        ]
        ```
     2. Check each path exists
     3. If any missing:
        - Log error with missing list
        - Raise RuntimeError with instructions to run download script
     4. If all exist: Log success

2. **`def update_heartbeat(redis_client: redis.Redis) -> None`**:

   - **Purpose**: Let API know worker is alive
   - **Logic**:
     ```python
     redis_client.set(
         "worker:heartbeat",
         datetime.utcnow().isoformat(),
         ex=120  # Expire after 2 minutes
     )
     ```
   - Called periodically to update timestamp

3. **`def main() -> None`**:
   - **Purpose**: Start RQ worker
   - **Flow**:
     1. Configure logging:
        - Remove default loguru handler
        - Add custom handler with format
        - Set level from settings.debug
     2. Log worker starting with version and name
     3. Verify models: `verify_models()`
        - On RuntimeError: log error, `sys.exit(1)`
     4. Connect to Redis: `redis_client = redis.from_url(settings.redis_url, decode_responses=False)`
     5. Create RQ worker:
        ```python
        worker = Worker(
            ["default"],  # Queue names
            connection=redis_client,
            name=settings.worker_name
        )
        ```
     6. Update heartbeat: `update_heartbeat(redis_client)`
     7. Log ready
     8. Start worker:
        ```python
        worker.work(
            with_scheduler=False,
            burst=False,  # Run continuously
            logging_level="INFO"
        )
        ```
     9. Handle KeyboardInterrupt: log graceful shutdown
     10. Handle other exceptions: log error, `sys.exit(1)`

**Entry Point**:

```python
if __name__ == "__main__":
    main()
```

---

## Testing Layer

### File: `tests/conftest.py`

**Purpose**: Shared fixtures and mocks for all tests

**Fixtures**:

1. **`@pytest.fixture(autouse=True) def setup_test_env(monkeypatch, tmp_path)`**:

   - **Purpose**: Set test environment variables for every test
   - **Logic**:
     ```python
     monkeypatch.setenv("SECRET_API_KEY", "test-api-key-12345")
     monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")  # Test DB
     monkeypatch.setenv("DEBUG", "true")
     monkeypatch.setenv("AUDIO_DIR", str(tmp_path / "audio"))
     monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
     # etc.
     ```
   - Auto-applies to all tests

2. **`@pytest.fixture def mock_redis()`**:

   - **Purpose**: Mock Redis client for tests without Redis server
   - **Logic**:
     ```python
     with patch("redis.from_url") as mock:
         client = Mock()
         client.ping.return_value = True
         client.hset.return_value = True
         client.hgetall.return_value = {}
         mock.return_value = client
         yield client
     ```

3. **`@pytest.fixture def mock_vllm_model()`**:

   - **Purpose**: Mock vLLM for testing without GPU
   - **Logic**:
     ```python
     with patch("vllm.LLM") as mock:
         model = Mock()
         output = Mock()
         output.text = "mocked output"
         model.generate.return_value = [Mock(outputs=[output])]
         mock.return_value = model
         yield model
     ```

4. **`@pytest.fixture def mock_whisper_model()`**:

   - **Purpose**: Mock Whisper for testing without GPU
   - **Logic**:
     ```python
     with patch("faster_whisper.WhisperModel") as mock:
         model = Mock()
         segment = Mock(text="Test transcript", avg_logprob=-0.5)
         info = Mock(duration=10.0, duration_after_vad=9.0)
         model.transcribe.return_value = ([segment], info)
         mock.return_value = model
         yield model
     ```

5. **`@pytest.fixture def mock_ffprobe()`**:

   - **Purpose**: Mock ffprobe subprocess call
   - **Logic**:
     ```python
     with patch("subprocess.run") as mock:
         mock.return_value = Mock(
             returncode=0,
             stdout='{"format": {"duration": "10.5"}, "streams": [{"codec_name": "pcm_s16le", "sample_rate": "16000", "channels": "1"}]}'
         )
         yield mock
     ```

6. **`@pytest.fixture def test_audio_file(tmp_path)`**:
   - **Purpose**: Create minimal valid WAV file in task directory structure
   - **Logic**:
     1. Create task directory: `task_dir = tmp_path / "audio" / str(uuid.uuid4())`
     2. `task_dir.mkdir(parents=True, exist_ok=True)`
     3. Use `wave` module to create 1 second of silence at 16kHz mono
     4. Save as: `task_dir / "raw.wav"`
     5. Return path to file
   - Used in integration tests

---

### File: `tests/builders.py`

**Purpose**: Builder pattern for creating test data

**Classes**:

1. **`class TaskBuilder`**:

   - **Purpose**: Build task data dicts for testing
   - **Attributes**: task_id, status, submitted_at, features, template_id, audio_path, etc.
   - **Methods**:
     - `with_status(status)`: Set status, return self
     - `with_features(features)`: Set features, return self
     - `completed()`: Set to COMPLETE with results, return self
     - `failed(error)`: Set to FAILED with error, return self
     - `build()`: Return dict with all data
   - **Usage**: `task = TaskBuilder().with_status("COMPLETE").completed().build()`

2. **`class GenerationConfigBuilder`**:
   - **Purpose**: Build GenerationConfig for testing
   - **Methods**:
     - `with_temperature(temp)`: Set temperature, return self
     - `with_top_p(top_p)`: Set top_p, return self
     - `deterministic()`: Set temp=0.0, top_k=1, return self
     - `creative()`: Set temp=0.8, top_p=0.95, return self
     - `build()`: Return GenerationConfig instance
   - **Usage**: `config = GenerationConfigBuilder().deterministic().build()`

---

### Test Files Structure

Each test file follows this pattern:

1. **Multiple test classes** grouping related tests
2. **Descriptive test names**: `test_{what}_{scenario}_{expected}`
3. **AAA pattern**: Arrange, Act, Assert
4. **Fixtures** for setup/teardown
5. **Parametrize** for multiple scenarios

**Example test structure**:

```python
class TestConfigurationBasics:
    """Test basic configuration loading."""

    def test_settings_requires_api_key(self):
        """Settings should raise error without API key."""
        # Arrange: (setup is in conftest)
        # Act & Assert:
        with pytest.raises(ValidationError, match="secret_api_key"):
            Settings()

    def test_settings_loads_from_env(self, monkeypatch):
        """Settings should load from environment."""
        # Arrange:
        monkeypatch.setenv("SECRET_API_KEY", "test-key")

        # Act:
        settings = Settings()

        # Assert:
        assert settings.secret_api_key == "test-key"
```

---

## Docker & Deployment

### File: `Dockerfile`

**Purpose**: Single image for both API and Worker

**Structure**:

1. **Base Stage**:

   - FROM: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
   - Install: python3.11, ffmpeg, curl, git
   - Install UV package manager
   - Set WORKDIR: `/app`

2. **Dependencies Stage**:

   - COPY: `pyproject.toml`, `uv.lock`, `.python-version`
   - Create venv: `uv venv`
   - Install dependencies: `uv pip install -e .`

3. **Application Stage**:

   - COPY: `src/`, `templates/`, `assets/`
   - Create directories: `data/audio`, `data/models`, `data/redis`
   - EXPOSE: 8000

4. **Default CMD**: `.venv/bin/python -m src.api.main`
   - Override in docker-compose for worker

---

### File: `docker-compose.yml`

**Purpose**: Multi-service orchestration

**Services**:

1. **`redis`**:

   - Image: `redis:7-alpine`
   - Command: `redis-server --appendonly yes --appendfsync everysec`
   - Ports: `6379:6379`
   - Volumes: `./data/redis:/data` (CRITICAL for persistence)
   - Healthcheck: `redis-cli ping`

2. **`api`**:

   - Build: `.`
   - Command: `.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
   - Ports: `8000:8000`
   - Volumes:
     - `./data/audio:/app/data/audio` (read-write for uploads)
     - `./templates:/app/templates:ro` (read-only)
     - `./assets:/app/assets:ro` (read-only)
     - `./.env:/app/.env:ro` (read-only)
   - Environment: `REDIS_URL=redis://redis:6379/0`
   - Depends_on: `redis` (with health condition)
   - Healthcheck: `curl -f http://localhost:8000/health`

3. **`worker`**:

   - Build: `.`
   - Command: `.venv/bin/python -m src.worker.main`
   - Volumes:
     - `./data/audio:/app/data/audio:ro` (read-only for processing)
     - `./data/models:/app/data/models:ro` (read-only)
     - `./templates:/app/templates:ro`
     - `./assets:/app/assets:ro`
     - `./.env:/app/.env:ro`
   - Environment: `REDIS_URL=redis://redis:6379/0`
   - Depends_on: `redis` (with health condition)
   - Deploy: Reserve GPU 0
     ```yaml
     resources:
       reservations:
         devices:
           - driver: nvidia
             device_ids: ["0"]
             capabilities: [gpu]
     ```

4. **`dashboard`** (optional):
   - Image: `eoranged/rq-dashboard:latest`
   - Ports: `9181:9181`
   - Environment: `RQ_DASHBOARD_REDIS_URL=redis://redis:6379/0`

---

## Scripts

### File: `scripts/download_models.sh`

**Purpose**: Download all required models before deployment

**Structure**:

1. Set MODEL_DIR from env or default to `./data/models`
2. Create subdirectories: `whisper/`, `llm/`
3. Check if `huggingface-cli` installed, install if not
4. Download each model:
   ```bash
   huggingface-cli download erax-ai/EraX-WoW-Turbo-V1.1-CT2 \
     --local-dir "$MODEL_DIR/whisper/erax-wow-turbo" \
     --local-dir-use-symlinks False
   ```
5. Verify downloads (check directories exist and not empty)
6. Show disk usage summary

---

### File: `scripts/load_test.sh`

**Purpose**: Simple load testing script

**Logic**:

1. Parse arguments: audio_file, num_requests
2. Loop to submit jobs:
   - POST to `/v1/process`
   - Store task_ids in array
3. Poll until all complete:
   - GET `/v1/status/{task_id}` for each
   - Count COMPLETE/FAILED
   - Show progress
4. Calculate metrics:
   - Total time
   - Average time per job
5. Display results summary

---

## TDD Implementation Order

### Week 1: Foundation

**Day 1-2: Configuration**

- Write tests for Settings class
- Implement Settings with Pydantic
- Write tests for GenerationConfig
- Implement GenerationConfig and hierarchy
- Write tests for config loading
- Implement load_model_generation_config

**Day 3: API Schemas**

- Write tests for all Pydantic models
- Implement all schema classes
- Test validation rules
- Test serialization/deserialization

### Week 2: Audio & ASR

**Day 4-5: Audio Preprocessing**

- Write tests for AudioPreprocessor
- Implement \_needs_normalization (easiest)
- Mock ffprobe, test \_probe_audio
- Implement \_probe_audio
- Mock ffmpeg, test \_normalize_audio
- Implement \_normalize_audio
- Test full preprocess flow

**Day 6-7: ASR Layer**

- Write tests for ASRFactory
- Implement factory with registration
- Write tests for WhisperBackend (without GPU)
- Implement WhisperBackend structure
- Mock Whisper model for testing
- Implement execute() with mocks passing
- Skip GPU tests initially with `@pytest.mark.skip`

### Week 3: LLM Processors

**Day 8-10: LLM Enhancement**

- Write tests for EnhancementLLMProcessor init
- Implement **init** with config hierarchy
- Write tests for config merging
- Implement load_model (with mocks)
- Write tests for enhance_text (with mocks)
- Implement enhance_text logic
- Test unload and version_info

**Day 11-13: LLM Summarization**

- Write tests for SummaryLLMProcessor init
- Implement **init** (similar to enhancement)
- Write tests for generate_summary with retries
- Implement generate_summary with JSON parsing
- Write tests for schema validation
- Implement validation with retry logic
- Test all error cases

### Week 4: Integration

**Day 14-16: API Endpoints**

- Write tests for health endpoint
- Implement HealthController
- Write tests for process endpoint (with mocks)
- Implement ProcessingController.submit_job
- Write tests for status endpoint
- Implement ProcessingController.get_status
- Write tests for discovery endpoints
- Implement DiscoveryController

**Day 17-19: Worker Pipeline**

- Write tests for pipeline stages (with mocks)
- Implement process_audio_task structure
- Test state transitions
- Implement status updates
- Test error handling
- Implement error recovery
- Test full pipeline (still mocked)

**Day 20-21: Worker Main**

- Write tests for model verification
- Implement verify_models
- Write tests for heartbeat
  Implement update_heartbeat
- Write tests for main() startup
- Implement main() with RQ worker setup
- Test graceful shutdown

### Week 5: End-to-End & GPU Tests

**Day 22-23: Integration Tests**

- Write integration tests for API + Redis (no GPU)
- Test file upload flow
- Test task lifecycle
- Test backpressure (429 responses)
- Test concurrent requests
- All tests use mocked processors

**Day 24-26: GPU Tests** (requires GPU hardware)

- Remove `@pytest.mark.skip` from GPU tests
- Download models: `./scripts/download_models.sh`
- Test WhisperBackend.execute() with real model
- Test EnhancementLLMProcessor with real LLM
- Test SummaryLLMProcessor with real LLM
- Test full pipeline end-to-end
- Measure performance (RTF, processing time)

**Day 27-28: Load Testing & Optimization**

- Run load tests with real audio
- Profile bottlenecks
- Optimize hot paths
- Test concurrent load (6+ requests)
- Verify GPU memory management
- Test error scenarios (OOM, invalid files)

---

## Key Interactions & Data Flow

### Request Flow

```
1. Client uploads audio
   ↓
2. API Server (main.py)
   - Validates file (size, format)
   - Checks backpressure (queue depth)
   - Generates task_id
   - Saves file to disk
   - Writes PENDING to Redis DB 1
   - Enqueues job to Redis DB 0
   ↓
3. Returns 202 with task_id immediately
```

### Processing Flow

```
4. Worker dequeues job from Redis DB 0
   ↓
5. Pipeline (pipeline.py)
   a. PREPROCESSING
      - AudioPreprocessor.preprocess()
      - Normalizes audio if needed

   b. PROCESSING_ASR
      - ASRFactory.create("whisper")
      - WhisperBackend.execute()
      - Returns transcript + metrics
      - WhisperBackend.unload() [CRITICAL]

   c. PROCESSING_LLM (if needed)
      - EnhancementLLMProcessor (if no native punctuation)
        - load_model()
        - enhance_text()
        - unload() [CRITICAL]

      - SummaryLLMProcessor (if summary requested)
        - load_model()
        - generate_summary() with JSON validation
        - unload() [CRITICAL]

   d. COMPLETE
      - Assemble versions, metrics, results
      - Write to Redis DB 1
   ↓
6. Worker updates heartbeat periodically
```

### Status Check Flow

```
7. Client polls GET /v1/status/{task_id}
   ↓
8. API Server (main.py)
   - Queries Redis DB 1
   - Returns current status
   - If COMPLETE: includes full results
   - If FAILED: includes error message
```

---

## Configuration Hierarchy in Action

### Example: Enhancement Temperature

**Scenario**: User wants deterministic punctuation

**Hierarchy Resolution**:

```python
# 1. Library defaults (vLLM)
temperature = 1.0  # Very creative

# 2. Model's generation_config.json
# File: data/models/llm/Qwen3-4B-Instruct/generation_config.json
{
  "temperature": 0.7  # Model author's recommendation
}
# Merged: temperature = 0.7

# 3. Environment variables (.env)
LLM_ENHANCEMENT_TEMPERATURE=0.1  # Deployment override
# Merged: temperature = 0.1

# 4. Runtime parameters (if provided)
processor = EnhancementLLMProcessor(temperature=0.05)
# Merged: temperature = 0.05 (highest priority)

# Final: temperature = 0.05
```

**Implementation in code**:

```python
# In enhancement_processor.py load_model():
self.final_config = build_generation_config(
    model_path=model_path,
    env_overrides=self.env_config,      # From .env
    runtime_overrides=self.runtime_config  # From constructor
)
# build_generation_config does the merging
```

---

## Critical Design Patterns

### 1. Sequential GPU Processing

**Why**: Single GPU (16-24GB VRAM) can't hold multiple large models

**Pattern**:

```python
# In pipeline.py
try:
    asr = ASRFactory.create("whisper")
    result = asr.execute(audio_path)
finally:
    asr.unload()  # CRITICAL: Free GPU before next stage

# GPU is now free

try:
    enhancement_llm = EnhancementLLMProcessor()
    enhancement_llm.load_model()
    enhanced = enhancement_llm.enhance_text(transcript)
finally:
    enhancement_llm.unload()  # CRITICAL

# GPU is now free

try:
    summary_llm = SummaryLLMProcessor()
    summary_llm.load_model()
    summary = summary_llm.generate_summary(enhanced, template_id)
finally:
    summary_llm.unload()  # CRITICAL
```

**Key Points**:

- Always use try/finally for unload
- Call `torch.cuda.empty_cache()` after unload
- Never overlap model loading
- Peak VRAM: ~8GB (ASR phase)

### 2. Factory Pattern (ASR Backends)

**Purpose**: Pluggable ASR implementations

**Registration**:

```python
# At module level in whisper.py:
ASRFactory.register("whisper", WhisperBackend)

# Future: in chunkformer.py:
ASRFactory.register("chunkformer", ChunkFormerBackend)
```

**Usage**:

```python
# Worker doesn't know about specific backends
backend = ASRFactory.create("whisper")
result = backend.execute(audio_path)
# Protocol ensures all backends have same interface
```

**Benefits**:

- Easy to add new backends
- Worker code unchanged
- Testable in isolation

### 3. Dependency Injection (Litestar)

**Pattern**:

```python
# In dependencies.py:
from redis.asyncio import Redis as AsyncRedis

async def get_redis_client() -> AsyncRedis:
    return AsyncRedis.from_url(settings.redis_url)

# In controller:
class ProcessingController(Controller):
    dependencies = {"redis_client": Provide(get_redis_client)}

    @post("/process")
    async def submit_job(
        self,
        redis_client: AsyncRedis,  # Injected automatically
        file: UploadFile
    ):
        # Use redis_client
```

**Benefits**:

- Testable (mock dependencies)
- Clean separation of concerns
- Type-safe

### 4. Builder Pattern (Tests)

**Purpose**: Consistent test data creation

**Usage**:

```python
# In tests:
task = (TaskBuilder()
    .with_status("COMPLETE")
    .with_features(["summary"])
    .completed()
    .build())

# Instead of:
task = {
    "task_id": str(uuid4()),
    "status": "COMPLETE",
    "features": ["summary"],
    "results": {...},
    "completed_at": datetime.utcnow().isoformat(),
    # ... many more fields
}
```

**Benefits**:

- DRY principle
- Readable tests
- Easy to modify

---

## Error Handling Strategy

### Validation Errors (4xx)

**Where**: API layer
**Examples**:

- 400: Malformed request
- 401: Missing/invalid API key
- 413: File too large
- 415: Unsupported format
- 422: Missing required field
- 429: Queue full

**Pattern**:

```python
# In main.py
if file_size > max_size:
    raise HTTPException(
        status_code=413,
        detail=f"File too large: {file_size}MB > {max_size}MB"
    )
```

### Processing Errors (5xx)

**Where**: Worker layer
**Examples**:

- Model loading failed
- GPU OOM
- Invalid JSON from LLM
- Schema validation failed

**Pattern**:

```python
# In pipeline.py
try:
    # Processing logic
    result = process()
except Exception as e:
    # Log with context
    logger.error("Processing failed", task_id=task_id, error=str(e), exc_info=True)

    # Update Redis
    redis.hset(task_key, mapping={
        "status": "FAILED",
        "error_message": str(e)
    })

    # Re-raise for RQ retry
    raise
```

**RQ Retry Logic**:

- Automatic retries: max 2
- Exponential backoff
- Failed jobs move to FailedJobRegistry
- Manual requeue possible

---

## Logging Strategy

### Structured Logging with Loguru

**Pattern**:

```python
from loguru import logger

# Basic log
logger.info("Task started", task_id=task_id, features=features)

# With context binding
logger.bind(task_id=task_id).info("ASR complete", rtf=0.06)

# Error with traceback
logger.error("Failed to load model", model=model_name, exc_info=True)
```

**Configuration**:

```python
# In create_app() and main():
logger.remove()  # Remove default
logger.add(
    sink=sys.stderr,
    format="<green>{time}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
    level="DEBUG" if settings.debug else "INFO",
    serialize=True  # JSON output for production
)
```

**Log Levels**:

- DEBUG: Detailed execution (model loading, config values)
- INFO: Major events (task started, stage complete)
- WARNING: Recoverable issues (retry triggered, queue high)
- ERROR: Failures (job failed, model crash)

---

## Performance Considerations

### Target Metrics

**For 45-minute audio**:

- Total processing time: 2-3 minutes
- RTF (Real-Time Factor): ~0.06
- Peak VRAM: 6-8GB

**Breakdown**:

```
ASR (Whisper):        45s  (37.5%)
ASR Unload:           1s   (0.8%)
Enhancement Load:     20s  (16.7%)
Enhancement Process:  15s  (12.5%)
Enhancement Unload:   1s   (0.8%)
Summary Load:         20s  (16.7%)
Summary Process:      18s  (15.0%)
Summary Unload:       1s   (0.8%)
────────────────────────
Total:               ~120s
```

### Optimization Opportunities

1. **Model Loading** (40s overhead):

   - Future: Keep models loaded between jobs
   - Requires: More VRAM or model swapping

2. **VAD Filtering**:

   - Enabled by default in Whisper
   - Skips silent portions
   - Improves speed on sparse audio

3. **Beam Size**:

   - Default: 5 (quality)
   - Can reduce to 1-2 for speed
   - Trade-off: accuracy vs latency

4. **Batch Processing**:
   - Current: One job at a time
   - Future: Batch multiple jobs
   - Requires: Queue management changes

---

## Testing Strategy Summary

### Unit Tests (Fast, No External Deps)

**What to test**:

- Configuration loading
- Schema validation
- Config hierarchy merging
- Audio validation logic
- Factory pattern
- Builder pattern
- Helper functions

**Characteristics**:

- Run in <10 seconds
- No GPU required
- No Redis required
- Heavy use of mocks

**Run**: `pytest tests/unit/ -v`

### Integration Tests (Medium Speed)

**What to test**:

- API endpoints with Redis
- File upload flow
- Task lifecycle
- Backpressure
- Concurrent requests

**Characteristics**:

- Requires Redis (Docker)
- No GPU required
- Processors are mocked
- Run in <1 minute

**Run**: `pytest tests/integration/ -v`

### E2E Tests (Slow, Full System)

**What to test**:

- Complete pipeline with real models
- Real audio processing
- Performance benchmarks
- Error scenarios

**Characteristics**:

- Requires GPU
- Requires downloaded models
- Run in several minutes
- Marked with `@pytest.mark.gpu`

**Run**: `pytest tests/e2e/ -m gpu -v`

---

## Deployment Checklist

### Pre-Deployment

1. **Environment Setup**:

   - Copy `.env.template` to `.env`
   - Set `SECRET_API_KEY` (generate with `secrets.token_urlsafe(32)`)
   - Review all configuration values
   - Adjust `MAX_QUEUE_DEPTH` for your load

2. **Download Models**:

   ```bash
   ./scripts/download_models.sh
   # Verify: ls -lh data/models/
   ```

3. **Build Images**:
   ```bash
   docker-compose build
   ```

### Deployment

4. **Start Services**:

   ```bash
   docker-compose up -d
   ```

5. **Verify Health**:

   ```bash
   # Check all containers running
   docker-compose ps

   # Test health endpoint
   curl http://localhost:8000/health

   # Should return:
   # {"status": "healthy", "version": "1.0.0", ...}
   ```

6. **Test Processing**:

   ```bash
   # Submit test job
   curl -X POST http://localhost:8000/v1/process \
     -H "X-API-Key: YOUR_KEY" \
     -F "file=@test.mp3" \
     -F "features=clean_transcript" \
     -F "features=summary" \
     -F "template_id=meeting_notes_v1"

   # Poll status
   curl http://localhost:8000/v1/status/TASK_ID \
     -H "X-API-Key: YOUR_KEY"
   ```

### Post-Deployment

7. **Monitor**:

   - Check logs: `docker-compose logs -f`
   - Watch RQ dashboard: http://localhost:9181
   - Monitor GPU: `watch -n 1 nvidia-smi`

8. **Load Test**:
   ```bash
   ./scripts/load_test.sh test.mp3 10
   ```

---

## Common Issues & Solutions

### Issue: Models Not Found

**Symptoms**: "Model not found" in worker logs

**Solution**:

```bash
# Verify models exist
ls -lh data/models/whisper/
ls -lh data/models/llm/

# Re-download if missing
./scripts/download_models.sh
```

### Issue: GPU Out of Memory

**Symptoms**: "CUDA out of memory" errors

**Solutions**:

1. Reduce `LLM_GPU_MEMORY_UTILIZATION` in `.env` (try 0.85)
2. Ensure models are being unloaded (check logs)
3. Kill other GPU processes: `nvidia-smi` → `kill`

### Issue: Redis Connection Failed

**Symptoms**: "Connection refused" in API logs

**Solution**:

```bash
# Check Redis container
docker-compose ps redis

# Restart if needed
docker-compose restart redis

# Check logs
docker-compose logs redis
```

### Issue: Worker Not Processing

**Symptoms**: Jobs stuck in PENDING

**Solutions**:

```bash
# Check worker logs
docker-compose logs worker

# Verify GPU visible
docker-compose exec worker nvidia-smi

# Restart worker
docker-compose restart worker
```

### Issue: Queue Full (429 Errors)

**Symptoms**: API returns 429 Too Many Requests

**Solutions**:

1. Increase `MAX_QUEUE_DEPTH` in `.env`
2. Add more workers (multi-GPU)
3. Optimize processing time
4. Clear old jobs from queue

---

## Critical Implementation Best Practices

### Security Best Practices

1. **File Upload Security**:

   - ✅ **DO**: Validate file.size BEFORE reading content
   - ✅ **DO**: Validate both MIME type AND file extension
   - ✅ **DO**: Sanitize filenames (check for `..`, `/`, `\`)
   - ✅ **DO**: Stream files to disk (use `aiofiles` with chunks)
   - ✅ **DO**: Use UUID-based filenames, not user input
   - ❌ **DON'T**: Load entire files into memory
   - ❌ **DON'T**: Trust user-supplied filenames directly

2. **API Key Validation**:

   - ✅ **DO**: Use `hmac.compare_digest()` for timing-safe comparison
   - ✅ **DO**: Check both case variants of headers (`X-API-Key` and `x-api-key`)
   - ✅ **DO**: Validate key format and length
   - ❌ **DON'T**: Use `==` for secret comparison (timing attacks)

3. **Error Handling**:
   - ✅ **DO**: Log full error context (with request details)
   - ✅ **DO**: Return generic errors to clients in production
   - ✅ **DO**: Include detailed errors in debug mode
   - ❌ **DON'T**: Expose internal state or stack traces to clients
   - ❌ **DON'T**: Log sensitive data (API keys, tokens)

### Redis Connection Management

1. **Async Redis Clients**:

   - ✅ **DO**: Use `redis.asyncio` from redis-py 5.x+
   - ✅ **DO**: Include `encoding="utf-8"` and `decode_responses=True`
   - ✅ **DO**: Set appropriate timeouts (`socket_timeout`, `socket_connect_timeout`)
   - ✅ **DO**: Use connection pools in production
   - ✅ **DO**: Close pools on application shutdown (`await pool.aclose()`)
   - ❌ **DON'T**: Forget to configure timeout values

2. **RQ and Synchronous Redis**:
   - ✅ **DO**: Use synchronous `Redis` client from `redis` package
   - ✅ **DO**: Wrap RQ operations with `asyncio.to_thread()` in async contexts
   - ✅ **DO**: Set `is_async=False` on Queue instances
   - ❌ **DON'T**: Use async Redis client with RQ (it won't work)
   - ❌ **DON'T**: Call queue.enqueue() directly from async functions

### Litestar Patterns

1. **Dependency Injection**:

   - ✅ **DO**: Wrap all dependencies with `Provide()` in Litestar
   - ✅ **DO**: Define dependencies at appropriate levels (app, controller, handler)
   - ✅ **DO**: Use type hints for automatic injection
   - ❌ **DON'T**: Pass functions directly without `Provide()` wrapper

2. **Guards**:

   - ✅ **DO**: Use correct signature: `(ASGIConnection, BaseRouteHandler) -> None`
   - ✅ **DO**: Raise `NotAuthorizedException` on auth failures
   - ✅ **DO**: Apply guards at controller level for all endpoints
   - ❌ **DON'T**: Return False or boolean values from guards
   - ❌ **DON'T**: Use incorrect type hints (like `Any`)

3. **Exception Handlers**:
   - ✅ **DO**: Define handlers for specific exception types
   - ✅ **DO**: Include request context in logs
   - ✅ **DO**: Differentiate debug vs production responses
   - ❌ **DON'T**: Let exceptions bubble to default handler

### GPU Memory Management

1. **Sequential Processing**:

   - ✅ **DO**: Unload models in `finally` blocks
   - ✅ **DO**: Call `torch.cuda.empty_cache()` after unloading
   - ✅ **DO**: Log GPU memory before/after operations
   - ✅ **DO**: Process: ASR → unload → Enhancement → unload → Summary → unload
   - ❌ **DON'T**: Load multiple models simultaneously on single GPU
   - ❌ **DON'T**: Skip unload operations (even on errors)

2. **Memory Monitoring**:

   ```python
   import torch

   def log_gpu_memory():
       if torch.cuda.is_available():
           allocated = torch.cuda.memory_allocated() / 1024**3
           reserved = torch.cuda.memory_reserved() / 1024**3
           logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
   ```

### Configuration Management

1. **Pydantic Settings**:

   - ✅ **DO**: Use `SettingsConfigDict` with `env_file=".env"`
   - ✅ **DO**: Set `extra="ignore"` to ignore unknown env vars
   - ✅ **DO**: Use `field_validator` for complex validation
   - ✅ **DO**: Set `validate_default=True` for early error detection
   - ❌ **DON'T**: Use `BaseSettings` from `pydantic` (use `pydantic_settings`)

2. **Configuration Hierarchy**:
   - Priority: Runtime > Environment > Model Config > Library Defaults
   - ✅ **DO**: Merge configs with `.merge_with()` method
   - ✅ **DO**: Log final resolved configuration
   - ✅ **DO**: Keep `None` values to indicate "use lower priority"
   - ❌ **DON'T**: Replace None with defaults prematurely

### Testing Best Practices

1. **Mocking External Dependencies**:

   - ✅ **DO**: Mock GPU operations in unit tests
   - ✅ **DO**: Mock Redis for fast tests
   - ✅ **DO**: Mock file I/O with temp directories
   - ✅ **DO**: Use `@pytest.mark.skip` for GPU tests until ready
   - ❌ **DON'T**: Require GPU for unit tests

2. **Test Data Builders**:
   - ✅ **DO**: Use builder pattern for test data
   - ✅ **DO**: Make builders composable and chainable
   - ✅ **DO**: Provide sensible defaults
   - ❌ **DON'T**: Duplicate test data setup across tests

### Production Deployment

1. **Resource Management**:

   - ✅ **DO**: Use connection pooling for Redis
   - ✅ **DO**: Set appropriate timeouts everywhere
   - ✅ **DO**: Implement health checks that actually check things
   - ✅ **DO**: Monitor queue depths and worker heartbeats
   - ❌ **DON'T**: Return hardcoded health check responses

2. **Logging and Observability**:

   - ✅ **DO**: Use structured logging (JSON in production)
   - ✅ **DO**: Include request IDs for tracing
   - ✅ **DO**: Log at appropriate levels (DEBUG/INFO/WARNING/ERROR)
   - ✅ **DO**: Include context (path, method, client) in logs
   - ❌ **DON'T**: Log sensitive data (API keys, PII)
   - ❌ **DON'T**: Log at DEBUG level in production

3. **Error Recovery**:
   - ✅ **DO**: Implement graceful degradation
   - ✅ **DO**: Retry transient failures (network, Redis)
   - ✅ **DO**: Clean up resources in finally blocks
   - ❌ **DON'T**: Retry GPU OOM errors
   - ❌ **DON'T**: Silently swallow errors

### Common Pitfalls to Avoid

1. **❌ DON'T** load entire uploaded files into memory → Stream with `aiofiles`
2. **❌ DON'T** use `==` for API key comparison → Use `hmac.compare_digest()`
3. **❌ DON'T** forget to wrap RQ calls with `asyncio.to_thread()` → Always wrap
4. **❌ DON'T** skip GPU model unloading → Always unload in `finally`
5. **❌ DON'T** use async Redis with RQ → Use sync `Redis` client
6. **❌ DON'T** trust user filenames → Generate UUIDs
7. **❌ DON'T** expose internal errors to clients → Generic messages in prod
8. **❌ DON'T** forget to close Redis connection pools → Use lifespan hooks
9. **❌ DON'T** skip MIME type validation → Check both type and extension
10. **❌ DON'T** load multiple models on single GPU → Sequential only

---

## Summary: Key Principles

### Architecture

1. **Separation of Concerns**: API / Worker / Storage
2. **Sequential Processing**: One model at a time on GPU
3. **Asynchronous API**: 202 Accepted, poll for results
4. **Configuration Hierarchy**: Runtime > Env > Model > Library
5. **Factory Pattern**: Pluggable backends

### Code Quality

1. **Type Safety**: Full type hints everywhere
2. **Validation**: Pydantic for all data
3. **Logging**: Structured with context
4. **Error Handling**: Specific exceptions, retry logic
5. **Testing**: TDD with comprehensive coverage

### Operations

1. **Reproducibility**: Full versioning (NFR-1)
2. **Observability**: Logs, metrics, health checks
3. **Reliability**: Backpressure, retries, heartbeat
4. **Performance**: Optimized for 16-24GB GPU
5. **Documentation**: Inline comments, external docs

---

## Next Steps: Implementation Order

1. **Start with Config** (Day 1-2)

   - Test Settings loading
   - Test GenerationConfig merging
   - Foundation for everything else

2. **Build API Schemas** (Day 3)

   - Test all Pydantic models
   - Quick wins to build momentum

3. **Audio Preprocessing** (Day 4-5)

   - Start with simple logic
   - Mock external tools initially

4. **ASR Layer** (Day 6-7)

   - Factory first (no GPU)
   - Backend structure with mocks
   - GPU tests later

5. **LLM Processors** (Day 8-13)

   - Enhancement first (simpler)
   - Summarization second
   - Mock vLLM initially

6. **API Endpoints** (Day 14-16)

   - Health first (simplest)
   - Process endpoint
   - Status endpoint

7. **Worker Pipeline** (Day 17-21)

   - Structure first
   - State management
   - Full pipeline with mocks

8. **Integration** (Day 22-23)

   - API + Redis
   - No GPU yet

9. **GPU Tests** (Day 24-26)

   - Download models
   - Real processing
   - Performance validation

10. **Load Testing** (Day 27-28)
    - Concurrent requests
    - Error scenarios
    - Optimization

**Follow TDD religiously**: Test first, minimal implementation, refactor, repeat.

Good luck with your implementation! 🚀

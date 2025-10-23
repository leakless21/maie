## MAIE API Dataflow Overview

This document explains the request dataflow for each API endpoint, including the functions and modules involved, and provides a deep dive into the end-to-end processing for POST `/v1/process`.

### Runtime Components
- **Framework**: Litestar (`src/api/main.py`, `src/api/routes.py`)
- **Auth Guard**: API key guard (`src/api/dependencies.py#api_key_guard`)
- **Schemas**: Pydantic models (`src/api/schemas.py`)
- **Queue**: RQ (Redis-backed) for background jobs (`src/api/dependencies.py#get_rq_queue`)
- **Results Store**: Redis DB for task status/results (`src/api/dependencies.py#get_results_redis`)
- **Worker**: Pipeline executor (`src/worker/pipeline.py`)
- **Settings/Config**: `src/config/settings` exposed via `from src.config import settings`

### Logging
- All service logs flow through Loguru (`src/config/logging.py`).
- Standard-library and third‑party loggers (e.g., uvicorn, vLLM) are intercepted into the same sinks.
- In production, console logs default to JSON; files rotate and retain logs.
- A correlation id is supported and can be bound per request/task.

### Global App Wiring
- App entry `Litestar` and OpenAPI: `src/api/main.py`
- Registered items:
  - `route_handlers`: `ProcessController`, `StatusController`, `ModelsController`, `TemplatesController`, plus `GET /health`
  - `dependencies`: `validate_request_data`
  - `exception_handlers` for `NotAuthorizedException` (401), `HTTPException`, `ValidationException`, and generic `Exception`
  - `cors_config` if available


## Endpoints and Dataflow

### GET `/health`
- Handler: `health()` in `src/api/main.py`
- Computes and returns `HealthResponse` (`src/api/schemas.py`):
  - `status`, `version`, `redis_connected`, `queue_depth`, `worker_active`
- Involved modules/functions:
  - `src/api/main.py:health`
  - `src/api/schemas.py:HealthResponse`
  - Reads from `settings`


### POST `/v1/process` (guarded)
- Route/controller: `ProcessController.process_audio` in `src/api/routes.py`
- Guard: `api_key_guard` (`src/api/dependencies.py`)
- Request schema: `ProcessRequestSchema` (`src/api/schemas.py`)
- Dataflow (high level):
  1. Validate API key via `api_key_guard`
  2. Parse multipart form into `ProcessRequestSchema`, normalize `features` and `asr_backend`
  3. Validate file extension/MIME and queue capacity
  4. Generate `task_id` and stream-save file to `settings.audio_dir`
  5. Create initial task hash in results Redis (DB = `settings.redis_results_db`)
  6. Enqueue background job `process_audio_task` via RQ
  7. Return `ProcessResponse` with `task_id` and status `PENDING`
- Helper functions involved:
  - `check_queue_depth` → RQ queue depth
  - `save_audio_file_streaming` → upload streaming + size enforcement
  - `create_task_in_redis` → writes `task:{uuid}` to results Redis
  - `enqueue_job` → enqueues `src/worker/pipeline.process_audio_task`


### GET `/v1/status/{task_id}` (guarded)
- Route/controller: `StatusController.get_status` (`src/api/routes.py`)
- Guard: `api_key_guard` (`src/api/dependencies.py`)
- Fetches task by key `task:{task_id}` from results Redis via `get_task_from_redis`
- Deserializes fields: `features`, `results`, `metrics`, `versions` if JSON
- Returns `StatusResponseSchema` (`src/api/schemas.py`) or 404


### GET `/v1/models`
- Route/controller: `ModelsController.get_models` (`src/api/routes.py`)
- Calls `get_available_models()` which enumerates `ASRFactory.BACKENDS`
- Returns `ModelsResponseSchema` with list of `ModelInfoSchema`


### GET `/v1/templates`
- Route/controller: `TemplatesController.get_templates` (`src/api/routes.py`)
- Calls `scan_templates_directory()` to read `settings.templates_dir/*.json`
- Returns `TemplatesResponseSchema` with list of `TemplateInfoSchema`


## Detailed Flow: POST `/v1/process`

This section details the end-to-end flow, including functions, files, Redis keys, and error paths.

### 1) Request ingress and authentication
- Entry: Litestar routing → `ProcessController.process_audio` in `src/api/routes.py`
- Guard: `api_key_guard(connection, route_handler)` (`src/api/dependencies.py`)
  - Reads `X-API-Key`/`x-api-key`
  - Validates format and constant-time compares against `settings.secret_api_key` and `settings.fallback_api_keys`
  - On failure: raises `NotAuthorizedException` → handled as HTTP 401 by `src/api/main.py`

### 2) Payload parsing and schema validation
- Incoming multipart is bound to `data: Dict[str, Any]` with `Body(media_type=RequestEncodingType.MULTI_PART)`
- `ProcessRequestSchema.model_validate(prepared_data)` (`src/api/schemas.py`)
  - Coerces `file` to `UploadFile` when possible
  - Default `features`: `[clean_transcript, summary]`
  - `model_validator` enforces that `template_id` is present if `summary` requested
- Manual normalization and checks in controller:
  - Normalize `features` (supports string, JSON array string, or list)
  - Normalize and validate `asr_backend` against `ASRFactory.BACKENDS`
  - Validate file extension and MIME (`allowed_extensions`, `allowed_mime_types`)
  - Errors: `HTTP_422_UNPROCESSABLE_ENTITY` (schema/parameter), `HTTP_415_UNSUPPORTED_MEDIA_TYPE` (MIME), each raised via `HTTPException`

### 3) Backpressure via queue depth
- `check_queue_depth()` (`src/api/routes.py`) uses `get_rq_queue()` to read `queue.count`
- If full, returns 429 with `APIValidationError` details

### 4) Task creation and upload persistence
- `task_id = uuid.uuid4()`
- `file_path = await save_audio_file_streaming(file, task_id)` (`src/api/routes.py`)
  - Streams to disk (`settings.audio_dir/<task_id>.<ext>`, ext sanitized/validated)
  - Enforces `settings.max_file_size_mb` while streaming; on overflow cleans up partial file and raises 413

### 5) Initialize task record in results Redis
- Function: `create_task_in_redis(task_id, request_params)` (`src/api/routes.py`)
- Client: `get_results_redis()` (`src/api/dependencies.py`) → DB = `settings.redis_results_db`
- Key: `task:{task_id}` (Hash)
- Initial fields:
  - `task_id`: string
  - `status`: `PENDING`
  - `submitted_at`: ISO8601
  - `features`: JSON array (string)
  - `template_id`: string
  - `file_path`: string
  - `asr_backend`: string
- TTL: `settings.result_ttl`

### 6) Enqueue background job
- Function: `enqueue_job(task_id, file_path, request_params)` (`src/api/routes.py`)
- Queue: `get_rq_queue()` (synchronous Redis client)
- Enqueues `src/worker/pipeline.process_audio_task` with parameters:
  - `task_id`, `audio_path`, `features`, `template_id`, `asr_backend`
  - `redis_host`, `redis_port`, `redis_db` (results DB)
- Options: `job_id = str(task_id)`, `job_timeout = settings.job_timeout`, `result_ttl = settings.result_ttl`

### 7) Immediate response
- Returns `ProcessResponse(task_id=<uuid>, status="PENDING")` (`src/api/schemas.py`)

---

## Worker Pipeline: `process_audio_task`
Location: `src/worker/pipeline.py`

### Parameters (from enqueue)
- `audio_path`: persisted upload path
- `asr_backend`: e.g., `whisper`, `chunkformer`
- `features`: e.g., `["clean_transcript", "summary"]`
- `template_id`: required if `summary` requested
- `redis_host`, `redis_port`, `redis_db`: results Redis connection

### Lifecycle and status transitions
- Job context: `rq.get_current_job()` → `job.id == task_id`; `task_key = f"task:{job_id}"`
- Status updates stored in hash `task:{job_id}` via `_update_status(redis, task_key, TaskStatus, details)`

#### Stage 1: PREPROCESSING
1. `_update_status(..., PREPROCESSING)`
2. Validate `audio_path` → raise `AudioValidationError` if invalid
3. `AudioPreprocessor().preprocess(Path(audio_path))` → may normalize to 16kHz mono WAV; returns `metadata` with duration/path
4. Choose `processing_audio_path` (normalized if present), `audio_duration = metadata.duration`
5. Errors handled and recorded via `handle_processing_error(..., stage="preprocessing")` with `TaskStatus.FAILED`

#### Stage 2: PROCESSING_ASR
1. `_update_status(..., PROCESSING_ASR)`
2. `load_asr_model(asr_backend, **config)` using `ASRFactory.create(...)`
3. `execute_asr_transcription(asr_model, processing_audio_path, audio_duration)` → `(transcription, asr_rtf, confidence, asr_metadata)`
4. Always `unload_asr_model(asr_model)` in `finally` (GPU memory cleanup)
5. On errors, raise `ASRProcessingError` and mark FAILED via `handle_processing_error`

#### Stage 3: PROCESSING_LLM
1. `_update_status(..., PROCESSING_LLM, {transcription_length})`
2. Guard: non-empty `transcription` or raise `LLMProcessingError`
3. `load_llm_model()` (lazy load)
4. `execute_llm_processing(llm_model, transcription, features, template_id, asr_backend)`
   - If `clean_transcript` requested and enhancement needed: `llm_model.enhance_text(...)` with metrics
   - If `summary` requested: `llm_model.generate_summary(transcript, template_id)` (requires `template_id`)
5. `get_version_metadata(asr_metadata, llm_model)` while model loaded
6. Always `unload_llm_model(llm_model)` in `finally`
7. Errors raise `LLMProcessingError` and mark FAILED

#### Stage 4: COMPLETE
1. `metrics = calculate_metrics(transcription, clean_transcript, start_time, audio_duration, asr_rtf)`
2. Ensure `version_metadata` present via `get_version_metadata(asr_metadata, None)` if needed
3. Build `result = { versions, metrics, results: {} }` and include requested features only:
   - `raw_transcript` → transcription
   - `clean_transcript` → enhanced transcript or raw if enhancement skipped
   - `summary` → structured summary (if present)
4. `_update_status(..., COMPLETE, { completed_at, versions, metrics, results })`
5. Return `result` dict (also available in RQ job metadata depending on configuration)

### Redis Keys and Fields Summary
- Key format: `task:{uuid}` (Hash in results DB)
- Fields (subset, may be updated over time):
  - `task_id`: string
  - `status`: one of `PENDING`, `PREPROCESSING`, `PROCESSING_ASR`, `PROCESSING_LLM`, `COMPLETE`, `FAILED`
  - `submitted_at`, `updated_at`, `completed_at`: timestamps
  - `features`: JSON string (array)
  - `template_id`: string
  - `file_path`: string (upload location)
  - `asr_backend`: string
  - `versions`: JSON string
  - `metrics`: JSON string
  - `results`: JSON string (contains selected outputs)
  - `error`, `stage`, `error_code`: on failure paths

### Error Handling Summary
- API surface:
  - 401 on invalid/missing API key (`NotAuthorizedException`)
  - 413 on upload exceeding `settings.max_file_size_mb` during streaming
  - 415 on unsupported file type/MIME
  - 422 on invalid payload or parameters
  - 429 when queue is full
- Worker:
  - Domain-specific exceptions (`AudioValidationError`, `AudioPreprocessingError`, `ASRProcessingError`, `LLMProcessingError`, `ModelLoadError`) mark task FAILED and persist error info into Redis via `_update_status`
  - Unexpected exceptions are logged with full traceback and surfaced as `FAILED` tasks with structured error metadata

### Primary Functions by Stage (Quick Index)
- Ingress: `ProcessController.process_audio` (`src/api/routes.py`)
- Guard: `api_key_guard` (`src/api/dependencies.py`)
- Upload: `save_audio_file_streaming` (`src/api/routes.py`)
- Task init: `create_task_in_redis` (`src/api/routes.py`)
- Queue: `enqueue_job` (`src/api/routes.py`)
- Worker entry: `process_audio_task` (`src/worker/pipeline.py`)
- Preprocess: `AudioPreprocessor.preprocess` (`src/processors/audio/preprocessor.py`)
- ASR model: `load_asr_model` → `ASRFactory.create` (`src/processors/asr/factory.py`)
- ASR run: `execute_asr_transcription`
- ASR unload: `unload_asr_model`
- LLM model: `load_llm_model`
- LLM run: `execute_llm_processing`
- LLM unload: `unload_llm_model`
- Versioning: `get_version_metadata`
- Metrics: `calculate_metrics`
- Status updates: `_update_status`
- Failure updates: inline try/except blocks within `process_audio_task` (`src/worker/pipeline.py`)

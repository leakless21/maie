# Edge Deployment Plan – MAIE (Single-Task Device)

This document is an engineering plan for the edge/single‑task deployment mode. It focuses on architecture and required code changes.

## 1. Objectives & Constraints

- **Goal**: Run the Modular Audio Intelligence Engine on an edge device that processes **one audio task at a time** while reusing the existing MAIE codebase and avoiding forks.
- **Constraints**:
  - Limited compute (CPU/GPU memory, RAM, disk space).
  - Likely offline or restricted network (no dynamic model downloads).
  - Need deterministic, single-threaded behavior for predictable resource usage.
- **Decision**: Keep the upstream pipeline intact; expose it via configuration and/or a lightweight wrapper so the same tracked codebase can continue to evolve with upstream fixes and tests.
- **Execution model for this device**: Implement **Option C – Minimal Synchronous Edge API** as the primary path; keep Options A and B documented as alternatives for deployments that prefer a CLI-only interface or reuse of the existing async API + worker model.

## 2. Target Architecture for the Edge Device

### 2.1 Execution Models

1. **Option C – Minimal synchronous edge API (recommended)** – Expose `process_audio_task` via `src/api/edge_main.py` as a single-process HTTP handler that executes tasks inline (no Redis/RQ) with an application-level lock. This is the execution model detailed in §4.
2. **Option A – Embedded pipeline CLI** – Import `process_audio_task` from `src/worker/pipeline.py` and invoke it directly in a small script (no Redis, no RQ, no HTTP API). One process handles one job, sequentially.
3. **Option B – Local API + worker** – Keep the Litestar API (`src/api/main.py`) and an RQ worker, but configure them with concurrency=1 and a shallow queue so at most one task is processed at a time and backlog remains bounded.

### 2.2 Shared Characteristics

- The pipeline remains sequential (preprocess → ASR → optional VAD → optional diarization → optional LLM → metrics/cleanup).
- Use the existing `AppSettings` configuration system (`src/config/model.py` and `src/config/profiles.py`) to tailor behavior without introducing forks.
- Models must be placed under local directories (`data/models/…`), and environment variables ensure MAIE never makes runtime network calls.

## 3. Configuration Strategy – introduce an `edge` profile

1. **Add `EDGE_PROFILE` to `src/config/profiles.py`**, derived from production but tuned for edge:
   - `environment`: `"edge"`
   - `debug`: `False`, `verbose_components`: `False`
   - `logging`: low retention/rotation (e.g., `"log_rotation": "50 MB"`, `"log_retention": "7 days"`)
   - `api.max_file_size_mb`: set to expected maximum (e.g., `100`)
   - `redis.max_queue_depth`: small (e.g., `5`)
   - `worker`: `worker_concurrency=1`, `worker_prefetch_multiplier=1`, `job_timeout` tuned, `result_ttl` shorter
   - `diarization.enabled`: optional; keep `require_cuda=False`
   - `features.enable_enhancement`: `False` unless improvement is required
   - `paths`: set to device-specific directories (overwrite in `.env` if needed)
2. **Register** the profile in the `PROFILES` dict alongside `development` and `production`.
3. **Select the profile** on-device via the `ENVIRONMENT` setting (for example, `ENVIRONMENT=edge` in the process environment or `.env` file).

## 4. Option C – Minimal Synchronous Edge API (`src/api/edge_main.py`)

Primary execution model for this edge device: provides an HTTP surface while keeping processing single-task, single-process, and Redis-free.

### 4.1 Purpose and behavior

- Provide a **minimal HTTP interface** on the edge device without Redis/RQ queueing.
- Expose a simple synchronous endpoint that:
  - Accepts an uploaded audio file and processing parameters.
  - Runs `process_audio_task` **inline** in the same process.
  - Returns the **full pipeline result payload** in the HTTP response.
- Enforce **single-task semantics** via an application-level lock, so only one request is processed at a time.

### 4.2 Implementation overview

- Module: `src/api/edge_main.py`
- Key components:
  - `app: Litestar` – minimal app instance for edge deployments.
  - `GET /health` – simple health check returning `{"status": "healthy"}`.
  - `POST /v1/process-sync` – synchronous processing endpoint.
  - Global `asyncio.Lock` (`_process_lock`) to serialize processing calls.
  - Reuse of:
    - `api_key_guard` for API key authentication.
    - `save_audio_file_streaming` for streamed upload and size validation.
    - `ProcessRequestSchema` and `Feature` enums for request validation.
    - `process_audio_task` as the core pipeline implementation.

### 4.3 Request specification – `POST /v1/process-sync`

- The request body matches the existing `ProcessRequestSchema` used by the main API.
- At a high level, the endpoint accepts:
  - A multipart/form-data upload containing the audio file (`file`).
  - Optional parameters such as `features`, `template_id`, `asr_backend`, `enable_diarization`, `enable_vad`, and `vad_threshold`.
- The full external API contract (fields, defaults, and error codes) is documented in `docs/EDGE_DEPLOYMENT_GUIDE.md` under “API Reference → POST /v1/process-sync” and should be treated as the source of truth for clients.

### 4.4 Processing and pipeline invocation

- After validation:
  - A `task_id` UUID is generated for logging and directory isolation.
  - The audio file is streamed to disk under `settings.paths.audio_dir / <task_id> / raw<ext>`.
  - A `task_params` dict is built:
    - `task_id`, `audio_path`, `features`, `template_id`, `asr_backend`
    - `enable_diarization`, `enable_vad`, `vad_threshold`
    - `redis_host`, `redis_port`, `redis_db` (present but effectively unused because there is no RQ job)
    - `correlation_id` set to `task_id` for tracing.
  - `process_audio_task(task_params)` is invoked via `anyio.to_thread.run_sync` to avoid blocking the event loop.
  - Because `get_current_job()` returns `None` when no RQ worker is present, `process_audio_task` does not attempt Redis status updates; it simply returns the result structure.

### 4.5 Single-task enforcement

- Global `asyncio.Lock` (`_process_lock`) in `edge_main.py` wraps the pipeline call:
  - Only one request can hold the lock at a time.
  - Additional concurrent `/v1/process-sync` calls will queue at the application level until the lock is released.
- This mirrors the hardware constraint of “one task at a time” regardless of HTTP clients.

### 4.6 Response specification

- On success, the handler returns the dictionary produced by `process_audio_task` (augmented with `task_id` if missing) as JSON with HTTP `200 OK`.
- The shape is identical to the existing queue-based pipeline result (`task_id`, `versions`, `metrics`, `results`, and optional `error`/`status` fields); see `docs/EDGE_DEPLOYMENT_GUIDE.md` “API Reference → POST /v1/process-sync” for an example payload that client integrators should rely on.

### 4.7 Running the edge API

- Start via Python:
  - `python -m src.api.edge_main`
- Or via Uvicorn:
  - `uvicorn src.api.edge_main:app --host 0.0.0.0 --port 8000 --log-config none --access-log false`
- Required environment:
  - `ENVIRONMENT=edge` (or appropriate profile selector).
  - `APP_API__secret_key` set to a strong API key.
  - Model and audio/template paths configured and populated as described in earlier sections.

## 5. Resource & Feature Tuning

This section captures the high-level tuning intent for edge. Concrete configuration matrices and environment variable examples live in `docs/EDGE_DEPLOYMENT_GUIDE.md` (“Resource Management” and “Performance Tuning”).

### 5.1 Models

- Pre-download required ASR/LLM/diarization models into local `data/models/`.
- Tailor to the device:
  - Smaller Whisper or ChunkFormer variant for CPU.
  - Smaller/on-device LLM or disable enhancement/summary if memory constrained.

### 5.2 Diarization

- Use `settings.diarization.require_cuda=False` to keep the system alive even if CUDA is missing.
- Enable only when the device has sufficient compute and you need speaker labels.
- Provide CLI flag or API parameter (`enable_diarization`) to opt into diarization per task.

### 5.3 GPU/CPU Mode

- In `EDGE_PROFILE`, set:
  - `asr.whisper_device`/`chunkformer_device` to `"cpu"` or `"cuda"` depending on hardware.
  - Batch sizes (if available) to values that balance latency/memory.
  - Disable GPU-specific features if running headless CPU (for example, by forcing `has_cuda()` false or setting `PYTORCH_CUDA_ALLOC_CONF`).

## 6. Logging, Cleanup, and Monitoring

At a high level, edge deployments should favor aggressive cleanup and small log retention; see `docs/EDGE_DEPLOYMENT_GUIDE.md` (“Configuration → Edge Profile Settings” and “Monitoring & Troubleshooting”) for concrete defaults and environment overrides.

1. Use `CleanupSettings` to schedule:
   - Frequent audio/log cleanup (for example `audio_cleanup_interval=600` seconds, `audio_retention_days=1`).
   - Cache/disk monitoring to avoid exhaustion.
2. Logging:
   - Small rotation/retention (for example `log_rotation="50 MB"`, `log_retention="7 days"`).
   - Log to device-local `logs/` or other persistent path.
3. Health:
   - If using an API, `/health` can be used for basic liveness checks.
   - The CLI wrapper can output pass/fail status plus metrics to help automation detect issues.

## 7. Verification & Hardening

This section complements the operational testing guidance in `docs/EDGE_DEPLOYMENT_GUIDE.md` (“Testing” and “Maintenance”).

1. **Local dev verification**:
   - Run targeted tests such as `tests/integration/processors/audio/test_diarizer_integration.py`.
   - Execute the primary edge API (Option C) against sample audio; optionally also test the CLI (Option A) or existing async API + worker (Option B) if you plan to support those variants.
   - Confirm `process_audio_task` returns structured results and metrics.
2. **On-device validation**:
   - Deploy code and `.env`.
   - Run sample file to confirm the pipeline completes without hitting queueing or memory issues.
   - Monitor resource usage (CPU/GPU, RAM, disk).
   - Inject failure scenarios (missing file, corrupted audio) to test error handling.
3. **Operationalization**:
   - Define startup scripts (systemd, Docker run, etc.).
   - Document log locations, output file format, and instructions for model updates.
   - Keep a maintenance guide describing how to upgrade MAIE or models in the future.

## 8. Optional Enhancements

1. Add a minimal status CLI command to show the last run metadata (`maie-edge status`).
2. Provide a small web-based admin UI (if using an API) for local uploads and results viewing.
3. Integrate remote config updates if deploying a fleet of such devices (for example, syncing `.env` or zipped profiles).

# MAIE — Modular Audio Intelligence Engine

A lightweight API and worker skeleton for audio processing tasks (ASR, summarization, etc.). This repository contains the API, worker pipeline skeletons, templates, and helper scripts so you can run and test the service locally.

## What this repo provides

- `src/` — application source
- `src/api/` — API endpoints and controllers
- `src/worker/` — worker entrypoint and pipeline (skeleton)
- `templates/` — example prompt/templates used by summarization features
- `data/` — runtime data (audio, models, etc.)
- `scripts/` — development helpers (`dev.sh`, `test.sh`, `lint.sh`)

Note: Several parts of the business logic are intentionally skeletons/placeholders so you can iterate on the API and integration points first. See "Implementation notes" below.

## Prerequisites

- Python 3.11+ (project configured for this workspace)
- `uv` (used by `scripts/dev.sh` to manage the virtual environment)
- `uvicorn` (if you want to run the ASGI app directly without `uv`)

If you don't have `uv` installed, you can still run the app with `uvicorn` after installing dependencies in your preferred environment.

## Quick start — run locally

1. Optional: copy and edit environment variables

```bash
cp .env.template .env
# Edit .env to set values such as REDIS_URL, SECRET_API_KEY, etc.
```

2. Ensure runtime directories exist

```bash
mkdir -p data/audio data/models data/redis templates
```

3. Start the development environment

Start API only (auto-reload enabled by default):

```bash
./scripts/dev.sh --api-only --host 0.0.0.0 --port 8000
```

Start worker only:

```bash
./scripts/dev.sh --worker-only
```

Start both API and worker (default):

```bash
./scripts/dev.sh
```

If you don't want to use `uv`, you can run the API directly with `uvicorn`:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API endpoints (smoke tests)

After the API is running, try these quick checks.

- Health

```bash
curl -fsS http://localhost:8000/health
```

- List available models

```bash
curl -fsS http://localhost:8000/v1/models
```

- List available templates

```bash
curl -fsS http://localhost:8000/v1/templates
```

- Submit audio for processing (will save file and return a `task_id`, but background processing is a placeholder until Redis/worker wiring is implemented)

```bash
curl -v -F "file=@tests/assets/audio/sample.wav" \
		 -F 'features=["clean_transcript","summary"]' \
		 -F "template_id=meeting_notes_v1" \
		 http://localhost:8000/v1/process
```

## Important files and where to look

- `scripts/dev.sh` — development script that invokes `uv` and starts API and worker processes
- `src/api/routes.py` — API controllers and route handlers. Notable functions:
  - `ProcessController.process_audio` — accepts multipart uploads and saves audio
  - `save_audio_file` — helper that writes uploaded bytes to `data/audio/`
  - `create_task_in_redis`, `enqueue_job`, `get_task_from_redis` — placeholder stubs for Redis/RQ integration
- `src/config.py` — application settings and defaults (e.g., `max_file_size_mb`, `templates_dir`, `audio_dir`)
- `templates/` — sample JSON templates used by summarization features

## Implementation notes & TODOs

- The API skeleton will accept uploads and persist audio files on disk. However, task persistence and background job queuing are not implemented in this branch:

  - `create_task_in_redis` and `enqueue_job` are placeholders and need Redis/RQ wiring.
  - `get_task_from_redis` currently returns `None`, so `GET /v1/status/{task_id}` will return `404 Not Found` until task storage is implemented.

- Add Redis connection, create task records, and enqueue jobs (RQ or alternative) so workers can pick up and process audio asynchronously.

## Running tests and linters

Run unit tests:

```bash
./scripts/test.sh
```

Run linters/formatters:

```bash
./scripts/lint.sh
```

## Where to go next

- Implement Redis task storage and RQ enqueueing in `src/api/routes.py`.
- Implement worker processing pipeline in `src/worker/` to consume queued tasks and write results back to Redis.
- Add auth and rate limiting in `src/api/dependencies.py` if required for production.

---

If you need a trimmed README for packaging or a CONTRIBUTING guide, tell me what tone/level of detail you want and I will add it.

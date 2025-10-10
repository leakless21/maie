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

- Python 3.13+ (project configured for this workspace)
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

## Project Status

The API skeleton is functional and can accept audio uploads. Redis/RQ task persistence and background worker processing are not yet implemented - these are tracked as integration tasks in `docs/tdd-followups.md`.

Current capabilities:

- ✅ API endpoints for audio processing requests
- ✅ File upload validation and persistence
- ✅ Template and model discovery endpoints
- ✅ Health check endpoint
- ⏳ Redis task storage (placeholder)
- ⏳ RQ job enqueueing (placeholder)
- ⏳ Background worker processing (placeholder)

## Next Steps

To complete the implementation:

1. Integrate Redis client for task persistence (see `docs/tdd-followups.md`)
2. Implement RQ job enqueueing
3. Build worker processing pipeline in `src/worker/`
4. Add authentication and rate limiting if needed

See `docs/TDD.md` and `docs/PRD.md` for full architecture and requirements.

## Development

### Running tests and linters

Run unit tests:

```bash
./scripts/test.sh
```

Run linters/formatters:

```bash
./scripts/lint.sh
```

### GPU Support and cuDNN Setup

For GPU acceleration with Whisper models, cuDNN libraries must be accessible at runtime. The project uses cuDNN from the Python virtual environment (`nvidia-cudnn-cu12` package).

**Why LD_LIBRARY_PATH is Required:**

CTranslate2 (the inference engine used by faster-whisper) loads cuDNN libraries using the system's dynamic linker (`dlopen`), which doesn't respect Python's import paths. The libraries must be in `LD_LIBRARY_PATH` **before the Python process starts**. This is a known limitation of CTranslate2's PyPI distribution, which doesn't include RPATH configuration.

**Automatic Setup (Recommended):**

The `scripts/test.sh` script automatically configures the environment to find cuDNN libraries. Just run:

```bash
./scripts/test.sh --integration
```

**Manual Setup:**

If running tests or the application manually, you can source the cuDNN environment helper:

```bash
source scripts/use-cudnn-env.sh
```

This adds the venv's cuDNN library path to `LD_LIBRARY_PATH`. After sourcing, GPU-based ASR models will work correctly.

**For Production Deployments:**

Set `LD_LIBRARY_PATH` in your process manager or Docker container:

```bash
# In systemd unit file
Environment="LD_LIBRARY_PATH=/path/to/venv/lib/python3.11/site-packages/nvidia/cudnn/lib"

# In Docker Compose
environment:
  - LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib
```

**Verification:**

Check if cuDNN is accessible:

```bash
python -c "import site, os; sp = site.getsitepackages(); print([os.path.join(s, 'nvidia', 'cudnn', 'lib') for s in sp if os.path.isdir(os.path.join(s, 'nvidia', 'cudnn', 'lib'))])"
```

**Troubleshooting:**

If you see "Unable to load libcudnn_ops.so" errors:

1. Ensure the Python environment has `nvidia-cudnn-cu12` installed
2. Use `scripts/use-cudnn-env.sh` to set up the environment **before** starting Python
3. For system-wide issues, see `scripts/purge-cudnn.sh` for cleanup instructions

**Note:** Setting `LD_LIBRARY_PATH` from within Python doesn't work because the dynamic linker reads this variable only at process startup, before Python even loads.

## Additional Resources

- **Architecture & Design**: See `docs/TDD.md` for technical design
- **Requirements**: See `docs/PRD.md` for product requirements
- **Implementation Tasks**: See `docs/tdd-followups.md` for remaining work
- **Configuration**: See `.env.template` for environment variables

For questions or contributions, refer to the documentation in `docs/`.

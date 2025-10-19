## Developer API Workflow

This guide shows how to run the API locally, submit an audio request, and retrieve the results.

### Prerequisites
- Python 3.13+
- `pixi` (recommended for dev) or `uvicorn`
- Optional: Docker/Docker Compose for a full stack run
- Optional: export configuration overrides via `AppSettings` environment variables
  (e.g. `export APP_LOGGING__LOG_LEVEL=debug`)

### Start the API (and worker)
From the project root (`/home/cetech/maie`):

```bash
mkdir -p data/audio data/models data/redis templates

# Start API + worker (default)
./scripts/dev.sh

# API only
./scripts/dev.sh --api-only --host 0.0.0.0 --port 8000

# Without pixi (direct uvicorn)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Quick smoke checks
```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/v1/models
curl -fsS http://localhost:8000/v1/templates
```

### Submit audio for processing
This endpoint enqueues an asynchronous job and returns a `task_id` you can poll.

```bash
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'features=["clean_transcript","summary"]' \
  -F "template_id=meeting_notes_v1" \
  -F "asr_backend=whisper" \
  http://localhost:8000/v1/process

# Alternative: repeated features fields (also supported)
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F "features=clean_transcript" \
  -F "features=summary" \
  -F "template_id=meeting_notes_v1" \
  -F "asr_backend=chunkformer" \
  http://localhost:8000/v1/process

# Single-line variant (avoids line-continuation issues)
curl -X POST -H "X-API-Key: your-secret-key" -F "file=@tests/e2e/assets/sample.wav" -F 'features=["clean_transcript","summary"]' -F "template_id=meeting_notes_v1" -F "asr_backend=whisper" http://localhost:8000/v1/process
```

**ASR Backend Options:**
- `asr_backend`: Optional field (default: `"whisper"`)
- Valid values: `"whisper"`, `"chunkformer"`
- Case-insensitive (automatically normalized to lowercase)

Expected immediate response:

```json
{ "task_id": "abc123def456", "status": "PENDING" }
```

### Retrieve results (poll status)
```bash
TASK_ID="abc123def456"
curl -H "X-API-Key: your-secret-key" \
     http://localhost:8000/v1/status/$TASK_ID | jq '.'
```

When `status` becomes `COMPLETE`, the response includes:
- `results.clean_transcript` — cleaned transcript
- `results.summary` — structured summary `{ title, main_points[], tags[] }`
- `metrics` — e.g., `rtf`, `processing_time_seconds`, `asr_confidence_avg`
- `versions` — model and pipeline metadata for reproducibility

### How to “see the result”
- The result JSON is returned by `GET /v1/status/{task_id}`. Use `jq` to pretty-print.
- Example: `curl -H "X-API-Key: ..." http://localhost:8000/v1/status/$TASK_ID | jq '.'`

### Monitoring during development
- Watch the worker terminal for logs when jobs run
- Optional (Docker stack): RQ Dashboard at `http://localhost:9181`, Jaeger at `http://localhost:16686`

### Docker Compose workflow (alternative)
```bash
docker compose up -d

# Wait for API to be healthy
timeout 300 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'

# Submit and fetch results (same as above; include X-API-Key)
```

### Reference
Status response structure (fields vary by run):

```json
{
  "task_id": "...",
  "status": "COMPLETE",
  "metrics": { "rtf": 0.06, "processing_time_seconds": 162.8 },
  "results": {
    "clean_transcript": "...",
    "summary": { "title": "...", "main_points": ["..."], "tags": ["..."] }
  },
  "versions": { "pipeline_version": "1.0.0", "asr_backend": { ... }, "llm": { ... } }
}
```


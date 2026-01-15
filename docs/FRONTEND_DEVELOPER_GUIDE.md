# MAIE Frontend Developer Guide (Jetson Edge API)

**Version:** 2.0  
**Last Updated:** December 6, 2025  
**Target Audience:** Frontend & Mobile Developers targeting Jetson devices

> **Scope:** This document describes the **MAIE Edge API running on Jetson
> (Nano/Orin)**. It focuses on **ASR-only** workflows exposed by
> `src/api/edge_main.py`. For the full cloud/main API (LLM workflows,
> Redis queue), see `docs/CLIENT_DEVELOPER_GUIDE.md`.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [API Deployments](#api-deployments)
4. [Authentication](#authentication)
5. [Core Endpoints](#core-endpoints)
6. [SSE Streaming API](#sse-streaming-api)
7. [Request/Response Formats](#requestresponse-formats)
8. [Error Handling](#error-handling)
9. [Code Examples](#code-examples)
10. [Best Practices](#best-practices)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Base URL (Jetson Edge API)

The Jetson Edge API listens on port `8000` by default:

```
http://<jetson-ip>:8000
```

Examples:
- On the Jetson itself: `http://localhost:8000`
- From another machine on the LAN: `http://192.168.1.42:8000`

### Minimal Example – Synchronous Transcription

```javascript
// Submit audio for transcription (Jetson Edge API)
const formData = new FormData();
formData.append('file', audioFile);           // WAV, MP3, M4A, FLAC
formData.append('asr_backend', 'whisper');   // or 'chunkformer'

const response = await fetch('http://<jetson-ip>:8000/v1/transcribe', {
  method: 'POST',
  body: formData
});

if (!response.ok) {
  const errorBody = await response.json().catch(() => ({}));
  throw new Error(errorBody.detail || `HTTP ${response.status}`);
}

const result = await response.json();
console.log(result.transcript);
```

For real-time progress updates and partial segments, use
`POST /v1/transcribe/stream` with Server-Sent Events (see
[SSE Streaming API](#sse-streaming-api)).

---

## Architecture Overview

### System Components

```
┌─────────────┐     ┌──────────────────────────────┐
│   Client    │────▶│   MAIE Edge API (Litestar)   │
│ (Web/Mobile)│     └──────────────┬───────────────┘
└─────────────┘                    │
                                   ▼
                           ┌──────────────┐
                           │   ASR Core   │  (Whisper / ChunkFormer)
                           └──────────────┘
                                   │
                                   ▼
                           ┌──────────────┐
                           │ JSON Result  │
                           └──────────────┘
```

### Processing Pipeline

```
Audio Upload → Preprocessing → ASR → Post-processing → JSON Result
                   ↓              ↓
                VAD /           Trans-
              Duration           cribe
```

### Deployment Types

| Deployment              | Use Case                         | Features                         |
|-------------------------|----------------------------------|----------------------------------|
| **Edge API (Sync)**     | Single-shot transcription        | ASR only (Whisper/ChunkFormer)  |
| **Edge API (Streaming)**| Real-time progress + segments    | ASR + SSE progress events       |

On Jetson, the Edge API runs in a **single process** with an
application-level lock (`_process_lock`) to guarantee that **only one
transcription runs at a time**. Additional requests wait for the lock
instead of being queued in Redis.

---

## API Deployments

### Edge API (Jetson Only)

**Capabilities (from `src/api/edge_main.py`):**
- Synchronous ASR transcription (`POST /v1/transcribe`)
- Streaming ASR with progress and segment events
  (`POST /v1/transcribe/stream`)
- Simple health reporting (`GET /health`)
- Model metadata (`GET /v1/models`)
- Template metadata + CRUD (`GET/POST/PUT/DELETE /v1/templates`)
- Single-task semantics enforced via `_process_lock`

**Endpoints:**
- `GET /` - Basic API info and endpoints
- `GET /health` - Health check (`status: healthy | busy | unhealthy`)
- `GET /v1/models` - List available ASR models and defaults
- `GET /v1/templates` - List templates available on-device (read-only)
- `GET /v1/templates/{id}` + `/schema` - Fetch template prompt + schema bundle
- `POST /v1/templates` - Create/update template bundles locally
- `PUT /v1/templates/{id}` - Modify an existing template
- `DELETE /v1/templates/{id}` - Remove template bundle from disk
- `POST /v1/transcribe` - Synchronous transcription
- `POST /v1/transcribe/stream` - Streaming transcription with SSE

**What is intentionally not available on Jetson:**
- ❌ LLM summarization / enhancement
- ❌ Speaker diarization
- ❌ Redis/RQ queue

Use the **main MAIE API** on a server-class machine if you need LLM
summaries or async/queued processing.

---

## Authentication

The Jetson Edge API **does not enforce API-key authentication by
default**. All endpoints in `src/api/edge_main.py` are open on the
configured host/port.

Recommended patterns for production:

- Put Jetson behind a **reverse proxy or API gateway** (Nginx, Envoy,
  Cloudflare Tunnel, etc.) that:
  - Terminates TLS.
  - Enforces authentication (API key, OAuth2, mTLS, etc.).
- Keep the Jetson API bound to a **private network** IP whenever
  possible.
- If your gateway uses headers like `X-API-Key`, send them to the
  gateway; the Jetson process itself ignores them.

From a frontend perspective, you typically only need:
- The base URL to the gateway or Jetson device.
- Any auth headers **required by that gateway**, not by MAIE Edge
  itself.

---

## Core Endpoints

### 1. POST /v1/transcribe (Jetson Edge API)

Synchronous transcription endpoint for edge devices.

**Request:**
```http
POST /v1/transcribe HTTP/1.1
Host: <jetson-ip>:8000
Content-Type: multipart/form-data

file: <audio-file>
asr_backend: "whisper"
enable_vad: true
vad_threshold: 0.5
language: "en"
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Audio file |
| `asr_backend` | String | `whisper` | ASR engine: `whisper` or `chunkformer` |
| `enable_vad` | Boolean | `true` | Voice Activity Detection |
| `vad_threshold` | Float | `0.5` | VAD confidence (0.0-1.0) |
| `language` | String | `null` | Language code (e.g., `vi`, `en`). Auto-detect if null |

**Response (200 OK):**
```json
{
  "task_id": "abc-123",
  "status": "completed",
  "transcript": "Full transcript text here",
  "segments": [
    {
      "start": 0.0,
      "end": 5.5,
      "text": "Hello world",
      "speaker": null
    }
  ],
  "duration_seconds": 60.0,
  "processing_time_seconds": 3.2,
  "asr_backend": "whisper",
  "language": "en"
}
```


### 2. GET /v1/templates (Jetson Edge API)

Use this endpoint to discover which template bundles are bundled with the
Jetson image. Each entry tells you the template identifier, example
output, and where to fetch the JSON schema.

**Request:**
```http
GET /v1/templates HTTP/1.1
Host: <jetson-ip>:8000
```

**Response (200 OK):**
```json
{
  "templates": [
    {
      "id": "meeting_notes_v2",
      "name": "Meeting Notes v2",
      "description": "Structured meeting notes summary",
      "schema_url": "/v1/templates/meeting_notes_v2/schema",
      "parameters": {
        "summary": {"type": "string"}
      },
      "example": {
        "summary": "Agenda recap...",
        "action_items": ["Follow up with Alex"]
      }
    }
  ]
}
```

Follow-up endpoints:
- `GET /v1/templates/{id}` – returns the prompt template + schema bundle
  for rendering previews or configuring forms.
- `GET /v1/templates/{id}/schema` – raw JSON schema (handy if you only
  need validation rules).

Template changes take effect instantly on Jetson because the API writes
directly to the on-device bundle directory. Use the POST/PUT/DELETE
endpoints (below) to manage them remotely, or ship updated template
directories with your device image.

---

### 3. POST /v1/templates (Jetson Edge API)

Create or upload a new template bundle directly on the Jetson device.
Useful for field deployments where you need to tweak prompts without
rebuilding the system image.

**Request:**
```http
POST /v1/templates HTTP/1.1
Host: <jetson-ip>:8000
Content-Type: application/json

{
  "id": "edge_notes_v1",
  "schema_data": {
    "title": "Edge Notes",
    "type": "object",
    "properties": {
      "summary": {"type": "string"},
      "action_items": {"type": "array", "items": {"type": "string"}}
    }
  },
  "prompt_template": "Summarize {{ transcript }}",
  "example": {
    "summary": "Quick recap",
    "action_items": ["Email recap"]
  }
}
```

**Response (201 Created):**
```json
{
  "id": "edge_notes_v1",
  "name": "Edge Notes",
  "prompt_template": "Summarize {{ transcript }}",
  "schema_data": { "...": "..." }
}
```

### 4. PUT /v1/templates/{id} (Jetson Edge API)

Update any part of an existing template bundle. Supply whichever fields
you need to change (`schema_data`, `prompt_template`, `example`).

```http
PUT /v1/templates/edge_notes_v1 HTTP/1.1
Host: <jetson-ip>:8000
Content-Type: application/json

{
  "prompt_template": "Updated prompt {{ transcript }}"
}
```

Response mirrors the template detail payload (200 OK).

### 5. DELETE /v1/templates/{id} (Jetson Edge API)

Remove a template bundle from the Jetson filesystem.

```http
DELETE /v1/templates/edge_notes_v1 HTTP/1.1
Host: <jetson-ip>:8000
```

Response: `204 No Content`.

> **Security note:** The Jetson edge API does not enforce authentication
> by default, so place it behind a trusted gateway when exposing these
> template mutating endpoints over the network.

---

## SSE Streaming API

### POST /v1/transcribe/stream (Jetson Edge API)

Real-time progress updates via Server-Sent Events.

**Features:**
- Real-time progress percentage
- Stage-by-stage updates
- Individual segment streaming
- Final result delivery
- Error notifications

If the Jetson device is already processing another request, the stream
begins with a `progress` event at stage `queued` and message
`"Waiting for current task to complete"` (see
`transcribe_audio_stream()` in `src/api/edge_main.py`).

**Request:**
```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('asr_backend', 'whisper');

const response = await fetch('http://<jetson-ip>:8000/v1/transcribe/stream', {
  method: 'POST',
  body: formData
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      handleEvent(event);
    }
  }
}
```

### SSE Event Types

#### 1. Progress Event
```json
{
  "event": "progress",
  "data": {
    "stage": "transcribing",
    "progress": 45.2,
    "message": "Transcribing... (15.2s elapsed)",
    "details": { "segments_count": 5 }
  }
}
```

**Progress Stages:**

| Stage | Progress Range | Description |
|-------|---------------|-------------|
| `queued` | 0% | Request received |
| `uploading` | 0-10% | Saving audio file |
| `preprocessing` | 10-20% | Analyzing audio |
| `transcribing` | 20-95% | ASR processing |
| `postprocessing` | 95-99% | Finalization |
| `completed` | 100% | Done |

#### 2. Segment Event
```json
{
  "event": "segment",
  "data": {
    "index": 1,
    "start": 0.0,
    "end": 5.5,
    "text": "Hello world",
    "speaker": "SPEAKER_00"
  }
}
```

#### 3. Result Event
```json
{
  "event": "result",
  "data": {
    "task_id": "abc-123",
    "transcript": "Full transcript...",
    "segments": [...],
    "duration_seconds": 60.0,
    "processing_time_seconds": 3.2,
    "asr_backend": "whisper",
    "language": "en"
  }
}
```

#### 4. Error Event
```json
{
  "event": "error",
  "data": {
    "error": "File format not supported",
    "error_code": "TRANSCRIPTION_ERROR",
    "stage": "preprocessing"
  }
}
```

### Client Implementation

**JavaScript/TypeScript:**
```typescript
interface ProgressData {
  stage: string;
  progress: number;
  message?: string;
  details?: any;
}

interface SegmentData {
  index: number;
  start: number;
  end: number;
  text: string;
  speaker?: string;
}

async function streamTranscription(
  file: File,
  onProgress: (data: ProgressData) => void,
  onSegment: (data: SegmentData) => void,
  onComplete: (result: any) => void,
  onError: (error: any) => void
) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/v1/transcribe/stream', {
    method: 'POST',
    body: formData
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      
      if (line.startsWith('event: ')) {
        const eventType = line.slice(7);
        continue;
      }
      
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        
        if (data.stage) {
          onProgress(data);
        } else if (data.index !== undefined) {
          onSegment(data);
        } else if (data.task_id) {
          onComplete(data);
        } else if (data.error) {
          onError(data);
        }
      }
    }
  }
}
```

**Python (httpx):**
```python
import httpx
import json

async def stream_transcription(audio_path: str, on_event=None):
    async with httpx.AsyncClient(timeout=None) as client:
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path, f, 'audio/wav')}
            
            async with client.stream(
                'POST',
                'http://localhost:8000/v1/transcribe/stream',
                files=files
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if on_event:
                            on_event(data)
                        
                        if data.get('task_id'):
                            return data
```

---

## Request/Response Formats

### File Upload (multipart/form-data)

**Supported Audio Formats (Jetson Edge API):**
- WAV (`.wav`)
- MP3 (`.mp3`)
- M4A (`.m4a`)
- FLAC (`.flac`)

**Maximum File Size (Jetson profile):**
- Default: **50 MB** (`settings.api.max_file_size_mb`, overridden by
  `JETSON_PROFILE` in `src/config/profiles.py`)
- Requests exceeding this limit return **`413 Payload Too Large`**.

**Example (JavaScript → `/v1/transcribe`):**
```javascript
const formData = new FormData();
formData.append('file', audioFile, 'recording.wav');
formData.append('asr_backend', 'whisper');    // or 'chunkformer'
formData.append('enable_vad', 'true');        // optional
formData.append('vad_threshold', '0.5');      // optional
formData.append('language', 'en');            // optional

const res = await fetch('http://<jetson-ip>:8000/v1/transcribe', {
  method: 'POST',
  body: formData
});
```

**Example (cURL → `/v1/transcribe`):**
```bash
curl -X POST "http://<jetson-ip>:8000/v1/transcribe" \
  -F "file=@audio.mp3" \
  -F "asr_backend=whisper" \
  -F "enable_vad=true" \
  -F "vad_threshold=0.5" \
  -F "language=en"
```

### Text Processing

The Jetson Edge API is **ASR-only** and does **not** expose text-only
processing or LLM summarization endpoints (such as `/v1/process_text`).
If your application needs summarization, tagging, or other LLM-driven
features, route those requests to the **main MAIE API** on a server
instead of the Jetson device.

---

## Error Handling

### HTTP Status Codes (Jetson Edge API)

| Code | Status | Meaning |
|------|--------|---------|
| `200` | OK | Transcription completed successfully |
| `400` | Bad Request | Validation error (missing file, invalid params) |
| `404` | Not Found | Invalid path or method |
| `413` | Payload Too Large | File exceeds configured size limit |
| `415` | Unsupported Media Type | Invalid or malformed multipart upload |
| `500` | Internal Server Error | Unhandled server-side error |

`create_edge_app()` in `src/api/edge_main.py` sets
`request_max_body_size` based on `settings.api.max_file_size_mb`, so
oversized uploads are rejected automatically with `413`.

### Error Response Format (JSON)

Non-2xx responses from the Jetson Edge API follow one of these shapes:

- **Validation errors** (`400`, via `_handle_validation_exception`):

```json
{
  "detail": "Validation Error",
  "message": "Request validation failed",
  "path": "/v1/transcribe"
}
```

- **Not found** (`404`, via `_handle_not_found`):

```json
{
  "detail": "Not Found",
  "path": "/unknown"
}
```

- **Other HTTP/Generic errors**:

```json
{
  "detail": "Human-readable error message"
}
```

### Application-Level Errors

The synchronous transcription endpoint can return `200 OK` with a
payload whose `status` is `"failed"` (see `EdgeTranscribeResponse` in
`src/api/edge_main.py`):

```json
{
  "task_id": "abc-123",
  "status": "failed",
  "transcript": "",
  "duration_seconds": 0.0,
  "processing_time_seconds": 1.2,
  "asr_backend": "whisper",
  "error": "Transcription backend unavailable"
}
```

Frontend code should check both `response.ok` **and** the JSON
`status` field.

### SSE Error Events

For streaming, errors are delivered as SSE `error` events (see
`ErrorEvent` in `src/streaming/events.py`):

```json
{
  "event": "error",
  "data": {
    "error": "Something went wrong",
    "error_code": "TRANSCRIPTION_ERROR",
    "stage": "transcribing"
  }
}
```

`error_code` is optional and may be omitted for some failures. Clients
should always inspect `data.error` for a human-readable message.

### Error Handling Best Practices

```javascript
async function transcribeOnEdge(file, baseUrl) {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${baseUrl}/v1/transcribe`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));

      switch (response.status) {
        case 400:
          throw new Error(error.message || 'Request validation failed.');
        case 413:
          throw new Error('File is too large. Check Jetson max file size.');
        case 415:
          throw new Error('Unsupported file format. Use WAV, MP3, M4A, or FLAC.');
        default:
          throw new Error(error.detail || `Server error (HTTP ${response.status}).`);
      }
    }

    const result = await response.json();
    if (result.status === 'failed') {
      throw new Error(result.error || 'Transcription failed.');
    }

    return result;
  } catch (error) {
    console.error('Jetson transcription error:', error);
    throw error;
  }
}
```

---

## Code Examples

### React Hook for Audio Processing

```typescript
import { useState, useCallback } from 'react';

interface EdgeProcessingState {
  taskId: string | null;
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';
  transcript: string | null;
  segments: any[];
  error: string | null;
}

export function useEdgeTranscription(apiUrl: string) {
  const [state, setState] = useState<EdgeProcessingState>({
    taskId: null,
    status: 'idle',
    transcript: null,
    segments: [],
    error: null,
  });

  const transcribe = useCallback(
    async (
      file: File,
      options: {
        asrBackend?: 'whisper' | 'chunkformer';
        enableVad?: boolean;
        vadThreshold?: number;
        language?: string;
      } = {},
    ) => {
      setState(prev => ({ ...prev, status: 'uploading', error: null }));

      try {
        const formData = new FormData();
        formData.append('file', file);
        if (options.asrBackend) {
          formData.append('asr_backend', options.asrBackend);
        }
        if (options.enableVad !== undefined) {
          formData.append('enable_vad', String(options.enableVad));
        }
        if (options.vadThreshold !== undefined) {
          formData.append('vad_threshold', String(options.vadThreshold));
        }
        if (options.language) {
          formData.append('language', options.language);
        }

        const res = await fetch(`${apiUrl}/v1/transcribe`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          throw new Error(await res.text());
        }

        const data = await res.json();
        setState({
          taskId: data.task_id ?? null,
          status: data.status,
          transcript: data.transcript ?? null,
          segments: data.segments ?? [],
          error: data.error ?? null,
        });
      } catch (err: any) {
        setState(prev => ({
          ...prev,
          status: 'failed',
          error: err.message ?? String(err),
        }));
      }
    },
    [apiUrl],
  );

  return { state, transcribe };
}
```

### Vue 3 Composition API

```typescript
import { ref, computed } from 'vue';

export function useMAIEEdge(baseUrl: string) {
  const taskId = ref<string | null>(null);
  const status = ref<'idle' | 'uploading' | 'processing' | 'completed' | 'failed'>('idle');
  const transcript = ref<string | null>(null);
  const segments = ref<any[]>([]);
  const error = ref<string | null>(null);
  
  const isProcessing = computed(() => 
    ['uploading', 'processing'].includes(status.value)
  );
  
  async function processAudio(
    file: File,
    options: {
      asrBackend?: 'whisper' | 'chunkformer';
      enableVad?: boolean;
      vadThreshold?: number;
      language?: string;
    } = {},
  ) {
    status.value = 'uploading';
    error.value = null;
    
    const formData = new FormData();
    formData.append('file', file);
    if (options.asrBackend) {
      formData.append('asr_backend', options.asrBackend);
    }
    if (options.enableVad !== undefined) {
      formData.append('enable_vad', String(options.enableVad));
    }
    if (options.vadThreshold !== undefined) {
      formData.append('vad_threshold', String(options.vadThreshold));
    }
    if (options.language) {
      formData.append('language', options.language);
    }
    
    try {
      const res = await fetch(`${baseUrl}/v1/transcribe`, {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) throw new Error(await res.text());
      
      const data = await res.json();
      taskId.value = data.task_id ?? null;
      status.value = data.status;
      transcript.value = data.transcript ?? null;
      segments.value = data.segments ?? [];
      error.value = data.error ?? null;
    } catch (err: any) {
      status.value = 'failed';
      error.value = err.message ?? String(err);
    }
  }
  
  return {
    taskId,
    status,
    transcript,
    segments,
    error,
    isProcessing,
    processAudio
  };
}
```

### Python Client

```python
import requests
from typing import Any, Dict, Optional


class MAIEEdgeClient:
    """Minimal Python client for the Jetson Edge API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def transcribe(
        self,
        file_path: str,
        asr_backend: str = "whisper",
        enable_vad: bool = True,
        vad_threshold: float = 0.5,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit audio to POST /v1/transcribe and return the JSON result."""
        with open(file_path, "rb") as f:
            files = {"file": f}
            data: Dict[str, Any] = {
                "asr_backend": asr_backend,
                "enable_vad": str(enable_vad).lower(),
                "vad_threshold": str(vad_threshold),
            }
            if language:
                data["language"] = language

            response = self.session.post(
                f"{self.base_url}/v1/transcribe",
                files=files,
                data=data,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "failed":
                raise RuntimeError(result.get("error") or "Transcription failed")

            return result


# Usage
client = MAIEEdgeClient("http://localhost:8000")
result = client.transcribe("audio.mp3", asr_backend="whisper")
print(result["transcript"])
```

---

## Best Practices

### 1. File Upload Optimization

✅ **Do:**
- Validate file type and size before upload
- Show upload progress
- Compress large files when possible
- Use appropriate audio formats (MP3 for size, WAV for quality)

❌ **Don't:**
- Upload files without validation
- Block UI during upload
- Send unnecessarily high-quality audio

```javascript
// File validation
function validateAudioFile(file) {
  const maxSize = 50 * 1024 * 1024; // 50MB (Jetson default)
  const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/m4a', 'audio/flac'];
  
  if (file.size > maxSize) {
    throw new Error('File too large. Maximum size is 50MB by default on Jetson.');
  }
  
  if (!allowedTypes.includes(file.type)) {
    throw new Error('Unsupported file type. Use WAV, MP3, M4A, or FLAC.');
  }
  
  return true;
}
```

### 2. Progress Strategy (Jetson)

✅ **Do:**
- Use `POST /v1/transcribe` for simple, single-shot transcription.
- Use `POST /v1/transcribe/stream` + SSE for long audio or when you
  need live progress updates.
- Surface clear UI states: *idle → uploading → processing → completed /
  failed*.
- Respect the fact that Jetson only processes **one request at a time**
  (`_process_lock`); avoid firing multiple parallel transcriptions.

❌ **Don't:**
- Implement custom polling against a non-existent `/v1/status` on Jetson.
- Spam the Edge API with concurrent requests from the same client.
- Block the UI while waiting; always show a spinner/progress indicator.

Combine the synchronous and streaming APIs where it makes sense:
- Use SSE streaming when the user actively watches the screen.
- Fall back to a single synchronous call in background tasks (e.g.,
  mobile upload + local notification).

### 3. Error Handling

✅ **Do:**
- Handle all error types gracefully
- Provide user-friendly error messages
- Log errors for debugging
- Implement retry logic for transient failures

❌ **Don't:**
- Show technical error messages to users
- Ignore errors silently
- Retry infinitely

### 4. Performance Optimization

**Client-Side:**
- Debounce file selection
- Use Web Workers for large file processing
- Implement request cancellation
- Cache model lists (`GET /v1/models`) on the client

**Server-Side:**
- Use appropriate ASR backend for use case
- Enable VAD to skip silence
- Adjust model size based on device capabilities
- Monitor Jetson device load (GPU/CPU, memory, temperature)

### 5. Security

✅ **Do:**
- Keep the Jetson Edge API on a trusted/private network or behind a
  gateway.
- Use HTTPS on the gateway that exposes the API to the internet.
- Validate all user inputs
- Implement request timeouts

❌ **Don't:**
- Expose a raw Jetson Edge API directly to the public internet
- Trust client-side validation alone
- Log sensitive data

---

## Testing

### Unit Testing

```typescript
// Jest/Vitest example
import { describe, it, expect, vi } from 'vitest';
import { transcribeOnEdge } from './maie-client';

describe('MAIE Client', () => {
  it('should submit audio and return transcript', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        task_id: 'test-123',
        status: 'completed',
        transcript: 'hello world'
      })
    });
    
    const file = new File([''], 'test.wav', { type: 'audio/wav' });
    const result = await transcribeOnEdge(file, 'http://localhost:8000');
    
    expect(result.task_id).toBe('test-123');
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/transcribe'),
      expect.objectContaining({ method: 'POST' })
    );
  });
});
```

### Integration Testing

```javascript
// Playwright/Cypress example
describe('Audio Transcription Flow', () => {
  it('should upload audio and display transcript', () => {
    cy.visit('/transcribe');
    
    // Upload file
    cy.get('input[type="file"]').attachFile('test-audio.wav');
    cy.get('button[type="submit"]').click();
    
    // Wait for processing
    cy.contains('Processing', { timeout: 30000 });
    
    // Check result
    cy.contains('Transcript:', { timeout: 60000 });
    cy.get('[data-testid="transcript"]').should('not.be.empty');
  });
});
```

---

## Troubleshooting

### Common Issues

#### 1. Cannot reach Jetson / network errors

**Symptoms:** Requests fail with `ERR_CONNECTION_REFUSED`,
`ECONNRESET`, or time out.

**Causes:**
- Jetson device is offline or MAIE Edge API is not running.
- Incorrect base URL or port.
- Firewall or VPN blocking access.

**Solutions:**
- Verify the service from a terminal:
  - `curl http://<jetson-ip>:8000/health`
- Confirm the API process is running (e.g.,
  `pixi run api` or `uvicorn src.api.edge_main:app`).
- Double-check IP/hostname and port in your frontend config.

#### 2. 413 Payload Too Large

**Symptoms:** Upload fails with 413

**Causes:**
- File exceeds server limit
- Network timeout

**Solutions:**
- Compress audio before upload
- Use lower quality settings
- Split long recordings
- Check Jetson max file size (`APP_API__MAX_FILE_SIZE_MB` /
  `settings.api.max_file_size_mb`)

#### 3. Long latency or “stuck” requests

**Symptoms:**
- `POST /v1/transcribe` takes longer than expected.
- SSE stream stays in `queued` stage for a while.

**Causes:**
- Jetson Edge API processes **only one request at a time**
  (`_process_lock`).
- A previous transcription is still running.
- Audio is very long or the device is heavily loaded.

**Solutions:**
- Check `/health` – if `status` is `busy`, wait until the current task
  finishes.
- For SSE, show the initial `queued` progress event in the UI so users
  know they are waiting in line.
- Avoid firing multiple concurrent transcriptions from the same client.
- Split very long files into smaller chunks when possible.

#### 4. CORS Errors

**Symptoms:** Browser blocks requests

**Solutions:**
```javascript
// For local development, proxy requests
// vite.config.ts
export default {
  server: {
    proxy: {
      '/v1': 'http://localhost:8000'
    }
  }
}
```

#### 5. SSE Connection Drops

**Symptoms:** Streaming stops unexpectedly

**Causes:**
- Network timeout
- Server restart
- Client navigation

**Solutions:**
```javascript
// Implement reconnection
let reconnectAttempts = 0;
const maxReconnects = 3;

async function connectSSE() {
  try {
    await streamTranscription(file, handlers);
  } catch (error) {
    if (reconnectAttempts < maxReconnects) {
      reconnectAttempts++;
      setTimeout(connectSSE, 2000 * reconnectAttempts);
    }
  }
}
```

### Debug Checklist

- [ ] Base URL points to the correct Jetson device or gateway
- [ ] File format is supported (WAV, MP3, M4A, FLAC)
- [ ] File size is under the configured Jetson limit (≈50 MB by default)
- [ ] CORS is configured correctly
- [ ] Network connection is stable
- [ ] Jetson Edge API is running and healthy (`GET /health`)
- [ ] Check browser console for errors
- [ ] Check network tab for request/response details

---

## Appendix

### Full TypeScript Client

```typescript
// maie-edge-client.ts
export interface EdgeClientConfig {
  baseUrl: string;
  timeoutMs?: number;
}

export interface EdgeTranscribeOptions {
  asrBackend?: 'whisper' | 'chunkformer';
  enableVad?: boolean;
  vadThreshold?: number;
  language?: string;
}

export interface EdgeTranscribeResponse {
  task_id: string;
  status: 'completed' | 'failed';
  transcript: string;
  segments: Array<{
    start: number;
    end: number;
    text: string;
    speaker?: string | null;
  }>;
  duration_seconds: number;
  processing_time_seconds: number;
  asr_backend: string;
  language?: string | null;
  error?: string | null;
}

export class MAIEEdgeClient {
  private baseUrl: string;
  private timeoutMs: number;
  
  constructor(config: EdgeClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.timeoutMs = config.timeoutMs ?? 300000; // 5 minutes default
  }
  
  async transcribe(
    file: File,
    options: EdgeTranscribeOptions = {},
  ): Promise<EdgeTranscribeResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const {
      asrBackend = 'whisper',
      enableVad = true,
      vadThreshold = 0.5,
      language,
    } = options;
    
    formData.append('asr_backend', asrBackend);
    formData.append('enable_vad', String(enableVad));
    formData.append('vad_threshold', String(vadThreshold));
    if (language) {
      formData.append('language', language);
    }
    
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    
    try {
      const response = await fetch(`${this.baseUrl}/v1/transcribe`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = (await response.json()) as EdgeTranscribeResponse;
      
      if (result.status === 'failed') {
        throw new Error(result.error || 'Transcription failed');
      }
      
      return result;
    } finally {
      clearTimeout(timeout);
    }
  }
}
```

### Environment Configuration

**.env.development**
```bash
VITE_MAIE_EDGE_URL=http://localhost:8000
```

**.env.production**
```bash
VITE_MAIE_EDGE_URL=https://edge.yourcompany.com
```

---

**Document Version:** 2.0  
**Status:** ✅ Ready for Implementation  
**Last Updated:** December 6, 2025

For questions or issues, please refer to the main documentation or contact the backend team.

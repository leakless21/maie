# MAIE API Reference

**MAIE (Modular Audio Intelligence Engine)** provides a REST API for processing audio files with advanced AI-powered features including transcription, summary, and content enhancement.

## API Overview

The MAIE API is a RESTful service built with [Litestar](https://litestar.dev/) that enables asynchronous audio processing through a simple HTTP interface. The API supports multiple audio formats and provides both real-time processing capabilities and comprehensive result retrieval.

### Base URL

```
http://localhost:8000
```

All API endpoints are prefixed with `/v1/` for version 1 of the API.

### Content Type

The API primarily accepts:

- `multipart/form-data` for file uploads (`/v1/process`)
- `application/json` for other requests

## Authentication

### API Key Authentication

All processing endpoints require authentication using an API key in the request header:

```
X-API-Key: your_api_key_here
```

**Authentication Rules:**

- API key must be provided in the `X-API-Key` header (case-insensitive)
- Minimum key length: 32 characters
- Uses timing-safe comparison to prevent timing attacks
- Multiple fallback keys supported for key rotation

**Example Request:**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@audio.mp3'
```

## Response Formats

Successful responses return the requested data payload directly as a JSON object.

Error responses use a standardized format:

```json
{
  "detail": "A human-readable error message."
}
```

In cases of validation errors, an `extra` field may be included with more context.

### Status Codes

| Code  | Description                         |
| ----- | ----------------------------------- |
| `200` | Success                             |
| `202` | Accepted (async processing started) |
| `400` | Bad Request                         |
| `401` | Unauthorized                        |
| `413` | Payload Too Large                   |
| `415` | Unsupported Media Type              |
| `422` | Unprocessable Entity                |
| `429` | Too Many Requests                   |
| `500` | Internal Server Error               |

## Error Handling

### Common Error Scenarios

#### File Upload Errors

**413 Payload Too Large**

```json
{
  "error": {
    "code": "FILE_TOO_LARGE",
    "message": "File too large: 150.5MB (max 100MB)",
    "details": {
      "file_size_mb": 150.5,
      "max_size_mb": 100.0,
      "filename": "large_audio.wav"
    }
  }
}
```

**415 Unsupported Media Type**

```json
{
  "error": {
    "code": "UNSUPPORTED_FORMAT",
    "message": "Unsupported file type: audio.mkv",
    "details": {
      "filename": "audio.mkv",
      "extension": ".mkv",
      "mime_type": "video/x-matroska",
      "allowed_extensions": [".wav", ".mp3", ".m4a", ".flac"],
      "allowed_mime_types": [
        "audio/wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/flac"
      ]
    }
  }
}
```

#### Validation Errors

**422 Unprocessable Entity**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "template_id is required when summary is in features",
    "details": {
      "features": ["raw_transcript", "summary"],
      "required_field": "template_id",
      "condition": "summary feature requested"
    }
  }
}
```

#### Queue Management Errors

**429 Too Many Requests**

```json
{
  "error": {
    "code": "QUEUE_FULL",
    "message": "Queue is full. Please try again later.",
    "details": {
      "queue_status": "full",
      "retry_after": "Please wait and retry"
    }
  }
}
```

## Endpoint Details

### POST /v1/process

Submit an audio file for asynchronous processing with configurable features and parameters.

#### Description

This endpoint accepts multipart form data containing an audio file and processing parameters. The request is queued for asynchronous processing, returning immediately with a task ID for status tracking.

**Supported Audio Formats:**

- WAV (`audio/wav`, `audio/wave`, `audio/x-wav`)
- MP3 (`audio/mpeg`)
- M4A (`audio/mp4`, `audio/x-m4a`)
- FLAC (`audio/flac`)

**File Size Limits:**

- Maximum file size: 500MB (configurable via `max_file_size_mb`)
- Files exceeding limit are rejected during upload streaming

#### Request Format

**Headers:**

```
Content-Type: multipart/form-data
X-API-Key: your_api_key_here
```

**Form Parameters:**

| Parameter     | Type   | Required    | Description                                                          |
| ------------- | ------ | ----------- | -------------------------------------------------------------------- |
| `file`        | binary | Yes         | Audio file to process                                                |
| `features`    | array  | No          | List of desired outputs (default: `["clean_transcript", "summary"]`) |
| `template_id` | string | Conditional | Template ID for summary format (required if `summary` in features)   |
| `asr_backend` | string | No          | ASR backend selection (default: `"whisper"`)                         |

**Features Options:**

- `raw_transcript` - Raw ASR transcript without cleaning
- `clean_transcript` - Processed transcript with noise reduction
- `summary` - Structured summary using template format
- `enhancement_metrics` - Audio quality and processing metrics

**ASR Backend Options:**

- `whisper` - OpenAI Whisper model (default)
- `chunkformer` - ChunkFormer ASR model

#### Request Body Examples

**Basic Processing (Default Settings):**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@/path/to/audio.mp3' \
  -F 'features=clean_transcript' \
  -F 'features=summary' \
  -F 'template_id=meeting_notes_v1'
```

**Raw Transcript Only:**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@/path/to/audio.wav' \
  -F 'features=raw_transcript'
```

**All Features with ChunkFormer Backend:**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@/path/to/audio.m4a' \
  -F 'features=raw_transcript' \
  -F 'features=clean_transcript' \
  -F 'features=summary' \
  -F 'features=enhancement_metrics' \
  -F 'template_id=meeting_notes_v1' \
  -F 'asr_backend=chunkformer'
```

**Python Client Example:**

```python
import requests

def process_audio(file_path, api_key, features=None, template_id=None, asr_backend="whisper"):
    url = "http://localhost:8000/v1/process"
    headers = {"X-API-Key": api_key}

    # Set default features if not provided
    if features is None:
        features = ["clean_transcript", "summary"]

    # Prepare multipart form data
    files = {"file": open(file_path, "rb")}
    data = {}

    # Add features as separate form fields
    for feature in features:
        data[f"features"] = feature

    if template_id:
        data["template_id"] = template_id

    if asr_backend != "whisper":
        data["asr_backend"] = asr_backend

    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()

# Usage
result = process_audio(
    file_path="meeting_audio.wav",
    api_key="your_api_key_here",
    features=["clean_transcript", "summary"],
    template_id="meeting_notes_v1"
)
print(f"Task ID: {result['data']['task_id']}")
```

#### Response Format

**202 Accepted**

```json
{
  "data": {
    "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
    "status": "PENDING"
  },
  "meta": {
    "request_id": "req_12345678-1234-1234-1234-123456789abc",
    "timestamp": "2025-10-20T07:18:17.637Z",
    "version": "1.0.0"
  }
}
```

#### Status Codes

| Code  | Description                                           |
| ----- | ----------------------------------------------------- |
| `202` | Processing request accepted and queued                |
| `413` | File too large (exceeds configured limit)             |
| `415` | Unsupported media type or file format                 |
| `422` | Validation error (missing fields, invalid parameters) |
| `429` | Queue is full, try again later                        |

---

### GET /v1/status/{task_id}

Retrieve the current status and results of a processing task.

#### Description

Check the processing status of a previously submitted task. Returns comprehensive information including processing state, results (when complete), and metadata for reproducibility.

**Task States:**

- `PENDING` - Task queued, awaiting processing
- `PREPROCESSING` - Audio preprocessing in progress
- `PROCESSING_ASR` - ASR transcription running
- `PROCESSING_LLM` - LLM processing (summary/enhancement) running
- `COMPLETE` - Processing finished successfully
- `FAILED` - Processing failed with error

#### Request Format

**Path Parameters:**

- `task_id` (UUID) - Unique identifier returned from `/v1/process`

**Headers:**

```
X-API-Key: your_api_key_here
```

**Example Request:**

```bash
curl -X GET 'http://localhost:8000/v1/status/c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b' \
  -H 'X-API-Key: your_api_key_here'
```

**Python Client Example:**

```python
import requests

def get_task_status(task_id, api_key):
    url = f"http://localhost:8000/v1/status/{task_id}"
    headers = {"X-API-Key": api_key}

    response = requests.get(url, headers=headers)
    return response.json()

# Usage
status = get_task_status(
    task_id="c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
    api_key="your_api_key_here"
)
print(f"Status: {status['data']['status']}")
```

#### Response Format

**200 OK - Processing Complete**

```json
{
  "data": {
    "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
    "status": "COMPLETE",
    "submitted_at": "2025-10-20T07:18:17.637Z",
    "completed_at": "2025-10-20T07:20:45.123Z",
    "versions": {
      "pipeline_version": "1.0.0",
      "asr_backend": {
        "name": "whisper",
        "model_variant": "erax-wow-turbo",
        "model_path": "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
        "checkpoint_hash": "a1b2c3d4e5f6...",
        "compute_type": "int8_float16",
        "decoding_params": {
          "beam_size": 5,
          "vad_filter": true
        }
      },
      "llm": {
        "name": "qwen3",
        "checkpoint_hash": "z9y8x7w6v5u4...",
        "quantization": "awq-4bit",
        "thinking": false,
        "reasoning_parser": null,
        "structured_output": {
          "title": "string",
          "main_points": ["string"],
          "tags": ["string"]
        },
        "decoding_params": {
          "temperature": 0.3,
          "top_p": 0.9
        }
      }
    },
    "metrics": {
      "input_duration_seconds": 2701.3,
      "processing_time_seconds": 162.8,
      "rtf": 0.06,
      "vad_coverage": 0.88,
      "asr_confidence_avg": 0.91,
      "edit_rate_cleaning": 0.15
    },
    "results": {
      "raw_transcript": "The meeting on October 4th...",
      "clean_transcript": "The meeting on October 4th covered...",
      "summary": {
        "title": "Q4 Budget Planning Meeting",
        "main_points": [
          "Budget approved for Q4 initiatives",
          "New hiring plan discussed",
          "Timeline set for product launch"
        ],
        "tags": ["Finance", "Budget", "Planning"]
      }
    }
  },
  "meta": {
    "request_id": "req_87654321-4321-4321-4321-fedcba987654",
    "timestamp": "2025-10-20T07:21:30.456Z",
    "version": "1.0.0"
  }
}
```

**200 OK - Processing In Progress**

```json
{
  "data": {
    "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
    "status": "PROCESSING_ASR",
    "submitted_at": "2025-10-20T07:18:17.637Z",
    "versions": {
      "pipeline_version": "1.0.0"
    }
  },
  "meta": {
    "request_id": "req_87654321-4321-4321-4321-fedcba987654",
    "timestamp": "2025-10-20T07:19:45.123Z",
    "version": "1.0.0"
  }
}
```

**404 Not Found**

```json
{
  "error": {
    "code": "TASK_NOT_FOUND",
    "message": "Task c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b not found"
  },
  "meta": {
    "request_id": "req_87654321-4321-4321-4321-fedcba987654",
    "timestamp": "2025-10-20T07:21:30.456Z"
  }
}
```

#### Status Codes

| Code  | Description                 |
| ----- | --------------------------- |
| `200` | Task found, status returned |
| `404` | Task not found or expired   |

---

### GET /v1/models

Retrieve information about available audio processing models and backends.

#### Description

Returns a list of available ASR (Automatic Speech Recognition) models and backends that can be used for audio processing.

#### Request Format

**Headers:**

```
X-API-Key: your_api_key_here
```

**Example Request:**

```bash
curl -X GET 'http://localhost:8000/v1/models' \
  -H 'X-API-Key: your_api_key_here'
```

#### Response Format

**200 OK**

```json
{
  "data": {
    "models": [
      {
        "id": "whisper",
        "name": "Whisper Backend",
        "description": "ASR backend using whisper",
        "type": "ASR",
        "version": "1.0",
        "supported_languages": ["en", "vi", "zh", "ja", "ko"]
      },
      {
        "id": "chunkformer",
        "name": "ChunkFormer Backend",
        "description": "ASR backend using chunkformer",
        "type": "ASR",
        "version": "1.0",
        "supported_languages": ["en", "vi", "zh", "ja", "ko"]
      }
    ]
  },
  "meta": {
    "request_id": "req_12345678-1234-1234-1234-123456789abc",
    "timestamp": "2025-10-20T07:18:17.637Z",
    "version": "1.0.0"
  }
}
```

#### Status Codes

| Code  | Description                   |
| ----- | ----------------------------- |
| `200` | Models retrieved successfully |

---

### GET /v1/templates

Retrieve information about available processing templates for structured output formatting.

#### Description

Returns a list of available templates that define the structure and format of summary outputs. Templates are JSON-based configurations that control how processed content is formatted.

#### Request Format

**Headers:**

```
X-API-Key: your_api_key_here
```

**Example Request:**

```bash
curl -X GET 'http://localhost:8000/v1/templates' \
  -H 'X-API-Key: your_api_key_here'
```

#### Response Format

**200 OK**

```json
{
  "data": {
    "templates": [
      {
        "id": "meeting_notes_v1",
        "name": "Meeting Notes v1",
        "description": "Structured format for meeting transcripts",
        "schema_url": "/v1/templates/meeting_notes_v1/schema",
        "parameters": {}
      },
      {
        "id": "interview_transcript_v1",
        "name": "Interview Transcript v1",
        "description": "Format for interview recordings",
        "schema_url": "/v1/templates/interview_transcript_v1/schema",
        "parameters": {}
      }
    ]
  },
  "meta": {
    "request_id": "req_12345678-1234-1234-1234-123456789abc",
    "timestamp": "2025-10-20T07:18:17.637Z",
    "version": "1.0.0"
  }
}
```

#### Status Codes

| Code  | Description                      |
| ----- | -------------------------------- |
| `200` | Templates retrieved successfully |

## Best Practices

### File Upload Optimization

1. **Chunked Uploads**: For large files (>100MB), consider implementing client-side chunking
2. **Format Selection**: Use lossless formats (WAV, FLAC) for highest quality transcription
3. **Preprocessing**: Remove silence and normalize audio before upload when possible

### Error Handling

1. **Retry Logic**: Implement exponential backoff for `429` (queue full) errors
2. **Task Monitoring**: Poll `/v1/status/{task_id}` every 5-10 seconds for updates
3. **Timeout Management**: Set appropriate timeouts for long-running audio files

### Performance Considerations

1. **Queue Management**: Monitor queue depth and implement circuit breakers if needed
2. **Batch Processing**: Submit multiple files sequentially rather than in parallel
3. **Resource Planning**: Consider API rate limits when scaling your application

### Security Best Practices

1. **API Key Management**: Rotate API keys regularly and use environment variables
2. **File Validation**: Always validate file types and sizes before upload
3. **Input Sanitization**: Ensure filenames don't contain malicious characters

## Troubleshooting

### Common Issues

**"Queue is full" (429 Error)**

- Wait and retry the request
- Implement exponential backoff (start with 5-second delay)
- Consider reducing submission rate

**"Task not found" (404 Error)**

- Verify the task_id is correct (UUID format)
- Tasks expire after the configured TTL (default: 24 hours)
- Check if the task completed successfully before expiration

**"File too large" (413 Error)**

- Check file size against configured limits (default: 500MB)
- Compress audio file if possible while maintaining quality
- Consider splitting long recordings into smaller segments

**"Unsupported file type" (415 Error)**

- Verify file extension matches actual audio format
- Check MIME type in file headers
- Convert to supported format (WAV, MP3, M4A, FLAC) if needed

### Support

For additional support or questions:

1. Check the application logs for detailed error information
2. Verify API configuration and environment variables
3. Review the OpenAPI specification available at `/docs` (if enabled)

## OpenAPI Specification

A complete OpenAPI 3.1 specification is available for this API. The specification provides detailed schema information and can be used to generate client SDKs in various programming languages.

**OpenAPI Document Location:**

```
/docs/openapi.json
```

**Interactive API Documentation:**

```
/docs
```

---

_This API documentation was generated for MAIE v1.0.0. For the latest updates, refer to the project repository or contact the development team._

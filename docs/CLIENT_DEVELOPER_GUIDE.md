# Client Developer Guide for MAIE

This guide is designed for frontend and mobile developers building applications that interact with the Modular Audio Intelligence Engine (MAIE) API.

## 1. Introduction

MAIE is an audio processing platform that provides:
- **Transcription**: Converting speech to text (ASR).
- **Summarization**: Generating structured summaries from transcripts using LLMs.
- **Diarization**: Identifying different speakers in the audio.
- **Enhancement**: Improving audio quality (metrics only).

## 2. Authentication

All API requests must include the `X-API-Key` header.

```http
X-API-Key: <your-api-key>
```

## 3. Core Workflow

The typical workflow for processing audio is asynchronous:

1.  **Upload Audio**: Submit an audio file to `/v1/process`. You will receive a `task_id`.
2.  **Poll Status**: Periodically check the status of the task using `/v1/status/{task_id}`.
3.  **Get Results**: When the status is `COMPLETE`, the response will contain the results (transcript, summary, etc.).

### Step 1: Upload Audio

**Endpoint**: `POST /v1/process`
**Content-Type**: `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | File | Yes | Audio file (WAV, MP3, M4A, FLAC). Max 500MB. |
| `features` | Array/String | No | List of features: `clean_transcript`, `raw_transcript`, `summary`, `enhancement_metrics`. Default: `['clean_transcript']`. |
| `template_id` | String | Conditional | Required if `summary` feature is requested. ID of the template to use (e.g., `meeting_notes_v2`). |
| `asr_backend` | String | No | `whisper` (default) or `chunkformer`. |
| `enable_diarization` | Boolean | No | `true` or `false` (default). Enables speaker identification. |
| `enable_vad` | Boolean | No | `true` or `false`. Explicitly enable/disable Voice Activity Detection. |
| `vad_threshold` | Float | No | `0.0` to `1.0`. VAD confidence threshold. Default: `0.5`. |

**Example Request (cURL):**

```bash
curl -X POST "https://api.example.com/v1/process" \
  -H "X-API-Key: your-key" \
  -F "file=@meeting.mp3" \
  -F "features=clean_transcript" \
  -F "features=summary" \
  -F "template_id=meeting_notes_v2"
```

**Example Response:**

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "PENDING"
}
```

### Step 2: Poll Status

**Endpoint**: `GET /v1/status/{task_id}`

**Polling Strategy**:
- Poll every 2-5 seconds.
- Stop polling when `status` is `COMPLETE` or `FAILED`.

**Example Request:**

```bash
curl "https://api.example.com/v1/status/c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b" \
  -H "X-API-Key: your-key"
```

**Response (Processing):**

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "PROCESSING_ASR",
  "stage": "asr"
}
```

**Response (Complete):**
 
 ```json
 {
   "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
   "status": "COMPLETE",
   "submitted_at": "2023-10-27T10:00:00Z",
   "completed_at": "2023-10-27T10:02:45Z",
   "versions": {
     "pipeline_version": "1.0.0",
     "asr_backend": {
       "name": "whisper",
       "model_variant": "erax-wow-turbo",
       "version": "1.0"
     },
     "llm": {
       "name": "qwen3",
       "quantization": "awq-4bit"
     }
   },
   "metrics": {
     "input_duration_seconds": 180.5,
     "processing_time_seconds": 45.2,
     "rtf": 0.25,
     "vad_coverage": 0.95,
     "asr_confidence_avg": 0.98,
     "vad_segments": 12,
     "edit_rate_cleaning": 0.02
   },
   "results": {
     "raw_transcript": "hello everyone welcome to the meeting...",
     "clean_transcript": "Hello everyone, welcome to the meeting...",
     "summary": {
       "title": "Project Kickoff",
       "main_points": [
         "Timeline agreed for Q4",
         "Budget approved with 10% contingency"
       ],
       "action_items": [
         "John to send email to stakeholders",
         "Sarah to set up Jira board"
       ],
       "tags": ["Project Management", "Kickoff", "Budget"]
     }
   }
 }
 ```
 
 **Response Fields:**
 
 -   `versions`: Metadata about the models used, ensuring reproducibility.
 -   `metrics`: Performance data.
     -   `rtf` (Real-Time Factor): Processing time / Audio duration. Lower is faster.
     -   `asr_confidence_avg`: Average confidence of the transcription (0.0 - 1.0).
 -   `results`: The core output.
     -   `clean_transcript`: The text with punctuation and capitalization fixed.
     -   `summary`: The structured summary. The exact fields (`title`, `main_points`, etc.) depend on the `template_id` used.
     -   `tags`: Automatically generated category tags (embedded within the summary object).

## 4. Helper Endpoints

### Get Available Templates

**Endpoint**: `GET /v1/templates`

Use this to populate a dropdown or selection screen for the user to choose a summary format.

**Response:**

```json
{
  "templates": [
    {
      "id": "meeting_notes_v2",
      "name": "Meeting Notes",
      "description": "Standard meeting summary with action items.",
      "schema_url": "...",
      "example": { ... }
    },
    ...
  ]
}
```

### Get Available Models

**Endpoint**: `GET /v1/models`

Use this if you want to allow advanced users to select the ASR backend.

## 5. Error Handling

Check the `status` field in the response. If `status` is `FAILED`, check the `error` and `error_code` fields.

| Error Code | Description | Action |
| :--- | :--- | :--- |
| `ASR_PROCESSING_ERROR` | Failed to transcribe audio. | Retry, check audio quality/format. |
| `MODEL_LOAD_ERROR` | Server-side model issue. | Retry later, contact support. |
| `VALIDATION_ERROR` | Invalid input parameters. | Check request parameters. |

## 6. Best Practices

- **File Size**: Keep uploads under 500MB.
- **Audio Quality**: 16kHz WAV is optimal, but MP3/M4A are supported.
- **Timeouts**: Processing can take time (approx 10-20% of audio duration). Ensure your client has appropriate timeouts or background processing capabilities.
- **Rate Limiting**: The API is rate-limited. Handle `429 Too Many Requests` responses gracefully.

## 7. Template Management

The API provides full CRUD capabilities for managing summary templates.

### Get Template Details

**Endpoint**: `GET /v1/templates/{template_id}`

**Response:**

```json
{
  "id": "meeting_notes_v2",
  "name": "Meeting Notes",
  "description": "Standard meeting summary with action items.",
  "schema_url": "/v1/templates/meeting_notes_v2/schema",
  "parameters": {},
  "example": { ... },
  "prompt_template": "Summarize this: {{ transcript }}",
  "schema_data": {
    "title": "Meeting Notes",
    "type": "object",
    "properties": { ... }
  }
}
```

### Create Template

**Endpoint**: `POST /v1/templates`
**Content-Type**: `application/json`

**Body:**

```json
{
  "id": "new_template_v1",
  "schema_data": {
    "title": "New Template",
    "type": "object",
    "properties": {
      "summary": { "type": "string" },
      "tags": { "type": "array", "items": { "type": "string" } }
    },
    "required": ["summary", "tags"]
  },
  "prompt_template": "Summarize this: {{ transcript }}",
  "example": {
    "summary": "Example summary",
    "tags": ["tag1"]
  }
}
```

### Update Template

**Endpoint**: `PUT /v1/templates/{template_id}`
**Content-Type**: `application/json`

**Body:** (Partial updates allowed)

```json
{
  "prompt_template": "Updated prompt: {{ transcript }}"
}
```

### Delete Template

**Endpoint**: `DELETE /v1/templates/{template_id}`

**Response**: `204 No Content`

# Client Developer Guide for MAIE

This guide is designed for frontend and mobile developers building applications that interact with the Modular Audio Intelligence Engine (MAIE) API.

## 1. System Architecture & Internals

### 1.1 Project Overview
MAIE (Modular Audio Intelligence Engine) is a sophisticated, enterprise-grade audio processing system designed to transform raw audio data into structured intelligence. Unlike simple transcription services, MAIE focuses on deep understanding of content through a pipeline of Automatic Speech Recognition (ASR), Speaker Diarization, and Large Language Model (LLM) analysis.

### 1.2 Technology Stack
The system is built on a modern, high-performance stack:
-   **API Framework**: [Litestar](https://litestar.dev/) (Python) - An asynchronous ASGI framework chosen for its performance and strict data validation.
-   **Task Queue**: [Redis](https://redis.io/) + [RQ (Redis Queue)](https://python-rq.org/) - Handles asynchronous background processing and job management.
-   **AI/ML Core**: [PyTorch](https://pytorch.org/) - The underlying deep learning framework.
-   **ASR Engines**:
    -   **Faster-Whisper**: An optimized implementation of OpenAI's Whisper model using CTranslate2 for faster inference.
    -   **ChunkFormer**: A specialized model architecture for processing long-form audio efficiently.
-   **LLM Engine**: [vLLM](https://github.com/vllm-project/vllm) - A high-throughput and memory-efficient serving engine for LLMs.
-   **Diarization**: [pyannote.audio](https://github.com/pyannote/pyannote-audio) - State-of-the-art speaker diarization.
-   **Validation**: [Pydantic](https://docs.pydantic.dev/) - Ensures strict schema validation for all inputs and outputs.

### 1.3 Detailed Architecture
The system operates on a distributed, asynchronous model:

#### A. API Layer
-   **Role**: Entry point for all client requests.
-   **Streaming Uploads**: Audio files are streamed directly to disk (`save_audio_file_streaming`) to handle large files (up to 500MB) without loading them into RAM.
-   **Validation**: Validates file types (magic numbers/MIME), extensions, and JSON schemas before accepting a task.
-   **State Management**: Creates an initial task record in Redis with status `PENDING` and returns a `task_id` immediately (HTTP 202 Accepted).

#### B. Worker Layer
-   **Role**: Consumes tasks from the Redis queue and executes the processing pipeline.
-   **Pipeline Stages**:
    1.  **Preprocessing**:
        -   Validates audio integrity.
        -   Normalizes audio to 16kHz mono WAV using `ffmpeg`.
    2.  **Voice Activity Detection (VAD)**:
        -   (Optional) Uses Silero VAD to detect speech segments.
        -   Optimizes processing by identifying silence.
    3.  **ASR Processing**:
        -   Loads the selected backend (Whisper or ChunkFormer).
        -   Transcribes audio to text with word-level timestamps.
        -   **Diarization**: If enabled, identifies speakers ("Speaker 0", "Speaker 1").
        -   **Hallucination Filter**: Post-processes the transcript to remove common neural network artifacts (e.g., repeated phrases, silence hallucinations).
    4.  **LLM Processing**:
        -   **Template Loading**: Loads the selected template (Schema + Prompt).
        -   **Prompt Rendering**: Uses Jinja2 to combine the transcript with the prompt template.
        -   **Generation**: Sends the prompt to the vLLM engine.
        -   **Structured Output**: Enforces the JSON schema to ensure the output matches the client's expectations exactly.
    5.  **Finalization**:
        -   Calculates metrics (RTF, Confidence).
        -   Saves results to Redis.
        -   Updates status to `COMPLETED`.

#### C. Data Storage
-   **Redis**:
    -   **Queue**: Stores pending job information.
    -   **Results**: Stores the final JSON output, status, and metrics. Keys expire automatically (TTL).
-   **Filesystem**:
    -   **Audio**: Temporary storage for uploaded and normalized audio files.
    -   **Models**: Local cache for large model weights (Whisper, LLM, etc.).

### 1.4 Key Internal Features

#### Template System
The template system is the core of MAIE's flexibility. It decouples the *processing logic* from the *output format*.
-   **Structure**:
    -   `schema.json`: A standard JSON Schema defining the output fields.
    -   `prompt.jinja`: Instructions for the LLM.
-   **Workflow**: The API exposes endpoints to Create/Read/Update/Delete templates, allowing dynamic adjustment of the system's capabilities without code changes.

#### Hallucination Filtering
Neural ASR models often generate text during silence or noise. MAIE implements a robust filtering mechanism:
-   **Exact Match**: Removes known hallucination phrases (e.g., "Subtitles by...").
-   **Repetition Detection**: Identifies and removes looping phrases.
-   **Confidence Thresholds**: Filters out low-confidence segments.

#### Performance Metrics
Every task generates detailed metrics for monitoring:
-   `rtf` (Real-Time Factor): Ratio of processing time to audio duration.
-   `asr_confidence_avg`: Average confidence score of the transcription.
-   `processing_time_seconds`: Total wall-clock time for the task.

---

## 2. API Endpoint Specifications

### Overview
Base URL: `http://<host>:<port>` (e.g., `http://localhost:8000`)

| Method | Endpoint | Description | Auth |
| :--- | :--- | :--- | :--- |
| `POST` | `/v1/process` | Upload audio for transcription & summarization | API Key |
| `POST` | `/v1/process_text` | Submit text for summarization/enhancement | API Key |
| `GET` | `/v1/status/{task_id}` | Check processing status & get results | API Key |
| `GET` | `/v1/models` | List available ASR/LLM models | None |
| `GET` | `/v1/templates` | List available summary templates | None |
| `GET` | `/v1/templates/{id}` | Get template details | None |
| `POST` | `/v1/templates` | Create a new template | API Key |
| `PUT` | `/v1/templates/{id}` | Update an existing template | API Key |
| `DELETE` | `/v1/templates/{id}` | Delete a template | API Key |
| `GET` | `/health` | System health check | None |

### Authentication
All state-changing or resource-intensive endpoints require an API Key.
**Header**: `X-API-Key: <your-api-key>`

---

## 3. Processing & Output

### Audio Processing (`POST /v1/process`)
Submits an audio file for asynchronous processing.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body Parameters**:
  - `file` (File, Required): Audio file (WAV, MP3, M4A, FLAC). Max 100MB.
  - `features` (String[], Optional): List of desired outputs.
    - Options: `raw_transcript`, `clean_transcript`, `summary`, `enhancement_metrics`.
    - Default: `["clean_transcript", "summary"]`.
  - `template_id` (String, Conditional): Required if `summary` is in features.
  - `asr_backend` (String, Optional): `whisper` (default) or `chunkformer`.
  - `enable_diarization` (Boolean, Optional): Enable speaker identification (default `false`).
  - `enable_vad` (Boolean, Optional): Enable Voice Activity Detection (default system setting).
  - `vad_threshold` (Float, Optional): VAD confidence threshold (0.0-1.0, default 0.5).

**Response (Success):**
```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "PENDING"
}
```

### Text Processing (`POST /v1/process_text`)
Submits text for summarization or enhancement without audio processing.

**Request:**
- **Content-Type**: `application/json`
- **Body Parameters**:
  - `text` (String, Required): Input text to process.
  - `features` (String[], Optional): `["summary"]` (default), `["clean_transcript"]` (for enhancement).
  - `template_id` (String, Conditional): Required if `summary` is in features.

**Response (Success):**
```json
{
  "task_id": "a1b2c3d4-5e6f-7g8h-9i0j-1k2l3m4n5o6p",
  "status": "PENDING"
}
```

### Tag/Metadata Extraction
Tag extraction is handled as part of the **Summary** feature.
- It is defined in the **Template Schema** (e.g., a `tags` field in the JSON schema).
- The LLM extracts tags based on the content and the schema definition.
- Output is found in `results.summary.tags` (or similar, depending on template).

---

## 4. Template System

Templates define the structure and instructions for the LLM summarization.

### Template Structure
A template consists of:
1.  **ID**: Unique identifier (e.g., `meeting_notes_v2`).
2.  **Schema Data**: A JSON Schema defining the output structure (fields, types, descriptions).
3.  **Prompt Template**: A Jinja2 template string (e.g., `Summarize this: {{ transcript }}`).
4.  **Example**: A sample JSON output.

### Template Usage
- **Selection**: Users select a template (e.g., "Meeting Notes", "Interview").
- **Output Control**: The template's `schema_data` strictly controls the JSON structure of the `summary` output.
    - If the schema has `attendees`, `decisions`, `action_items`, the output will have those fields.
- **Custom Templates**: Admins/Developers can create custom templates via the API.

### Template Management Endpoints
- **Create**: `POST /v1/templates`
- **Update**: `PUT /v1/templates/{template_id}`
- **Delete**: `DELETE /v1/templates/{template_id}`
- **Get Details**: `GET /v1/templates/{template_id}` (includes prompt & schema)
- **Get Schema**: `GET /v1/templates/{template_id}/schema` (raw JSON schema)

---

## 5. Processing Performance

### Processing Times
- **Real-Time Factor (RTF)**: Typically 0.1 - 0.3 (e.g., 10 minutes of audio takes 1-3 minutes).
- **Latency**: Depends on queue depth and audio length. Longer audio takes proportionally longer.
- **Metrics**: Returned in the `metrics` field of the status response (`processing_time_seconds`, `rtf`).

### Concurrency & Limits
- **Queue**: Requests are queued in Redis.
- **Limits**:
    - **Max File Size**: 100MB (configurable).
    - **Rate Limits**: Configured per API key/IP (default 60 req/min for lightweight, stricter for processing).
    - **Queue Depth**: Rejects requests with `429 Too Many Requests` if the queue is full.

### Output Quality
- **Confidence**: `asr_confidence_avg` (0.0 - 1.0) indicates transcription certainty.
    - > 0.9: High quality.
    - < 0.7: Low quality (check audio clarity, background noise).
- **Limitations**: Heavy background noise or overlapping speech can degrade diarization and transcription accuracy.

---

## 6. Error Handling & Status

### Processing States (`status` field)
- `PENDING`: Task accepted, waiting in queue.
- `PREPROCESSING`: Audio validation/conversion in progress.
- `PROCESSING_ASR`: Transcribing audio.
- `PROCESSING_LLM`: Generating summary/enhancement.
- `COMPLETE`: Finished successfully. Results available.
- `FAILED`: Error occurred.

### Error Responses
**HTTP Errors (Immediate):**
- `413 Payload Too Large`: File exceeds 100MB.
- `415 Unsupported Media Type`: Invalid file format.
- `422 Unprocessable Entity`: Missing parameters, invalid template ID.
- `429 Too Many Requests`: Rate limit exceeded or Queue full.

**Task Errors (in `GET /v1/status/{id}`):**
If `status` is `FAILED`, check `error` and `error_code`:
- `ASR_PROCESSING_ERROR`: Transcription failed.
- `LLM_PROCESSING_ERROR`: Summarization failed.
- `AUDIO_VALIDATION_ERROR`: Corrupt or empty audio file.

### Async Workflow
1.  **Submit**: `POST /v1/process` → Returns `task_id` immediately (HTTP 202).
2.  **Poll**: `GET /v1/status/{task_id}` every 2-5 seconds.
3.  **Finish**: Stop when status is `COMPLETE` or `FAILED`.

---

## 7. Data Retention & Privacy

### Storage
- **Audio Files**: Stored temporarily in the configured `audio_dir` (under a `task_id` folder) for processing.
- **Results**: Stored in Redis.

### Retention Policy
- **Results (Redis)**: Auto-expire after a configured TTL (default: 24 hours).
- **Audio Files**: Currently persisted in the task directory. Cleanup policy depends on server configuration (manual or cron job recommended).
- **Logging**: Access logs and processing steps are logged. No audio content is logged, but metadata (filename, duration) is.

---

## 8. Cost & Quotas

### Usage Tracking
- **Metrics**: Every completed task returns `input_duration_seconds` and `transcription_length`. These can be used for billing (e.g., cost per minute).
- **Reporting**: Usage data is not stored in a permanent SQL database by MAIE itself; it relies on Redis for transient results. External logging/monitoring is required for long-term usage tracking.

### Rate Limiting
- **Global/Key Limit**: Enforced via Redis.
- **Headers**: Standard `X-RateLimit-*` headers are returned to indicate remaining quota.

# Technical Design Document (TDD)

## Modular Audio Intelligence Engine (MAIE)

|                       |                                            |
| :-------------------- | :----------------------------------------- |
| **Project Name**      | Modular Audio Intelligence Engine (MAIE)   |
| **Version**           | 1.3                                        |
| **Status**            | **Production Ready**                       |
| **Related Documents** | PRD V1.3 (docs/PRD.md), Project Brief V1.3 |
| **Last Updated**      | October 15, 2025                           |
| **Authors**           | Engineering Team                           |

---

## 1. Executive Summary

This Technical Design Document provides the architectural blueprint for V1.0 of the Modular Audio Intelligence Engine (MAIE). The system delivers reproducible audio intelligence capabilities (transcription, enhancement, summarization, categorization) through a well-architected API designed for on-premises deployment under GPU resource constraints.

**Key Design Principles:**

- Sequential model processing for VRAM-constrained environments (16-24GB single GPU)
- Asynchronous task queue architecture for API responsiveness
- Full reproducibility through comprehensive versioning (NFR-1)
- Configuration-driven behavior via environment variables (NFR-4)
- First-class developer experience with OpenAPI 3.1 specification (NFR-2)

---

## 2. System Architecture

### 2.1. High-Level Architecture

The system follows a **three-tier architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Host Machine (Single Node)                    │
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────────┐ │
│  │ API Server   │      │    Redis     │      │  GPU Worker   │ │
│  │  (Litestar)  │◄────►│   (AOF On)   │◄────►│  (RQ Worker)  │ │
│  │  Stateless   │      │  Dual DB     │      │  Sequential   │ │
│  └──────┬───────┘      └──────┬───────┘      └───────┬───────┘ │
│         │                     │                       │          │
│         │                     │                       │          │
│         └─────────────────────┴───────────────────────┘          │
│                               │                                  │
│              ┌────────────────┴────────────────┐                │
│              │  Shared Filesystem (Host Mounts) │                │
│              ├──────────────────────────────────┤                │
│              │  /data/audio      (Uploads)      │                │
│              │  /data/models     (AI Models)    │                │
│              │  /data/redis      (AOF)          │                │
│              │  /app/templates   (Schemas)      │                │
│              │  /app/assets      (Chat Tpl)     │                │
│              └───────────────────────────────────┘                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
         ▲
         │ HTTPS (X-API-Key Authentication)
         │
    [Client Application]
```

**Component Responsibilities:**

| Component         | Purpose                                                 | Stateful/Stateless |
| ----------------- | ------------------------------------------------------- | ------------------ |
| API Server        | HTTP endpoint handling, request validation, job enqueue | Stateless          |
| Redis             | Job queue (DB 0) + Results store (DB 1)                 | Stateful           |
| GPU Worker        | Sequential AI model execution (ASR + LLM)               | Stateless          |
| Shared Filesystem | Audio files, model weights, schemas, chat templates     | Stateful           |

### 2.2. Data Flow Sequence

**Standard Processing Flow (Happy Path):**

1. **Client → API Server**: `POST /v1/process` with audio file + parameters
2. **API Server**: Validates request, saves file, generates `task_id`
3. **API Server → Redis DB 1**: Creates task record with status `PENDING`
4. **API Server → Redis DB 0**: Enqueues job with parameters
5. **API Server → Client**: Returns `202 Accepted` with `task_id`
6. **GPU Worker ← Redis DB 0**: Dequeues next job
7. **GPU Worker**: Executes sequential pipeline (ASR → LLM)
8. **GPU Worker → Redis DB 1**: Updates status to `PROCESSING_ASR` → `PROCESSING_LLM` → `COMPLETE`
9. **Client → API Server**: Polls `GET /v1/status/{task_id}`
10. **API Server ← Redis DB 1**: Retrieves complete result
11. **API Server → Client**: Returns full response with versions, metrics, results

**Failure Flow:**

- RQ automatically retries failed jobs (configurable retry policy)
- After retry exhaustion, job moves to **FailedJobRegistry** (not DLQ)
- Failed jobs remain indexed for manual inspection via `queue.failed_job_registry`
- Manual requeue possible via RQ CLI or programmatically

### 2.3. Technology Stack

| Layer                  | Technology                  | Version       | Justification                                                                                                                                                                                                                                                                                                                           |
| ---------------------- | --------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Package Management** | Pixi                        | Latest        | Fast, reproducible Python environment management (Conda + PyPI)                                                                                                                                                                                                                                                                         |
| **Runtime**            | Python                      | 3.11+         | Modern Python with performance improvements                                                                                                                                                                                                                                                                                             |
| **API Framework**      | Litestar                    | 2.x           | Auto OpenAPI 3.1, type-safe, async-native                                                                                                                                                                                                                                                                                               |
| **Task Queue**         | Redis Queue (RQ)            | 1.15+         | Simple, reliable, low overhead for single-node                                                                                                                                                                                                                                                                                          |
| **Message Broker**     | Redis                       | 7.x           | Dual-DB setup (queue + results) with AOF                                                                                                                                                                                                                                                                                                |
| **Audio Processing**   | ffmpeg                      | 6.0+          | Audio format normalization and validation                                                                                                                                                                                                                                                                                               |
| **ASR Backend**        | faster-whisper, chunkformer | CT2 / PyTorch | faster-whisper (CTranslate2-optimized Whisper) for general workloads; ChunkFormer for long-form audio. Whisper default variant `erax-wow-turbo` is an org CT2 default (set via `WHISPER_MODEL_VARIANT`); use official variants like `large-v3` or `distil-large-v3` as needed. Official docs: https://github.com/systran/faster-whisper |
| **LLM Backend**        | Qwen3-4B-Instruct           | AWQ-4bit      | Direct vLLM Python library inference                                                                                                                                                                                                                                                                                                    |
| **Logging**            | Loguru                      | Latest        | Zero-config structured logging with JSON output                                                                                                                                                                                                                                                                                         |
| **Monitoring**         | rq-dashboard                | Latest        | Queue visualization                                                                                                                                                                                                                                                                                                                     |
| **Observability**      | OpenTelemetry               | 1.20+         | Distributed tracing (optional)                                                                                                                                                                                                                                                                                                          |
| **Container Runtime**  | Docker + Compose            | Latest        | Simplified deployment                                                                                                                                                                                                                                                                                                                   |
| **GPU Runtime**        | NVIDIA CUDA                 | 12.1+         | GPU acceleration                                                                                                                                                                                                                                                                                                                        |

---

## 3. Detailed Component Design

### 3.1. API Server (Litestar Application)

**Responsibilities:**

- HTTP request handling and validation (FR-1, FR-7)
- Authentication via X-API-Key header (implied security requirement)
- File upload processing and storage
- Job enqueueing to Redis queue
- Status polling and result retrieval
- Discovery endpoints for models and templates (FR-8)

**Key Design Decisions:**

**Authentication Mechanism:**

- Static API key validation via dependency injection
- Key comparison against `SECRET_API_KEY` environment variable
- Returns `401 Unauthorized` for missing/invalid keys

**File Upload Handling:**

- Maximum file size: 500MB (server-side validation)
- Supported formats: `.wav`, `.mp3`, `.m4a`, `.flac` (FR-1)
- Files saved to `{AUDIO_DIR}/{task_id}/raw{ext}`
- Asynchronous file I/O to avoid blocking event loop

**Backpressure Management (NFR-5):**

- Pre-enqueue check: `queue.count >= MAX_QUEUE_DEPTH`
- Returns `429 Too Many Requests` when queue full
- Prevents worker overload (target: 6+ concurrent requests)
- Note: `queue.count` can be slow on large queues; consider `started_job_registry` count for active jobs

**OpenAPI Specification (NFR-2):**

- Litestar auto-generates OpenAPI 3.1 schema
- Available at `/schema`
- Interactive Swagger UI at `/schema/swagger`
- Pydantic schemas ensure type-safe validation

**Endpoint Specifications:**

| Endpoint               | Method | Purpose                      | Status Code | PRD Reference |
| ---------------------- | ------ | ---------------------------- | ----------- | ------------- |
| `/v1/process`          | POST   | Submit audio for processing  | 202         | FR-7.1        |
| `/v1/status/{task_id}` | GET    | Retrieve job status/results  | 200/404     | FR-7.2        |
| `/v1/models`           | GET    | List available ASR backends  | 200         | FR-8          |
| `/v1/templates`        | GET    | List summarization templates | 200         | FR-8          |
| `/schema`              | GET    | OpenAPI specification        | 200         | NFR-2         |

**Request Validation Logic (FR-7.1):**

For `POST /v1/process`:

1. Validate `file` is present and acceptable format
2. Validate `features` array (optional, default: `["clean_transcript", "summary"]`)
3. Validate `asr_backend` (optional, default: `"chunkformer"`)
   - Valid values: `"whisper"`, `"chunkformer"` (V1.0)
   - For `"whisper"`, model variant determined by `WHISPER_MODEL_VARIANT` (default: `erax-wow-turbo`, org CT2 default)
   - For `"chunkformer"`, model name determined by `CHUNKFORMER_MODEL_NAME` (default: `khanhld/chunkformer-rnnt-large-vie`)
4. Validate `template_id` is provided if `"summary"` in features (FR-4)
5. Generate UUID v4 as `task_id`
6. Check queue depth for backpressure
7. Initialize task record in Redis DB 1 with status `PENDING`
8. Enqueue job to Redis DB 0 with timeout=600s, retries=2
9. Return `{"task_id": "..."}`

**Response Contract (FR-7.2):** Refer to the canonical schema in PRD Appendix B. This TDD intentionally avoids duplicating the response schema to prevent drift.

### 3.2.1. Prompting & Templates (Jinja + vLLM)

- Chat template: Use a Jinja ChatML template compatible with Qwen3 (non-thinking
  by default). Serve it explicitly (e.g., vLLM `--chat-template
/app/assets/chat-templates/qwen3_nonthinking.jinja`). The selected model is
  instruct-only (no “thinking” mode), so reasoning toggles do not apply by
  default.
- Prompt templates: For each `template_id`, render a concise Jinja prompt into
  OpenAI-style `messages` (system + user with transcript). Emphasize JSON-only
  output.
- Structured outputs: Enforce with `response_format={type:"json_schema"}` using
  the exact JSON Schema file under `templates/`. Validate again post-generation;
  retry once with error hints if validation fails.

### 3.2. GPU Worker (RQ Worker Process)

**Responsibilities:**

- Dequeue jobs from Redis queue
- Execute sequential processing pipeline
- Manage GPU memory through load/unload cycles
- Calculate runtime metrics (FR-5)
- Store versioning metadata (NFR-1)
- Update task status throughout pipeline

**Sequential Processing Design (KAD-2):**

The worker enforces **strict sequential execution** to operate within VRAM constraints:

**Why Sequential?**

- Single GPU with 16-24GB VRAM cannot simultaneously hold multiple large models
- EraX-WoW-Turbo V1.1 (CT2): ~6-8GB VRAM (int8_float16 compute type)
- Qwen3-4B-Instruct (AWQ 4-bit): ~3-4GB VRAM
- Sequential load/unload guarantees stability and prevents OOM errors

**Latency Budget:**

- Model loading overhead per model: 5-25 seconds
- EraX-WoW-Turbo V1.1: 8-12s load time (CT2 optimized)
- ChunkFormer: ~5–10s load time (implementation-dependent)
- Qwen3-4B-Instruct (AWQ): 15-25s load time (direct vLLM inference)
- Target total: 2-3 minutes for 45-minute audio (NFR-5)

**Pipeline State Machine:**

```
PENDING
   ↓
PREPROCESSING
   ↓ (Audio normalization, format validation)
PROCESSING_ASR
   ↓ (ASR model unload + VRAM cleanup)
PROCESSING_LLM
   ↓ (LLM model load → inference → unload + VRAM cleanup)
COMPLETE / FAILED
```

**State Transition Logic:**

1. **PENDING → PREPROCESSING**:

   - **Audio Format Validation:** Verify file is valid audio (not corrupted)
   - **Format Normalization:** Convert to WAV 16kHz mono if needed (ffmpeg)
   - **Audio Quality Checks:**
     - Verify sample rate (must be ≥16kHz for ASR accuracy)
     - Check for silence/empty audio (reject if <1s of audio detected)
     - Validate duration (reject if exceeds max allowed duration)
   - **Update job metadata:** Record original format, duration, sample rate
   - **Error handling:** Invalid/corrupted files → FAILED status with `AUDIO_DECODE_ERROR`

2. **PREPROCESSING → PROCESSING_ASR**:

   - Load ASR processor based on `asr_backend` parameter
   - Execute transcription with VAD filtering (if enabled) on normalized audio
   - Record ASR confidence, VAD coverage for metrics
   - **Critical**: Call `processor.unload()` → `torch.cuda.empty_cache()`

3. **PROCESSING_ASR → PROCESSING_LLM**:

   - **Check Context Length:** Count tokens in transcript and compare to `LLM_CONTEXT_LENGTH`
   - **Determine Processing Strategy (Task-Dependent Thresholds):**
     - **Text Enhancement:** Threshold at 40% of context limit (output ≈ input length, needs room for both)
     - **Summarization:** Threshold at 70% of context limit (output << input, compression task)
     - If below threshold: Use direct processing
     - If above threshold: Use appropriate chunking strategy (MapReduce for summarization, overlapping chunks for enhancement)
   - Determine if text enhancement needed (FR-3 conditional logic)
   - Whisper with erax-wow-turbo variant outputs punctuated text → skip enhancement
   - If ASR backend lacks punctuation (not applicable in V1.0), call LLM punctuation
   - Calculate edit distance for `edit_rate_cleaning` metric

4. **PROCESSING_LLM → COMPLETE**:
   - Load Qwen3-4B-Instruct model via vLLM Python library
   - Process requested features via direct inference:
     - `clean_transcript`: Enhancement (if needed) - separate inference
     - `summary`: Structured output with JSON schema validation (FR-4) - includes tags field (FR-6)
   - **Note**: Tags are generated as part of the summary template, not separately
   - **Single inference**: Summary + tags generated together for efficiency
   - **Critical**: Call `llm_processor.unload()` → `torch.cuda.empty_cache()`
   - Prompting, structured outputs, and chat template details are defined in
     `docs/prompt-management.md`
   - Construct `versions` block with full model metadata (NFR-1)
   - Calculate `metrics` block (FR-5)
   - Store complete result in Redis DB 1

**Key Architectural Decision:** V1.0 uses a single sequential model loading strategy. Any form of LLM preloading is deferred to a future release.

**Failure Handling:**

- Wrap pipeline in try/except block
- Transient failures: RQ auto-retries (max 2 retries)
- Permanent failures: Move to `FailedJobRegistry`
- Failed jobs can be manually requeued:
  - `queue.failed_job_registry.requeue(job_id)`
  - `rq requeue --queue default <job_id>`

**OpenTelemetry Instrumentation:**

- Worker creates span for each job execution
- `trace_id` passed through RQ job payload from API request
- Enables end-to-end distributed tracing correlation

**Long Context Handling:** Deferred to V1.1. See section 10.1.1 for planned implementation.

### 3.3. Audio Preprocessing

**Purpose:** Normalize audio inputs to ensure consistent ASR quality and prevent processing failures due to format incompatibilities.

**Audio Storage Structure:**

Files are organized in task-specific directories for better isolation and cleanup:

```
data/audio/
└── {task-id}/
    ├── raw.{original-ext}     # Original uploaded file (immutable)
    └── preprocessed.wav        # Normalized 16kHz mono WAV
```

**Benefits of Task-Based Directories:**

- Clear isolation between different processing jobs
- Simple cleanup: `rm -rf data/audio/{task-id}/`
- Better traceability and debugging
- Preserves original file for reproducibility
- Scalable to thousands of tasks

**Preprocessing Pipeline:**

```python
# src/processors/audio/preprocessor.py
from pathlib import Path
import subprocess
from loguru import logger

class AudioPreprocessor:
    """Normalize audio files for ASR processing"""

    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1  # Mono
    MIN_DURATION_SEC = 1.0

    def preprocess(self, input_path: Path) -> dict:
        """
        Validate and normalize audio file.

        Args:
            input_path: Path to raw audio file (e.g., data/audio/{task-id}/raw.mp3)

        Returns metadata dict with:
        - original_format, duration, sample_rate
        - normalized_path (e.g., data/audio/{task-id}/preprocessed.wav)
        """
        # 1. Probe audio metadata
        metadata = self._probe_audio(input_path)

        # 2. Validate audio quality
        if metadata["duration"] < self.MIN_DURATION_SEC:
            raise ValueError(f"Audio too short: {metadata['duration']}s < {self.MIN_DURATION_SEC}s")

        # 3. Normalize if needed
        if self._needs_normalization(metadata):
            normalized_path = self._normalize_audio(input_path, metadata)
            metadata["normalized_path"] = normalized_path
            logger.info(
                "Audio normalized",
                original_format=metadata["format"],
                target_format="wav_16khz_mono"
            )

        return metadata

    def _probe_audio(self, path: Path) -> dict:
        """Use ffprobe to extract audio metadata"""
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=sample_rate,channels,codec_name",
            "-of", "json", str(path)
        ], capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)
        return {
            "format": data["format"].get("codec_name", "unknown"),
            "duration": float(data["format"]["duration"]),
            "sample_rate": int(data["streams"][0]["sample_rate"]),
            "channels": int(data["streams"][0]["channels"])
        }

    def _needs_normalization(self, metadata: dict) -> bool:
        """Check if audio needs format conversion"""
        return (
            metadata["sample_rate"] != self.TARGET_SAMPLE_RATE
            or metadata["channels"] != self.TARGET_CHANNELS
            or metadata["format"] not in ["pcm_s16le", "wav"]
        )

    def _normalize_audio(self, input_path: Path, metadata: dict) -> Path:
        """Convert audio to WAV 16kHz mono using ffmpeg"""
        # Save preprocessed file in same directory as raw
        task_dir = input_path.parent
        output_path = task_dir / "preprocessed.wav"

        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", str(self.TARGET_SAMPLE_RATE),
            "-ac", str(self.TARGET_CHANNELS),
            "-sample_fmt", "s16",
            str(output_path)
        ], check=True, capture_output=True)

        return output_path
```

**Preprocessing Validations:**

| Check          | Threshold           | Error Code            | Action |
| -------------- | ------------------- | --------------------- | ------ |
| Duration       | ≥1 second           | `AUDIO_TOO_SHORT`     | Reject |
| Sample Rate    | ≥8kHz               | `INVALID_SAMPLE_RATE` | Reject |
| File Integrity | Valid audio codec   | `AUDIO_DECODE_ERROR`  | Reject |
| Format         | Supported by ffmpeg | `UNSUPPORTED_FORMAT`  | Reject |

**Normalization Criteria:**

Audio is normalized to **WAV 16kHz mono** if:

- Sample rate ≠ 16kHz
- Channels ≠ 1 (mono)
- Format is not PCM WAV (e.g., MP3, AAC, FLAC)

**Benefits:**

- ✅ Consistent ASR input quality (16kHz is optimal for Whisper)
- ✅ Early detection of corrupted/invalid files
- ✅ Reduced ASR errors from format incompatibilities
- ✅ Metadata tracking for debugging and metrics

**Error Handling:**

```python
try:
    metadata = preprocessor.preprocess(audio_path)
except subprocess.CalledProcessError as e:
    raise ProcessingError(
        error_code="AUDIO_DECODE_ERROR",
        message=f"Failed to decode audio: {e.stderr}",
        stage="preprocessing",
        retry=False
    )
except ValueError as e:
    raise ProcessingError(
        error_code="AUDIO_TOO_SHORT",
        message=str(e),
        stage="preprocessing",
        retry=False
    )
```

**Dependencies:** Requires `ffmpeg` and `ffprobe` installed in worker container

### 3.4. ASR Processor Abstraction (FR-2)

**Design Pattern: Factory Pattern + Strategy Pattern for Pluggable Backends**

The ASR processing system uses a **Factory Pattern** to instantiate backend-specific processors, combined with the **Strategy Pattern** for polymorphic execution. This architecture enables:

- **Separation of Concerns:** Each ASR backend implementation in its own module
- **Easy Extension:** Add new backends without modifying existing code
- **Type Safety:** Factory validates backend names and returns correct types
- **Testability:** Mock specific backends independently

**V1.0 Implementation Scope:**

- ✅ Basic segment-level transcription (text + start/end timestamps)
- ✅ VAD filtering for improved speed and accuracy
- ✅ Language detection
- ✅ Segment-level confidence scores
- ✅ Sequential processing pattern (load → execute → unload)
- ❌ Word-level timestamps → Deferred to V1.1+
- ❌ Batched inference → Deferred to V1.2+
- ✅ Speaker diarization — Implemented; see `docs/archive/diarization/DIARIZATION_FINAL_STATUS.md`

**Module Structure:**

```
src/processors/asr/
├── __init__.py          # Public exports: ASRFactory, ASRBackend
├── factory.py           # ASRFactory + ASRBackend protocol
├── whisper.py           # WhisperBackend implementation (V1.0)
└── chunkformer.py       # ChunkFormerBackend implementation (V1.0)
```

**Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Worker Pipeline                           │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Job: { "asr_backend": "whisper", ... }             │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ASRFactory.create(backend_id)                │  │
│  │                                                       │  │
│  │  BACKENDS = {                                        │  │
│  │    "whisper": WhisperBackend,                        │  │
│  │    "chunkformer": ChunkFormerBackend                 │  │
│  │  }                                                    │  │
│  └──────────┬───────────────────────┬───────────────────┘  │
│             │                       │                        │
│    "whisper" backend       "chunkformer" backend            │
│             │                       │                        │
│             ▼                       ▼                        │
│  ┌────────────────┐      ┌──────────────────┐              │
│  │ WhisperBackend │      │ChunkFormerBackend│              │
│  ├────────────────┤      ├──────────────────┤              │
│  │ • execute()    │      │ • execute()      │              │
│  │ • unload()     │      │ • unload()       │              │
│  │ • get_version()│      │ • get_version()  │              │
│  └────────────────┘      └──────────────────┘              │
│             │                       │                        │
│             └───────────┬───────────┘                        │
│                         ▼                                    │
│              ┌────────────────────┐                          │
│              │    ASRResult       │                          │
│              └────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

**Factory Implementation Pattern:**

```python
# src/processors/asr/factory.py
from typing import Protocol

class ASRBackend(Protocol):
    """Protocol defining the ASR backend interface"""
    def execute(self, file_path: str) -> ASRResult: ...
    def unload(self) -> None: ...
    def get_version_info(self) -> dict: ...

class ASRFactory:
    """Factory for instantiating ASR backend implementations"""

    BACKENDS: dict[str, type[ASRBackend]] = {
        "whisper": WhisperBackend,
        "chunkformer": ChunkFormerBackend,
    }

    @classmethod
    def create(cls, backend_id: str, **config) -> ASRBackend:
        if backend_id not in cls.BACKENDS:
            raise ValueError(f"Unknown backend: {backend_id}")
        return cls.BACKENDS[backend_id](**config)
```

_See `ASR_Factory_Pattern.md` for complete details._

**Usage in Worker Pipeline:**

```python
# src/worker/pipeline.py
from processors.asr import ASRFactory

def process_audio_task(task_params: dict) -> dict:
    backend_id = task_params.get("asr_backend", "whisper")
    asr = ASRFactory.create(backend_id=backend_id)

    try:
        return asr.execute(task_params["audio_path"])
    finally:
        asr.unload()
```

**Abstract Interface:**

Each concrete processor must implement:

- `execute(file_path: str) → ASRResult`: Process audio file
- `unload()`: Release GPU memory and resources
- `get_version_info() → dict`: Return model metadata for NFR-1

**Concrete Implementations:**

**Concrete Implementations:**

**ASR Defaults (V1.0)**

- `asr_backend`: `chunkformer`
- Whisper (faster-whisper):
  - `whisper_model_variant`: `erax-wow-turbo` (org CT2 default; use `large-v3`/`distil-large-v3` for official)
  - `whisper_beam_size`: `5`
  - `whisper_vad_filter`: `true`
  - `whisper_compute_type`: `int8_float16`
  - `whisper_device`: `cuda`
  - `whisper_condition_on_previous_text`: `true`
- ChunkFormer:
  - `chunkformer_model_name`: `khanhld/chunkformer-rnnt-large-vie`
  - `chunkformer_chunk_size`: `64`
  - `chunkformer_left_context`: `128`
  - `chunkformer_right_context`: `128`
  - `chunkformer_total_batch_duration`: `14400`
  - `chunkformer_return_timestamps`: `true`

Installation:

- Python API and CLI: `pip install chunkformer` (provides `chunkformer-decode`)

**1. WhisperBackend (Default Backend)**

**Module:** `src/processors/asr/whisper.py`

**Purpose:** Whisper-based ASR with support for multiple model variants

**Default Model: EraX-WoW-Turbo V1.1**

- Model: `erax-ai/EraX-WoW-Turbo-V1.1-CT2`
- Runtime: CTranslate2 with int8_float16 compute type
- VRAM: ~6-8GB (int8_float16 compute type)
- Base Architecture: Optimized Whisper variant
- VAD Filter: Silero VAD for non-speech segment filtering
- Beam Size: 5 (configurable; 1-2 for faster with minimal quality loss)

**Key Features:**

- Native punctuation and capitalization (FR-3: skip enhancement)
- VAD filtering significantly improves speed on silent audio
- CTranslate2 optimized inference engine
- Batch-friendly architecture
- WoW (Whisper on Whisper) optimization for speed

**Alternative Whisper Models (Configurable):**

- Model selection via `WHISPER_MODEL_VARIANT` environment variable
- Supported variants: `large-v3`, `turbo`, `distil-large-v3`, custom CT2 models

**Distil-Whisper Support:**

For faster, smaller Distil-Whisper models (e.g., `distil-large-v3`):

**Required Configuration:**

```env
WHISPER_MODEL_VARIANT=distil-large-v3
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false  # REQUIRED for Distil models
```

**Why `condition_on_previous_text=False`?**

Distil-Whisper models are trained without context conditioning and perform better with this setting disabled. This is documented in the official faster-whisper repository.

**Configuration Example:**

```python
# In config.py or .env
whisper_model_variant = "distil-large-v3"
whisper_condition_on_previous_text = False  # Critical for Distil
whisper_language = "en"  # Optional: force language
```

**Benefits of Distil-Whisper:**

- **Speed:** ~5-7x faster than large-v3
- **Size:** ~40% smaller model
- **Quality:** Minimal accuracy loss (<2% WER increase)
- **VRAM:** Lower memory requirements

**Version Metadata (NFR-1):**

- Backend name: `whisper`
- Model variant: `erax-wow-turbo` (or configured variant)
- Model path: `erax-ai/EraX-WoW-Turbo-V1.1-CT2`
- Checkpoint hash: Auto-detected or manually configured
- Compute type: `int8_float16`
- Decoding params: `beam_size`, `vad_filter`

**2. ChunkFormerBackend** (Long-Form Backend)

**Module:** `src/processors/asr/chunkformer.py`

**Purpose:** Single-request latency optimization

**Configuration:**

- Model: `khanhld/chunkformer-rnnt-large-vie` (Hugging Face Hub)
- Architecture: Chunk-wise processing (ICASSP 2025)
- Processing: Unbatched, sequential chunks
- Context: Configurable left/right context windows

**Key Features:**

- Significantly faster than Whisper for single requests
- Foundation for future streaming capabilities
- Offline processing support

**Version Metadata (NFR-1):**

- Backend name: `chunkformer`
- Model variant: `rnnt-large-vie`
- Checkpoint hash: Auto-detected from HuggingFace
- Architecture params: `chunk_size_frames`, `left_context_frames`, `right_context_frames`, `total_batch_duration_seconds`

Implementation note: This project wraps the underlying implementation behind a stable factory interface. Configuration fields (`chunk_size`, `left_context`, `right_context`, `total_batch_duration`, `return_timestamps`) are mapped to the chosen decode routine. The backend handles multiple ChunkFormer API response formats (dict, list, or string) and normalizes them to a consistent ASRResult structure.

**ASRResult Data Structure:**

```
ASRResult:
  - text: str                    # Full transcript text
  - segments: list[dict] | None  # Segments with timestamps
  - language: str | None         # Auto-detected language code (e.g., "en", "vi")
  - language_probability: float | None  # Confidence in language detection (0-1)
  - confidence: float | None     # Average confidence score (if available)
  - error: dict | None           # Structured error information (if execution failed)
```

**Note:** The `language` and `language_probability` fields are automatically populated by faster-whisper's language detection when `language` parameter is not explicitly set.

**Adding New ASR Backends:**

To extend the system with a new ASR backend, follow these steps:

1. **Create Backend Module:** `src/processors/asr/new_backend.py`

   - Implement `execute()`, `unload()`, `get_version_info()` methods
   - Follow lazy loading pattern for GPU resources
   - Ensure proper cleanup in `unload()`

2. **Register in Factory:** Update `src/processors/asr/factory.py`

   ```python
   from .new_backend import NewBackend

   class ASRFactory:
       BACKENDS = {
           "whisper": WhisperBackend,
           "chunkformer": ChunkFormerBackend,
           "new_backend": NewBackend,  # Add here
       }
   ```

3. **Add Configuration:** Update `.env.template` with backend-specific environment variables

4. **Write Tests:** Create `tests/unit/test_new_backend.py` with interface compliance tests

5. **Update Documentation:** Backend automatically appears in `/v1/models` via `ASRFactory.list_available()`

**Implementation Template:** See `ASR_Factory_Pattern.md` for complete code template and detailed examples.

**Benefits of Factory Pattern:**

- ✅ **Open/Closed Principle:** Add backends without modifying existing code
- ✅ **Single Responsibility:** Each backend in isolated module
- ✅ **Type Safety:** Factory validates backend names at runtime
- ✅ **Discoverability:** Centralized registry for all backends
- ✅ **Testing:** Mock factory in tests, unit test backends independently

---

### 3.5. LLM Processor (FR-3, FR-4, FR-6)

**Design: Direct vLLM Python Library Integration**

**Architecture Decision:**

- LLM loaded **directly in worker process** using vLLM Python library
- **Sequential execution**: Load model → Run inference → Unload model → Clear VRAM
- Eliminates HTTP overhead and separate service complexity
- Fits within single-GPU VRAM budget through sequential processing

**Preloading Mode:** Deferred post V1.0.

**vLLM Python Configuration:**

- Model: `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`
- Quantization: AWQ 4-bit (memory efficient, ~3-4GB VRAM)
- Library: `vllm` Python package for direct inference
- Sampling Params: Configurable temperature, max_tokens, top_p
- GPU: Same GPU as ASR (sequential usage prevents memory conflicts)

**Model Characteristics:**

- Base: Qwen3-4B-Instruct
- Quantization: AWQ (Activation-aware Weight Quantization) 4-bit
- VRAM Footprint: ~3-4GB when loaded
- Instruction-tuned for chat and task following
- Optimized for production deployment with vLLM

**Key Implementation Details:**

```python
from vllm import LLM, SamplingParams

# Load model once per job
llm = LLM(
    model="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
    quantization="awq",
    gpu_memory_utilization=0.9,
    max_model_len=32768
)

# Configure sampling
sampling = SamplingParams(temperature=0.7, max_tokens=4096, top_p=0.95)

# Run inference
outputs = llm.generate(prompts, sampling)

# Cleanup
del llm
torch.cuda.empty_cache()
```

_Refer to implementation guide for complete error handling and retry logic._

**Memory Management Strategy:**

1. ASR completes → Unload ASR model → Clear VRAM
2. Load LLM model (3-4GB)
3. Run LLM tasks for the job (enhancement if needed, structured summary with embedded tags)
4. Unload LLM model → Clear VRAM
5. Ready for next job

**Processing Capabilities:**

**1. Text Enhancement (FR-3 - Conditional):**

- **When**: ASR backend lacks punctuation (e.g., chunkformer)
- **Task**: Add punctuation, fix capitalization, remove fillers
- **Metric**: Calculate `edit_rate_cleaning` via Levenshtein distance
- **Skip**: When Whisper with erax-wow-turbo variant is used (native punctuation)

**2. Structured Summarization with Embedded Categorization (FR-4, FR-6):**

- **Input**: Enhanced transcript
- **Output**: JSON object matching template schema (including tags field)
- **Templates**: Stored in `/app/templates/{template_id}.json`
- **Chat Templates**: Use a Jinja ChatML template by default. Mount templates
  under `/app/assets/chat-templates/` and configure vLLM with
  `--chat-template /app/assets/chat-templates/qwen3_nonthinking.jinja`.
- **Tags Integration**: Templates include a `tags` field (array of 1-5 strings) for content categorization
- **Validation**: JSON Schema validation post-generation
- **Constrained Decoding**: LLM generates valid JSON with all required fields in single pass
- **Efficiency**: Single LLM inference generates both summary and tags simultaneously

**Example Template Schema:**

```json
{
  "title": "Meeting Notes V1",
  "type": "object",
  "properties": {
    "title": { "type": "string" },
    "abstract": { "type": "string" },
    "main_points": {
      "type": "array",
      "items": { "type": "string" }
    },
    "tags": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1,
      "maxItems": 5
    }
  },
  "required": ["title", "abstract", "main_points", "tags"]
}
```

_Full template schemas with detailed descriptions available in `/app/templates/`. Chat template guidance in §3.2.1 and PRD._

**LLM Prompt Structure:**

The system uses a dedicated prompt management subsystem to render prompts for the LLM. This subsystem uses Jinja2 templates to allow for configurable and dynamic prompt generation.

See section 3.5.1 for more details.

### 3.5.1. Prompt Management

**Design: Externalized Jinja2 Templates**

To ensure that prompts are easy to manage, version, and iterate on, they are externalized from the application code and managed using the Jinja2 templating engine.

**Directory Structure:**

- **`templates/prompts/`**: This directory stores all Jinja2 prompt templates (e.g., `text_enhancement_v1.jinja`, `meeting_notes_v1.jinja`).
- **`src/processors/prompt/`**: This directory contains the prompt rendering logic.

**Key Components:**

1.  **`TemplateLoader` Class (`src/processors/prompt/template_loader.py`):**

    - Responsible for loading Jinja2 templates from the `templates/prompts` directory.
    - Uses a cache (`@lru_cache`) to avoid redundant file system access.

2.  **`PromptRenderer` Class (`src/processors/prompt/renderer.py`):**
    - Uses the `TemplateLoader` to get a template.
    - Renders the template with the provided context (e.g., transcript, JSON schema) to generate the final prompt string.

**Workflow:**

1.  The LLM processor (e.g., `SummaryLLMProcessor`) receives a request with a `template_id`.
2.  It calls the `PromptRenderer` with the `template_id` and the necessary context.
3.  The `PromptRenderer` uses the `TemplateLoader` to load the corresponding Jinja2 template.
4.  The template is rendered with the context, and the final prompt is returned to the LLM processor.

**Benefits:**

- ✅ Single LLM inference (not separate calls)
- ✅ Faster processing time (reduces latency)
- ✅ Better semantic coherence (tags aligned with summary)
- ✅ Simpler pipeline logic

**3. Template Validation Strategy**

**Overview:** Three-layer validation ensures LLM summaries conform to JSON Schema templates for type safety and API contract compliance.

**Validation Layers:**

1. **Request-Time (API Server):** Validate template_id exists, load schema, verify required "tags" field
2. **Generation-Time (LLM Processor):** Use jsonschema.validate() on LLM output, retry on malformed JSON (max 2 retries with lower temperature)
3. **Response-Time (Worker):** Re-validate before storing to catch data corruption

**Template Schema Requirements:**

- Root type: `"object"` with `properties` and `required` fields
- Must include `tags` array field (1-5 items, FR-6)
- Follow JSON Schema Draft 7 standard

**Example Template Structure:**

```json
{
  "type": "object",
  "properties": {
    "title": { "type": "string", "maxLength": 200 },
    "main_points": { "type": "array", "items": { "type": "string" } },
    "tags": { "type": "array", "minItems": 1, "maxItems": 5 }
  },
  "required": ["title", "main_points", "tags"]
}
```

**Error Taxonomy:**

| Validation Failure         | Error Code                 | HTTP Status | Retry? |
| -------------------------- | -------------------------- | ----------- | ------ |
| Template not found         | `TEMPLATE_NOT_FOUND`       | 422         | No     |
| Invalid template JSON      | `INVALID_TEMPLATE`         | 500         | No     |
| Missing tags field         | `INVALID_TEMPLATE_SCHEMA`  | 500         | No     |
| LLM output invalid JSON    | `INVALID_JSON`             | 500         | Yes    |
| LLM output schema mismatch | `SCHEMA_VALIDATION_FAILED` | 500         | Yes    |

**Template Discovery:** GET `/v1/templates` endpoint lists available templates with metadata (template_id, name, required_fields, has_tags)

**Benefits:** Early validation prevents wasted GPU cycles, type-safe responses, automatic retry on transient failures

> **Implementation Reference:** See `docs/Template_Validation_Guide.md` for complete code examples and integration patterns

**4. Long Context Handling:** Deferred to V1.1. See section 10.1.1 for planned implementation details (MapReduce and overlapping chunking strategies).

**Version Metadata (NFR-1):**

- Model name: `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`
- Checkpoint hash: Auto-detected or configured
- Quantization: `awq-4bit`
- Generation params: `temperature`, `max_tokens`

**LLMResult Data Structure:**

```python
LLMResult:
  - clean_transcript: str | None    # FR-3: Enhanced text
  - summary: dict | None            # FR-4: Structured summary + tags
  - model_version: str              # NFR-1: Version metadata
  - duration_ms: int                # FR-5: Processing time
```

_Tags are embedded within summary dict, not separate field._

### 3.6. Redis Configuration

**Dual-Database Strategy:**

| Database | Purpose      | Key Pattern      | Persistence  |
| -------- | ------------ | ---------------- | ------------ |
| DB 0     | RQ job queue | RQ internal      | Ephemeral OK |
| DB 1     | Task results | `task:{task_id}` | AOF required |

**Persistence Configuration:**

```
appendonly yes
appendfsync everysec
```

**Why AOF?**

- Write durability with 1-second fsync interval
- Balance between safety and performance/
- Survives container restarts

**Critical Requirement:**
Docker Compose **must** mount volume for Redis data persistence:

```
volumes:
  - ./data/redis:/data  # REQUIRED for AOF file persistence
```

Without volume mount, Redis data lost on container restart.

**Task Data Structure (Redis Hash):**

```
Key: task:{task_id}

Fields:
  status              → "PENDING" | "PROCESSING_ASR" | "PROCESSING_LLM" | "COMPLETE" | "FAILED"
  submitted_at        → ISO 8601 timestamp
  request_params      → JSON string (features, asr_backend, template_id)
  versions            → JSON string (full versioning block, NFR-1)
  metrics             → JSON string (FR-5 metrics block)
  results             → JSON string (conditional output based on features)
  error_message       → String (if status = FAILED)
```

### 3.7. Deployment Architecture

**Package Management: Pixi (prefix.dev)**

All Python dependencies managed through Pixi for:

- Fast dependency resolution
- Reproducible builds via `pixi.lock` (optional)
- Consistent environments across dev/prod

**Project Structure:**

```
/maie/
├── .env.template           # Configuration template
├── .python-version         # Python 3.11+
├── pyproject.toml          # Project metadata
├── pixi.toml               # Pixi environment manifest
├── docker-compose.yml      # Multi-service orchestration
├── Dockerfile              # Single image for API + Worker
├── README.md
│
├── src/
│   ├── api/                # Litestar application
│   │   ├── main.py
│   │   ├── schemas.py      # Pydantic request/response models
│   │   └── dependencies.py # Auth, validation
│   ├── config.py           # Pydantic Settings (v2)
│   ├── processors/         # Pluggable AI processors
│   │   ├── base.py         # Abstract interfaces
│   │   ├── audio/          # Audio preprocessing
│   │   │   ├── __init__.py
│   │   │   └── preprocessor.py # AudioPreprocessor (ffmpeg wrapper)
│   │   ├── asr/            # ASR backend modules
│   │   │   ├── __init__.py # ASRFactory and exports
│   │   │   ├── factory.py  # ASRFactory + ASRBackend protocol
│   │   │   ├── whisper.py  # WhisperBackend
│   │   │   └── chunkformer.py # ChunkFormerBackend
│   │   ├── llm.py          # Direct vLLM inference
│   │   └── prompt/         # Prompt rendering logic
│   │       ├── __init__.py
│   │       ├── renderer.py
│   │       └── template_loader.py
│   └── worker/
│       ├── main.py         # RQ worker entrypoint
│       └── pipeline.py     # Sequential processing logic
│
├── scripts/
│   └── download_models.sh  # Model download script
│
├── templates/              # JSON Schema definitions (FR-4)
│   ├── meeting_notes_v1.json
│   └── prompts/            # Jinja2 prompt templates
│       ├── text_enhancement_v1.jinja
│       └── meeting_notes_v1.jinja
├── assets/
│   └── chat-templates/     # Jinja ChatML templates for Qwen3 (optional)
│
├── data/                   # Runtime data (mounted volumes)
│   ├── audio/              # Uploaded audio files
│   ├── redis/              # Redis persistence (AOF)
│   └── models/             # AI model weights
│       ├── whisper/
│       │   └── erax-wow-turbo/
│       ├── chunkformer-rnnt-large-vie/
│       └── llm/
│           └── qwen3-4b-awq/
│
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

**Docker Compose Services:**

| Service      | Purpose               | GPU     | Ports       | Volumes               |
| ------------ | --------------------- | ------- | ----------- | --------------------- |
| api          | Litestar HTTP server  | No      | 8000        | audio (RW)            |
| worker       | RQ worker + ASR + LLM | Yes (0) | -           | audio (R), models (R) |
| redis        | Queue + results       | No      | 6379        | redis-data (RW)       |
| rq-dashboard | Queue monitoring      | No      | 9181        | -                     |
| jaeger       | Tracing (optional)    | No      | 16686, 4318 | -                     |

Note: By default, we use the tokenizer’s ChatML template and do not mount or
configure a server chat template. If you need to override globally, mount
`./assets/chat-templates` into the container (e.g., `/app/assets/chat-templates`)
and configure vLLM with `--chat-template /app/assets/chat-templates/<file>.jinja`.

**GPU Allocation Strategies:**

**Single GPU (16-24GB) - Recommended Configuration:**

- Worker: GPU 0 for sequential ASR → LLM processing
- Whisper (erax-wow-turbo variant): ~6-8GB VRAM (sequential phase 1)
- Qwen3-4B-AWQ: ~3-4GB VRAM (sequential phase 2)
- Peak VRAM: ~8GB max (never overlapping)
- Benefits: Simple configuration, works on single GPU systems

**Docker Compose GPU Syntax (Single GPU):**

```yaml
worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["0"]
            capabilities: [gpu]
```

**CUDA Library Setup (Linux):**

For GPU acceleration with faster-whisper, the following NVIDIA libraries must be installed and properly configured:

**Required Libraries:**

```bash
# Install CUDA 12 libraries
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

**Environment Configuration:**

```bash
# Set library path for Python runtime
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

**Docker Integration:**

```dockerfile
# In Dockerfile
RUN pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

# Set environment variable
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib
```

**Docker Compose:**

```yaml
worker:
  environment:
    - LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib
    - OMP_NUM_THREADS=4 # For GPU performance optimization
```

**GPU Performance Tuning:**

For optimal GPU performance, ensure proper CUDA setup:

```yaml
# Docker Compose
worker:
  environment:
    - OMP_NUM_THREADS=4 # Adjust based on CPU cores
    - WHISPER_DEVICE=cuda
    - WHISPER_COMPUTE_TYPE=float16
```

**Recommendation:** Ensure CUDA is properly installed and GPU is available for optimal performance.

### 3.8. Model Downloading Strategy

**Overview:** All AI models must be downloaded before worker startup to ensure deterministic behavior and avoid runtime download failures.

**Model Inventory (V1.0):**

| Model                  | Size   | Download Source | Local Path                                |
| ---------------------- | ------ | --------------- | ----------------------------------------- |
| EraX-WoW-Turbo V1.1    | ~3GB   | HuggingFace Hub | `/data/models/whisper/erax-wow-turbo`     |
| ChunkFormer RNNT Large | ~1.5GB | HuggingFace Hub | `/data/models/chunkformer-rnnt-large-vie` |
| Qwen3-4B-Instruct AWQ  | ~2.5GB | HuggingFace Hub | `/data/models/llm/qwen3-4b-awq`           |

**Download Strategy:**

**1. Pre-Deployment Download (Recommended for V1.0)**

Download all models before `docker-compose up`:

```bash
# Create download script: scripts/download_models.sh
#!/bin/bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./data/models}"
mkdir -p "$MODEL_DIR"/{whisper,chunkformer,llm}

echo "Downloading ASR models..."
hf download erax-ai/EraX-WoW-Turbo-V1.1-CT2 \
  --local-dir "$MODEL_DIR/whisper/erax-wow-turbo" \
  --local-dir-use-symlinks False

hf download khanhld/chunkformer-rnnt-large-vie \
  --local-dir "$MODEL_DIR/chunkformer-rnnt-large-vie" \
  --local-dir-use-symlinks False

echo "Downloading LLM models..."
hf download cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit \
  --local-dir "$MODEL_DIR/llm/qwen3-4b-awq" \
  --local-dir-use-symlinks False

echo "✓ All models downloaded successfully"
```

**Usage:**

```bash
# Run before first deployment
chmod +x scripts/download_models.sh
./scripts/download_models.sh

# Verify downloads
ls -lh data/models/{whisper,chunkformer,llm}/
```

**2. Model Loading in Worker**

Workers load models from local filesystem only:

```python
# src/processors/asr/whisper.py
class WhisperBackend:
    def __init__(self, model_variant: str = "erax-wow-turbo"):
        model_path = f"/data/models/whisper/{model_variant}"

        # Fail fast if model not found
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run scripts/download_models.sh before starting workers."
            )

        self.model = ct2.Model(model_path, device="cuda")
```

**3. Model Verification on Startup**

Add health check to worker startup:

```python
# src/worker/main.py
def verify_models():
    """Verify all required models exist before accepting jobs"""
    required = [
        "/data/models/whisper/erax-wow-turbo",
        "/data/models/chunkformer-rnnt-large-vie",
        "/data/models/llm/qwen3-4b-awq"
    ]

    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Missing models: {missing}. "
            f"Run scripts/download_models.sh"
        )

    from loguru import logger
    logger.info("✓ All required models verified")

if __name__ == "__main__":
    verify_models()
    # Start RQ worker...
```

**Benefits:**

- ✅ Deterministic startup (no network dependency)
- ✅ Faster job processing (no download latency)
- ✅ Offline deployment capability
- ✅ Version pinning (no unexpected updates)

**Future Enhancements (Post V1.0):**

- Automatic model download on first worker startup
- Model version checking and auto-update
- Model caching layer for shared deployments
- S3/object storage model distribution

---

## 4. Data Models & Contracts

### 4.1. Request Schema (FR-7.1)

**POST /v1/process**

```
Content-Type: multipart/form-data

Fields:
  file: UploadFile               # Required, .wav or .mp3
  features: list[str]            # Optional, default: ["clean_transcript", "summary"]
                                 # Valid: "raw_transcript", "clean_transcript", "summary", "enhancement_metrics"

  asr_backend: str               # Optional, default: "chunkformer"
                                 # Valid: "whisper", "chunkformer"
                                 # For "whisper": model variant set via WHISPER_MODEL_VARIANT config (default: erax-wow-turbo)
  template_id: str               # Required if "summary" in features
                                 # Example: "meeting_notes_v1"
                                 # Template must include "tags" field for categorization

Response: 202 Accepted
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b"
}
```

**Important Changes:**

- Tags are automatically included in the `summary` output via the template schema
- Templates should define a `tags` field (array, 1-5 items) for categorization
- This reduces LLM inference calls and improves coherence

### 4.2. Response Schema (FR-7.2)

Refer to PRD Appendix B for the authoritative response schema and examples. Tags are embedded within the `summary` object per the template schema.

**Status: FAILED**

```json
{
  "task_id": "...",
  "status": "FAILED",
  "error": "Model loading failed: CUDA out of memory"
}
```

### 4.3. Discovery Endpoints (FR-8)

**GET /v1/models**

**Implementation:** Uses `ASRFactory.list_available()` for dynamic backend discovery

```json
{
  "models": [
    {
      "backend_id": "whisper",
      "name": "Whisper (CTranslate2)",
      "default_variant": "erax-wow-turbo",  # org CT2 default; replace with official variants as needed
      "model_path": "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
      "description": "Default ASR backend with native punctuation",
      "capabilities": ["transcription", "punctuation", "vad_filtering"],
      "compute_type": "int8_float16",
      "vram_gb": "6-8"
    }
  ]
}
```

**Note:** Backend list auto-populated from `ASRFactory.BACKENDS` registry.

**GET /v1/templates**

```json
{
  "templates": [
    {
      "template_id": "meeting_notes_v1",
      "name": "Meeting Notes V1",
      "description": "Structured format for meeting transcripts",
      "schema_url": "/v1/templates/meeting_notes_v1/schema"
    }
  ]
}
```

---

## 5. Configuration Management (NFR-4)

### 5.1. Environment Variables

All system behavior configured via environment variables loaded from `.env` file.

**Minimal Configuration (V1.0):**

| Variable                             | Purpose                                | Default                              |
| ------------------------------------ | -------------------------------------- | ------------------------------------ |
| `PIPELINE_VERSION`                   | Version stamping for NFR-1             | `1.0.0`                              |
| `WHISPER_MODEL_VARIANT`              | Whisper model variant                  | `erax-wow-turbo` (org default)       |
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | Use context (False for Distil-Whisper) | `true`                               |
| `WHISPER_LANGUAGE`                   | Force language code or auto-detect     | `None` (auto-detect)                 |
| `CHUNKFORMER_MODEL_NAME`             | ChunkFormer model name                 | `khanhld/chunkformer-rnnt-large-vie` |
| `REDIS_URL`                          | Queue and results store                | `redis://redis:6379/0`               |
| `SECRET_API_KEY`                     | API authentication                     | —                                    |
| `MAX_QUEUE_DEPTH`                    | Backpressure threshold                 | `50`                                 |
| `MAX_FILE_SIZE_MB`                   | Upload size limit                      | `500`                                |
| `OMP_NUM_THREADS`                    | CPU threads for performance            | `4` (recommended)                    |

**Example `.env` (V1.0)**

```env
# Pipeline
PIPELINE_VERSION=1.0.0
WHISPER_MODEL_VARIANT=erax-wow-turbo
WHISPER_CONDITION_ON_PREVIOUS_TEXT=true
WHISPER_LANGUAGE=  # Empty = auto-detect

# Runtime
REDIS_URL=redis://redis:6379/0
SECRET_API_KEY=CHANGE_ME
MAX_QUEUE_DEPTH=50
MAX_FILE_SIZE_MB=500

# CPU Performance (when running on CPU)
OMP_NUM_THREADS=4
```

**Note on DEFAULT_FEATURES:** The `tags` feature has been removed as tags are now embedded within summary templates (FR-6). Templates should define a `tags` field for categorization.

**Note:** Long-context strategies and related tuning parameters are out of scope for V1.0 and will be reconsidered post-release if needed.

### 5.2. Pydantic Settings (v2)

Configuration loaded via `pydantic-settings` package (separate from `pydantic` core):

**Import Pattern:**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
```

**Configuration Pattern:**

```python
model_config = SettingsConfigDict(
    env_file='.env',
    env_file_encoding='utf-8',
    case_sensitive=False
)
```

**Version Tracking Fields (NFR-1):**

- `pipeline_version`: System version
- `whisper_model_checkpoint_hash`: Whisper model hash (auto-detect or manual, variant-specific)
- `chunkformer_model_checkpoint_hash`: ChunkFormer hash
- `vllm_model_checkpoint_hash`: LLM model hash

---

## 6. Testing Strategy

### 6.1. Unit Tests

**Scope:** Individual components in isolation

**Key Test Cases:**

- Pydantic schema validation (request/response contracts)
- Configuration loading from environment variables
- Utility functions (edit distance, JSON schema validation)
- **ASR Factory:** Backend registration, instantiation, error handling
- **Individual Backends:** WhisperBackend, ChunkFormerBackend interface compliance
- Redis operations (key generation, serialization, state transitions)

**Testing Framework:** `pytest` with fixtures and mocks

**Test Structure Examples:**

```python
# Factory pattern tests
def test_factory_lists_backends():
    assert "whisper" in ASRFactory.list_available()

def test_factory_creates_valid_backend():
    backend = ASRFactory.create("whisper")
    assert hasattr(backend, "execute")

def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        ASRFactory.create("invalid_backend")

# Backend implementation tests
def test_whisper_backend_interface_compliance():
    backend = WhisperBackend()
    assert callable(backend.execute)
    assert callable(backend.unload)
    assert callable(backend.get_version_info)
```

**Mock Factory Pattern:**

To maintain clean, reusable test fixtures without brittle inline Mock objects, the codebase uses centralized mock factories:

```python
# tests/fixtures/mock_factories.py
def create_mock_asr_output(text="...", confidence=0.95):
    """Factory returning real ASRResult dataclass instances"""
    return ASRResult(text=text, segments=[], ...)

def create_mock_asr_processor(backend="whisper"):
    """Factory returning properly configured Mock processor"""
    return Mock(spec=ASRBase, execute=Mock(return_value=ASRResult(...)))

# Usage in fixtures
@pytest.fixture
def mock_asr_processor():
    return create_mock_asr_processor(backend="whisper")

# Usage in tests
def test_something(mock_asr_processor):
    result = mock_asr_processor.execute(audio_data)
```

This approach enables:

- Reusable mock objects across test suite
- Type-safe factories returning proper dataclass instances
- Easy customization for specific test scenarios
- Reduced duplication and maintenance burden

_See `tests/fixtures/mock_factories.py` for complete factory implementations._

**Coverage Target:** >80% for non-ML code  
**Current Status:** 836 unit tests passing, 100% pass rate

### 6.2. Integration Tests

**Scope:** Component interactions validating API↔queue↔worker↔storage flow and schema enforcement.

**Current Status:**

- **34 integration tests passing** (marked with `@pytest.mark.integration`)
- **Execution time:** ~1.2 minutes
- **Coverage:** API routes, worker pipeline, Redis integration, diarization, verbose logging

**Running Integration Tests:**

```bash
# Run only integration tests (fast, uses mocks)
pytest -m "integration" --tb=short -v

# Run with specific focus
pytest tests/integration/test_worker_pipeline_real.py -v
```

**Key Test Cases:**

1. **API → Redis Integration**

   - Enqueue job with correct parameters; returns `202` with `task_id`
   - Initialize task record in results DB with `PENDING`
   - Backpressure: when `MAX_QUEUE_DEPTH` exceeded, returns `429`
   - Mock factories ensure consistent processor behavior

2. **Worker → Redis Integration (with mock processors)**

   - Dequeue job, update status transitions (ASR → LLM → COMPLETE)
   - Store final result document and metrics; retrievable via `/v1/status/{task_id}`
   - Failed job registry populated on exceptions
   - Uses `create_mock_asr_processor()` and `create_mock_llm_processor()` factories

3. **File Upload Flow**

   - Multipart handling with size/type validation
   - Persist under `/data/audio/{task_id}/raw.{ext}`
   - Cleanup policy on `FAILED` (configurable, default keep)

4. **Processor Loading (with mocks)**

   - ASR factory instantiation for default backend (`whisper`)
   - LLM processor initialization path exercised
   - Memory cleanup verification (unload hooks called)
   - Factory validation and error handling on invalid backend

5. **Template and Schema Validation**

   - `/v1/templates` lists available templates with `schema_url`
   - Missing `template_id` when `summary` requested → `422`
   - LLM mocked invalid JSON once → retry → valid JSON
   - Response validated against JSON Schema; on mismatch → `500` with retryable flag

6. **Real Audio Processing**
   - Full pipeline with real ASR models (not mocked)
   - Diarization with speaker attribution
   - Audio preprocessing and metrics collection
   - Enhancement logic validation

**Execution:**

```bash
# Standard run
pytest -m "integration" --tb=short -v

# With specific test file
pytest tests/integration/test_worker_pipeline_real.py -xvs

# With coverage reporting
pytest -m "integration" --cov=src --cov-report=html
```

**Mock Factory Usage Pattern:**

All integration tests use centralized mock factories to ensure consistency:

```python
# In tests/integration/test_worker_pipeline_real.py
from tests.fixtures.mock_factories import (
    create_mock_asr_output,
    create_mock_asr_processor,
    create_mock_llm_processor,
    create_mock_redis_client
)

@pytest.fixture
def mock_asr_model(mock_asr_output):
    return create_mock_asr_processor(backend="whisper")

@pytest.fixture
def mock_llm_model():
    return create_mock_llm_processor(backend="vllm")

def test_full_pipeline(mock_asr_model, mock_llm_model):
    # Test implementation using factories
    result = pipeline(mock_asr_model, mock_llm_model)
    assert result is not None
```

**Deprecated Tests Removed:**

- 22 deprecated IoU calculation tests removed (old diarization algorithm)
- All remaining tests are actively maintained and passing

### 6.3. End-to-End (E2E) Tests

**Scope:** Full system with real models on GPU hardware

**Prerequisites:**

- GPU-enabled environment (16GB+ VRAM, RTX 3060+ recommended)
- All models downloaded to `/data/models`
- Full docker-compose stack running
- For LLM tests: Set `LLM_TEST_MODEL_PATH` environment variable

**Available E2E Test Suites:**

1. **Standard E2E Tests (Always Run)**

   - Happy path with real audio processing
   - Feature selection and combinations
   - Error handling and edge cases
   - Current status: 34 passing integration tests

2. **Real LLM Integration Tests (Optional - Expensive)**

   These tests perform actual LLM model inference and should only be run for model validation:

   ```bash
   # Enable by setting model path
   export LLM_TEST_MODEL_PATH="/path/to/qwen3-4b-instruct"

   # Run LLM integration tests
   pytest -m real_llm tests/integration/test_real_llm_integration.py -v
   ```

   - **Duration:** 30-120 seconds per test
   - **Resource:** 4-12GB GPU VRAM
   - **When to run:** Before releases, model validation, LLM feature development
   - **When to skip:** Regular development, CI/CD pipelines

   Tests covered:

   - Real text enhancement with actual LLM
   - Structured summarization generation
   - Model loading and caching
   - Error handling with real models
   - Performance benchmarking

3. **API Routes Tests (Optional - Configuration Dependent)**

   Redis and API endpoint validation:

   ```bash
   # Ensure Redis is running
   docker-compose up -d redis

   # Run API integration tests
   pytest tests/api/test_routes_redis_integration.py -v
   ```

   - **When to run:** API feature development, pre-deployment validation
   - **When to skip:** ASR/LLM-focused work
   - **Requirements:** Redis running, template database initialized

4. **GPU-Specific Tests (Manual Only - Safety-Gated)**

   Tests requiring GPU execution are explicitly marked to skip for safety:

   ```bash
   # These tests may cause segmentation faults
   # Only run manually in isolated environment
   # See OPTIONAL_TESTS_GUIDE.md for details
   ```

**Key Test Scenarios:**

1. **Happy Path Test:**

   - Submit 1-minute audio file
   - Verify all requested features generated
   - Validate response schema compliance
   - Check versioning metadata completeness

2. **Feature Selection Tests:**

   - Test each feature independently
   - Test various feature combinations
   - Verify conditional field presence in results

3. **LLM JSON Compliance (when enabled):**

   - Force invalid JSON once; ensure single retry and final valid JSON
   - Validate final payload matches schema and golden sample

4. **Error Handling:**

   - Invalid file format submission
   - Corrupted audio file
   - Missing template_id when summary requested
   - Queue overflow (backpressure)

5. **Concurrent Load Test:**
   - Submit 6+ concurrent requests
   - Verify all jobs complete successfully
   - Check queue management and task ordering

**Validation Criteria:**

- Processing time: <3 minutes for 45-minute audio
- Success rate: >99% for valid inputs
- Versioning data: Complete and accurate
- Metrics: RTF, confidence, VAD coverage within expected ranges
- Speaker attribution: Correct assignment for diarized segments

**Running Different Test Subsets:**

```bash
# Fast unit tests only (~1.8 min)
pytest -m "not integration" --ignore=tests/e2e --tb=no -q

# Integration tests only (~1.2 min)
pytest -m "integration" --tb=short -v

# Full suite excluding expensive tests (~2.8 min)
pytest --ignore=tests/e2e -m "not real_llm" --tb=short -q

# Full suite including all tests (~4-5 min with LLM tests)
export LLM_TEST_MODEL_PATH="/path/to/model"
pytest --ignore=tests/e2e --tb=short -q
```

**CI/CD Pipeline Recommendations:**

| Stage              | Command                                             | Duration | Resources  |
| ------------------ | --------------------------------------------------- | -------- | ---------- |
| Pre-commit         | `pytest -m "not integration" --ignore=tests/e2e -q` | ~1.8 min | Minimal    |
| PR validation      | `pytest --ignore=tests/e2e -m "not real_llm" -q`    | ~2.8 min | Moderate   |
| Nightly build      | `pytest --ignore=tests/e2e -q`                      | ~4-5 min | High (GPU) |
| Production release | `pytest -m "integration" -v` + LLM tests            | ~3 min   | High (GPU) |

### 6.4. Performance Testing

**Scope:** Benchmarking and optimization validation

**Key Metrics:**

1. **Latency Benchmarks:**

   - Model loading times (ASR: <12s, LLM: <25s)
   - End-to-end processing time vs audio duration
   - Real-Time Factor (RTF) for different audio lengths

2. **Throughput Testing:**

   - Maximum concurrent requests before degradation
   - Queue processing rate (jobs/hour)
   - GPU utilization patterns

3. **Resource Monitoring:**
   - Peak VRAM usage per pipeline stage
   - CPU utilization during non-GPU phases
   - Redis memory consumption with queue depth

**Test Scenarios:**

- **Short Audio (1-5 min):** Measure overhead-to-processing ratio
- **Medium Audio (30-60 min):** Target use case validation
- **Long Audio (2-3 hours):** Stress test and memory leak detection
- **Burst Load:** 20 requests submitted simultaneously

**Tools:**

- `locust` for load generation
- `nvidia-smi` for GPU monitoring
- `prometheus` + `grafana` for metrics visualization

### 6.5. Regression Testing

**Scope:** Ensure changes don't break existing functionality

**Strategy:**

- Maintain golden dataset of audio files with expected outputs
- Run full E2E suite on each commit to main branch
- Compare output hashes and versioning metadata
- Alert on RTF degradation >10%

**Automation:**

- CI/CD pipeline with GPU runners
- Automated comparison of results against baselines

---

## 7. Security Considerations

### 7.1. API Authentication

**Mechanism:** Static API key via X-API-Key header

**Implementation:**

```python
from litestar import Request
from litestar.exceptions import NotAuthorizedException

async def api_key_guard(request: Request, _: Any) -> None:
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != settings.SECRET_API_KEY:
        raise NotAuthorizedException("Invalid or missing API key")
```

**Deployment Considerations:**

- Store `SECRET_API_KEY` in `.env` file (never commit)
- Use strong random keys (32+ characters)
- Rotate keys periodically in production
- Consider JWT tokens for multi-user scenarios (future)

### 7.2. File Upload Security

**Threats:**

- Malicious file uploads (malware, exploits)
- Path traversal attacks
- Disk space exhaustion

**Mitigations:**

1. **File Type Validation:**

   - Whitelist only `.wav`, `.mp3` extensions
   - Verify MIME type matches extension
   - Use `python-magic` for content-based detection

2. **File Size Limits:**

   - Hard limit: 500MB per upload
   - Configurable via `MAX_FILE_SIZE_MB` env var
   - Return `413 Payload Too Large` on violation

3. **Secure File Storage:**

   - Generate UUIDs for task directories (ignore client names)
   - Store in task-specific directories with restricted permissions
   - Pattern: `{AUDIO_DIR}/{task_id}/raw.{ext}` (original upload)
   - Pattern: `{AUDIO_DIR}/{task_id}/preprocessed.wav` (normalized audio)
   - Task-based isolation prevents cross-contamination

4. **Disk Quotas:**
   - Monitor `/data/audio` disk usage
   - Implement cleanup policy (delete files >7 days old)
   - Alert when usage >80%

### 7.3. File Cleanup Strategy

**V1.1 Approach: Task-Based Directory Cleanup**

**Cleanup Policy:**

- Task directories retained for 7 days after job completion
- Failed job directories retained for 14 days (debugging)
- Successful job directories eligible for immediate cleanup if disk pressure

**Directory Structure Benefits:**

- Atomic cleanup: Delete entire task directory in one operation
- No orphaned files from incomplete processing
- Preserves both raw and preprocessed files until cleanup
- Easier to implement retention policies

**Implementation:**

```bash
# Simple cron-based cleanup (V1.1)
# Add to host crontab or docker-compose healthcheck

# Daily cleanup: Remove task directories older than 7 days
0 2 * * * find /data/audio -maxdepth 1 -type d -mtime +7 -exec rm -rf {} +

# Emergency cleanup: Remove task directories if disk >90%
*/15 * * * * [ $(df -h /data/audio | awk 'NR==2 {print $5}' | sed 's/%//') -gt 90 ] && \
  find /data/audio -maxdepth 1 -type d -mtime +1 -exec rm -rf {} +
```

**Manual Cleanup Commands:**

```bash
# Remove all task directories older than 7 days
docker exec maie-api find /data/audio -maxdepth 1 -type d -mtime +7 -exec rm -rf {} +

# Remove specific task directory (includes raw + preprocessed)
docker exec maie-api rm -rf /data/audio/{task_id}

# Check disk usage
docker exec maie-api df -h /data/audio

# List task directories with sizes
docker exec maie-api du -sh /data/audio/*
```

**Future Enhancements (Post V1.0):**

- Automatic cleanup triggered by worker after successful processing
- Configurable retention policies via environment variables
- Archive to S3/object storage before deletion
- Cleanup API endpoint for on-demand purging

### 7.4. Data Privacy

**Considerations:**

- Audio files may contain sensitive conversations
- On-premises deployment reduces third-party exposure
- No data sent to external APIs (self-contained)

**Best Practices:**

- Document data retention policies
- Provide API endpoint for file deletion: `DELETE /v1/tasks/{task_id}`
- Consider encryption at rest for `data/audio` volume
- Log access to sensitive endpoints (audit trail)

### 7.4. Dependency Security

**Strategy:**

- Use `pixi` for reproducible dependency management
- Optionally pin all dependencies in `pixi.lock`
- Regular security audits: `pixi run pip-audit` (or `pip-audit`)
- Subscribe to security advisories for critical packages

**High-Risk Dependencies:**

- `litestar`: Web framework (check for CVEs)
- `redis`: Potential RCE vulnerabilities
- `torch`, `vllm`: ML frameworks (supply chain risks)

### 7.5. Redis Security

**Configuration:**

- Bind to `127.0.0.1` only (not `0.0.0.0`)
- Require password via `requirepass` directive
- Disable dangerous commands: `rename-command FLUSHALL ""`
- Use separate Redis instances for queue vs results (isolation)

**Docker Compose:**

```yaml
redis:
  command: redis-server --requirepass ${REDIS_PASSWORD} --bind 127.0.0.1
  environment:
    - REDIS_PASSWORD=${REDIS_PASSWORD}
```

### 7.6. Error Taxonomy

**Overview:** Comprehensive error classification for debugging, monitoring, and user feedback.

**Error Categories:**

| Category                 | HTTP Status   | Retry? | User Action       | Examples                          |
| ------------------------ | ------------- | ------ | ----------------- | --------------------------------- |
| **Client Error**         | 400, 413, 415 | No     | Fix request       | Invalid format, file too large    |
| **Auth Error**           | 401, 403      | No     | Check credentials | Missing/invalid API key           |
| **Resource Error**       | 404           | No     | Check identifier  | Task ID not found                 |
| **Rate Limit**           | 429           | Yes    | Wait and retry    | Queue full, backpressure          |
| **Validation Error**     | 422           | No     | Fix parameters    | Invalid backend, missing template |
| **Processing Error**     | 500           | Maybe  | Contact support   | Model crash, CUDA OOM             |
| **Infrastructure Error** | 503           | Yes    | Wait and retry    | Redis down, worker unavailable    |

**Common Error Codes:**

```python
# Client Errors
"INVALID_FILE_FORMAT"     # 415 - Unsupported audio format
"FILE_TOO_LARGE"          # 413 - Exceeds 500MB limit
"VALIDATION_ERROR"        # 422 - Missing required field
"UNKNOWN_BACKEND"         # 422 - Invalid asr_backend value
"QUEUE_FULL"              # 429 - System at capacity

# Processing Errors
"MODEL_LOAD_FAILED"       # 500 - Model files not found
"CUDA_OOM"                # 500 - GPU memory exhausted
"LLM_GENERATION_FAILED"   # 500 - Invalid JSON/schema mismatch (retry)
"AUDIO_DECODE_ERROR"      # 500 - Corrupted audio file
"AUDIO_TOO_SHORT"         # 422 - Audio duration < 1 second
"INVALID_SAMPLE_RATE"     # 422 - Sample rate < 8kHz
"UNSUPPORTED_FORMAT"      # 415 - Audio format not supported by ffmpeg

# Infrastructure Errors
"REDIS_UNAVAILABLE"       # 503 - Redis connection failed (retry)
"NO_WORKERS"              # 503 - No workers available (retry)
```

**Error Response Structure:**

```python
# API Error (immediate failure)
{
  "error_code": "INVALID_FILE_FORMAT",
  "message": "Unsupported audio format. Allowed: .wav, .mp3",
  "status_code": 415,
  "retry": false
}

# Job Error (async failure)
{
  "task_id": "...",
  "status": "FAILED",
  "error_code": "CUDA_OOM",
  "error": "CUDA out of memory during ASR inference",
  "stage": "asr_execute",
  "retry": false
}
```

**Pipeline Stage Identifiers:**

`request_validation` → `file_upload` → `job_enqueue` → `audio_preprocess` → `audio_normalize` → `asr_init` → `asr_decode` → `asr_execute` → `llm_init` → `llm_enhance` → `llm_summarize` → `schema_validation` → `result_store`

**Error Handling Best Practices:**

1. **Fail Fast:** Validate inputs before enqueueing jobs
2. **Specific Errors:** Use precise codes (not generic "INTERNAL_ERROR")
3. **Actionable Messages:** Tell users what to do next
4. **Stage Tracking:** Include stage for debugging
5. **Retry Logic:** Mark transient vs permanent failures

**Monitoring Integration:** Log errors with structured data using loguru's `.bind()` method (task_id, error_code, stage, duration); emit Prometheus error_counter metrics per error_code and stage

**Example:**

```python
from loguru import logger

logger.bind(
    task_id=task_id,
    error_code="CUDA_OOM",
    stage="asr_execute",
    duration_sec=elapsed_time
).error("Job failed")
```

> **Implementation Reference:** See `docs/Error_Handling_Guide.md` for complete error handling patterns and recovery strategies

---

## 8. Monitoring and Observability

### 8.1. Logging Strategy

**Log Levels:**

- `DEBUG`: Detailed model loading, inference timing
- `INFO`: Request received, job started/completed, status updates
- `WARNING`: Queue depth approaching limit, retries triggered
- `ERROR`: Model loading failures, processing errors, validation failures

**Structured Logging with Loguru:**

```python
from loguru import logger

# Configure JSON output for production
logger.add(
    sink=sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    serialize=True  # Output JSON for log aggregation
)

# Usage: context is automatically serialized
logger.bind(task_id=task_id, asr_backend=asr_backend).info(
    "Job started",
    features=features,
    audio_duration=duration_sec
)
```

**Benefits:** Zero-config structured logging, automatic exception formatting, thread-safe, performance optimized

**Status Values (V1.0):**

- `PENDING`: Job queued, not yet started
- `PREPROCESSING`: Audio normalization and validation
- `PROCESSING_ASR`: ASR model processing audio
- `PROCESSING_LLM`: LLM processing transcript
- `COMPLETE`: Job finished successfully
- `FAILED`: Job failed with error

**Note:** Extended status values for long-context processing (MapReduce chunks, etc.) are planned for V1.1.

**Log Aggregation:**

- Output JSON logs to stdout
- Collect via Docker logging driver (json-file, fluentd)
- Forward to centralized system (ELK stack, Loki)

### 8.2. Metrics Collection

**Key Metrics:**

1. **API Metrics:**

   - Request rate (requests/sec)
   - Response times (p50, p95, p99)
   - Status code distribution (2xx, 4xx, 5xx)
   - Queue depth over time

2. **Worker Metrics:**

   - Job processing time distribution
   - RTF by audio duration bucket
   - Model loading times (ASR, LLM)
   - VRAM usage per pipeline stage

3. **System Metrics:**
   - GPU utilization (%)
   - GPU memory usage (GB)
   - CPU utilization per container
   - Redis memory usage and key count

**Implementation:**

- Instrument code with Prometheus client
- Expose `/metrics` endpoint (separate port for API)
- Worker publishes metrics to Prometheus pushgateway

### 8.2.1. Stage-Level Timing Metrics

**Overview:** Detailed timing breakdown for each pipeline stage to identify bottlenecks and optimize performance.

**Metric Structure:**

```json
{
  "task_id": "...",
  "status": "COMPLETE",
  "timing": {
    "total_ms": 162800,
    "queue_wait_ms": 1200,
    "stages": {
      "asr_init_ms": 8500,
      "asr_decode_ms": 450,
      "asr_inference_ms": 42000,
      "asr_unload_ms": 1200,
      "llm_init_ms": 22000,
      "llm_enhance_ms": 15000,
      "llm_summarize_ms": 28000,
      "llm_unload_ms": 1500,
      "schema_validate_ms": 120,
      "result_store_ms": 350
    },
    "breakdown_pct": {
      "asr_total": 52.1,
      "llm_total": 40.8,
      "overhead": 7.1
    }
  }
}
```

**Pipeline Stages (11 total):**

| Stage                | Description                         | Typical Duration | Critical?   |
| -------------------- | ----------------------------------- | ---------------- | ----------- |
| `queue_wait_ms`      | Time from enqueue to worker dequeue | 0-5s             | No          |
| `asr_init_ms`        | ASR model loading (cold start)      | 5-12s            | Yes         |
| `asr_decode_ms`      | Audio file decoding                 | 0.2-1s           | No          |
| `asr_inference_ms`   | Actual transcription                | 30-90s           | Yes         |
| `asr_unload_ms`      | ASR model cleanup                   | 0.5-2s           | No          |
| `llm_init_ms`        | LLM model loading (cold start)      | 15-25s           | Yes         |
| `llm_enhance_ms`     | Text enhancement (if needed)        | 5-20s            | Conditional |
| `llm_summarize_ms`   | Structured summary + tags           | 10-40s           | Yes         |
| `llm_unload_ms`      | LLM model cleanup                   | 1-3s             | No          |
| `schema_validate_ms` | JSON schema validation              | 0.05-0.5s        | No          |
| `result_store_ms`    | Redis write operation               | 0.1-1s           | No          |

**Implementation:** Use `PipelineTiming` dataclass with `record(stage, duration_ms)` method; track start_time, stages dict, and calculate percentage breakdown (asr_total, llm_total, overhead)

**Prometheus Metrics:**

```python
stage_duration = Histogram(
    'maie_stage_duration_seconds',
    'Duration of pipeline stages',
    ['stage', 'backend_id'],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120]
)
```

**Optimization Targets (NFR-5):**

| Metric                         | V1.0 Target | V1.1 Goal     |
| ------------------------------ | ----------- | ------------- |
| Total processing (45min audio) | <180s       | <120s         |
| ASR inference RTF              | <0.06       | <0.04         |
| Model loading overhead         | <40s        | <5s (preload) |
| LLM summary generation         | <30s        | <20s          |

> **Implementation Reference:** See `docs/Performance_Monitoring_Guide.md` for complete PipelineTiming class and PromQL query examples

### 8.3. Distributed Tracing

**OpenTelemetry Integration:**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.jaeger import JaegerExporter

tracer = trace.get_tracer(__name__)

@app.post("/v1/process")
async def process_audio():
    with tracer.start_as_current_span("api.process_audio") as span:
        span.set_attribute("audio.duration", duration_sec)
        # ... processing logic
```

**Trace Flow:**

1. API creates root span with `task_id`
2. Span context passed to RQ job metadata
3. Worker continues span for ASR phase
4. Worker creates child span for LLM phase
5. Complete trace visible in Jaeger UI

**Benefits:**

- End-to-end latency breakdown
- Identify bottlenecks (model loading vs inference)
- Correlate logs with traces via `trace_id`

**Note:** Enhanced tracing for long-context processing (MapReduce, chunking) will be added in V1.1.

### 8.4. Alerting Rules

**Critical Alerts:**

1. **Worker Down:**

   - Condition: No job completions in 5 minutes
   - Action: Page on-call, restart worker container

2. **Queue Backlog:**

   - Condition: Queue depth >100 for >10 minutes
   - Action: Notify team, consider scaling workers

3. **High Error Rate:**

   - Condition: >5% of jobs failing
   - Action: Check logs, investigate model issues

4. **GPU OOM:**

   - Condition: CUDA out of memory errors
   - Action: Reduce `gpu_memory_utilization`, check for memory leaks

5. **Disk Space:**
   - Condition: `/data/audio` usage >90%
   - Action: Trigger cleanup job, expand volume

**Alerting Tools:**

- Prometheus Alertmanager for metric-based alerts
- PagerDuty/Opsgenie for on-call escalation

### 8.5. Health Checks

**API Health Endpoint:**

`GET /health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "redis_connected": true,
  "queue_depth": 3,
  "worker_active": true
}
```

**Worker Health:**

- Heartbeat mechanism: Worker updates timestamp in Redis every 30s
- API checks timestamp freshness: `redis.get("worker:heartbeat")`
- Report worker as unhealthy if timestamp >60s old

**Container Healthchecks:**

```yaml
api:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 5s
    retries: 3
```

---

## 9. Deployment Procedures

### 9.1. Initial Setup

**Prerequisites:**

- Docker + Docker Compose installed
- NVIDIA Container Toolkit configured
- GPU with 16GB+ VRAM

**Steps:**

1. Clone repository and navigate to project root
2. Copy `.env.template` to `.env` and configure:

   ```bash
   cp .env.template .env
   # Edit .env with your SECRET_API_KEY and other settings
   ```

3. Create required directories:

   ```bash
   mkdir -p data/audio data/models data/redis templates
   ```

4. Download models (first run will auto-download, or pre-fetch):

   ```bash
   # EraX-WoW-Turbo
   hf download erax-ai/EraX-WoW-Turbo-V1.1-CT2 --local-dir data/models/erax-wow-turbo

   # Qwen3-4B-Instruct AWQ
   hf download cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit --local-dir data/models/qwen3-4b-awq

   # ChunkFormer
   hf download khanhld/chunkformer-rnnt-large-vie --local-dir data/models/chunkformer-rnnt-large-vie
   ```

5. Build and start services:

   ```bash
   docker-compose build
   docker-compose up -d
   ```

6. Verify services are running:

   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   ```

7. Access Swagger UI: `http://localhost:8000/schema/swagger`

8. Access RQ Dashboard: `http://localhost:9181`

### 9.2. Configuration Updates

**To change ASR backend default:**

```bash
# Edit .env
DEFAULT_ASR_BACKEND=whisper

# Restart API and worker
docker-compose restart api worker
```

**To enable LLM preloading (KAD-8):**

```bash
# Edit .env
WORKER_MODEL_STRATEGY=preload_llm

# Restart worker to apply
docker-compose restart worker
```

**To add new summarization template:**

1. Create JSON schema: `templates/my_template.json`
2. Restart API: `docker-compose restart api`
3. Verify: `curl http://localhost:8000/v1/templates`

### 9.3. Scaling Workers (Dual GPU)

**Edit `docker-compose.yml`:**

```yaml
services:
  worker-gpu0:
    extends: worker
    environment:
      - NVIDIA_VISIBLE_DEVICES=0

  worker-gpu1:
    extends: worker
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
```

**Apply changes:**

```bash
docker-compose up -d --scale worker-gpu0=1 --scale worker-gpu1=1
```

### 9.4. Backup and Recovery

**Critical Data:**

- Redis AOF file: `data/redis/appendonly.aof`
- Audio files: `data/audio/` (optional, can be deleted after processing)
- Templates: `templates/` (version controlled)

**Backup Procedure:**

```bash
# Stop Redis to ensure consistent backup
docker-compose stop redis

# Backup Redis data
tar -czf redis-backup-$(date +%Y%m%d).tar.gz data/redis/

# Restart Redis
docker-compose start redis
```

**Recovery Procedure:**

```bash
# Stop services
docker-compose down

# Restore Redis data
tar -xzf redis-backup-YYYYMMDD.tar.gz

# Start services
docker-compose up -d
```

### 9.5. Troubleshooting

**Worker Not Processing Jobs:**

```bash
# Check worker logs
docker-compose logs -f worker

# Verify Redis connection
docker-compose exec worker redis-cli -h redis ping

# Check GPU availability
docker-compose exec worker nvidia-smi
```

**GPU Out of Memory:**

```bash
# Check current VRAM usage
nvidia-smi

# Reduce LLM memory utilization in .env
VLLM_GPU_MEMORY_UTILIZATION=0.85

# Restart worker
docker-compose restart worker
```

**Queue Stuck:**

```bash
# Check queue status
docker-compose exec redis redis-cli -h redis LLEN rq:queue:default

# Inspect failed jobs
docker-compose exec worker rq info --url redis://redis:6379/0

# Requeue all failed jobs
docker-compose exec worker rq requeue --all --url redis://redis:6379/0
```

---

## 10. Future Enhancements (Post V1.0)

### V1.0 ASR Feature Scope

**What's Included in V1.0:**

- ✅ Basic segment-level transcription (text + timestamps)
- ✅ VAD filtering (Voice Activity Detection)
- ✅ Language detection
- ✅ Segment-level confidence scores
- ✅ Sequential processing (load → execute → unload)
- ✅ Distil-Whisper support

**What's Deferred to Future Releases:**

- ❌ Word-level timestamps → V1.1+
- ❌ Batched inference → V1.2+
- ❌ Model preloading → V1.2+
- ✅ Speaker diarization — Implemented; see `docs/archive/diarization/DIARIZATION_FINAL_STATUS.md`
- ❌ Streaming transcription → V1.3+

**V1.0 Philosophy:** Simple, stable, sequential - one job, one GPU, load → process → unload.

---

### 10.1. V1.1: Advanced ASR & WhisperX-like Capabilities

**Planned ASR Enhancements:**

- **Word-Level Timestamps:** Enable fine-grained timeline features via faster-whisper's `word_timestamps=True` parameter
  - **Use Cases:** Video subtitle generation, interactive transcript navigation, speaker diarization alignment
  - **Not in V1.0 Because:** Not required by PRD metrics (FR-5); adds storage overhead; V1.0 focuses on transcript → summary pipeline
  - **Implementation:** Update `WhisperBackend.execute()` to capture word-level data when enabled
- **Speaker Diarization:** Integrate pyannote.audio for multi-speaker identification
  - **Requires:** Word-level timestamps for alignment accuracy
  - **Output:** Speaker labels per segment/word
- **Subtitle Generation:** Output `.srt` and `.vtt` formats
  - **Requires:** Word-level timestamps
  - **Use Cases:** Accessibility, video platforms

**Implementation Notes:**

- Add new `features`: `diarization`, `word_timestamps`, `subtitles`
- New processor: `DiarizationProcessor` (pyannote-audio)
- Update response schema with speaker labels and word-level timestamps
- Update `ASRResult` dataclass to include optional `words` field

---

### 10.1.1 V1.1: Performance & Long-Context

**Planned Performance Enhancements:**

- **LLM Preloading (Performance Mode):** Optional worker mode to load the LLM at startup and reuse across jobs on high-VRAM systems to reduce latency.
  - **Not in V1.0 Because:** Contradicts sequential architecture (TDD Section 3.2); requires >24GB VRAM; V1.0 prioritizes stability
  - **Prerequisites:** Multi-GPU support OR larger VRAM, revised memory management
- **Long-Context Handling Strategies:** Introduce task-dependent strategies for transcripts exceeding LLM context window:
  - Overlapping chunking for text enhancement (local task)
  - MapReduce for summarization and tags (global task)

**Configuration Notes:**

- Add `WORKER_MODEL_STRATEGY` with values `sequential` (default) and `preload_llm`.
- Add `LLM_CONTEXT_LENGTH`, `LLM_CHUNK_SIZE`, `LLM_CHUNK_OVERLAP` with safe defaults.

### 10.2. V1.2: Batched Inference & Multi-GPU Support

**Goal:** Improve throughput on high-VRAM systems through parallel processing

**Planned ASR Enhancements:**

- **Batched Inference Pipeline:** Use faster-whisper's `BatchedInferencePipeline` for processing multiple audio chunks in parallel
  - **Not in V1.0 Because:** Contradicts sequential architecture; requires model preloading; adds complexity without PRD requirement
  - **Prerequisites:** Model preloading strategy, >24GB VRAM OR multi-GPU setup
  - **Expected Speedup:** 3-5x for longer audio files
  - **Implementation:** New `execute_batched()` method in `WhisperBackend`
- **Multi-GPU Scaling:** Distribute ASR and LLM across separate GPUs
  - **Architecture Change:** Load ASR on GPU 0, LLM on GPU 1 (eliminates unload/reload cycles)
  - **VRAM Requirement:** 2x 16GB GPUs OR 1x 32GB+ GPU
- **Model Preloading:** Keep models resident in VRAM between jobs
  - **Configuration:** `WORKER_MODEL_STRATEGY=preload_llm` or `preload_all`
  - **Memory Planning:** Pre-calculate peak VRAM usage across pipeline stages

**Configuration Example:**

```bash
# V1.2 High-Performance Mode
WORKER_MODEL_STRATEGY=preload_all
WHISPER_BATCH_SIZE=16
WHISPER_USE_BATCHED_PIPELINE=true
GPU_ASR_DEVICE=cuda:0
GPU_LLM_DEVICE=cuda:1
```

**Benefits:**

- Higher throughput (6+ concurrent → 15+ concurrent)
- Lower latency (no model loading overhead)
- Better GPU utilization

**Trade-offs:**

- Higher VRAM requirements
- More complex deployment
- Reduced flexibility (models stay loaded)

---

### ASR Feature Roadmap Summary

| Feature                     | V1.0 | V1.1 | V1.2 | V1.3 | Reason for Deferral                          |
| --------------------------- | ---- | ---- | ---- | ---- | -------------------------------------------- |
| **Basic Transcription**     | ✅   | ✅   | ✅   | ✅   | Core feature                                 |
| Segment-level timestamps    | ✅   | ✅   | ✅   | ✅   | Core feature                                 |
| Segment-level confidence    | ✅   | ✅   | ✅   | ✅   | Core feature (FR-5 metrics)                  |
| VAD filtering               | ✅   | ✅   | ✅   | ✅   | Performance optimization                     |
| Language detection          | ✅   | ✅   | ✅   | ✅   | Core feature                                 |
| Distil-Whisper support      | ✅   | ✅   | ✅   | ✅   | Simple model swap                            |
| Sequential processing       | ✅   | ✅   | ✅   | ✅   | V1.0 architecture                            |
| **Word-Level Timestamps**   | ❌   | ✅   | ✅   | ✅   | Not in PRD metrics; adds complexity          |
| **Speaker Diarization**     | ✅   | ✅   | ✅   | ✅   | Requires word timestamps                     |
| **Subtitle Generation**     | ❌   | ✅   | ✅   | ✅   | Requires word timestamps                     |
| **Batched Inference**       | ❌   | ❌   | ✅   | ✅   | Contradicts sequential architecture          |
| **Model Preloading**        | ❌   | ❌   | ✅   | ✅   | Requires >24GB VRAM; deferred per TDD 3.2    |
| **Multi-GPU Support**       | ❌   | ❌   | ✅   | ✅   | Deployment complexity; not needed for 6+ RPS |
| **Streaming Transcription** | ❌   | ❌   | ❌   | ✅   | Major architecture change; WebSocket needed  |

**Key Insights:**

- **V1.0:** Focus on stability and simplicity (sequential, single GPU, basic features)
- **V1.1:** Add timeline/subtitle features (word timestamps, diarization)
- **V1.2:** Optimize for throughput (batching, preloading, multi-GPU)
- **V1.3:** Enable real-time use cases (streaming)

---

### 10.2.1 V1.2: Streaming Support

**Goal:** Real-time transcription with websocket API (V1.3+ feature preview)

**Technical Approach:**

– ChunkFormer long-form decode is supported via the backend wrapper

- WebSocket endpoint: `/v1/stream`
- Chunk-by-chunk transcription with incremental updates

### 10.3. V1.3: Multi-Language Support

**Goal:** Automatic language detection and translation

**Implementation:**

- Add language detection preprocessor
- Integrate NLLB-200 for translation
- New feature: `translation` with target language param

### 10.4. V1.4: Distributed Architecture with CPU-Only Worker

**Goal:** Offload non-GPU tasks to separate CPU-only machine for better resource utilization

**Motivation:**

- Free up GPU resources for inference-intensive tasks
- Reduce costs by using cheaper CPU-only machines for I/O and preprocessing
- Improve overall system throughput

**Architecture Changes:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Machine (Primary)                         │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────────┐ │
│  │ API Server   │      │    Redis     │      │  GPU Worker   │ │
│  │  (Litestar)  │◄────►│   (Network)  │◄────►│  (ASR + LLM)  │ │
│  └──────────────┘      └──────┬───────┘      └───────────────┘ │
└────────────────────────────────┼──────────────────────────────────┘
                                 │
                                 │ Network
                                 │
┌────────────────────────────────┼──────────────────────────────────┐
│                    CPU Machine (Secondary)                        │
│                         ┌──────┴───────┐                          │
│                         │ CPU Worker   │                          │
│                         │ (RQ Worker)  │                          │
│                         └──────────────┘                          │
│                                                                   │
│  Tasks:                                                           │
│  - File preprocessing (format conversion, validation)             │
│  - Audio normalization and resampling                             │
│  - Result post-processing (export, formatting)                    │
│  - Cleanup jobs (old file deletion)                               │
│  - Metrics aggregation and reporting                              │
└───────────────────────────────────────────────────────────────────┘
```

**Implementation Approach:**

1. **Queue Separation:**
   - GPU queue: `rq:queue:gpu` for ASR and LLM inference
   - CPU queue: `rq:queue:cpu` for preprocessing and post-processing
2. **Two-Phase Processing:**

   ```
   POST /v1/process
     → Enqueue to CPU queue (preprocessing)
       → Audio validation, format conversion
       → Enqueue to GPU queue (inference)
         → ASR + LLM processing
         → Enqueue to CPU queue (post-processing)
           → Format results, cleanup
           → Mark COMPLETE
   ```

3. **Shared Storage:**
   - NFS mount or S3-compatible storage
   - Both machines access `/data/audio` and `/data/models`
4. **Redis Configuration:**
   - Network-accessible Redis (remove `bind 127.0.0.1`)
   - TLS encryption for cross-machine communication
   - Separate databases for GPU and CPU queues

**Benefits:**

- **GPU Efficiency:** GPU only does inference, not I/O
- **Cost Optimization:** CPU machine can be cheaper, lower-spec
- **Scalability:** Scale CPU workers independently
- **Throughput:** Parallel preprocessing + inference

**Challenges:**

- Network latency for file transfers
- Shared storage setup complexity
- More complex deployment and monitoring

**Configuration Example:**

```yaml
# GPU Machine - docker-compose.gpu.yml
services:
  api:
    # ... existing config
  worker-gpu:
    environment:
      - REDIS_URL=redis://redis.internal:6379
      - QUEUE_NAME=gpu

# CPU Machine - docker-compose.cpu.yml
services:
  worker-cpu:
    environment:
      - REDIS_URL=redis://redis.internal:6379
      - QUEUE_NAME=cpu
    volumes:
      - /mnt/nfs/audio:/data/audio
```

**Timeline:** Post V1.3 (when CPU machine becomes available)

### 10.5. Operational Improvements

- **Kubernetes Deployment:** Helm charts for production
- **Auto-Scaling:** Scale workers based on queue depth
- **Model Caching:** Persistent model loading across jobs
- **Result Caching:** Deduplicate identical audio files

---

## 11. Appendices

### Appendix A: Glossary

| Term     | Definition                                          |
| -------- | --------------------------------------------------- |
| **ASR**  | Automatic Speech Recognition                        |
| **AOF**  | Append-Only File (Redis persistence)                |
| **AWQ**  | Activation-aware Weight Quantization                |
| **CT2**  | CTranslate2 inference engine                        |
| **LLM**  | Large Language Model                                |
| **RTF**  | Real-Time Factor (processing_time / audio_duration) |
| **VAD**  | Voice Activity Detection                            |
| **VRAM** | Video Random Access Memory (GPU memory)             |

### Appendix B: Model Licenses

| Model               | License    | Commercial Use |
| ------------------- | ---------- | -------------- |
| EraX-WoW-Turbo V1.1 | Apache 2.0 | ✅ Allowed     |
| ChunkFormer         | MIT        | ✅ Allowed     |
| Qwen3-4B-Instruct   | Apache 2.0 | ✅ Allowed     |

### Appendix C: Hardware Requirements

**Minimum:**

- GPU: NVIDIA with 16GB VRAM (e.g., RTX 4060 Ti 16GB)
- CPU: 8 cores
- RAM: 32GB
- Storage: 100GB SSD (for models + data)

**Recommended:**

- GPU: NVIDIA with 24GB VRAM (e.g., RTX 4090, A5000)
- CPU: 16 cores
- RAM: 64GB
- Storage: 500GB NVMe SSD
- Preloading guidance: For `preload_llm`, ensure free VRAM for resident LLM + largest ASR + job buffers; 24GB+ recommended.

**Optimal (Dual GPU):**

- GPU: 2x NVIDIA with 24GB VRAM each
- CPU: 32 cores
- RAM: 128GB
- Storage: 1TB NVMe SSD

### Appendix D: Performance Targets

| Audio Duration | Target Processing Time | Max RTF |
| -------------- | ---------------------- | ------- |
| 1 minute       | 5-10 seconds           | 0.17    |
| 15 minutes     | 30-60 seconds          | 0.07    |
| 45 minutes     | 2-3 minutes            | 0.07    |
| 2 hours        | 8-10 minutes           | 0.08    |

---

**Document Revision History:**

| Version | Date        | Author           | Changes                                                                                                                                              |
| ------- | ----------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | Oct 4, 2025 | Engineering Team | Initial draft                                                                                                                                        |
| 1.1     | Oct 5, 2025 | Engineering Team | Updated ASR backend to whisper with erax-wow-turbo default variant                                                                                   |
| 1.2     | Oct 6, 2025 | Engineering Team | Changed vLLM to direct Python inference, completed testing/security/monitoring sections                                                              |
| 1.3     | Oct 6, 2025 | Engineering Team | Simplified V1.0: single ASR backend (Whisper), sequential-only, minimal config; deferred long-context (KAD-7) and preloading (KAD-8) to V1.1 roadmap |

---

**End of Technical Design Document**

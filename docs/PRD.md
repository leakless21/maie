## Product Requirements Document (PRD)

|                       |                                          |
| :-------------------- | :--------------------------------------- |
| **Project Name**      | Modular Audio Intelligence Engine (MAIE) |
| **Version**           | 1.3 (V1.0 Production Ready)              |
| **Status**            | **Approved**                             |
| **Related Documents** | Project Brief V1.3, TDD V1.3             |
| **Last Updated**      | October 15, 2025                         |

#### 1. Introduction

This document defines the product requirements for the V1.0 release of the Modular Audio Intelligence Engine. It details the project's features and non-functional requirements, serving as the foundational guide for design, development, and testing.

Architecture Overview: The system uses a three-tier architecture with API server, Redis queue/storage, and GPU worker running sequential ASR and LLM inference on a single GPU. This design prioritizes simplicity and reliability on resource-constrained environments. Prompting relies on the default ChatML formatting bundled with the Qwen3 model plus per-`template_id` Jinja prompt templates that render OpenAI-style `messages`.

#### 2. V1.0 Features & Functional Requirements (FR)

- FR-1: Audio Ingestion — The system MUST accept at least `.wav` and `.mp3` audio formats via `multipart/form-data` upload.

- FR-2: ASR Backend — The system provides two ASR backends for V1.0:

  - whisper: Default backend for V1.0, using Whisper-based models via CTranslate2 runtime. The default model variant is EraX-WoW-Turbo V1.1 (`erax-wow-turbo`), which provides native punctuation and capitalization, optimized for throughput with VAD filtering.

  - chunkformer: Alternative backend for long-form audio transcription, using ChunkFormer models optimized for single-request latency. The default model variant is `khanhld/chunkformer-rnnt-large-vie`, which provides chunk-wise processing with configurable context windows and significantly faster processing for long audio files.

  **V1.0 ASR Feature Scope:**

  - ✅ Segment-level transcription (text + start/end timestamps per segment)
  - ✅ VAD filtering (Voice Activity Detection) for improved speed/accuracy
  - ✅ Language detection
  - ✅ Segment-level confidence scores (for FR-5 metrics)
  - ✅ Sequential processing (load → execute → unload pattern)
  - ✅ Distil-Whisper model support (faster alternative models)
  - ✅ ChunkFormer model support (optimized for long-form audio)

  **Deferred to Post-V1.0:**

  - ❌ Word-level timestamps → V1.1+ (not required by FR-5 metrics; for timeline/subtitle features)
  - ❌ Batched inference → V1.2+ (contradicts sequential architecture; requires model preloading)
  - ❌ Speaker diarization → V1.1+ (requires additional pyannote.audio integration)
  - ❌ Streaming transcription → V1.3+ (requires WebSocket API and architectural changes)

- FR-3: Text Enhancement — Text enhancement is an optional pipeline step. It MUST be bypassed when the selected ASR backend provides adequate punctuation and casing (e.g., `whisper` with `erax-wow-turbo` variant). When required (with backends that lack punctuation; not applicable to V1.0 default), the pipeline will use the LLM for punctuation and capitalization correction. Note: ChunkFormer may require text enhancement depending on model configuration.

- FR-4: Structured Summarization — The system MUST generate a summary from the enhanced transcript using the Qwen3-4B-Instruct LLM with AWQ 4-bit quantization. Summaries MUST validate against a versioned JSON Schema corresponding to the chosen `template_id`. The LLM generation will use constrained decoding or structured output techniques to ensure compliance (vLLM `response_format` with JSON Schema). Prompting MUST use Jinja templates (system + user) rendered to `messages`, relying on the model’s default chat formatting. The prompts themselves are configurable and managed as Jinja2 templates in the `templates/prompts` directory, allowing for easy modification and versioning.

- FR-5: Runtime Metrics — Result MUST include self-reported metrics to help users gauge the quality and performance of the processing. These include: `rtf` (Real-Time Factor), `asr_confidence_avg`, `vad_coverage`, and `edit_rate_cleaning`.

- FR-6: Auto-Categorization (Embedded in Summarization) — The system MUST generate relevant category tags (1-5 tags) for the input audio. This is accomplished by including a `tags` field in the summarization template schema (FR-4). Tags are generated in the same LLM inference pass as the summary, ensuring semantic coherence and reducing processing time. Tags are NOT a separate feature but are embedded within the summary structure.

- FR-7: API Endpoints & Contracts

  - FR-7.1: Asynchronous Processing Endpoint — `POST /v1/process`
    - Request: See Appendix A. Includes optional `features` and `template_id`.
    - Response: `202 Accepted` with `{ "task_id": "..." }`.
  - FR-7.2: Status & Result Retrieval Endpoint — `GET /v1/status/{task_id}`
    - Response: See Appendix B for the full, detailed data contract.

- FR-8: Discovery Endpoints — The API MUST expose discovery endpoints to allow developers to query for available resources.
  - `GET /v1/models`: Returns a list of available ASR backends and model variants. In V1.0, this returns both the Whisper backend (default) and ChunkFormer backend with their respective model variants and metadata.
  - `GET /v1/templates`: Returns a list of available summarization templates and links to their JSON Schemas. (Future: may include prompt metadata version.)

#### 3. Non-Functional Requirements (NFR)

- NFR-1: Reproducibility — All results generated by the API MUST include a `versions` block, containing detailed information about the `pipeline_version`, model names, checkpoint hashes, quantization, key decoding parameters, and prompt template variant to ensure every result is reproducible and auditable.

- NFR-2: Developer Experience — The system MUST expose a valid OpenAPI 3.1 specification.

- NFR-3: Deployment — The entire system MUST be containerized (Docker) and deployable on-premises via a `docker-compose.yml` file. The architecture is optimized for single-GPU deployments (16-24GB VRAM) through sequential model execution only in V1.0.

- NFR-4: Configurability — Use opinionated defaults with a minimal set of environment variables (approximately 5–6) for operators (e.g., Redis URL, API key, queue depth). Additional configurability may be introduced progressively.

- NFR-5: Reliability & Backpressure — The system MUST implement queue depth checks and return `429 Too Many Requests` when the queue is full to prevent overload. Target 6+ concurrent requests without instability on reference hardware.

- NFR-6: Security — The system MUST implement secure file handling practices:
  - Validate file sizes BEFORE loading content to prevent memory exhaustion DoS attacks
  - Validate both MIME types and file extensions for uploaded audio files
  - Sanitize filenames to prevent path traversal attacks
  - Use timing-safe comparison for API key validation
  - Stream large files to disk instead of buffering in memory
  - Store files in isolated task-specific directories (`data/audio/{task-id}/raw.{ext}` and `preprocessed.wav`)

---

#### Appendix A: Request Contract for `POST /v1/process`

Request Body (`multipart/form-data`)

- `file` (file): The audio file. (Required)
- `features` (list[str]): Desired outputs. (Optional)
  - Values: `raw_transcript`, `clean_transcript`, `summary`, `enhancement_metrics`.
  - Default: `["clean_transcript", "summary"]`.
  - Note: `tags` is NO LONGER a separate feature. Tags are embedded in the `summary` output via the template schema.
- `asr_backend`: Backend selection between `"whisper"` (default) and `"chunkformer"` for different use cases.
- `template_id` (str): The summary format. (Required if `summary` is in `features`)
  - Templates should include a `tags` field (array of 1-5 strings) for automatic categorization.

---

#### Appendix B: Final API Data Contract for `GET /v1/status/{task_id}`

Successful Final Response (JSON Body)

**Note:** The example below shows Whisper backend response. For ChunkFormer backend, the `asr_backend` section would include `name: "chunkformer"`, `model_variant: "rnnt-large-vie"`, `model_path: "khanhld/chunkformer-rnnt-large-vie"`, and architecture parameters like `chunk_size`, `left_context_size`, `right_context_size`, `total_batch_duration`, `return_timestamps`.

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "COMPLETE",
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
    "summarization_llm": {
      "name": "cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
      "checkpoint_hash": "f6g7h8i9j0k1...",
      "quantization": "awq-4bit",
      "thinking": false,
      "reasoning_parser": null,
      "structured_output": {
        "backend": "json_schema",
        "schema_id": "meeting_notes_v1",
        "schema_hash": "sha256:..."
      },
      "decoding_params": {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 20,
        "repetition_penalty": 1.05
      }
    }
  },
  "metrics": {
    "input_duration_seconds": 2701.3,
    "processing_time_seconds": 162.8,
    "rtf": 0.06,
    "vad_coverage": 0.88,
    "asr_confidence_avg": 0.91,
    "edit_rate_cleaning": 0.05
  },
  "results": {
    "raw_transcript": "the meeting on oct 4 focused on q4 budgets...",
    "clean_transcript": "The meeting on October 4th focused on Q4 budgets.",
    "summary": {
      "title": "Q4 Budget Planning Meeting",
      "abstract": "A review of the fourth-quarter budget proposal, focusing on marketing and R&D allocations.",
      "main_points": [
        "Marketing budget approved with a 5% increase.",
        "R&D budget for 'Project Phoenix' is pending final review."
      ],
      "tags": [
        "Finance",
        "Budget Planning",
        "Marketing Spend",
        "R&D Allocation"
      ]
    }
  }
}
```

**V1.0 ASR Output Scope:**

V1.0 transcription provides **segment-level** data only:

- ✅ Segment text with start/end timestamps
- ✅ Segment-level confidence scores
- ✅ VAD-filtered speech regions
- ✅ Language detection metadata

**Deferred to Future Releases:**

- ❌ Word-level timestamps → V1.1+ (for timeline/subtitle features)
- ❌ Speaker labels → V1.1+ (requires diarization)
- ❌ Word-level confidence → V1.1+ (requires word timestamps)

The above response contract is complete for V1.0 requirements. Word-level data structures will be added in V1.1 when timeline and subtitle features are implemented.

---

Important Note on Tags:

- Tags (FR-6) are embedded within the `summary` object, not as a separate top-level field
- This ensures tags are generated in a single LLM inference alongside the summary
- Templates must include a `tags` field in their JSON schema
- If `summary` is not requested, tags will not be generated

# MAIE API Guide for Android Developers

## Quick Start

**Base URL:** `http://localhost:8000` or `http://YOUR_SERVER_IP:8000`

⚠️ **CRITICAL:** Use `http://` NOT `https://` - The API runs on plain HTTP by default.

---

## Common SSL Error

If you see this error:

```
write EPROTO 61108795834368:error:100000f7:SSL routines:OPENSSL_internal:WRONG_VERSION_NUMBER
```

**Solution:** You're using `https://` when you should use `http://`. The server doesn't have SSL/TLS enabled by default.

---

## Authentication

All processing endpoints require an API key in the request header:

```
X-API-Key: your_api_key_here
```

- Minimum key length: 32 characters
- Case-insensitive header name
- Contact your backend team to get your API key

---

## API Workflow Overview

```
1. Upload Audio → Get task_id (202 Accepted)
2. Poll Status → Check progress (200 OK)
3. Get Results → When status = "COMPLETE" (200 OK)
```

---

## Endpoint 1: Submit Audio for Processing

### `POST /v1/process`

Upload an audio file for transcription and analysis.

#### Request

**Headers:**

```
X-API-Key: your_api_key_here
Content-Type: multipart/form-data
```

**Form Data Parameters:**

| Parameter            | Type     | Required | Description                                                                               |
| -------------------- | -------- | -------- | ----------------------------------------------------------------------------------------- |
| `file`               | File     | Yes      | Audio file (WAV, MP3, M4A, FLAC). Max 100MB                                               |
| `features`           | String[] | No       | Requested outputs: `raw_transcript`, `clean_transcript`, `summary`, `enhancement_metrics` |
| `template_id`        | String   | No       | Template for summary format (e.g., `meeting_notes_v2`)                                    |
| `asr_backend`        | String   | No       | ASR engine: `whisper` or `chunkformer`                                                    |
| `enable_diarization` | Boolean  | No       | Enable speaker identification (default: true)                                             |
| `enable_vad`         | Boolean  | No       | Enable voice activity detection                                                           |
| `vad_threshold`      | Float    | No       | VAD threshold 0.0-1.0 (default: 0.5)                                                      |

#### Response (202 Accepted)

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "PENDING"
}
```

**Save the `task_id`** - you'll need it to check status and get results!

#### Example Kotlin Code

```kotlin
// Using Retrofit + OkHttp
interface MaieApi {
    @Multipart
    @POST("v1/process")
    suspend fun processAudio(
        @Header("X-API-Key") apiKey: String,
        @Part file: MultipartBody.Part,
        @Part("features") features: List<String>,
        @Part("template_id") templateId: String? = null,
        @Part("enable_diarization") enableDiarization: Boolean = true
    ): ProcessResponse
}

data class ProcessResponse(
    @SerializedName("task_id")
    val taskId: String,
    val status: String
)

// Usage
val audioFile = File("path/to/audio.mp3")
val requestBody = audioFile.asRequestBody("audio/mpeg".toMediaType())
val filePart = MultipartBody.Part.createFormData("file", audioFile.name, requestBody)

val response = maieApi.processAudio(
    apiKey = "your_api_key_here",
    file = filePart,
    features = listOf("clean_transcript", "summary"),
    templateId = "meeting_notes_v2"
)

val taskId = response.taskId
```

#### cURL Example

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@audio.mp3' \
  -F 'features=clean_transcript' \
  -F 'features=summary' \
  -F 'template_id=meeting_notes_v2' \
  -F 'enable_diarization=true'
```

---

## Endpoint 2: Check Task Status & Get Results

### `GET /v1/status/{task_id}`

Retrieve the current status and results of a processing task.

#### Request

**Headers:**

```
X-API-Key: your_api_key_here
```

**Path Parameters:**

- `task_id` (String) - UUID from the `/v1/process` response

#### Response States

The response **changes** as the task progresses:

### State 1: Processing (200 OK)

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "PROCESSING_ASR",
  "submitted_at": "2025-10-20T07:18:17.637Z"
}
```

**Possible Status Values:**

- `PENDING` - Queued, waiting to start
- `PREPROCESSING` - Audio preprocessing
- `PROCESSING_ASR` - Transcription in progress
- `PROCESSING_LLM` - AI analysis in progress
- `COMPLETE` - ✅ Done! Results available
- `FAILED` - ❌ Error occurred

### State 2: Complete (200 OK)

```json
{
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
      "compute_type": "int8_float16"
    },
    "llm": {
      "name": "qwen3",
      "quantization": "awq-4bit"
    }
  },
  "chỉ_số": {
    "thời_lượng_đầu_vào_giây": 2701.3,
    "thời_gian_xử_lý_giây": 162.8,
    "hệ_số_thời_gian_thực": 0.06,
    "độ_phủ_vad": 0.88,
    "độ_tin_cậy_asr_trung_bình": 0.91,
    "tỷ_lệ_chỉnh_sửa_làm_sạch": 0.15
  },
  "kết_quả": {
    "bản_ghi_thô": "The meeting on October 4th covered several important topics...",
    "bản_ghi_sạch": "The meeting on October 4th covered several important topics...",
    "tóm_tắt": {
      "tiêu_đề": "Q4 Budget Planning Meeting",
      "chủ_đề_chính": [
        "Budget approved for Q4 initiatives",
        "New hiring plan discussed",
        "Timeline set for product launch"
      ],
      "thẻ": ["Finance", "Budget", "Planning"]
    }
  }
}
```

**Key Fields in Complete Response:**

| Field                        | Type     | Description                                    |
| ---------------------------- | -------- | ---------------------------------------------- |
| `status`                     | String   | Will be `"COMPLETE"`                           |
| `completed_at`               | ISO 8601 | Timestamp when processing finished             |
| `versions`                   | Object   | Model versions used (for reproducibility)      |
| `metrics`                    | Object   | Performance metrics                            |
| `metrics.rtf`                | Float    | Real-Time Factor (lower is faster)             |
| `metrics.asr_confidence_avg` | Float    | Average transcription confidence (0-1)         |
| `results`                    | Object   | **YOUR ACTUAL DATA**                           |
| `results.clean_transcript`   | String   | Cleaned transcript text                        |
| `results.summary`            | Object   | Structured summary (schema varies by template) |

### State 3: Failed (200 OK)

```json
{
  "task_id": "a1b2c3d4-5e6f-7g8h-9i0j-1k2l3m4n5o6p",
  "status": "FAILED",
  "error": "ASR transcription failed: Audio file not found",
  "error_code": "ASR_PROCESSING_ERROR",
  "stage": "asr",
  "submitted_at": "2025-10-20T07:43:53Z",
  "completed_at": "2025-10-20T07:45:20Z"
}
```

### State 4: Not Found (404 Not Found)

```json
{
  "error": {
    "code": "TASK_NOT_FOUND",
    "message": "Task c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b not found"
  }
}
```

Tasks expire after a configured TTL (check with backend team).

#### Example Kotlin Code

```kotlin
interface MaieApi {
    @GET("v1/status/{task_id}")
    suspend fun getTaskStatus(
        @Header("X-API-Key") apiKey: String,
        @Path("task_id") taskId: String
    ): StatusResponse
}

data class StatusResponse(
    @SerializedName("task_id")
    val taskId: String,
    val status: String,
    @SerializedName("submitted_at")
    val submittedAt: String?,
    @SerializedName("completed_at")
    val completedAt: String?,
    val error: String?,
    @SerializedName("error_code")
    val errorCode: String?,
    val stage: String?,
    val versions: Versions?,
    val metrics: Metrics?,
    val results: Results?
)

data class Versions(
    @SerializedName("pipeline_version")
    val pipelineVersion: String,
    @SerializedName("asr_backend")
    val asrBackend: AsrBackend?,
    val llm: Llm?
)

data class AsrBackend(
    val name: String,
    @SerializedName("model_variant")
    val modelVariant: String?
)

data class Llm(
    val name: String,
    val quantization: String?
)

data class Metrics(
    @SerializedName("input_duration_seconds")
    val inputDurationSeconds: Double?,
    @SerializedName("processing_time_seconds")
    val processingTimeSeconds: Double?,
    val rtf: Double?,
    @SerializedName("asr_confidence_avg")
    val asrConfidenceAvg: Double?
)

data class Results(
    @SerializedName("raw_transcript")
    val rawTranscript: String?,
    @SerializedName("clean_transcript")
    val cleanTranscript: String?,
    val summary: JsonObject? // Use JsonObject or define specific template schemas
)

// Polling implementation
suspend fun pollForResults(taskId: String): StatusResponse {
    repeat(60) { // Poll for up to 60 seconds
        val response = maieApi.getTaskStatus(
            apiKey = "your_api_key_here",
            taskId = taskId
        )

        when (response.status) {
            "COMPLETE" -> return response
            "FAILED" -> throw Exception("Task failed: ${response.error}")
            else -> delay(1000) // Wait 1 second before next check
        }
    }
    throw TimeoutException("Task did not complete within timeout")
}
```

#### cURL Example

```bash
curl -H 'X-API-Key: your_api_key_here' \
  http://localhost:8000/v1/status/c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b
```

---

## Endpoint 2: Get Results

### `GET /v1/results/{task_id}`

Retrieve the results of the audio processing task.

#### Request

**Headers:**

```
X-API-Key: your_api_key_here
```

#### Response (200 OK)

Depending on the requested features and template, the response will vary. Here are examples for different templates:

**1. Structured Analysis Template**

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "COMPLETE",
  "data": {
    "structured_analysis": {
      "summary": "This is a summary of the audio content.",
      "key_points": ["Key point 1", "Key point 2"],
      "detailed_analysis": {
        "speaker_1": {
          "transcript": "Speaker 1's transcript here.",
          "duration": 120
        },
        "speaker_2": {
          "transcript": "Speaker 2's transcript here.",
          "duration": 90
        }
      }
    }
  }
}
```

**2. Clean Transcript Template**

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "COMPLETE",
  "data": {
    "clean_transcript": "This is the cleaned transcript of the audio content."
  }
}
```

**3. Raw Transcript Template**

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "COMPLETE",
  "data": {
    "raw_transcript": "This is the raw transcript of the audio content with all filler words and pauses."
  }
}
```

---

## Key Differences: Submission vs Status Check

| Field          | POST /v1/process | GET /v1/status (Processing) | GET /v1/status (Complete) |
| -------------- | ---------------- | --------------------------- | ------------------------- |
| HTTP Status    | `202 Accepted`   | `200 OK`                    | `200 OK`                  |
| `task_id`      | ✅               | ✅                          | ✅                        |
| `status`       | `"PENDING"`      | `"PROCESSING_*"`            | `"COMPLETE"`              |
| `submitted_at` | ❌               | ✅                          | ✅                        |
| `completed_at` | ❌               | ❌                          | ✅                        |
| `versions`     | ❌               | ❌                          | ✅                        |
| `metrics`      | ❌               | ❌                          | ✅                        |
| **`results`**  | ❌               | ❌                          | **✅ (YOUR DATA!)**       |

---

## Additional Endpoints

### GET /v1/models

List available ASR models and backends (no authentication required).

```bash
curl http://localhost:8000/v1/models
```

**Response:**

```json
{
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
}
```

### GET /v1/templates

List available summary templates (no authentication required).

```bash
curl http://localhost:8000/v1/templates
```

**Response:**

```json
{
  "templates": [
    {
      "id": "meeting_notes_v2",
      "name": "Meeting Notes v2",
      "description": "Structured format for meeting transcripts",
      "schema_url": "/v1/templates/meeting_notes_v2/schema"
    },
    {
      "id": "interview_transcript_v2",
      "name": "Interview Transcript v2",
      "description": "Format for interview recordings",
      "schema_url": "/v1/templates/interview_transcript_v2/schema"
    }
  ]
}
```

### GET /health

Basic health check (no authentication required).

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy"
}
```

---

## Complete Android Implementation Example

```kotlin
class MaieRepository(
    private val api: MaieApi,
    private val apiKey: String
) {
    suspend fun processAudioFile(
        file: File,
        features: List<String> = listOf("clean_transcript", "summary"),
        templateId: String = "meeting_notes_v2",
        enableDiarization: Boolean = true
    ): Flow<ProcessingState> = flow {
        // Step 1: Submit audio
        emit(ProcessingState.Uploading)

        val requestBody = file.asRequestBody("audio/*".toMediaType())
        val filePart = MultipartBody.Part.createFormData("file", file.name, requestBody)

        val submitResponse = api.processAudio(
            apiKey = apiKey,
            file = filePart,
            features = features,
            templateId = templateId,
            enableDiarization = enableDiarization
        )

        val taskId = submitResponse.taskId
        emit(ProcessingState.Submitted(taskId))

        // Step 2: Poll for completion
        var attempts = 0
        val maxAttempts = 120 // 2 minutes with 1-second intervals

        while (attempts < maxAttempts) {
            delay(1000) // Wait 1 second
            attempts++

            val statusResponse = api.getTaskStatus(apiKey, taskId)

            when (statusResponse.status) {
                "PENDING" -> emit(ProcessingState.Queued)
                "PREPROCESSING" -> emit(ProcessingState.Preprocessing)
                "PROCESSING_ASR" -> emit(ProcessingState.Transcribing)
                "PROCESSING_LLM" -> emit(ProcessingState.Analyzing)
                "COMPLETE" -> {
                    emit(ProcessingState.Complete(statusResponse))
                    return@flow
                }
                "FAILED" -> {
                    emit(ProcessingState.Failed(
                        statusResponse.error ?: "Unknown error",
                        statusResponse.errorCode
                    ))
                    return@flow
                }
            }
        }

        emit(ProcessingState.Failed("Timeout waiting for results", "TIMEOUT"))
    }
}

sealed class ProcessingState {
    object Uploading : ProcessingState()
    data class Submitted(val taskId: String) : ProcessingState()
    object Queued : ProcessingState()
    object Preprocessing : ProcessingState()
    object Transcribing : ProcessingState()
    object Analyzing : ProcessingState()
    data class Complete(val response: StatusResponse) : ProcessingState()
    data class Failed(val error: String, val errorCode: String?) : ProcessingState()
}

// Usage in ViewModel
class AudioProcessingViewModel(
    private val repository: MaieRepository
) : ViewModel() {

    fun processAudio(file: File) {
        viewModelScope.launch {
            repository.processAudioFile(file)
                .catch { e ->
                    _uiState.value = UiState.Error(e.message ?: "Unknown error")
                }
                .collect { state ->
                    when (state) {
                        is ProcessingState.Complete -> {
                            // Extract results
                            val transcript = state.response.results?.cleanTranscript
                            val summary = state.response.results?.summary
                            _uiState.value = UiState.Success(transcript, summary)
                        }
                        is ProcessingState.Failed -> {
                            _uiState.value = UiState.Error(state.error)
                        }
                        else -> {
                            _uiState.value = UiState.Processing(state)
                        }
                    }
                }
        }
    }
}
```

---

## Error Handling

### Common HTTP Status Codes

| Code  | Description            | Action                                        |
| ----- | ---------------------- | --------------------------------------------- |
| `202` | Accepted               | Task submitted successfully, poll for results |
| `200` | OK                     | Status check successful                       |
| `400` | Bad Request            | Check request parameters                      |
| `401` | Unauthorized           | Invalid API key                               |
| `404` | Not Found              | Task expired or doesn't exist                 |
| `413` | Payload Too Large      | Audio file exceeds size limit                 |
| `415` | Unsupported Media Type | Invalid audio format                          |
| `422` | Validation Error       | Missing required fields                       |
| `429` | Too Many Requests      | Rate limit exceeded, retry later              |
| `500` | Internal Server Error  | Server error, contact backend team            |

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "additional": "context"
    }
  }
}
```

---

## Best Practices

1. **Always use HTTP, not HTTPS** (unless your backend team configures SSL)
2. **Store API key securely** (use Android Keystore, not hardcoded)
3. **Implement exponential backoff** for polling to reduce server load
4. **Show progress indicators** based on status values
5. **Handle network errors gracefully** with retry logic
6. **Validate file size** before upload (check max 100MB limit)
7. **Cache results** if needed (tasks expire after TTL)
8. **Add request timeouts** (recommended: 30s for upload, 120s for processing)

---

## Testing Script

A test script is available at `scripts/test-api.sh`:

```bash
./scripts/test-api.sh
```

This will test all endpoints and show you example responses.

---

## Need Help?

- **Full API Documentation:** See `docs/API_REFERENCE.md`
- **E2E Testing Guide:** See `docs/E2E_TESTING_HYBRID_QUICKSTART.md`
- **Client Developer Guide:** See `docs/CLIENT_DEVELOPER_GUIDE.md`
- **Contact Backend Team:** For API keys, server URLs, and configuration

---

## Quick Reference

```kotlin
// 1. Submit audio
val response = api.processAudio(apiKey, file, features, templateId)
val taskId = response.taskId

// 2. Poll for completion
while (true) {
    val status = api.getTaskStatus(apiKey, taskId)
    when (status.status) {
        "COMPLETE" -> {
            val transcript = status.results?.cleanTranscript
            val summary = status.results?.summary
            break
        }
        "FAILED" -> throw Exception(status.error)
        else -> delay(1000)
    }
}
```

**Remember:** `http://` not `https://` ⚠️

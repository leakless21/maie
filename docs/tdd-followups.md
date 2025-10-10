# MAIE API Follow-Up Tasks

This note tracks temporary shims and placeholders introduced while getting the API
walking skeleton and tests online. Revisit each item as the full implementation
matures.

**Last Updated:** October 9, 2025  
**Status:** Config and API best practices implemented with 104/108 tests passing

---

## 🔴 High Priority - Blocking Worker Integration

### **Replace Redis/Queue Scaffolding**

- **Location:** `src/api/routes.py`
- **Current State:**
  - ✅ `check_queue_depth()` - Returns `True` (no-op placeholder)
  - ✅ `create_task_in_redis()` - Empty function, needs Redis client integration
  - ✅ `enqueue_job()` - Empty function, needs RQ job enqueueing
  - ✅ `get_task_from_redis()` - Returns `None`, needs Redis retrieval logic
- **Action Items:**
  1. Inject Redis client via dependency injection (Litestar `Provide`)
  2. Implement actual queue depth check against `MAX_QUEUE_DEPTH` setting
  3. Create task record in Redis DB 1 with `PENDING` status
  4. Enqueue RQ job to Redis DB 0 with proper timeout and retry settings
  5. Implement task retrieval with proper serialization/deserialization
- **Dependencies:** Redis connection setup, RQ worker implementation
- **Tests:** All scaffolding is fully tested with mocks - just swap implementations

### **File Storage Integration**

- **Location:** `src/api/routes.py::save_audio_file()`
- **Current State:** ✅ Implemented synchronously, saves to `settings.audio_dir`
- **Improvements Needed:**
  1. Consider async I/O for large file writes (use `aiofiles`)
  2. Add file integrity validation (magic number checking with `python-magic`)
  3. Implement cleanup policy (delete on failure, retention period)
  4. Add disk space monitoring before save
- **Priority:** Medium (works but could be optimized)

---

## 🟡 Medium Priority - Production Readiness

### **ASR Model Discovery**

- **Location:** `src/api/routes.py::get_available_models()`
- **Current State:** Returns hard-coded Whisper backend
- **Action Items:**
  1. Import and use `ASRFactory.list_available()` from `src/processors/asr/factory.py`
  2. Query actual model variants from `settings.models_dir`
  3. Return dynamic model metadata (VRAM requirements, capabilities, variants)
  4. Add model health checks (verify model files exist and are loadable)
- **Reference:** See TDD Section 3.4 for ASR Factory Pattern details

**V1.0 ASR Implementation Scope:**

Per V1.0 scope clarification (see `docs/V1.0_SCOPE_CLARIFICATION.md`), ASR implementation should focus on:

✅ **Included in V1.0:**

- Basic segment-level transcription (text + start/end timestamps)
- VAD filtering support with custom parameters
- Language detection
- Segment-level confidence tracking
- Sequential load → execute → unload pattern
- Distil-Whisper model variant support

❌ **Deferred to Post-V1.0:**

- Word-level timestamps → V1.1+ (not in PRD metrics; for timeline/subtitle features)
- Batched inference pipeline → V1.2+ (contradicts sequential architecture)
- Speaker diarization → V1.1+ (requires pyannote.audio integration)
- Streaming transcription → V1.3+ (requires WebSocket API)
- Model preloading → V1.2+ (explicitly deferred in TDD Section 3.2)

**Implementation Notes:**

- `WhisperBackend.execute()` should NOT enable `word_timestamps=True` in V1.0
- `WhisperBackend` should NOT implement `execute_batched()` method in V1.0
- Focus on stability and sequential processing: Load → Process → Unload → Clear VRAM
- See `docs/guide.md` "Future Enhancements (Post-V1.0 Roadmap)" section for detailed feature roadmap

### **Template Discovery Enhancement**

- **Location:** `src/api/routes.py::scan_templates_directory()`
- **Current State:** ✅ Scans filesystem, returns basic metadata
- **Improvements:**
  1. Cache template list to avoid repeated filesystem scans
  2. Validate JSON Schema compliance on startup
  3. Extract and return `required_fields` and `has_tags` metadata
  4. Add template versioning support (e.g., `meeting_notes_v1`, `meeting_notes_v2`)
- **Priority:** Low-Medium (works but could be more robust)

### **Authentication & Security**

- **Location:** Currently not implemented
- **Action Items:**
  1. Wire up `api_key_auth` dependency from `src/api/dependencies.py`
  2. Add to all controllers via dependency injection
  3. Implement rate limiting per API key
  4. Add request/response logging with sanitization
- **Reference:** TDD Section 7.1 for authentication patterns

---

## 🟢 Low Priority - Enhancements

### **Response Schema Refinements**

- **Location:** `src/api/schemas.py::StatusResponseSchema`
- **Current State:**
  - ✅ Added `error` field for failed tasks
  - ✅ Set `extra='allow'` for flexibility
- **Future Considerations:**
  1. Decide if `error` should be structured (error_code + message + stage)
  2. Consider moving to PRD Appendix B canonical schema format
  3. Add OpenAPI examples for each status state
- **Note:** Current implementation aligns with PRD requirements

### **Validation Hardening**

- **Location:** `src/api/routes.py::process_audio()`
- **Current State:** Basic validation implemented
- **Enhancements:**
  1. Reject malformed JSON in `features` field with detailed error
  2. Add content-type validation using `python-magic` (beyond extension check)
  3. Implement audio quality checks (sample rate, duration, silence detection)
  4. Add structured error responses with error codes from TDD Section 7.6
- **Priority:** Low (sufficient for V1.0)

### **Observability & Monitoring**

- **Action Items:**
  1. Add structured logging (loguru) to all route handlers
  2. Implement OpenTelemetry spans for distributed tracing
  3. Add Prometheus metrics for request counts, latencies, error rates
  4. Log task_id, features, file_size, processing_time for analytics
- **Reference:** TDD Section 8 for monitoring strategy

### **Testing Enhancements**

- **Current State:** ✅ 16/16 tests passing with comprehensive coverage
- **Future Tests:**
  1. Integration tests with real Redis instance
  2. E2E tests with actual audio files and processing pipeline
  3. Load testing for backpressure and queue management
  4. Performance benchmarks for file upload handling
- **Reference:** TDD Section 6 for full testing strategy

---

## 📝 Documentation Tasks

1. **API Documentation**

   - Generate OpenAPI spec and publish to `/schema/swagger`
   - Add request/response examples for each endpoint
   - Document error codes and troubleshooting guide

2. **Developer Guide**
   - Add "Getting Started" section for running locally
   - Document environment variables and configuration
   - Add troubleshooting section for common issues

---

## ✅ Recently Completed Items (October 9, 2025)

### **Config Module Best Practices** ✅

- **Location:** `src/config.py`
- **Completed:**
  - ✅ Added `validate_default=True` to SettingsConfigDict for early error detection
  - ✅ Added `env_nested_delimiter="__"` for hierarchical configuration support
  - ✅ Added comprehensive tests (4 new tests in `TestPydanticSettingsBestPractices`)
- **Tests:** 38/38 config tests passing

### **API Dependencies Security Verification** ✅

- **Location:** `src/api/dependencies.py`
- **Verified Already Correct:**
  - ✅ Timing-safe API key comparison with `hmac.compare_digest()` (prevents timing attacks)
  - ✅ Redis connection pooling with proper `encoding="utf-8"` and timeouts
  - ✅ Litestar guards use correct signature `(ASGIConnection, BaseRouteHandler) -> None`
  - ✅ Separate sync Redis client for RQ operations
- **Tests:** Added 17 new tests for security verification

### **API Routes Security Improvements** ✅

- **Location:** `src/api/routes.py`
- **Completed:**
  - ✅ Added `sanitize_filename()` function to prevent path traversal attacks
  - ✅ Updated `save_audio_file()` to use UUID-based filenames (not user input)
  - ✅ Enhanced MIME type validation (checks both extension AND content-type)
  - ✅ Improved file size validation with security comments
- **Tests:** Added 7 security tests (4 passing, 3 skipped for future streaming)

### **Best Practices Documentation** ✅

- **Files:** `docs/implementation-changes.md`, `docs/guide.md`, `docs/PRD.md`
- **Completed:**
  - ✅ Created comprehensive implementation changes document
  - ✅ Updated guide.md with best practices section
  - ✅ Added NFR-6 security requirements to PRD.md
- **Coverage:** All changes documented with code examples and test references

---

## 🔄 Remaining Best Practices TODOs

### **File Streaming Implementation** (Future Enhancement)

- **Location:** `src/api/routes.py::save_audio_file()`
- **Status:** Skipped for now (marked as TODO in code)
- **Action Items:**
  1. Implement `aiofiles` streaming with 8KB chunks to prevent memory exhaustion
  2. Add Content-Length header validation before reading
  3. Update tests to verify chunk-based streaming
- **Priority:** Medium (current implementation works but not optimal for large files)

### **Lifespan Hooks for Redis Pools** (Future Enhancement)

- **Location:** `src/api/main.py`
- **Status:** Not implemented yet
- **Action Items:**
  1. Add `init_redis_pools()` and `close_redis_pools()` lifespan hooks
  2. Implement proper Redis connection pool lifecycle management
  3. Add tests for lifespan hook execution
- **Priority:** Medium (current dependency injection works but pools recreated per request)

### **Structured Logging Enhancement** (Future Enhancement)

- **Location:** `src/api/main.py` exception handlers
- **Status:** Basic exception handlers implemented
- **Action Items:**
  1. Add request context (IP, user agent, request ID) to logs
  2. Implement debug vs production error detail levels
  3. Add structured logging with JSON format for production
- **Priority:** Low-Medium (current handlers work but could be more informative)

---

**Next Immediate Steps:**

1. Implement Redis client integration and task persistence
2. Wire up RQ job enqueueing
3. Add authentication guards to all endpoints
4. Integrate ASRFactory for dynamic model discovery
5. Consider implementing file streaming with `aiofiles` for production readiness

Keep this file updated as implementation progresses. Delete completed sections
to maintain clarity.

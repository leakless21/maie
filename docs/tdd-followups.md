# MAIE API Follow-Up Tasks

This note tracks temporary shims and placeholders introduced while getting the API
walking skeleton and tests online. Revisit each item as the full implementation
matures.

**Last Updated:** October 8, 2025  
**Status:** Routes implementation complete with 16/16 tests passing

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

## ✅ Completed Items

- ✅ Implemented all 4 route controllers (Process, Status, Models, Templates)
- ✅ Comprehensive test suite (16 tests, 100% pass rate)
- ✅ File upload validation (type, size, format)
- ✅ Request validation (features, template_id requirements)
- ✅ Backpressure placeholder (429 handling)
- ✅ Error field in StatusResponseSchema
- ✅ Template directory scanning
- ✅ Basic models endpoint
- ✅ Type-safe implementations with Pydantic schemas

---

**Next Immediate Steps:**

1. Implement Redis client integration and task persistence
2. Wire up RQ job enqueueing
3. Add authentication guards to all endpoints
4. Integrate ASRFactory for dynamic model discovery

Keep this file updated as implementation progresses. Delete completed sections
to maintain clarity.

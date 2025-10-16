## MAIE Best Practices

### Environment and Configuration
- **Prefer explicit env for runtime, sanitized env for tests**: Keep `.env` for local runs, but ensure tests start with a minimal environment. Clear or override variables like `SECRET_API_KEY`, `LLM_*`, `WHISPER_LANGUAGE`, `TOP_P`, `GPU_MEMORY_UTILIZATION` during tests to validate defaults.
- **Use Pydantic settings consistently**: All configuration should flow through `src/config/settings.py`. Avoid reading raw `os.environ` in business logic unless absolutely necessary; when needed, prefer env → Settings fallback patterns consistently.
- **Immutable defaults in code**: Choose sensible defaults in code (e.g., `secret_api_key` placeholder, `None` for sampling params) and document overrides clearly in `.env` template.

### Model and Cache Management
- **Pin model locations for offline**: Prefer local model directories under `data/models` for deterministic/offline runs. When using HF repos, prefetch via scripts (`scripts/download-models.sh`) and reference local paths.
- **Hugging Face cache isolation**: Set `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to writable, project-local paths (or tmp dirs in tests). This avoids permission issues and cross-project cache coupling.
- **No implicit downloads in production**: For production/offline, enforce `local_files_only=True` (as in Whisper backend) and fail fast with actionable errors if weights are missing.

### ASR (Whisper, ChunkFormer)
- **Device selection**: Default to GPU (`cuda`) where supported; validate CUDA availability early with clear errors. Provide an explicit CPU mode only where acceptable for functionality/perf.
- **Deterministic kwargs**: Build transcription parameters from settings with user overrides taking precedence. Keep language default `None` (auto) unless an explicit deployment requires a fixed language.
- **Resource cleanup**: Ensure models expose `unload()` paths and are released in context managers and pipeline teardown.

### LLM (vLLM)
- **Sampling parameter handling**: Centralize generation configuration (env defaults, runtime overrides) and convert to vLLM `SamplingParams` in one place. Expose helpers (e.g., `apply_overrides_to_sampling`) at import paths tests and callers expect.
- **Quantization detection**: Detect quantization from model names where possible and allow explicit overrides; document supported values.
- **Version info**: Provide `get_version_info()` with stable keys (`model_name`, `checkpoint_hash`, `backend`) to support auditability and tests.

### Version Metadata
- **Stable schema**: Return top-level keys `asr`, `llm`, `processing_pipeline`, `maie_worker`. Preserve sub-metadata exactly; avoid renaming/normalizing within version aggregation.
- **Sanitization**: Only strip non-serializable values or sensitive data; do not drop informative fields.

### Logging and Observability
- **Structured logs**: Prefer structured logs with consistent keys; allow JSON serialization in containers. Keep log levels configurable via settings.
- **Actionable errors**: Include remediation hints in exceptions (e.g., missing model path, CUDA unavailable, permission issues).

### Testing
- **Fixture-driven env control**: Use session-scoped, autouse fixtures to set HF cache to a tmp dir and sanitize env before imports. Avoid relying on developer machine env.
- **Unit vs integration boundaries**: Mock heavy dependencies (vLLM, CUDA, HF) in unit tests; let integration tests validate real paths/devices behind flags or markers.
- **Determinism**: Avoid tests that depend on external network or mutable global caches. If network is required, mark and skip by default.

### CI/CD
- **Reproducible runners**: Use containers or pinned runners with CUDA libs installed for GPU jobs. Cache models in CI where licensing allows.
- **Fail fast**: Separate fast unit tests from heavier integration/E2E; run unit tests on every PR, gate heavy suites behind labels or nightly jobs.

### Code Quality
- **Readable, explicit code**: Prefer descriptive names and clear control flow; avoid unnecessary try/except that hides errors.
- **Minimal comments, maximal clarity**: Comment only non-obvious rationale and invariants. Keep formatting consistent and avoid unrelated reformatting in edits.

### Security
- **Secret handling**: Never commit real secrets. Provide placeholders in templates and require overrides in deployment. Validate that defaults are clearly non-production.
- **Dependency vigilance**: Pin critical AI/runtime libs; track GPU library compatibility (e.g., cuDNN, cuBLAS versions) and document required env.

### Operational Tips
- **Warmup and health checks**: Optionally add warmup steps to preload models in long-running workers and expose health endpoints that include version metadata.
- **Backpressure and concurrency**: Configure worker concurrency, prefetch, and job timeouts based on GPU memory and model sizes; document recommended settings per SKU.



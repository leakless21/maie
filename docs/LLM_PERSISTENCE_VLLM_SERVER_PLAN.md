# LLM Persistence & vLLM Server Integration Plan

This document describes how to evolve MAIE's current in‑process vLLM integration
into a persistent, vLLM‑server–backed architecture, while keeping the existing
prompt templates and schema validation model.

The goal is:

- keep the LLM model loaded once in a long‑running vLLM server process
- have the MAIE worker pipeline call that server instead of constructing
  `vllm.LLM` for every job
- preserve or improve the current system‑prompt‑per‑request behaviour and
  structured JSON output

---

## 1. Current Architecture (LLM Path)

### 1.1 Components

- `src/processors/llm/processor.py:LLMProcessor`
  - Lazily constructs an in‑process `vllm.LLM` instance in `_load_model()`.
  - Builds prompts using Jinja templates via `PromptRenderer` and
    `TemplateLoader`.
  - For `task="summary"`:
    - Loads a JSON schema with `load_template_schema(template_id, templates_dir)`.
    - Renders a **system prompt** (instructions + schema) using
      `PromptRenderer.render(template_id, schema=...)`.
    - Builds OpenAI‑format `messages = [{"role": "system", ...}, {"role": "user", ...}]`.
    - Optionally sets `GuidedDecodingParams(json=...)` for vLLM's guided JSON decoding.
    - Calls `self.model.chat(messages, sampling_params=...)`.
    - Validates the JSON via `validate_llm_output` and post‑processes it.
  - For `task="enhancement"`:
    - Renders `text_enhancement_v1` Jinja template as a system prompt.
    - Builds analogous `[system, user]` messages and calls `model.chat`.
  - Keeps `self.model` and `self.tokenizer` cached per `LLMProcessor` instance
    (`_model_loaded` flag).

- `src/worker/pipeline.py:process_audio_task`
  - Performs ASR, VAD, diarization, then LLM enhancement/summary in **Stage 3**.
  - LLM usage pattern:
    - Creates a **temporary** `LLMProcessor` (`temp_llm`) just to call
      `needs_enhancement(asr_backend)` (if available).
    - If LLM is needed:
      - Ensures CUDA and clears GPU cache (`torch.cuda.empty_cache()`).
      - Instantiates a fresh `LLMProcessor()` as `llm_model`.
      - Calls `llm_model._load_model()` to initialize vLLM in‑process.
      - Uses `llm_model.enhance_text` and/or summary functionality.
      - Calls `get_version_metadata(asr_metadata, llm_model)` while model is
        still loaded.
    - In the `finally` block:
      - Calls `llm_model.unload()` (which deletes `self.model` and clears CUDA
        cache).
      - Clears GPU memory again via `torch.cuda.empty_cache()`.

- `src/tooling/vllm_utils.py`
  - Helper utilities for vLLM sampling params and checkpoint hashing.
  - Not currently aware of any HTTP / server‑mode calling.

### 1.2 Prompt / System Prompt Flow

- The **API client** never sends a system prompt; it only sends:
  - audio file
  - `features` list (`clean_transcript`, `summary`, etc.)
  - `template_id` for summary
- The **worker** calls `LLMProcessor`, which:
  - loads template + schema
  - renders **system prompts** on each call using Jinja
  - constructs `[{"role": "system", ...}, {"role": "user", ...}]`
  - passes those messages into `vllm.LLM.chat()`

So from MAIE's perspective, "sending the system prompt every request" means:
we rebuild the system prompt via Jinja and send it to the model for every
enhancement/summary run. This is currently done inside the worker process,
not over HTTP.

---

## 2. Target Architecture (vLLM Server + Persistence)

### 2.1 High‑Level Design

- Run a dedicated **vLLM server process** (OpenAI‑compatible API) with:
  - model loaded once at server startup
  - long‑lived GPU memory context
  - batching / KV‑cache optimizations handled by vLLM itself
- Treat MAIE as a **client** of that vLLM server:
  - MAIE worker no longer constructs `vllm.LLM` or manages CUDA for LLM.
  - All LLM calls go through a thin HTTP client layer.
  - The worker remains responsible for:
    - rendering system prompts via Jinja
    - building OpenAI‑style messages
    - performing JSON schema validation and post‑processing.

Key properties:

- **Persistence**: model weights stay loaded in vLLM server across all MAIE
  requests and even across worker restarts.
- **Isolation**: ASR / diarization GPU lifecycle can be managed independently
  from LLM GPU usage.
- **Configurable backend**: we keep the existing in‑process vLLM path as a
  fallback/backend option.

### 2.2 vLLM Server Setup (Operational)

We will assume use of vLLM's OpenAI‑compatible API server
(`vllm.entrypoints.openai.api_server:app`), e.g.:

```bash
vllm serve \
  cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 32768
```

Key operational decisions (to be reflected in docs / env):

- Env vars (examples):
  - `APP_LLM_BACKEND=vllm_server`
  - `APP_LLM_SERVER__BASE_URL=http://vllm:8001/v1`
  - `APP_LLM_SERVER__API_KEY=...` (optional, if vLLM server enforces keys).
- For local dev, the server can be:
  - run manually via `vllm serve ...`
  - or managed via `docker-compose` as a service (future enhancement).

---

## 3. Config & Abstraction Changes

### 3.1 Add LLM Backend Mode to Settings

Extend `AppSettings` with a logical LLM backend selector and server settings,
while keeping existing `llm_enhance` and `llm_sum` fields unchanged:

- New config model (conceptual):

```python
class LlmBackendType(str, Enum):
    LOCAL_VLLM = "local_vllm"
    VLLM_SERVER = "vllm_server"


class LlmServerSettings(BaseModel):
    base_url: AnyHttpUrl = Field(default="http://localhost:8001/v1")
    api_key: SecretStr | None = None
    model_enhance: str | None = None  # override if different from llm_enhance.model
    model_summary: str | None = None  # override if different from llm_sum.model
    request_timeout_seconds: float = 60.0
```

- Add to `AppSettings`:

```python
llm_backend: LlmBackendType = Field(default=LlmBackendType.LOCAL_VLLM)
llm_server: LlmServerSettings = Field(default_factory=LlmServerSettings)
```

- Expose env overrides, e.g.:
  - `APP_LLM_BACKEND=vllm_server`
  - `APP_LLM_SERVER__BASE_URL=...`
  - `APP_LLM_SERVER__MODEL_ENHANCE=...`

### 3.2 Introduce a Thin LLM Client Abstraction

Create a small interface to decouple `LLMProcessor` from the concrete backend:

- New module (conceptual): `src/tooling/llm_client.py`

Interface:

```python
class ChatCompletionClient(Protocol):
    def chat(self, *, messages: list[dict[str, str]], **params: Any) -> Any: ...
```

Implementations:

- `LocalVllmClient`
  - Wraps existing `self.model.chat()` / `self.model.generate()`.
  - Uses `vllm.SamplingParams` and `GuidedDecodingParams` the same way
    `LLMProcessor` does today.
- `VllmServerClient`
  - Uses an HTTP client (e.g. `httpx` or `openai` with `base_url` override) to
    call vLLM's OpenAI‑compatible `/v1/chat/completions` endpoint.
  - Accepts:
    - `messages` (OpenAI format)
    - `model` (from `settings.llm_server.model_enhance / model_summary`
      or fallback)
    - decoding params (temperature, top_p, top_k, max_tokens, stop, etc.).
  - Translates responses back to a "vLLM‑like" shape that `LLMProcessor` can
    interpret (text, token counts, finish_reason).

`LLMProcessor` will:

- construct the appropriate client based on `settings.llm_backend`
- delegate the actual generation/chat call to that client.

---

## 4. LLMProcessor Changes for Server Mode

### 4.1 Backend Selection & Lifecycle

Currently:

- `_load_model()`:
  - imports `vllm.LLM`
  - constructs `self.model = LLM(**llm_args)`
  - sets `_model_loaded = True`
  - calculates checkpoint hash and `model_info`.

Planned behaviour:

- When `settings.llm_backend == LOCAL_VLLM`:
  - Keep `_load_model()` exactly as-is (subject to small refactors for reuse).
  - `LLMProcessor` still holds `self.model` and uses local GPU.

- When `settings.llm_backend == VLLM_SERVER`:
  - `_load_model()`:
    - Do **not** construct `vllm.LLM`.
    - Instead:
      - set `self.model` to a lightweight proxy object or `None`.
      - initialize a `VllmServerClient` with `base_url`, `api_key`, and
        chosen model name.
      - set `self.checkpoint_hash` / `model_info` based on configuration
        (e.g. `hf:model_name`), or optionally by querying
        `GET /v1/models` once and caching the result.
    - Mark `_model_loaded = True` so the rest of the code path skips local
      model construction.

### 4.2 Execute Path: Chat vs Generate

`LLMProcessor.execute()` already builds:

- `messages` (with system prompt and user content), or
- `final_prompt` for non‑chat tasks.

In the vLLM server mode we will:

- Prefer the **chat** path for both `summary` and `enhancement` (matching
  current design).
- For `summary`:
  - Keep Jinja‑rendered system prompt (schema + instructions).
  - Keep user message with transcript.
  - Pass `messages` and decoding params to `VllmServerClient.chat()`.
  - The server returns a standard OpenAI chat completion; extract the first
    choice's content as `generated_text`.
  - Retain local JSON schema validation via `validate_llm_output`.
- For `enhancement`:
  - Same chat flow using `text_enhancement_v1` system prompt.

Guided decoding:

- Local vLLM uses `GuidedDecodingParams(json=schema_json)`.
- vLLM OpenAI server supports JSON‑structured outputs via its API; for
  compatibility we can:
  - Phase 1 (minimal): rely purely on system prompt + schema text and keep the
    existing JSON validation, **without** guided decoding.
  - Phase 2 (optional): use vLLM server's structured‑output features (e.g.
    `response_format={"type": "json_schema", ...}` or related options) to
    improve robustness, then adjust tests accordingly.

### 4.3 Token / Metadata Handling

The current code collects:

- `tokens_used` from `outputs[0].prompt_token_ids`
- `generated_tokens` from `first_output.token_ids`
- `finish_reason` and `output_length` for metrics.

For the server mode:

- Map OpenAI response fields to the same metadata keys where reasonable:
  - `prompt_tokens` and `completion_tokens` from `usage` (if available).
  - `finish_reason` from `choices[0].finish_reason`.
  - `generated_tokens` approximated by `usage.completion_tokens` (or left
    `None` if absent).
- Keep `result_metadata` shape stable so existing logging and metrics tests
  require minimal changes.

### 4.4 Unload Semantics

`LLMProcessor.unload()` currently:

- deletes the in‑process model (`del self.model`)
- clears CUDA cache
- resets `tokenizer`, `current_template_id`, `current_schema_hash`.

In server mode:

- `unload()` becomes effectively a **no‑op** with regard to GPU:
  - Do not touch `torch.cuda.*`.
  - Optionally clear cached tokenizer or local HTTP session.

This will require the worker pipeline to be aware of backend mode to avoid
unnecessary CUDA cache clears around LLM when using a remote server.

---

## 5. Worker Pipeline Adjustments

### 5.1 Removing Per‑Job Model Load/Unload in Server Mode

In `process_audio_task` (Stage 3: PROCESSING_LLM):

Current flow:

- Always:
  - instantiate `LLMProcessor` for `needs_enhancement(...)` check.
- If `wants_llm`:
  - clear GPU cache
  - create `llm_model = LLMProcessor()`
  - call `llm_model._load_model()`
  - run enhancement / summary
  - in `finally`: `llm_model.unload()` and `torch.cuda.empty_cache()`.

Planned behaviour:

- Keep the **decision logic** (`needs_enhancement`, `wants_llm`) unchanged.
- When `settings.llm_backend == LOCAL_VLLM`:
  - Preserve current behaviour (load local vLLM, clear CUDA, unload at end).
- When `settings.llm_backend == VLLM_SERVER`:
  - Skip GPU cache manipulation for LLM:
    - Do **not** call `torch.cuda.empty_cache()` before/after LLM stage.
  - Still instantiate `LLMProcessor()`, but:
    - `_load_model()` will just create a remote client and mark itself
      initialized.
    - `unload()` should be no‑op (except for cleaning local state).
  - This means each job still constructs an `LLMProcessor` instance, but the
    heavy state (model weights) lives in the separate vLLM server process.

Optional later optimization:

- Introduce a **singleton** `LLMProcessor` or `ChatCompletionClient` per worker
  process (in `src/worker/main.py`) to reuse HTTP session and any client‑side
  caches. This is not required for correctness and can be added after the
  basic server mode works.

### 5.2 Version Metadata

`get_version_metadata(asr_metadata, llm_model)` currently expects `llm_model`
to expose `get_version_info()`, which `LLMProcessor` implements using
`self.model_info` and `self.checkpoint_hash`.

In server mode:

- `LLMProcessor.get_version_info()` should continue to return:

```python
{
    "name": settings.llm_enhance.model (or server model name),
    "checkpoint_hash": "...",
    "quantization": "awq-4bit" or similar,
    ...
}
```

- For the initial iteration, use configured values:
  - `model_name` from `llm_server.model_*` or `llm_enhance.model` /
    `llm_sum.model`.
  - `checkpoint_hash` from:
    - config (`hf:model_name`) or a static tag (e.g. `remote:<model_name>`),
      or
    - an optional server metadata endpoint if we wire it up later.

This keeps the version metadata schema stable for API consumers and tests.

---

## 6. Prompt / System Prompt Strategy with vLLM Server

### 6.1 Keep Existing Behaviour (Recommended Baseline)

Short term, we preserve the current pattern:

- For every LLM call:
  - Render Jinja system prompt.
  - Build `[system, user]` messages.
  - Send full messages to vLLM server.

Rationale:

- vLLM server already implements KV caching internally; repeatedly sending the
  same system prompt for different requests is acceptable, especially when
  jobs vary in content and schema.
- MAIE keeps prompt engineering centralized in the worker (templates +
  schemas).

### 6.2 Optional Optimization: Prompt Profiles / Prefix Caching

If we want to reduce system‑prompt overhead further:

- Define **prompt profiles** keyed by template/schema combination, e.g.:

```text
profile_id = hash(template_id + schema_hash)
```

- vLLM already implements automatic prefix/KV caching when different requests
  share a common token prefix. Using identical system prompts across requests
  allows the engine to reuse that prefix internally; no explicit cache ID or
  HTTP parameter is required.
- A `profile_id` on the MAIE side would therefore be an **internal** grouping
  (for logging/metrics or future engine‑level extensions), not something that
  needs to be sent to the vLLM server in the current OpenAI‑compatible API.

This is an advanced optimization and should be considered phase 2+ once basic
server integration is stable.

---

## 7. Testing Strategy

### 7.1 Unit Tests

- Configuration:
  - Tests for new `LlmBackendType`, `LlmServerSettings`, and env overrides.
- LLMProcessor:
  - Tests that when `llm_backend == LOCAL_VLLM`, behaviour matches current
    expectations (backwards compatibility).
  - Tests that when `llm_backend == VLLM_SERVER`:
    - `_load_model()` does **not** attempt to import or construct `vllm.LLM`.
    - `execute()` calls the `VllmServerClient` with correctly shaped
      `messages` and parameters.
    - `unload()` is a no‑op for CUDA but clears internal state.
- Client abstraction:
  - Tests that `VllmServerClient` correctly maps OpenAI responses to the
    simplified metadata the processor expects (using mocked HTTP responses).

### 7.2 Integration Tests

- API / worker pipeline:
  - Use `settings.llm_backend = VLLM_SERVER` with a **mocked** HTTP client
    (no real vLLM server) to validate end‑to‑end flow:
    - `/v1/process` request with `summary` feature.
    - Worker job executes LLM stage and returns summary structure that passes
      schema validation.
  - Verify:
    - No CUDA clear calls are made in server mode (can be asserted indirectly
      by patching `torch.cuda.empty_cache`).
    - Version metadata includes an LLM block even in server mode.

### 7.3 Optional Real‑Server Tests

- Extend `tests/integration/test_real_llm_integration.py` to support a
  `vLLM_SERVER` mode guarded by marks/env vars, e.g.:
  - `LLM_TEST_BACKEND=vllm_server`
  - `LLM_TEST_SERVER_BASE_URL=...`

These would run only when a vLLM server is available and not in the default
CI path.

---

## 8. Rollout Plan

1. **Config & Abstraction**
   - Add `LlmBackendType`, `LlmServerSettings`, and `llm_backend` to
     `AppSettings`.
   - Implement `ChatCompletionClient` interface and `LocalVllmClient`.
   - Ensure existing tests pass (local vLLM still default).

2. **VLLM Server Client**
   - Implement `VllmServerClient` with HTTP client.
   - Wire `LLMProcessor` to select backend based on `settings.llm_backend`.
   - Add unit tests for server mode, mocking HTTP.

3. **Worker Pipeline Adjustments**
   - Make Stage 3 aware of backend mode; skip CUDA clear calls in
     `VLLM_SERVER` mode.
   - Adjust `LLMProcessor.unload()` semantics accordingly.
   - Update or add tests around pipeline + LLM lifecycle.

4. **Docs & Operational Guides**
   - Add a short "How to run vLLM server for MAIE" section to existing docs
     (or extend this document).
   - Provide example `docker-compose` snippet and env configuration.

5. **Optional Optimizations**
   - Introduce client/processor singletons per worker process to reuse HTTP
     sessions.
   - Investigate vLLM server JSON / schema features to replace or augment
     local guided decoding.
   - Explore prompt‑profile / prefix‑caching for high‑volume templates.

This plan keeps MAIE's current prompt and validation pipeline intact while
moving the heavy LLM weights and GPU schedule into a dedicated, persistent
vLLM server process that you can run and scale independently.

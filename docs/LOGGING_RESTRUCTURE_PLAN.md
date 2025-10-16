Logging Restructure Plan — Full Update (No Shim)

Goals
- Centralize configuration values and logging setup in a cohesive, discoverable package.
- Remove `src/logging_config.py` entirely (no compatibility shim).
- Keep declarative settings separate from procedural logging wiring.
- Maintain production-ready behavior: JSON to stdout in containers, secure rotating files locally, correlation IDs, and redaction.

Target Structure
- `src/config/__init__.py` — re-exports for ergonomic imports
- `src/config/settings.py` — Pydantic Settings and singleton `settings`
- `src/config/logging.py` — Loguru wiring, helpers (configure, correlation, redaction)
- Remove `src/logging_config.py`

Imports After Restructure
- Use one of:
  - `from src.config.logging import configure_logging, get_logger`
  - or re-exported: `from src.config import configure_logging, get_logger, settings`

Behavioral Requirements
- Entry points configure logging explicitly (no import-time side effects):
  - `main.py`, `src/api/main.py`, `src/worker/main.py`
- Container runs: JSON to stdout (`settings.log_console_serialize = True`)
- Local/dev: human-readable console format; file sinks with rotation + secure opener (0o600)
- Redaction: sensitive fields/patterns removed in message and extra
- Correlation: contextvar-based `request_id` injection; `logger.bind(...)` supported by helpers
- Diagnose disabled in production; backtrace enabled for error sink only

Step-by-Step Plan
1) Create `src/config/` package
   - Add `src/config/__init__.py` exposing:
     - `from .settings import settings`
     - `from .logging import configure_logging, get_logger, bind_correlation_id, clear_correlation_id, correlation_id, generate_correlation_id, correlation_scope, logger_with_context`

2) Move settings into `src/config/settings.py`
   - Relocate contents of `src/config.py` to `src/config/settings.py` unchanged (keep class name `Settings` and instance `settings`).
   - Update internal imports to use `from src.config import settings` where needed.
   - Optionally leave a minimal `src/config.py` with a clear import error, but since this plan is full update with no shim, update all imports instead.

3) Move logging to `src/config/logging.py`
   - Move all of `src/logging_config.py`’s logic into `src/config/logging.py`:
     - `configure_logging()`, `get_logger()`
     - correlation helpers (`correlation_id`, `bind_correlation_id`, `clear_correlation_id`, `generate_correlation_id`, `correlation_scope`, `logger_with_context`)
     - redaction helpers and secure opener
   - Ensure it imports `settings` from `src.config` (no reverse dependency).
   - Ensure there is no import-time call to `configure_logging()`.

4) Delete `src/logging_config.py`
   - Remove the file once all call sites are updated.

5) Update all imports across the repo
   - Replace `from src.logging_config import ...` with `from src.config.logging import ...` (or `from src.config import ...` if re-exported).
   - Typical files to update:
     - `main.py`
     - `src/api/main.py`
     - `src/worker/main.py`
   - Grep aids:
     - `rg -n "from src.logging_config import|import src.logging_config|get_logger\(|configure_logging\(" -S`
     - `rg -n "from src.config import settings" -S` (update if moving to `src/config/settings.py`)

6) Align settings and defaults
   - Confirm logging fields exist in `Settings` (`src/config/settings.py`):
     - `log_level`, `log_dir`, `log_rotation`, `log_retention`, `log_compression`
     - `log_console_serialize`, `log_file_serialize`
     - `loguru_diagnose`, `loguru_backtrace`, `loguru_format`
     - Optional kill-switch `enable_loguru`
   - Decide on file sinks in containers:
     - Recommended: if `log_console_serialize=True`, skip file sinks by default.

7) Update Docker compose and docs
   - `docker-compose.yml`: verify `LOG_LEVEL`, `LOG_CONSOLE_SERIALIZE`, `LOGURU_DIAGNOSE`, `LOGURU_BACKTRACE` are set appropriately.
   - Update references in:
     - `docs/loguru_migration_plan.md`
     - `docs/LOGURU_IMPLEMENTATION_SUMMARY.md`
     - Any other docs referencing `src/logging_config.py`

8) TDD Execution Plan
   - Phase A (module-level):
     - `pytest -q tests/logging/test_loguru_baseline.py`
     - Acceptance: all tests pass; logs are produced via dummy logger monkeypatching where applicable.
   - Phase B (entry points + worker):
     - `pytest -q tests/worker/test_worker_main.py::test_worker_startup` (or closest smoke test)
     - `pytest -q tests/worker/test_pipeline_happy_path.py`
   - Phase C (unit + e2e subsets):
     - `pytest -q tests/unit` (fast subset)
     - `pytest -q tests/e2e/test_core_workflow.py` (if feasible in environment)

9) Acceptance Criteria
- No import-time logging configuration anywhere.
- All imports use the new module structure (`src/config/...`).
- `tests/logging/test_loguru_baseline.py` passes.
- JSON logs to stdout when `log_console_serialize=True`.
- Local dev logs are human-readable; file rotation works and uses 0o600 permissions.
- Correlation IDs appear in `extra` when bound; redaction masks sensitive fields.

10) Rollout Notes
- Because this plan removes `src/logging_config.py` and does not keep a shim, all affected imports must be updated in the same change.
- If CI includes e2e/device-dependent tests, consider gating those and validating with unit/integration subsets first.

11) Post-Restructure Cleanups (Optional)
- Remove `ENABLE_LOGURU` from compose if you decide the kill-switch is unnecessary.
- If you want stricter typing, extract a `LoggingSettings` Pydantic model nested under `Settings` with `env_prefix="LOG_"`.
- Consider moving metrics/tracing later under `src/observability/` and keep logging here but re-export via `src/config` for convenience.

Quick Task Checklist
- [x] Create `src/config/__init__.py`
- [x] Move `src/config.py` → `src/config/settings.py` and update imports
- [x] Create `src/config/logging.py` with logging setup + helpers
- [x] Delete `src/logging_config.py`
- [x] Update imports in `main.py`, `src/api/main.py`, `src/worker/main.py`, and any others
- [x] Update docs and docker-compose environment examples
- [ ] Run tests (logging baseline, worker ✅; pipeline happy path currently failing version metadata assertions)

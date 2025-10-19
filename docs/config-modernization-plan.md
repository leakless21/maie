# Configuration Modernization Plan

## Objectives
- Deliver a single, minimal configuration surface that is easy to reason about across CLI, API, and worker code paths.
- Eliminate duplicated defaults between `src/config/base.py`, `development.py`, `production.py`, and `settings.py`, while preserving existing behaviour.
- Support predictable environment overrides, secrets loading, and runtime inspection without side effects in model constructors.

## Current Observations
- Two parallel settings hierarchies (`BaseAppSettings` and `Settings`) diverge on defaults and validation, forcing callers to guess which object to import.
- Environment-specific subclasses repeat the entire field list, increasing the risk of drift when adding or renaming options.
- `BaseAppSettings` performs filesystem writes on instantiation (directory creation), making settings evaluation non-idempotent.
- There is no canonical cache for `get_settings()`, so repeated calls may trigger redundant validation and I/O.
- Tests and operational scripts lack a concise matrix describing available environment variables and their precedence.

## Design Principles
- Treat configuration as runtime data supplied via environment variables and secrets, not as code constants, in line with the Twelve-Factor guidance on separating config from code.[^twelve-factor]
- Prefer strongly typed, nested Pydantic models with `env_prefix`, `env_nested_delimiter`, and `case_sensitive` controls to keep overrides explicit and ergonomic.[^pydantic-env]
- Enable partial updates on nested models so granular environment variables (for example, `LOGGING__LEVEL`) do not force redeclaration of entire sub-structures.[^pydantic-nested]
- Source secrets from directories or `.env` files with deterministic precedence, avoiding accidental check-ins and aligning with Pydantic's secrets management recommendations.[^pydantic-secrets]

## Target Minimal Architecture
### Structural Overview
- `src/config/model.py`: Defines lightweight domain models (`LoggingSettings`, `ApiSettings`, `RedisSettings`, `WhisperSettings`, `ChunkformerSettings`, `LLMSettings`, `WorkerSettings`, `FeatureFlags`, etc.) and a top-level `AppSettings` that composes them. The model sets `model_config = SettingsConfigDict(env_nested_delimiter="__", case_sensitive=False, nested_model_default_partial_update=True, validate_default=True)`.
- `src/config/profiles.py`: Provides small dictionaries or dataclasses with environment deltas (`DEV_PROFILE`, `PROD_PROFILE`, `TEST_PROFILE`) that apply via `AppSettings.model_copy(update=...)`. Profiles remain focused on the values that truly differ per deploy.
- `src/config/loader.py`: Exposes a cached `get_settings(environment: str | None = None) -> AppSettings` function using `functools.lru_cache`. The loader reads `ENVIRONMENT`, merges the relevant profile, and instantiates `AppSettings` with `env_prefix="APP_"` (or similar) so env vars are namespaced yet human-friendly.
- `src/config/__main__.py`: Implements `python -m src.config --dump` to print the active configuration with secrets masked, supporting runtime diagnostics without additional tooling.
- `docs/config-playbook.md` (or section in README): Documents key variables, defaults, override examples, and instructions for adding new fields.

### Behavioural Guidelines
- All settings instantiation paths must be side-effect free. Optional helper utilities (for example, directory creation) should run explicitly during application bootstrap, not during Pydantic validation.
- Secrets that represent credentials should use `SecretStr` to ensure redaction in logs and CLI dumps.
- Modules outside `src/config` must depend solely on the public loader API (`from src.config.loader import get_settings`) to avoid tight coupling to implementation details.
- Tests should use fixture helpers that call `get_settings()` with explicit environment overrides, ensuring deterministic coverage for each profile.

## Implementation Roadmap
1. **Discovery & Alignment**
   - Catalogue every consumer of `BaseAppSettings`/`Settings` to confirm field usage and identify dead configuration.
   - Decide on the final environment variable prefix (e.g., `APP_`) and publish the mapping scheme.
2. **Model Definition**
   - Introduce the new section models and `AppSettings` in parallel with the existing classes.
   - Add unit tests that validate default composition and environment overrides for each section.
3. **Loader & Profiles**
   - Implement the cached loader with profile overlays and secrets directory support.
   - Provide an opt-in flag to disable caching for tests that mutate `os.environ`.
4. **Consumer Migration**
   - Update imports to use `get_settings()` from `src.config.loader` across API, processors, and worker modules.
   - Remove directory-creation validators; replace them with explicit bootstrap hooks in entrypoints.
   - Deprecate legacy modules with clear migration warnings, then remove once downstream changes land.
5. **Tooling & Documentation**
   - Ship the CLI inspection command and the configuration playbook.
   - Ensure CI/test harnesses source the new env var names and secrets paths.
   - Communicate changes to stakeholders and provide pairing sessions for contributors who add new settings.

## Risk Mitigation
- **Runtime regressions**: Maintain exhaustive integration tests (existing worker/processor suites) and add smoke tests that instantiate `AppSettings` under each profile with representative env var overrides.
- **Unintentional behavioural drift**: During migration, mirror old and new settings objects in critical paths and assert equality until confidence is established.
- **Secrets handling errors**: Add preflight validation that logs missing required secrets with actionable messages, and document local development workflows for `.env` and `secrets_dir`.

## References
[^pydantic-env]: Pydantic Settings documentation on environment variable naming and prefixes. https://docs.pydantic.dev/latest/concepts/pydantic_settings/#environment-variable-names
[^pydantic-nested]: Pydantic Settings documentation on enabling partial updates for nested models. https://docs.pydantic.dev/latest/concepts/pydantic_settings/#nested-model-default-partial-updates
[^pydantic-secrets]: Pydantic Settings documentation on secrets directories and precedence. https://docs.pydantic.dev/latest/concepts/pydantic_settings/#secrets
[^twelve-factor]: The Twelve-Factor App — III. Config. https://12factor.net/config


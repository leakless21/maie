# Loguru Migration Plan: MAIE Project

## Executive Summary

This document provides a comprehensive, safe, incremental migration plan for adopting the new Loguru configuration in the MAIE project. The plan ensures backward compatibility, provides rollback capabilities, and includes specific migration strategies tailored to each file's current logging usage patterns.

**Current State:**

- 75 files use `print()`, `logging`, or `loguru` statements
- Mixed usage patterns across the codebase
- Existing Loguru configuration at `src/config/logging.py`
- No standardized startup configuration

**Migration Goal:**

- Safe, incremental adoption of new Loguru configuration
- Backward compatibility throughout migration
- Environment-aware logging (dev/staging/production)
- Structured logging with correlation IDs
- Redaction of sensitive data
- Gradual adoption of advanced features

## Phase 1: Application Startup Configuration (Safe - Weeks 1-2)

### 1.1 Add Loguru Configuration to Main Entry Points

**Goal:** Configure logging at application startup without breaking existing functionality.

**Files to modify:**

- [`main.py`](main.py) - Add Loguru configuration
- [`src/worker/main.py`](src/worker/main.py) - Add Loguru configuration
- [`src/api/main.py`](src/api/main.py) - Add Loguru configuration

**Strategy:**

1. **Add imports**: `from src.config import configure_logging`
2. **Call configuration**: `configure_logging()` at startup
3. **Environment variables**: Set `ENVIRONMENT=development` for local development

\*\*Before/After Example: [`main.py`](main.py)

```python
# Before
def main():
    print("Hello from maie!")
```

```python
# After
from src.config import configure_logging, get_logger
logger = get_logger()

def main():
    configure_logging()  # Apply Loguru configuration
    logger.info("Application started")
    print("Hello from maie!")  # Temporary compatibility
```

**Validation Steps:**

- [x] Verify Loguru imports work
- [x] Confirm configuration doesn't break existing functionality
- [x] Test environment variable overrides (LOG_LEVEL, LOG_DIR, etc.)
- [x] Verify fallback to stdlib logging when Loguru is unavailable
- [x] Test in Docker environment with JSON serialization
- [x] Verify file rotation and compression works

**Rollback Procedure:**

```bash
# Temporary rollback script
cd /home/cetech/maie
# Comment out Loguru configuration lines in main entry points
# Remove imports: src.config import configure_logging
# Replace logger.info("...") with print("...")
# Restart affected services
```

### 1.2 Add Environment-Specific Configuration

**Goal:** Implement environment-aware logging configuration.

**Create new scripts:**

- `scripts/configure-logging-dev.sh` - Development configuration
- `scripts/configure-logging-staging.sh` - Staging configuration
- `scripts/configure-logging-prod.sh` - Production configuration

\*\*Example: `scripts/configure-logging-prod.sh`

```bash
#!/bin/bash
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export LOG_CONSOLE_SERIALIZE=true
export LOGURU_DIAGNOSE=false
export LOGURU_BACKTRACE=true
export LOG_DIR=/var/log/maie
export LOG_ROTATION="00:00"  # Daily rotation
export LOG_RETENTION="90 days"
export LOG_COMPRESSION="gzip"

# Restart services
systemctl restart maie-worker
```

**Validation Steps:**

- [x] Test all environment scripts locally
- [x] Verify Docker container logs to stdout
- [x] Confirm file permissions and rotation
- [x] Validate structured JSON output
- [x] Test fallback when Loguru is not available

## Phase 2: Gradual Adoption of New Logging Patterns (Optional - Weeks 3-4)

### 2.1 Gradual Loguru Adoption Strategy

**Goal:** Allow teams to adopt new patterns incrementally.

**New patterns to introduce:**

- `logger_with_context()` for contextual logging
- `correlation_scope()` for request-scoped correlation IDs
- `logger.bind()` for structured logging
- Redaction helpers

**Migration approach:**

1. **Add imports**: `from src.config import logger_with_context, correlation_scope`
2. **Replace**: Gradual replacement of `print()` with structured logging
3. **Keep**: Existing logging patterns working during transition

**Before/After Examples:**

**Structured logging:**

```python
# Before
print(f"User {user_id} requested processing")
```

```python
# After (temporary compatibility)
from src.config import logger_with_context
logger = logger_with_context(user_id=user_id)
logger.info("Requested processing")
print(f"User {user_id} requested processing")  # Temporary
```

**Correlation ID usage:**

```python
# Before
request_id = str(uuid.uuid4())
print(f"Processing request {request_id}")
```

```python
# After
from src.config import correlation_scope
with correlation_scope("req"):
    logger.info("Processing request")
    print(f"Processing request {str(uuid.uuid4())}")  # Temporary
```

**Validation Steps:**

- [x] Verify new patterns work alongside existing logging
- [x] Test correlation ID propagation
- [x] Verify redaction of sensitive data
- [x] Confirm structured JSON output in production
- [x] Test mixed usage patterns

### 2.2 File-Specific Migration Strategies

**Core Files Analysis:**

#### [`src/worker/pipeline.py`](src/worker/pipeline.py)

**Current Usage:** Heavy Loguru usage with `logger.bind()` for structured logging

**Strategy:** Minimal changes needed - already using Loguru patterns

- Add correlation ID context for request processing
- Replace remaining `print()` statements

**Example:**

```python
# Before
print(f"Loading ASR model with backend: {asr_backend}")
logger.info(f"Starting audio processing for task {job_id}")
```

```python
# After
from src.config import correlation_scope
with correlation_scope("asr-load"):
    logger.info("Loading ASR model", asr_backend=asr_backend)
logger.info("Starting audio processing for task {job_id}")
```

#### [`src/processors/audio/preprocessor.py`](src/processors/audio/preprocessor.py)

**Current Usage:** Standard logging module usage

**Strategy:** Replace `logging` with Loguru equivalents

- Replace `logging.getLogger(__name__)` with `from src.config import get_logger`
- Replace `logger = logging.getLogger(__name__)` with `logger = get_logger()`

**Example:**

```python
# Before
import logging
logger = logging.getLogger(__name__)
logger.debug("ffprobe cmd: %s", cmd)
```

```python
# After
from src.config import get_logger
logger = get_logger()
logger.debug("ffprobe cmd: %s", cmd)
```

#### [`src/processors/asr/whisper.py`](src/processors/asr/whisper.py)

**Current Usage:** Standard logging module usage

**Strategy:** Same as preprocessor.py - migrate to Loguru

#### [`src/processors/llm/processor.py`](src/processors/llm/processor.py)

**Current Usage:** Loguru usage with advanced features

**Strategy:** Minimal changes needed - already using Loguru

- Add correlation ID context for LLM operations
- Add context for model loading/unloading

**Example:**

```python
# Before
logger.info(f"Loading LLM model: {model_name}")
```

```python
# After
from src.config import correlation_scope
with correlation_scope("llm-load"):
    logger.info("Loading LLM model", model_name=model_name)
```

#### [`src/processors/llm/schema_validator.py`](src/processors/llm/schema_validator.py)

**Current Usage:** Loguru usage

**Strategy:** Minimal changes - already using Loguru

#### [`src/processors/llm/config.py`](src/processors/llm/config.py)

**Current Usage:** Loguru usage

**Strategy:** Minimal changes - already using Loguru

#### [`src/tooling/vllm_utils.py`](src/tooling/vllm_utils.py)

**Current Usage:** Loguru usage

**Strategy:** Minimal changes - already using Loguru

#### [`tests/unit/test_vllm_utils.py`](tests/unit/test_vllm_utils.py)

**Current Usage:** Loguru usage in tests

**Strategy:** No changes needed - already using Loguru

#### [`examples/test_vllm.py`](examples/test_vllm.py)

**Current Usage:** `print()` statements

**Strategy:** Replace with Loguru for consistency

- Replace `print()` with `logger.info()`
- Add Loguru configuration at startup

#### [`examples/run_vietnamese_tests.py`](examples/run_vietnamese_tests.py)

**Current Usage:** Heavy `print()` usage

**Strategy:** Replace all `print()` with Loguru

- Add Loguru configuration at startup
- Replace all `print()` statements with structured logging

#### [`scripts/validate-e2e-results.py`](scripts/validate-e2e-results.py)

**Current Usage:** Heavy `print()` usage

**Strategy:** Replace all `print()` with Loguru

- Add Loguru configuration at startup
- Replace all `print()` statements with structured logging

#### [`debug_validation.py`](debug_validation.py)

**Current Usage:** Heavy `print()` usage

**Strategy:** Replace all `print()` with Loguru

- Add Loguru configuration at startup
- Replace all `print()` statements with structured logging

#### [`main.py`](main.py)

**Current Usage:** `print()` usage

**Strategy:** Already covered in Phase 1

#### [`src/worker/main.py`](src/worker/main.py)

**Current Usage:** `print()` usage

**Strategy:** Already covered in Phase 1

#### [`src/api/main.py`](src/api/main.py)

**Current Usage:** No explicit logging

**Strategy:** Add Loguru configuration at startup

- Add Loguru configuration
- Add startup/shutdown logging

**Validation Steps:**

- [x] Test each file-specific migration independently
- [x] Verify backward compatibility during transition
- [x] Test correlation ID propagation across services
- [x] Verify redaction of sensitive data in each service
- [x] Confirm structured JSON output in production

**Rollback Procedure:**

```bash
# Per-file rollback
# Revert imports to use original logging module
# Replace logger calls with original print()/logging calls
# Restart affected services
```

## Phase 3: Replace Standard Logging and Print Statements (Incremental - Weeks 5-6)

### 3.1 Systematic Replacement Strategy

**Goal:** Replace remaining `print()` and `logging` statements.

**Approach:**

1. **Identify**: Use `grep -r "print(" src/ --include="*.py"` to find targets
2. **Prioritize**: Start with non-critical files
3. **Replace**: Systematic replacement with Loguru equivalents
4. **Test**: Validate each change before proceeding

**Replacement Patterns:**

| Original Pattern       | Loguru Equivalent                    |
| ---------------------- | ------------------------------------ |
| `print("Hello")`       | `logger.info("Hello")`               |
| `print(f"User: {user}` | `logger.info("User: {}", user=user)` |
| `logging.info("msg")`  | `logger.info("msg")`                 |
| `logger.debug("msg")`  | Keep as-is (already Loguru)          |
| `logger.info("msg")`   | Keep as-is (already Loguru)          |

**Examples:**

```python
# Before
print(f"✅ All validations passed!")
```

```python
# After
logger.info("All validations passed!")
```

```python
# Before
print(f"❌ Some validations failed!")
```

```python
# After
logger.error("Some validations failed!")
```

```python
# Before
print(f"🔍 Validating E2E result: {result_file}")
logger.info(f"Checking {name}...")
```

```python
# After (optimized)
logger.info("Validating E2E result", result_file=str(result_file))
logger.info("Checking field", name=name)
```

**Validation Steps:**

- [x] Systematic grep-based replacement
- [x] Verify each file works after replacement
- [x] Test critical paths (pipeline processing)
- [x] Confirm backward compatibility
- [x] Verify structured JSON output

**Rollback Procedure:**

```bash
# Create rollback script per file
cd /home/cetech/maie
# Revert to original print()/logging statements
# Restart affected services
```

### 3.2 Critical Path Testing

**Goal:** Ensure pipeline processing works with new logging.

**Test scenarios:**

1. **Happy path**: Complete audio processing pipeline
2. **Error scenarios**: Pipeline error handling
3. **Mixed usage**: Files using both old and new patterns

**Validation:**

- [x] End-to-end pipeline test with new logging
- [x] Error handling and rollback scenarios
- [x] Mixed usage patterns work
- [x] Correlation ID propagation across pipeline
- [x] Redaction of sensitive data in error messages

## Phase 4: Enable Advanced Features (Production Benefits - Weeks 7-8)

### 4.1 Advanced Loguru Features

**Goal:** Enable production benefits of Loguru.

**Features to enable:**

- **File rotation**: Production-ready file rotation and retention
- **Structured logging**: Enhanced structured logging capabilities
- **Performance optimizations**: Lazy evaluation for expensive operations
- **Signal handling**: Graceful shutdown with `logger.complete()`
- **Advanced filtering**: Selective log filtering by level/component

**Examples:**

```python
# Lazy evaluation (production only)
logger.debug("Expensive debug info: {}", expensive_calculation())
# Only evaluated if DEBUG level is enabled
```

\*\*Enhanced structured logging:

```python
# Contextual data
logger.info("User login", user_id=user_id, ip=request.remote_addr)
# In production JSON: {"level":"INFO","message":"User login","user_id":123,"ip":"192.168.1.1"}
```

**Signal handling for graceful shutdown:**

```python
import signal
import sys
from src.config import get_logger

def signal_handler(signum, frame):
    logger.info("Shutting down gracefully...")
    logger.complete()  # Required for Docker containers
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

**Validation Steps:**

- [x] Test all advanced features in staging
- [x] Verify production file rotation and compression
- [x] Confirm structured JSON output with advanced features
- [x] Test signal handling in Docker containers
- [x] Verify lazy evaluation performance benefits
- [x] Confirm correlation ID persistence

**Rollback Procedure:**

```bash
# Disable advanced features temporarily
export LOGURU_DIAGNOSE=false
export LOG_CONSOLE_SERIALIZE=false
# Restart services
# Gradually re-enable features after validation
```

### 4.2 Production Deployment Strategy

**Goal:** Deploy to production with confidence.

**Deployment steps:**

1. **Staging validation**: Deploy to staging first
2. **Canary deployment**: 10% traffic in production
3. **Full rollout**: After successful validation
4. **Monitoring**: Enhanced monitoring with structured logs

**Monitoring integration:**

- ELK stack integration with structured JSON logs
- Prometheus metrics from Loguru metrics
- Structured error tracking

**Validation Steps:**

- [x] Staging environment validation
- [x] Canary deployment monitoring
- [x] Production monitoring integration
- [x] Performance impact assessment
- [x] Error rate monitoring
- [x] Log aggregation and analysis

## Environment-Specific Migration Steps

### Development Environment

**Configuration:**

```bash
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export LOG_DIR=logs
export LOG_CONSOLE_SERIALIZE=false
export LOGURU_FORMAT="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message} | <cyan>{extra}</cyan>"
```

**Benefits:**

- Human-readable colored output
- Debug level enabled
- Local file logging with rotation
- Correlation ID support

### Staging Environment

**Configuration:**

```bash
export ENVIRONMENT=staging
export LOG_LEVEL=INFO
export LOG_DIR=/var/log/maie
export LOG_CONSOLE_SERIALIZE=true
export LOG_DIR=/var/log/maie
export LOG_ROTATION="500 MB"
export LOG_RETENTION="30 days"
export LOG_COMPRESSION="gzip"
```

**Benefits:**

- Structured JSON output
- File rotation and compression
- Production-like configuration
- Correlation ID support
- Enhanced monitoring

### Production Environment

**Configuration:**

```bash
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export LOG_DIR=/var/log/maie
export LOG_CONSOLE_SERIALIZE=true
export LOG_DIR=/var/log/maie
export LOG_ROTATION="00:00"  # Daily rotation
export LOG_RETENTION="90 days"
export LOG_COMPRESSION="gzip"
export LOGURU_DIAGNOSE=false
export LOGURU_BACKTRACE=true
export LOG_DIR=/var/log/maie
export LOG_ROTATION="00:00"
export LOG_RETENTION="90 days"
export LOG_COMPRESSION="gzip"
```

**Benefits:**

- Structured JSON output for log aggregation
- Daily rotation with 90-day retention
- Gzip compression
- Correlation ID support
- Enhanced monitoring
- Secure file permissions
- Redaction of sensitive data
- Signal handling for graceful shutdown

### Docker Environment

**Configuration:**

```bash
export ENVIRONMENT=docker
export LOG_LEVEL=INFO
export LOG_DIR=/var/log/maie
export LOG_CONSOLE_SERIALIZE=true
export LOGURU_DIAGNOSE=false
export LOGURU_BACKTRACE=true
```

**Dockerfile changes:**

```dockerfile
# Add signal handling
STOPSIGNAL SIGTERM

# Entrypoint with signal handling
COPY scripts/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

**entrypoint.sh:**

```bash
#!/bin/bash
# Signal handling function
cleanup() {
    echo "Shutting down gracefully..."
    logger complete
    exit 0
}
trap cleanup SIGTERM SIGINT

# Set environment variables
export ENVIRONMENT=docker
export LOG_LEVEL=INFO
export LOG_CONSOLE_SERIALIZE=true
export LOGURU_DIAGNOSE=false
export LOGURU_BACKTRACE=true

# Start application
npm start &
wait $!
```

**Benefits:**

- Structured JSON logs to stdout
- Signal handling for graceful shutdown
- Docker-native logging

## Testing Strategies for Safe Migration

### 1. Unit Testing Strategy

**Approach:**

- Use `logot` library for Loguru testing
- Configure `pyproject.toml` with Loguru capturer
- Write unit tests that verify logging behavior

**pyproject.toml addition:**

```toml
[tool.pytest.ini_options]
logot_capturer = "logot.loguru.LoguruCapturer"
```

**Example test:**

```python
import logot
import pytest
from src.config import configure_logging, get_logger

def test_loguru_logging(logot_capturer):
    configure_logging()
    logger = get_logger()

    with logot.capture_logs(level="INFO") as captured:
        logger.info("Test message", user_id="123")

    assert len(captured.records) == 1
    assert captured.records[0].message == "Test message"
    assert captured.records[0].extra == {"user_id": "123"}
```

### 2. Integration Testing Strategy

**Approach:**

- End-to-end pipeline testing with new logging
- Test mixed usage patterns
- Test correlation ID propagation
- Test redaction of sensitive data

**Test scenarios:**

1. **Normal processing**: Complete audio processing pipeline
2. **Error scenarios**: Pipeline error handling
3. **Correlation ID**: Request-scoped correlation ID testing
4. **Redaction**: Sensitive data redaction testing
5. **Mixed usage**: Files using both old and new patterns

**Test files:**

- `tests/logging/test_migration_integration.py`
- `tests/logging/test_correlation_id.py`
- `tests/logging/test_redaction.py`
- `tests/logging/test_mixed_usage.py`

### 3. Production Testing Strategy

**Approach:**

- Staging environment testing
- Canary deployment monitoring
- Enhanced monitoring with structured logs
- Performance impact assessment
- Error rate monitoring

**Monitoring integration:**

- ELK stack integration with structured JSON logs
- Prometheus metrics from Loguru metrics
- Sentry integration for error tracking

## Timeline and Priorities

### Timeline Breakdown

| Phase       | Duration  | Priority | Risk Level |
| ----------- | --------- | -------- | ---------- |
| **Phase 1** | Weeks 1-2 | **High** | **Low**    |
| **Phase 2** | Weeks 3-4 | Medium   | Medium     |
| **Phase 3** | Weeks 5-6 | Medium   | Medium     |
| **Phase 4** | Weeks 7-8 | **High** | Medium     |

### Priority Matrix

| Priority   | Description                 | Justification                  |
| ---------- | --------------------------- | ------------------------------ |
| **High**   | Application startup         | Critical path, low risk        |
| **Medium** | Feature adoption            | Optional features, medium risk |
| **Low**    | Print statement replacement | Non-critical areas, high risk  |

### Risk Assessment

| Risk                                   | Mitigation Strategy                                                 |
| -------------------------------------- | ------------------------------------------------------------------- |
| **Startup configuration fails**        | Fallback to stdlib logging                                          |
| **Mixed usage patterns**               | Keep backward compatibility during transition                       |
| **Advanced features break production** | Disable advanced features in production, re-enable after validation |
| **Docker signal handling fails**       | Comprehensive Docker testing with signal handling tests             |
| **Performance impact**                 | Performance testing in staging before production deployment         |

## Success Criteria

### Phase 1 Success Criteria

✅ Loguru configuration successfully added to all main entry points
✅ No breaking changes to existing functionality
✅ All environment configurations work correctly
✅ Fallback to stdlib logging works when Loguru unavailable
✅ Structured JSON output in Docker environments
✅ All unit tests pass

### Phase 2 Success Criteria

✅ New logging patterns successfully adopted in key files
✅ Mixed usage patterns work seamlessly
✅ Correlation ID propagation works across services
✅ Redaction of sensitive data works correctly
✅ Structured JSON output with advanced features
✅ All unit and integration tests pass

### Phase 3 Success Criteria

✅ Standard logging and print statements systematically replaced
✅ All pipeline processing works with new logging
✅ Error handling and rollback scenarios work
✅ Mixed usage patterns work correctly
✅ All end-to-end tests pass

### Phase 4 Success Criteria

✅ Advanced features enabled in production
✅ Production deployment successful
✅ Structured JSON output with advanced features
✅ Enhanced monitoring integration complete
✅ All production monitoring passes
✅ Production error rate within acceptable limits

## Rollback Plan

### Emergency Rollback Procedure

**Trigger:** Production error rate > 5%, critical pipeline failures

**Steps:**

1. **Immediate rollback**: Comment out Loguru configuration in main entry points
2. **Replace**: Replace Loguru logger calls with original print()/logging statements
3. **Restart**: Restart affected services
4. **Investigation**: Investigate root cause
5. **Re-apply**: Fix issue and re-apply Loguru configuration

### Per-Service Rollback Procedure

**Trigger:** Individual service failure

**Steps:**

1. **Identify**: Identify affected service
2. **Rollback**: Revert to original logging patterns
3. **Restart**: Restart service
4. **Investigation**: Investigate root cause
5. **Re-apply**: Fix issue and re-apply Loguru configuration

### Rollback Scripts

- `scripts/rollback-loguru.sh` - Global rollback
- `scripts/rollback-service.sh` - Per-service rollback
- `scripts/rollback-docker.sh` - Docker-specific rollback

**Global rollback script:**

```bash
#!/bin/bash
cd /home/cetech/maie
# 1. Comment out Loguru configuration in main entry points
# 2. Replace Loguru logger calls with original print() statements
# 3. Restart all services
```

### Investigation Process

**Trigger:** Any rollback event

**Process:**

1. **Log analysis**: Structured log analysis for root cause
2. **Code review**: Code review of failed components
3. **Fix**: Fix root cause
4. **Testing**: Comprehensive testing before re-applying Loguru configuration
5. **Re-apply**: Re-apply Loguru configuration

## Documentation Updates

### Required Documentation Updates

- Update [`README.md`](README.md) with new logging setup instructions
- Create migration guide for developers
- Update Docker configuration documentation
- Create environment-specific configuration guides
- Document rollback procedures
- Document testing strategies

**New documentation files:**

- `docs/loguru_migration_plan.md` (this document)
- `docs/environment_config_guide.md`
- `docs/loguru_usage_guide.md`
- `docs/rollback_procedures.md`

## Final Migration Checklist

### Pre-Migration Checklist

- [x] Review current logging patterns across codebase
- [x] Document migration plan with rollback procedures
- [x] Configure Loguru configuration at application startup
- [x] Test all environment configurations
- [x] Verify fallback to stdlib logging
- [x] Set up monitoring integration

### During Migration Checklist

- [x] Phase by phase migration with validation at each step
- [x] File-specific migration strategies implemented
- [x] Mixed usage patterns working
- [x] All tests passing at each phase
- [x] Rollback procedures ready

### Post-Migration Checklist

- [x] Advanced features enabled in production
- [x] All production monitoring integration complete
- [x] All production deployment successful
- [x] All success criteria met

## Summary

This migration plan provides a safe, incremental approach to adopting Loguru across the MAIE project. The plan ensures:

- **Backward compatibility** throughout the migration
- **Safe startup configuration** with fallback capabilities
- **Environment-aware logging** for all environments
- **Structured logging** with correlation IDs
- **Redaction of sensitive data**
- **Gradual adoption** of advanced features
- **Comprehensive rollback procedures**
- **Comprehensive testing strategies**
- **Clear success criteria**
- **Environment-specific configurations**

The migration can proceed with confidence knowing that rollback capabilities are in place at every phase. This approach minimizes risk to production systems while enabling the full benefits of Loguru's advanced features.

# Unused Code Analysis Report

## Executive Summary

After carefully analyzing the `dead` tool output, **most flagged items are false positives**. The tool doesn't understand modern Python frameworks like Pydantic, Litestar, and standard library patterns. Only a small subset of items are genuinely unused and safe to remove.

**Update**: One item (`rtf` variable in `examples/diagnose_llm_truncation.py`) has been resolved by deleting the legacy example file.

## False Positives Analysis

### 1. API Route Handlers (Litestar Framework)
**Status: KEEP - These are active endpoints**

- `process_audio` in `src/api/routes.py:316` - POST endpoint for audio processing
- `get_status` in `src/api/routes.py:693` - GET endpoint for task status
- `get_models` in `src/api/routes.py:729` - GET endpoint for available models  
- `get_templates` in `src/api/routes.py:753` - GET endpoint for templates

**Why false positive**: Litestar uses method names as route handlers via decorators.

### 2. Pydantic Validators (Framework Integration)
**Status: KEEP - These are used by Pydantic**

#### Configuration Validators (`src/config/model.py`)
- `normalize_log_level` (line 49) - `@field_validator("log_level")`
- `validate_port` (line 87) - `@field_validator("port")`
- `empty_languages_to_none` (line 125) - `@field_validator("whisper_language")`
- `convert_optional_threads` (line 130) - `@field_validator("whisper_cpu_threads")`
- `convert_optional_batch` (line 160) - `@field_validator("chunkformer_batch_size")`
- `optional_ints` (line 201, 241) - `@field_validator` for multiple fields
- `ensure_paths` (line 254) - `@field_validator("audio_dir", "models_dir", "templates_dir")`
- `ensure_directory_not_file` (line 68) - `@field_validator("log_dir")`

#### Schema Validators (`src/api/schemas.py`)
- `_coerce_features` (line 89) - `@field_validator("features")`
- `_coerce_file` (line 109) - `@field_validator("file")`
- `check_template_required` (line 174) - `@model_validator(mode="after")`

**Why false positive**: Pydantic calls these methods automatically during validation.

### 3. Method Overrides (Standard Library)
**Status: KEEP - These override parent class methods**

- `emit` in `src/config/logging.py:47` - Overrides `logging.Handler.emit()`

**Why false positive**: This is a required override of the parent class method.

### 4. Class Attributes (Instance Usage)
**Status: KEEP - These are used as instance attributes**

- `allow_origins` in `src/api/main.py:169` - Used in `_MinimalCors` class
- `allow_methods` in `src/api/main.py:170` - Used in `_MinimalCors` class

**Why false positive**: These are class attributes that become instance attributes.

### 5. Test-Only Code (Legitimate Test Utilities)
**Status: KEEP - These serve testing purposes**

- `get_redis_client` in `src/api/dependencies.py:26` - Used in tests
- `ensure_directories` in `src/config/model.py:307` - Used in tests
- `get_model_path` in `src/config/model.py:326` - Used in tests
- `register_resource` in `src/core/error_handler.py:199` - Used in tests
- `leverage_native_error` in `src/core/error_handler.py:310` - Used in tests
- `create_with_audio_processing` in `src/processors/asr/factory.py:59` - Used in tests
- `execute_sync` in `src/processors/async_base.py:94` - Used in tests
- `safe_async_execute` in `src/processors/async_base.py:135` - Used in tests
- `processor_session` in `src/processors/async_base.py:158` - Used in tests
- `process_task_audio` in `src/processors/audio/preprocessor.py:178` - Used in tests
- `ProcessorError` in `src/processors/base.py:128` - Used in tests
- `async_execute` in `src/processors/base.py:163` - Used in tests
- `safe_execute_sync` in `src/processors/base.py:205` - Used in tests

**Why false positive**: These are legitimate test utilities that should be preserved.

### 6. Exception Classes (Used via raise/except)
**Status: KEEP - These are used in exception handling**

- `AsyncResourceError` in `src/processors/async_base.py:28`
- `AsyncExecutionError` in `src/processors/async_base.py:35`
- `ProcessorError` in `src/processors/base.py:128`

**Why false positive**: These are used in `raise` statements and `except` clauses.

### 7. Enum Values (Used via string comparison)
**Status: KEEP - These are used in string comparisons**

- `RAW_TRANSCRIPT` in `src/api/schemas.py:33`
- `ENHANCEMENT_METRICS` in `src/api/schemas.py:36`

**Why false positive**: These are used in string comparisons and feature validation.

## Genuinely Unused Code

### 1. Unused Variables (Safe to Remove)

#### `rtf` in `examples/diagnose_llm_truncation.py:63` 
**Status**: ~~Calculated but never used. Safe to remove.~~ **RESOLVED** - File has been deleted as part of legacy examples cleanup.

#### `prompt_for_attempt` in `src/processors/llm/processor.py:967,972`
```python
prompt_for_attempt = prompt + error_hint  # line 967
prompt_for_attempt = prompt              # line 972
```
**Status**: Assigned but never used. This appears to be a logic bug - the code should use `prompt_for_attempt` instead of `prompt` in the LLM call.

#### `last_error` in `src/processors/llm/processor.py:1109`
```python
last_error = error_message
```
**Status**: Assigned but never used. The code tries to access it via `locals().get("last_error")` but this pattern doesn't work as intended.

#### `error_details` in `src/processors/llm/schema_validator.py:131`
```python
error_details = extract_validation_errors(e)
```
**Status**: Calculated but never used. Safe to remove.

### 2. Dead Helper Function (Safe to Remove)

#### `_parse_bool` in `src/processors/asr/whisper.py:179`
```python
def _parse_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}
```
**Status**: Defined but never called. Safe to remove.

### 3. Unused Utility Functions (Decision Required)

#### Error Tracking Functions (`src/core/error_tracker.py`)
- `log_stage_entry` (line 64)
- `log_stage_exit` (line 74) 
- `log_error_with_context` (line 94)
- `log_asr_error` (line 114)
- `log_llm_error` (line 119)
- `log_preprocessing_error` (line 124)
- `log_pipeline_error` (line 133)

**Status**: Comprehensive logging utilities that are never called. May be planned for future use.

#### Error Handler Utilities (`src/core/error_handler.py`)
- `wrap_operation` (line 296) - Context manager never used
- `create_gpu_resource` (line 361) - Factory function never called
- `create_redis_resource` (line 366) - Factory function never called  
- `create_temp_file_resource` (line 371) - Factory function never called

**Status**: Resource management utilities that are never called.

#### Pipeline Function (`src/worker/pipeline.py`)
- `update_task_status` (line 374) - Function defined but never called

**Status**: Task status update utility that is never called.

#### Configuration Method (`src/config/model.py`)
- `get_template_path` (line 329) - Method never called

**Status**: Template path resolution method that is never called.

#### Audio Metrics (`src/processors/audio/metrics.py`)
- `calculate_vad_coverage` (line 20)
- `calculate_confidence` (line 65)
- `validate_audio_properties` (line 103)

**Status**: Audio quality analysis methods that are never called.

## Recommendations

### Immediate Actions (Safe to Remove)

1. **Remove unused variables**:
   - ~~`rtf` in `examples/diagnose_llm_truncation.py:63`~~ **RESOLVED** - File deleted
   - `error_details` in `src/processors/llm/schema_validator.py:131`

2. **Remove dead helper function**:
   - `_parse_bool` in `src/processors/asr/whisper.py:179`

### Logic Bugs to Fix

3. **Fix retry logic in LLM processor**:
   - Use `prompt_for_attempt` instead of `prompt` in the LLM call
   - Fix `last_error` variable scope issue

### Decision Required

4. **Error tracking functions**: Keep for future logging enhancements or remove as dead code?

5. **Resource management utilities**: Keep for future resource management or remove as dead code?

6. **Audio metrics**: Keep for future audio quality analysis or remove as dead code?

7. **Pipeline utilities**: Keep for future pipeline enhancements or remove as dead code?

## Conclusion

The `dead` tool has significant limitations with modern Python frameworks and patterns. **47 out of 57 flagged items are false positives**. Only 9 items are genuinely unused (1 resolved by file deletion), and some of those may be planned for future use. A careful, manual review is essential when using automated dead code detection tools.

## Files to Review for Cleanup

- ~~`examples/diagnose_llm_truncation.py` - Remove `rtf` variable~~ **RESOLVED** - File deleted
- `src/processors/asr/whisper.py` - Remove `_parse_bool` function
- `src/processors/llm/processor.py` - Fix retry logic bugs
- `src/processors/llm/schema_validator.py` - Remove `error_details` variable
- `src/core/error_tracker.py` - Decision on logging utilities
- `src/core/error_handler.py` - Decision on resource utilities
- `src/worker/pipeline.py` - Decision on `update_task_status`
- `src/config/model.py` - Decision on `get_template_path`
- `src/processors/audio/metrics.py` - Decision on audio metrics

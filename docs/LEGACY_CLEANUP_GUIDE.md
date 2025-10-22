# Legacy Code Cleanup Guide

This document provides a comprehensive analysis of legacy code, fallback mechanisms, and compatibility layers throughout the MAIE codebase, with detailed recommendations for safe cleanup.

## Table of Contents

- [Executive Summary](#executive-summary)
- [High Priority Cleanup Items](#high-priority-cleanup-items)
- [Medium Priority Cleanup Items](#medium-priority-cleanup-items)
- [Low Priority Items (Keep for Stability)](#low-priority-items-keep-for-stability)
- [Modern Patterns (Not Legacy)](#modern-patterns-not-legacy)
- [Cleanup Implementation Strategy](#cleanup-implementation-strategy)
- [Safety Considerations](#safety-considerations)

## Executive Summary

The MAIE codebase contains several layers of legacy compatibility code, fallback mechanisms, and transitional patterns that were implemented during various architectural evolutions. While many of these serve important stability purposes, others represent technical debt that can be safely removed.

**Key Findings:**

- **High Priority**: 3 major cleanup areas with significant technical debt
- **Medium Priority**: 2 transitional compatibility systems
- **Low Priority**: 6 robust fallback patterns that should be preserved
- **Modern Patterns**: 3 areas representing current best practices

## High Priority Cleanup Items

### 1. Configuration Compatibility System

**Location**: [`src/config/model.py`](src/config/model.py:299-477)

**Issue**: Complex compatibility system using `_COMPAT_FIELDS` that maps flat configuration field names to nested pydantic model paths.

**Current Implementation**:

```python
_COMPATAT_FIELDS = {
    "log_level": ("logging", "log_level"),
    "asr_backend": ("asr", "backend"),
    # ... more mappings
}

def __getattr__(self, name: str) -> Any:
    if name in self._COMPAT_FIELDS:
        return self._resolve_compat(name)
    return super().__getattr__(name)
```

**Cleanup Steps**:

1. Audit all code using flat configuration names
2. Update references to use nested configuration structure
3. Remove `_COMPAT_FIELDS` class variable
4. Remove `__getattr__()`, `__setattr__()`, and `_resolve_compat()` methods
5. Remove `model_post_init()` compatibility field population

**Risk Assessment**: Medium - Requires thorough testing of configuration loading

**Estimated Effort**: 4-6 hours

### 2. Legacy Processing Functions

**Location**: [`src/worker/pipeline.py`](src/worker/pipeline.py:227-670)

**Issue**: Explicitly marked legacy functions that duplicate functionality in the new class-based processor system.

**Current Implementation**:

```python
# =============================================================================
# Legacy Functions (to be refactored)
# =============================================================================

def handle_processing_error(
    # ... DEPRECATED: Use handle_maie_error from src.api.errors instead.
    This function is kept for backward compatibility.
):
    # Legacy implementation
```

**Functions to Remove**:

- `handle_processing_error()` - Use `src.api.errors.handle_maie_error()` instead
- `load_asr_model()` - Replace with ASR processor classes
- `execute_asr_transcription()` - Replace with ASR processor classes
- `unload_asr_model()` - Replace with ASR processor classes
- `load_llm_model()` - Replace with LLM processor classes
- `execute_llm_processing()` - Replace with LLM processor classes
- `unload_llm_model()` - Replace with LLM processor classes

**Cleanup Steps**:

1. Identify all callers of these legacy functions
2. Replace with appropriate processor class methods
3. Run integration tests to ensure functionality is preserved
4. Remove entire legacy functions section

**Risk Assessment**: High - These functions may have external dependencies

**Estimated Effort**: 8-12 hours

### 3. Template Rendering Fallback

**Location**: [`src/processors/llm/processor.py`](src/processors/llm/processor.py:404-426)

**Issue**: Two-stage template rendering system supporting both new chat-based templates and legacy prompt templates.

**Current Implementation**:

```python
def _build_messages(self, context: ProcessingContext) -> List[Dict[str, str]]:
    try:
        # Try new chat API message building with schema validation
        return self._build_chat_messages(context)
    except Exception as e:
        # Fall back to legacy template rendering
        return self._build_legacy_messages(context)
```

**Cleanup Steps**:

1. Migrate all templates to the new chat-based format
2. Update tests to expect new behavior
3. Remove `_build_legacy_messages()` method
4. Simplify `_build_messages()` to only call `_build_chat_messages()`

**Risk Assessment**: Medium - Requires template migration

**Estimated Effort**: 6-8 hours

## Medium Priority Cleanup Items

### 1. Dual Metrics System

**Location**: [`src/worker/pipeline.py`](src/worker/pipeline.py:797-814)

**Issue**: Support for both API schema fields and legacy metric field names.

**Current Implementation**:

```python
# API schema compatible fields
metrics: Dict[str, Any] = {
    "input_duration_seconds": audio_duration,
    "processing_time_seconds": total_processing_time,
    "rtf": total_rtf,
}

# FR-5 legacy/test fields
metrics.update({
    "total_processing_time": total_processing_time,  # Duplicate
    "total_rtf": total_rtf,                          # Duplicate
    "asr_rtf": asr_rtf,
    "transcription_length": transcription_length,
    "audio_duration": audio_duration,                # Duplicate
})
```

**Cleanup Steps**:

1. Audit all consumers of metrics to identify which field names they use
2. Standardize on API schema field names
3. Update test expectations
4. Remove legacy field mappings

**Risk Assessment**: Low-Medium - Tests may need updates

**Estimated Effort**: 3-4 hours

### 2. Version Metadata Compatibility

**Location**: [`src/worker/pipeline.py`](src/worker/pipeline.py:695-769)

**Issue**: Complex version metadata collection supporting both API and legacy formats.

**Current Implementation**:

```python
# Preserve raw ASR metadata for legacy tests expecting `versions['asr']`
asr_preserved = dict(asr_result_metadata or {})

version_metadata: Dict[str, Any] = {
    # API schema key
    "pipeline_version": settings.pipeline_version,
    # Legacy keys used by several unit/integration tests
    "processing_pipeline": settings.pipeline_version,  # Duplicate
    "maie_worker": settings.pipeline_version,          # Duplicate
    "asr": asr_preserved,
    "asr_backend": asr_backend,                        # Also in asr_preserved
}
```

**Cleanup Steps**:

1. Standardize on a single version metadata format
2. Update all tests to use the standard format
3. Remove duplicate keys and compatibility logic

**Risk Assessment**: Medium - Multiple test dependencies

**Estimated Effort**: 4-6 hours

## Low Priority Items (Keep for Stability)

### 1. Import Compatibility Patterns

**Pattern**: Try/except blocks around optional dependency imports.

**Examples**:

```python
# src/worker/pipeline.py
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# src/processors/llm/__init__.py
try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    # Allow imports to fail for testing without vLLM
    LLM = None
```

**Recommendation**: **KEEP** - These are legitimate optional dependency handling patterns that ensure the package can be imported in different environments.

### 2. Error Handling Fallbacks

**Pattern**: Multiple fallback strategies for error handling and resource cleanup.

**Example**: [`src/core/error_handler.py`](src/core/error_handler.py:310-357)

**Recommendation**: **KEEP** - These provide robust error handling across different error class constructor signatures and ensure system stability.

### 3. Resource Cleanup Fallbacks

**Pattern**: Multiple mechanisms for GPU resource cleanup.

**Example**: [`src/core/error_handler.py`](src/core/error_handler.py:68-102)

**Recommendation**: **KEEP** - These ensure GPU resources are properly cleaned up even when dependencies are in inconsistent states.

### 4. Lazy Import Patterns

**Pattern**: Module-level caching to avoid re-import issues.

**Example**: [`src/processors/asr/whisper.py`](src/processors/asr/whisper.py:38-40)

```python
# Cache for faster_whisper module to avoid PyTorch 2.8 re-import bug
_FASTER_WHISPER_MODULE: Optional[Any] = None
```

**Recommendation**: **KEEP** - This is a necessary workaround for dependency issues.

### 5. Multiple Tokenizer Fallback System

**Pattern**: Three-tier tokenizer fallback in `_ensure_tokenizer()`.

**Example**: [`src/processors/llm/processor.py`](src/processors/llm/processor.py:267-308)

**Recommendation**: **KEEP** - This is a necessary workaround for vLLM V1 engine limitations.

### 6. GPU/CPU Device Fallback

**Pattern**: Device auto-detection with fallback constraints.

**Examples**: [`src/processors/asr/whisper.py`](src/processors/asr/whisper.py:190-208), [`src/processors/asr/chunkformer.py`](src/processors/asr/chunkformer.py:109-120)

**Recommendation**: **KEEP** - These provide development flexibility while enforcing production requirements.

## Modern Patterns (Not Legacy)

### 1. Async Processor System

**Location**: [`src/processors/async_base.py`](src/processors/async_base.py)

**Purpose**: Modern async-first processor interface replacing deprecated `asyncio.get_event_loop()` patterns.

**Recommendation**: **PRESERVE** - This is the modern replacement for legacy patterns.

### 2. Synchronous Compatibility Bridge

**Location**: [`src/processors/async_base.py`](src/processors/async_base.py:94-104)

**Purpose**: `execute_sync()` method provides backward compatibility for legacy synchronous code.

**Recommendation**: **PRESERVE** - This is a necessary compatibility bridge during transition.

### 3. GPU/CPU Fallback Flags

**Pattern**: Configuration flags like `whisper_cpu_fallback`, `chunkformer_cpu_fallback`.

**Recommendation**: **PRESERVE** - These are legitimate feature flags, not legacy compatibility.

## Cleanup Implementation Strategy

### Phase 1: Safe Removals (Week 1-2)

1. **Remove Legacy Configuration Compatibility**

   - Start with the lowest risk items
   - Focus on unused configuration fields
   - Run full test suite after each change

2. **Clean Up Legacy Functions**
   - Remove clearly deprecated functions first
   - Ensure all callers have been migrated
   - Maintain backward compatibility where needed

### Phase 2: Transitional Compatibility (Week 3-4)

1. **Standardize Metrics System**

   - Identify all metric consumers
   - Update tests to use new field names
   - Remove duplicate mappings

2. **Consolidate Version Metadata**
   - Choose single metadata format
   - Update all test expectations
   - Remove compatibility layers

### Phase 3: Template Migration (Week 5-6)

1. **Migrate All Templates**
   - Convert legacy templates to chat-based format
   - Update template examples and documentation
   - Remove fallback rendering logic

### Phase 4: Validation and Testing (Week 7-8)

1. **Comprehensive Testing**

   - Run full test suite with removed code
   - Performance testing to ensure no regressions
   - Integration testing with external dependencies

2. **Update Documentation**
   - Remove references to deprecated features
   - Update API documentation
   - Create migration guides for external users

## Safety Considerations

### Before Any Cleanup

1. **Baseline Testing**

   ```bash
   # Run full test suite to establish baseline
   pytest -q --cov=src tests/

   # Run integration tests
   pytest -q tests/integration/

   # Run end-to-end tests
   pytest -q tests/e2e/
   ```

2. **Dependency Analysis**

   - Identify external consumers of APIs
   - Check for undocumented dependencies
   - Review internal code usage patterns

3. **Performance Baseline**
   - Benchmark current performance
   - Document response times and resource usage
   - Create regression test suite

### During Cleanup

1. **Incremental Changes**

   - Make small, targeted changes
   - Test after each change
   - Use feature flags for gradual rollout

2. **Comprehensive Testing**

   - Unit tests for each changed component
   - Integration tests for interface changes
   - End-to-end tests for workflow changes

3. **Monitoring and Rollback**
   - Monitor error rates and performance
   - Have rollback plan ready
   - Document all changes thoroughly

### After Cleanup

1. **Validation**

   - Full test suite passes
   - Performance meets or exceeds baseline
   - No increase in error rates

2. **Documentation Updates**

   - Remove deprecated API documentation
   - Update examples and tutorials
   - Update changelog

3. **Communication**
   - Notify external API users of changes
   - Provide migration guides
   - Document breaking changes

## Conclusion

This legacy code cleanup will reduce technical debt, improve maintainability, and simplify the codebase while preserving essential stability mechanisms. The phased approach ensures safe removal of legacy code while maintaining system reliability.

**Expected Benefits:**

- Reduced complexity in configuration system
- Elimination of duplicate functionality
- Cleaner, more maintainable codebase
- Improved performance through simplified code paths
- Better testing coverage with fewer compatibility branches

**Risks Mitigated:**

- Comprehensive testing before and after changes
- Incremental implementation with rollback capability
- Preservation of essential stability patterns
- Clear documentation and communication plan

Following this guide will result in a cleaner, more maintainable codebase while preserving the robustness and reliability that are essential for production systems.

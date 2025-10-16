<!-- 6c1b9cae-f1d8-4654-a3c8-d98a9eda689d 8371a406-7aa3-4663-99ee-d155693c8664 -->
# MAIE Codebase Cleanup Implementation

## Current Progress (2025-10-15)

- Phase 1
  - Dead code removal: completed (removed `src/processors/chat/` including `template_manager.py`).
  - Import standardization: completed via pixi `style`; fixed source lints in `src/api/main.py`, `src/config/logging.py`, `src/processors/__init__.py`, `src/processors/asr/factory.py`, `src/processors/asr/chunkformer.py`, `src/processors/llm/processor.py`.
  - Validation: in progress. Majority tests pass; remaining failures are related to expected test behaviors (not cleanup):
    - Whisper version info missing `language` field in `get_version_info` (tests expect it).
    - Pipeline version metadata structure expects `versions["asr"]` and `versions["llm"]` keys.
    - Unit tests patch `apply_overrides_to_sampling` from `src.processors.llm.processor` (currently defined in `src.tooling.vllm_utils`).

- Phase 2
  - New config modules scaffolded: `src/config/base.py`, `src/config/development.py`, `src/config/production.py`.
  - Refactor of `src/config/settings.py` to load env-specific settings is next.

- Phase 3
  - Error utilities scaffolded: `src/api/errors.py` (ErrorResponse + ErrorCodes).
  - Automation scripts added: `scripts/detect_dead_code.py`, `scripts/validate_imports.py`, `scripts/check_documentation_overlap.py`.
  - Documentation consolidation mapping added: `docs/documentation_mapping.md`.

Next steps
- Finish Phase 1 validation by aligning code with test expectations:
  - Add `language` field to Whisper `get_version_info`.
  - Ensure pipeline version metadata exposes `versions["asr"]` and `versions["llm"]` keys.
  - Re-export or alias `apply_overrides_to_sampling` in `src/processors/llm/processor.py` for test patching.
- Proceed with Phase 2 config refactor in `src/config/settings.py`.

---

## Phase 1: High Priority Cleanup (Week 1-2)

### 1.1 Dead Code Removal

- Remove `src/processors/chat/template_manager.py` entirely
- Remove `src/processors/chat/__init__.py` (empty file)
- Remove `src/processors/chat/` directory
- Verify no references to ChatTemplateManager exist in codebase

**Files to remove:**

- `src/processors/chat/template_manager.py`
- `src/processors/chat/__init__.py`
- `src/processors/chat/` (directory)

### 1.2 Import Pattern Standardization

- Run `isort` on entire codebase with black profile
- Run `black` for consistent formatting
- Run `ruff check --fix` for linting
- Verify no unused imports with `ruff --select F401`

**Commands to execute:**

```bash
isort src/ tests/ --profile black --skip-gitignore
black src/ tests/
ruff check src/ tests/ --fix
ruff check src/ tests/ --select F401
```

### 1.3 Validation

- Run full test suite to ensure no breakage (in progress)
- Check for any remaining references to deprecated code (none found)

---

## Phase 2: Medium Priority Improvements (Week 3-4)

### 2.1 Configuration Simplification

Create environment-specific configuration files to reduce complexity:

**New structure in `src/config/`:**

- `base.py` - Base configuration shared across environments
- `development.py` - Development-specific overrides
- `production.py` - Production-specific overrides
- `settings.py` - Update to use environment-based loading

**Configuration consolidation:**

- Reduce logging environment variables from 8 to 3-4 core variables
- Simplify validation hierarchy
- Add comprehensive configuration documentation

### 2.2 Documentation Structure Proposal

Create a consolidated documentation structure proposal without actually merging files:

**Target structure:**

```
docs/
├── guide.md (consolidated best practices)
├── implementation/
│   ├── streaming.md (merge STREAMING_IMPLEMENTATION.md + parts of MIGRATION_COMPLETE.md)
│   ├── migration.md (merge MIGRATION_COMPLETE.md + IMPLEMENTATION_SUMMARY.md)
│   ├── configuration.md (configuration best practices)
│   └── testing.md (testing strategies from multiple sources)
├── reference/
│   ├── api.md (API reference)
│   ├── error-handling.md (error handling patterns)
│   └── troubleshooting.md (troubleshooting guide)
└── archive/ (move completed/historical docs)
    ├── TORCH_TEST_FIX_COMPLETED.md
    ├── LOGURU_IMPLEMENTATION_SUMMARY.md
    └── FIXES_COMPLETED.md
```

**Create documentation mapping file:**

- Document which content from which files should merge
- Identify overlap percentages
- Provide consolidation recommendations

---

## Phase 3: Long-term Maintenance (Week 5-6)

### 3.1 Error Handling Standardization

Define and implement consistent error handling patterns across the codebase:

```python
try:
    result = operation()
except SpecificException as e:
    logger.bind(
        operation="operation_name",
        error_code="ERROR_CODE",
        context=context_data
    ).error("Operation failed: {error}", error=str(e))
    
    return {
        "error": {
            "code": "ERROR_CODE",
            "message": "Human-readable message",
            "details": {} if settings.environment == "production" else str(e)
        }
    }
```

**Files to update (20+ files):**

- All API route handlers in `src/api/routes.py`
- Worker pipeline functions in `src/worker/pipeline.py`
- Processor implementations in `src/processors/`
- Update `handle_processing_error` to follow new pattern

**Create error taxonomy:**

- Define error codes (e.g., AUDIO_DECODE_ERROR, MODEL_LOAD_ERROR, etc.)
- Document error categories and handling strategies
- Create `src/api/errors.py` for standardized error responses

### 3.2 Automation Scripts

Create scripts for ongoing code quality maintenance:

**Scripts to create in `scripts/`:**

1. `scripts/detect_dead_code.py`
   - Use `deadcode` library (already in dev dependencies)
   - Detect unused functions and imports
   - Exit with error if dead code found

2. `scripts/validate_imports.py`
   - Check import ordering consistency
   - Verify no in-function imports (except intentional ones)
   - Validate import patterns

3. `scripts/check_documentation_overlap.py`
   - Analyze documentation files for content similarity
   - Report overlap percentages
   - Suggest consolidation opportunities

4. `docs/documentation_mapping.md`
   - Create detailed mapping of current documentation structure
   - Show proposed consolidation
   - List specific sections to merge

### 3.3 Testing and Validation

- Run complete test suite after each phase
- Verify no performance regression
- Ensure all imports resolve correctly
- Validate configuration loading in all environments

---

## Success Metrics

**Code Quality:**

- Zero dead code detected by vulture/deadcode
- 100% import pattern consistency
- Zero unused imports (F401)
- All tests passing

**Configuration:**

- Simplified environment variable count
- Clear environment-specific configurations
- Comprehensive configuration documentation

**Error Handling:**

- Consistent error response format across all endpoints
- Structured error logging with proper context
- Clear error taxonomy and documentation

**Documentation:**

- Proposed structure with clear consolidation mapping
- Reduced overlap (mapped for future reduction to <10%)
- Clear separation of concerns (implementation vs reference vs archive)

### To-dos

- [x] Remove deprecated chat module and template_manager.py
- [x] Standardize imports using isort, black, and ruff
- [ ] Run tests and verify no breakage from Phase 1 cleanup
- [x] Create environment-specific configuration files (base.py, development.py, production.py)
- [ ] Refactor settings.py to use environment-based loading
- [x] Create documentation consolidation mapping and proposed structure
- [ ] Define error codes and create error taxonomy documentation
- [x] Create src/api/errors.py with standardized error response classes
- [ ] Update API routes and worker pipeline to use standardized error handling
- [x] Create scripts/detect_dead_code.py using deadcode library
- [x] Create scripts/validate_imports.py for import pattern validation
- [x] Create scripts/check_documentation_overlap.py for documentation analysis
- [ ] Run complete test suite and validate all automation scripts




# Naming Convention Resolution

## ASR Factory Naming

### Issue

The documentation and codebase had inconsistent naming:

- **TDD.md**: Uses `ASRFactory` (15 occurrences)
- **guide.md**: Mixed use of `ASRFactory` and `ASRProcessorFactory`
- **Implementation**: Was using `ASRProcessorFactory`

### Resolution (October 15, 2025)

**Renamed class to `ASRFactory`** to match documentation consistently.

```python
# In src/processors/asr/factory.py
class ASRFactory:
    """Factory class for creating ASR backend instances."""
    # ... implementation ...
```

### Current State

Single, consistent name throughout codebase:

- ✅ `ASRFactory` - The one and only class name

### Exports

```python
from src.processors.asr.factory import ASRFactory

__all__ = [
    "ASRFactory",
    "ASRBackend",
    "WhisperBackend",
    "ChunkFormerBackend"
]
```

### Usage

```python
# Import from factory module
from src.processors.asr.factory import ASRFactory
backend = ASRFactory.create("whisper")

# Import from module (recommended)
from src.processors.asr import ASRFactory
backend = ASRFactory.create("whisper")
```

### Files Updated (Completed October 15, 2025)

1. ✅ `src/processors/asr/factory.py` - Class named `ASRFactory`
2. ✅ `src/processors/asr/__init__.py` - Exports `ASRFactory`
3. ✅ `src/worker/main.py` - Uses `ASRFactory`
4. ✅ `src/worker/pipeline.py` - Uses `ASRFactory`
5. ✅ `tests/worker/conftest.py` - Mock uses `ASRFactory`
6. ✅ `tests/unit/test_asr_factory.py` - All tests updated to use `ASRFactory`
7. ✅ `tests/unit/test_audio_preprocessor.py` - Mock updated to use `ASRFactory`
8. ✅ `docs/guide.md` - All references updated to `ASRFactory`

### Test Results

All 8 ASRFactory unit tests pass:

- ✅ test_register_backend
- ✅ test_register_backend_and_overwrite
- ✅ test_create_backend
- ✅ test_create_unknown_backend
- ✅ test_create_backend_init_error
- ✅ test_create_with_audio_processing_success
- ✅ test_registering_non_class_backend_results_in_type_error_on_creation
- ✅ test_smoke_factory_creation_fast

### Consistency

All documentation examples in TDD.md and guide.md now match the implementation exactly.

# PyTorch Test Environment Fix Plan

## Root Cause Analysis

### Problem

Tests fail with: `RuntimeError: function '_has_torch_function' already has a docstring`

### Root Cause

**PyTorch cannot be reloaded after initial import.** Each test in `test_pipeline_helpers.py` contains:

```python
def test_something(self):
    from src.worker.pipeline import _update_status  # ← Imports torch EVERY test
```

This causes:

1. Test 1: `import src.worker.pipeline` → `import torch` (✅ works)
2. Test 2: `import src.worker.pipeline` → tries to reload `torch` (❌ fails)

When torch is reloaded, internal C++ libraries throw errors:

- `RuntimeError: function '_has_torch_function' already has a docstring`
- `RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace triton`

### Evidence

```bash
$ python -c "import torch; import importlib; importlib.reload(torch)"
RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace triton
```

PyTorch's C++ bindings register global state that cannot be re-initialized.

---

## Solution: Fix Import Strategy

### Option 1: Module-Level Imports (RECOMMENDED) ⭐

Move imports to **module level** instead of inside test functions.

**Implementation:**

```python
# tests/worker/test_pipeline_helpers.py
import pytest
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock

from src.api.schemas import TaskStatus
from src.worker.pipeline import _update_status, _calculate_edit_rate  # ← Move here


class TestUpdateStatus:
    def test_update_status_basic(self, mock_redis_sync):
        # No import needed - already available
        task_id = "test-task-123"
        ...
```

**Pros:**

- ✅ Standard Python/pytest pattern
- ✅ Imports happen once per test session
- ✅ No torch reload issues
- ✅ Faster test execution (no repeated imports)
- ✅ Better IDE support

**Cons:**

- None

**Effort:** 5 minutes (simple find/replace)

---

### Option 2: Pytest Fixture for Functions

Create a fixture that returns the functions.

**Implementation:**

```python
# tests/worker/conftest.py
import pytest

@pytest.fixture(scope="session")
def pipeline_helpers():
    """Import pipeline helpers once per test session."""
    from src.worker.pipeline import _update_status, _calculate_edit_rate
    return {
        'update_status': _update_status,
        'calculate_edit_rate': _calculate_edit_rate
    }
```

**Usage:**

```python
def test_update_status_basic(self, mock_redis_sync, pipeline_helpers):
    update_status = pipeline_helpers['update_status']
    update_status(mock_redis_sync, task_key, TaskStatus.PROCESSING_ASR)
```

**Pros:**

- ✅ Imports once per session
- ✅ Explicit dependency injection
- ✅ Can add setup/teardown logic

**Cons:**

- ❌ More verbose test code
- ❌ Extra fixture parameter in every test

**Effort:** 15 minutes

---

### Option 3: Mock Torch Entirely (For Unit Tests)

Don't import torch at all in unit tests.

**Implementation:**

```python
# tests/worker/test_pipeline_helpers.py
import sys
from unittest.mock import MagicMock

# Mock torch before any imports
sys.modules['torch'] = MagicMock()

from src.worker.pipeline import _update_status, _calculate_edit_rate
```

**Pros:**

- ✅ Unit tests don't need real torch
- ✅ Faster test execution
- ✅ Works on systems without CUDA

**Cons:**

- ❌ Doesn't test torch-dependent code
- ❌ May hide integration issues
- ❌ Need separate integration tests

**Effort:** 30 minutes (requires test restructuring)

---

### Option 4: pytest-forked Plugin

Run each test in a separate process.

**Installation:**

```bash
pixi add --pypi pytest-forked
```

**Usage:**

```bash
pytest --forked tests/worker/test_pipeline_helpers.py
```

**Pros:**

- ✅ Complete test isolation
- ✅ No code changes needed
- ✅ Handles any problematic imports

**Cons:**

- ❌ Slower (process spawning overhead)
- ❌ Extra dependency
- ❌ May not work on all platforms

**Effort:** 5 minutes (add dependency + update test command)

---

### Option 5: Separate Test Files by Import

Split tests that import torch vs don't import torch.

**Structure:**

```
tests/worker/
├── test_pipeline_helpers_stateless.py  # No torch imports (_calculate_edit_rate)
├── test_pipeline_helpers_redis.py      # No torch imports (_update_status)
└── test_pipeline_integration.py        # Real torch imports
```

**Pros:**

- ✅ Clean separation of concerns
- ✅ Fast unit tests without torch
- ✅ Slower integration tests isolated

**Cons:**

- ❌ More test files to maintain
- ❌ Reorganization needed

**Effort:** 45 minutes

---

## Recommended Solution: Option 1 (Module-Level Imports)

### Why Option 1?

1. **Standard Practice**: This is how pytest tests are normally written
2. **Simple Fix**: Just move 17 import statements
3. **No Dependencies**: No new packages needed
4. **Performance**: Tests run faster (imports once)
5. **Maintainability**: Clearer code structure

### Implementation Steps

#### Step 1: Update test file

```python
# tests/worker/test_pipeline_helpers.py
"""
Unit tests for worker pipeline helper functions.
"""
import pytest
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock

from src.api.schemas import TaskStatus
# Import once at module level
from src.worker.pipeline import _update_status, _calculate_edit_rate


class TestUpdateStatus:
    def test_update_status_basic(self, mock_redis_sync):
        # Remove: from src.worker.pipeline import _update_status
        task_id = "test-task-123"
        task_key = f"task:{task_id}"

        _update_status(mock_redis_sync, task_key, TaskStatus.PROCESSING_ASR)
        ...
```

#### Step 2: Remove all in-function imports

Delete these lines from all test functions:

- `from src.worker.pipeline import _update_status`
- `from src.worker.pipeline import _calculate_edit_rate`

#### Step 3: Run tests

```bash
pixi run test-debug tests/worker/test_pipeline_helpers.py -v
```

### Expected Result

All 17 tests should pass because:

- Torch imports only once (at module load)
- No reload attempts
- Functions available in all test methods

---

## Alternative: Quick Test with Option 4

If you want immediate verification without code changes:

```bash
# Install pytest-forked
pixi add --pypi pytest-forked

# Run with forked processes
pixi run pytest --forked tests/worker/test_pipeline_helpers.py -v
```

This proves the tests work correctly when torch reload is avoided.

---

## Long-Term Strategy

### For Unit Tests

- Use **Option 1** (module-level imports) for all unit tests
- Keep tests focused on logic, not torch integration

### For Integration Tests

- Create separate integration test file
- Mark with `@pytest.mark.integration`
- Test real torch/GPU functionality
- Run less frequently (CI only, or manually)

### Test Structure

```
tests/
├── unit/
│   └── test_pipeline_helpers.py        # Module-level imports, no torch reload
└── integration/
    └── test_pipeline_full.py           # Real torch, real GPU, marked @integration
```

---

## Verification Steps

After implementing Option 1:

1. **Clear cache:**

   ```bash
   rm -rf .pytest_cache __pycache__ src/__pycache__ src/*/__pycache__
   ```

2. **Run tests:**

   ```bash
   pixi run test-debug tests/worker/test_pipeline_helpers.py -v
   ```

3. **Expected output:**

   ```
   17 tests collected
   test_update_status_basic PASSED
   test_update_status_with_details PASSED
   ...
   17 passed in 2.5s
   ```

4. **Run full test suite:**
   ```bash
   pixi run test
   ```

---

## Summary

| Option                         | Effort | Pros                   | Best For           |
| ------------------------------ | ------ | ---------------------- | ------------------ |
| **1. Module-level imports** ⭐ | 5 min  | Standard, fast, simple | **All projects**   |
| 2. Fixture                     | 15 min | Explicit, flexible     | Special test setup |
| 3. Mock torch                  | 30 min | Fast, no CUDA needed   | Pure unit tests    |
| 4. pytest-forked               | 5 min  | No code changes        | Quick verification |
| 5. Split files                 | 45 min | Clean separation       | Large test suites  |

**Action:** Implement Option 1 immediately (5 minutes fix)

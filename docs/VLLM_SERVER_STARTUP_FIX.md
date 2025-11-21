# vLLM Server Startup Script Fix

**Date:** 2025-11-20  
**Issue:** `api_server.py: error: unrecognized arguments` when running `./scripts/start-vllm-server.sh`

## Problem Description

The vLLM server startup script was failing with an error indicating unrecognized arguments that appeared to be log messages:

```
api_server.py: error: unrecognized arguments: 15:48:30.397 | INFO | config.logging:configure_logging:184 | ...
```

## Root Cause

The issue was in `scripts/vllm_server_config.py`:

1. **Logging initialization**: The script called `configure_logging()` which outputted initialization messages to stdout
2. **Logger usage**: The `print_config_info()` function used `logger.info()` which also went to stdout
3. **Shell script capture**: The bash script `start-vllm-server.sh` captured ALL stdout output and passed it as arguments to vLLM

This caused log messages to be interpreted as command-line arguments, resulting in the error.

## Solution

Two changes were made to `/home/cetech/maie/scripts/vllm_server_config.py`:

### Change 1: Remove Logging Initialization (Lines 17-23)

**Before:**
```python
# Import MAIE configuration and logging
from src.config import settings
from src.config.logging import configure_logging, get_module_logger

# Configure logging using MAIE's loguru setup
configure_logging()
logger = get_module_logger(__name__)
```

**After:**
```python
# Import MAIE configuration (no logging to avoid stdout pollution)
from src.config import settings
```

**Rationale:** The script doesn't need the full logging infrastructure since it's just a simple configuration helper.

### Change 2: Use stderr for Info Messages (Lines 74-99)

**Before:**
```python
def print_config_info(model_type: str = "enhance"):
    """Print configuration information for debugging."""
    # ... config loading ...
    
    logger.info("=" * 70)
    logger.info("vLLM Server Configuration (from MAIE settings)")
    logger.info(f"Model Type: {model_type}")
    # ... more logger.info() calls ...
```

**After:**
```python
def print_config_info(model_type: str = "enhance"):
    """Print configuration information for debugging."""
    # ... config loading ...
    
    # Print to stderr to avoid mixing with vLLM args on stdout
    print("=" * 70, file=sys.stderr)
    print("vLLM Server Configuration (from MAIE settings)", file=sys.stderr)
    print(f"Model Type: {model_type}", file=sys.stderr)
    # ... more print(..., file=sys.stderr) calls ...
```

**Rationale:** Configuration info should go to stderr (for human viewing) while vLLM arguments go to stdout (for script capture).

## Verification

Test that the fix works:

```bash
# Only vLLM arguments should appear (no log messages)
python scripts/vllm_server_config.py --model-type enhance --show-config 2>/dev/null

# Expected output:
# --host 0.0.0.0 --port 8001 --model data/models/qwen3-4b-instruct-2507-awq --gpu-memory-utilization 0.9 --max-model-len 32768
```

Run the full startup script:

```bash
./scripts/start-vllm-server.sh

# Should successfully start vLLM server without argument errors
# Wait for: "Application startup complete"
```

## Documentation Updates

Updated the following documentation files:

1. **`docs/LLM_BACKEND_CONFIGURATION.md`**
   - Added troubleshooting section for vLLM server startup script errors
   - Included verification steps

2. **`docs/E2E_TESTING_HYBRID_QUICKSTART.md`**
   - Added optional Step 2b for running vLLM server mode
   - Added troubleshooting section for vLLM server startup errors
   - Updated terminal numbering to account for optional vLLM server

## Key Takeaways

1. **Separate stdout and stderr**: When scripts output both data (for capture) and info (for humans), use stdout for data and stderr for info
2. **Avoid logging in utility scripts**: Simple configuration scripts don't need full logging infrastructure
3. **Test script output**: Always verify what goes to stdout vs stderr when scripts are used in pipelines

## Related Files

- `scripts/vllm_server_config.py` - Configuration helper script (fixed)
- `scripts/start-vllm-server.sh` - Bash launcher script (unchanged)
- `docs/LLM_BACKEND_CONFIGURATION.md` - Backend configuration guide (updated)
- `docs/E2E_TESTING_HYBRID_QUICKSTART.md` - E2E testing guide (updated)

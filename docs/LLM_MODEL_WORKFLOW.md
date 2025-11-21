# LLM Workflow with Different Enhancement and Summary Models

## Overview

MAIE supports using **different models** for text enhancement and summarization tasks. Here's how the workflow handles this:

## Current Configuration Structure

### Two Separate Model Configurations

**Enhancement Model** (`LlmEnhanceSettings`):
```python
# Default: data/models/qwen3-4b-instruct-2507-awq (local)
LLM_ENHANCE_MODEL=data/models/qwen3-4b-instruct-2507-awq
LLM_ENHANCE_GPU_MEMORY_UTILIZATION=0.9
LLM_ENHANCE_MAX_MODEL_LEN=32768
LLM_ENHANCE_TEMPERATURE=0.5
```

**Summary Model** (`LlmSumSettings`):
```python
# Default: cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit (HuggingFace)
LLM_SUM_MODEL=cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit
LLM_SUM_GPU_MEMORY_UTILIZATION=0.9
LLM_SUM_MAX_MODEL_LEN=32768
LLM_SUM_TEMPERATURE=0.7
```

## Workflow Scenarios

### Scenario 1: Local vLLM Mode (DEFAULT)

**How it works:**
1. Worker creates `LLMProcessor()` instance
2. Processor uses `LlmEnhanceSettings.model` for initialization
3. **Single model loaded** - enhancement settings take precedence
4. Both enhancement and summary tasks use the **same loaded model**
5. Only **sampling parameters differ** (temperature, top_p, etc.)

**Pipeline flow:**
```
Audio Processing
  ↓
ASR (Whisper/ChunkFormer)
  ↓
LLMProcessor._load_model()  ← Uses LlmEnhanceSettings.model
  ↓
├─ enhance_text() ← Uses LlmEnhanceSettings sampling params
└─ generate_summary() ← Uses LlmSumSettings sampling params
  ↓
Unload model
```

**Key Point:** In local mode, you can only use ONE model at a time. The enhancement model path is used, but summary sampling parameters (temperature, etc.) are still applied.

### Scenario 2: vLLM Server Mode (RECOMMENDED)

**How it works:**
1. You can run **TWO separate vLLM servers** (one for each model)
2. Or run **ONE server** and specify which model to use

**Option A: Single Server (Same Model)**
```bash
# Start one server
./scripts/start-vllm-server.sh enhance

# Configure MAIE to use it for both tasks
LLM_BACKEND=vllm_server
LLM_SERVER__BASE_URL=http://localhost:8001/v1
LLM_SERVER__MODEL_ENHANCE=data/models/qwen3-4b-instruct-2507-awq
LLM_SERVER__MODEL_SUMMARY=data/models/qwen3-4b-instruct-2507-awq
```

**Option B: Two Servers (Different Models)**
```bash
# Start enhancement server on port 8001
VLLM_SERVER_PORT=8001 ./scripts/start-vllm-server.sh enhance

# Start summary server on port 8002
VLLM_SERVER_PORT=8002 ./scripts/start-vllm-server.sh summary

# Configure MAIE - but this requires code changes!
# Current implementation only supports ONE server URL
```

**Current Limitation:** The current `VllmServerClient` implementation uses a **single base_url**, so you can only connect to one server at a time.

## Code Implementation Details

### Local vLLM Mode

In `src/processors/llm/processor.py`:

```python
def _load_model(self, **kwargs):
    # Uses LlmEnhanceSettings for model initialization
    llm_args = {
        "model": model_name,  # From LlmEnhanceSettings.model
        "gpu_memory_utilization": settings.llm_enhance.gpu_memory_utilization,
        "max_model_len": settings.llm_enhance.max_model_len,
        # ... other enhancement settings
    }
    self.model = LLM(**llm_args)
```

```python
def execute(self, text: str, **kwargs):
    # Select config based on task
    env_config = (
        self.env_config_enhancement  # LlmEnhanceSettings
        if task == "enhancement"
        else self.env_config_summary  # LlmSumSettings
    )
    
    # Only sampling params differ, not the model!
    runtime_config = GenerationConfig(
        temperature=...,  # From respective config
        top_p=...,
        # ...
    )
```

### vLLM Server Mode

In `src/processors/llm/processor.py`:

```python
def _load_model(self, **kwargs):
    if settings.llm_backend == LlmBackendType.VLLM_SERVER:
        self.client = VllmServerClient(
            base_url=settings.llm_server.base_url,  # Single URL
            model_name=settings.llm_server.model_enhance  # Uses enhance model
        )
```

**Issue:** The `model_enhance` is used for initialization, but `model_summary` is never used in the current implementation!

## Recommendations

### If You Want Different Models:

**Option 1: Use Two Separate Workers (Recommended)**
```bash
# Worker 1: Enhancement only
# Configure with enhancement model in vLLM server

# Worker 2: Summary only  
# Configure with summary model in vLLM server
```

**Option 2: Modify Code to Support Multiple Servers**
You would need to modify `LLMProcessor` to:
1. Check the task type in `execute()`
2. Use different `VllmServerClient` instances for enhancement vs summary
3. Connect to different server URLs based on task

### If You Want Same Model (Current Design):

**Option: Use One Server**
```bash
# Start one vLLM server with your preferred model
./scripts/start-vllm-server.sh enhance

# Configure MAIE
LLM_BACKEND=vllm_server
LLM_SERVER__BASE_URL=http://localhost:8001/v1
LLM_SERVER__MODEL_ENHANCE=data/models/qwen3-4b-instruct-2507-awq
LLM_SERVER__MODEL_SUMMARY=data/models/qwen3-4b-instruct-2507-awq  # Same model
```

## Summary

| Mode | Models Used | How It Works |
|------|-------------|--------------|
| **Local vLLM** | 1 model | Uses `LlmEnhanceSettings.model`, different sampling params per task |
| **vLLM Server** | 1 model | Connects to one server, uses `model_enhance` setting |
| **Future: Multi-Server** | 2+ models | Would require code changes to support task-based routing |

## Current Design Philosophy

The current implementation assumes:
1. **Same model** for both enhancement and summary
2. **Different sampling parameters** for different tasks (temperature, top_p, etc.)
3. This is efficient because:
   - No model swapping needed
   - Fast task switching (just parameter changes)
   - Lower GPU memory usage

## If You Need Different Models

Let me know if you want me to implement support for different models per task! This would require:
1. Modifying `LLMProcessor` to maintain task-specific clients
2. Adding task-based routing logic
3. Supporting multiple vLLM server URLs

# Migration Guide - Dual-Endpoint LLM Configuration

## Overview

This guide helps you migrate from the old single-endpoint vLLM server configuration to the new dual-endpoint structure that supports separate servers for enhancement and summary tasks.

## ⚠️ Breaking Changes

**Old environment variables are NO LONGER SUPPORTED:**
- `LLM_SERVER__BASE_URL` → replaced by `LLM_SERVER__ENHANCE_BASE_URL`
- `LLM_SERVER__API_KEY` → replaced by `LLM_SERVER__ENHANCE_API_KEY`
- `LLM_SERVER__MODEL_ENHANCE` → replaced by `LLM_SERVER__ENHANCE_MODEL_NAME`
- `LLM_SERVER__MODEL_SUMMARY` → replaced by `LLM_SERVER__SUMMARY_MODEL_NAME`

## Migration Steps

### Step 1: Update Environment Variables

**Old Configuration:**
```bash
LLM_BACKEND=vllm_server
LLM_SERVER__BASE_URL=http://localhost:8000/v1
LLM_SERVER__API_KEY=your-key
LLM_SERVER__MODEL_ENHANCE=Qwen/Qwen2.5-3B-Instruct
LLM_SERVER__MODEL_SUMMARY=Qwen/Qwen2.5-3B-Instruct
```

**New Configuration (Single Server):**
```bash
LLM_BACKEND=vllm_server
LLM_SERVER__ENHANCE_BASE_URL=http://localhost:8001/v1
LLM_SERVER__ENHANCE_API_KEY=your-key
LLM_SERVER__ENHANCE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
# Summary endpoint defaults to enhance endpoint if not set
```

**New Configuration (Dual Servers):**
```bash
LLM_BACKEND=vllm_server

# Enhancement server
LLM_SERVER__ENHANCE_BASE_URL=http://localhost:8001/v1
LLM_SERVER__ENHANCE_API_KEY=your-key-1
LLM_SERVER__ENHANCE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct

# Summary server (different endpoint)

LLM_SERVER__SUMMARY_BASE_URL=http://localhost:8002/v1
LLM_SERVER__SUMMARY_API_KEY=your-key-2
LLM_SERVER__SUMMARY_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

### Step 2: Update Server Launch (if using local deployment)

**Single Server** (both tasks use same server):
```bash
# Start one vLLM server
./scripts/start-vllm-server.sh enhance
```

**Dual Servers** (separate optimized servers):
```bash
# Enhancement server (larger model, more GPU memory)
VLLM_SERVER_PORT=8001 ./scripts/start-vllm-server.sh enhance

# Summary server (smaller/faster model)
VLLM_SERVER_PORT=8002 ./scripts/start-vllm-server.sh summary
```

### Step 3: Restart Workers

After updating configuration:
```bash
# Stop existing workers
./scripts/kill_maie.sh

# Start workers with new config
pixi run worker
```

## Configuration Examples

### Local vLLM Mode (No Changes Required)

If you're using `LLM_BACKEND=local_vllm`, no migration is needed:
```bash
LLM_BACKEND=local_vllm
# Uses LlmEnhanceSettings and LlmSumSettings as before
```

### Single vLLM Server (Minimal Migration)

Use one server for both tasks:
```bash
LLM_BACKEND=vllm_server
LLM_SERVER__ENHANCE_BASE_URL=http://localhost:8001/v1
# Summary endpoint auto-defaults to enhancement endpoint
```

### Dual vLLM Servers (Advanced)

Run separate optimized servers:
```bash
LLM_BACKEND=vllm_server

# Large model for enhancement (better quality)
LLM_SERVER__ENHANCE_BASE_URL=http://gpu1.internal:8001/v1
LLM_SERVER__ENHANCE_MODEL_NAME=Qwen2.5-14B-Instruct-AWQ

# Small model for summary (faster)
LLM_SERVER__SUMMARY_BASE_URL=http://gpu2.internal:8002/v1
LLM_SERVER__SUMMARY_MODEL_NAME=Qwen2.5-1.5B-Instruct
```

## Benefits of Dual-Endpoint Setup

1. **Task-Specific Optimization:**
   - Enhancement: Larger model for better quality
   - Summary: Smaller/faster model for speed

2. **Resource Efficiency:**
   - Different GPU allocation per task
   - Different quantization levels

3. **Scalability:**
   - Scale enhancement and summary independently
   - Load balance across multiple GPUs

## Troubleshooting

### Configuration Not Loading

**Error:** `Field required` or `Validation error`

**Solution:** Ensure you've updated ALL old variable names:
```bash
# Check your .env file
grep "LLM_SERVER__BASE_URL" .env  # Should return nothing
grep "LLM_SERVER__ENHANCE_BASE_URL" .env  # Should find your config
```

### Summary Endpoint DefaultPath Issues

If summary tasks are going to the wrong endpoint:

1. Check if `LLM_SERVER__SUMMARY_BASE_URL` is set
2. If not set, it defaults to `LLM_SERVER__ENHANCE_BASE_URL`
3. Set explicitly if you want different endpoints

### Worker Not Finding Client

**Error:** `Client not available for task: summary`

**Solution:** Verify model loading succeeded:
```bash
# Check worker logs
tail -f logs/app.log | grep  "LLM"
```

## Rollback Plan

If you need to rollback:

1. This feature requires code changes - cannot rollback via config only
2. Revert to previous git commit:
   ```bash
   git log --oneline | grep "LLM"
   git revert <commit-hash>
   ```

## Support

For issues or questions:
- Check logs: `logs/app.log` and `logs/errors.log`
- Review configuration: `docs/LLM_BACKEND_CONFIGURATION.md`
- See examples: `README.md` LLM Backend section

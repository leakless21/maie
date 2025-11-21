# LLM Backend Configuration Guide

## Overview

MAIE supports two LLM backend modes for text enhancement and summarizatiTwo backend modes with flexible endpoint configuration:

1. **Local vLLM** (`LLM_BACKEND=local_vllm`) - In-process model loading
   - Single model loaded into GPU memory
   - Both enhancement and summary tasks use same model
   - Different sampling parameters per task type
   
2. **vLLM Server** (`LLM_BACKEND=vllm_server`) - Remote server mode
   - **Single Server**: Both tasks use one endpoint
   - **Dual Server**: Separate optimized endpoints per task

## Backend Selection in `.env`

### Local vLLM (Default)

The traditional mode where vLLM is loaded and unloaded with each job.

**Advantages:**
- Simple setup, no external dependencies
- Complete control over model lifecycle
- Automatic GPU memory management

**Disadvantages:**
- Model loading overhead on each job (~10-30 seconds)
- GPU memory fragmentation over time
- Higher per-job latency

**Configuration:**

```bash
LLM_BACKEND=local_vllm
```

### vLLM Server

Connects to a persistent vLLM server that keeps the model loaded in memory.

**Advantages:**
- No model loading overhead per job
- Consistent GPU memory usage
- Lower per-job latency
- Better resource isolation
- Easier horizontal scaling

**Disadvantages:**
- Requires separate vLLM server deployment
- Additional network latency
- More complex infrastructure

**Configuration:**

```bash
LLM_BACKEND=vllm_server
LLM_SERVER__BASE_URL=http://localhost:8000/v1
LLM_SERVER__API_KEY=EMPTY
LLM_SERVER__MODEL_ENHANCE=Qwen/Qwen2.5-3B-Instruct
LLM_SERVER__MODEL_SUMMARY=Qwen/Qwen2.5-3B-Instruct
LLM_SERVER__REQUEST_TIMEOUT_SECONDS=120
```

## vLLM Server Deployment

### Using MAIE's Launch Script (Recommended for Local Deployment)

MAIE includes a script to launch vLLM server with settings automatically read from your MAIE configuration:

```bash
# Launch vLLM server using enhancement model config
./scripts/start-vllm-server.sh

# Or use summary model config
./scripts/start-vllm-server.sh summary
```

The script will:
- Read model path from `LlmEnhanceSettings.model` (or `LlmSumSettings.model`)
- Use GPU settings from your `.env` configuration
- Launch on port 8001 by default (configurable via `VLLM_SERVER_PORT`)
- Auto-detect local vs HuggingFace models

**Configuration:**
```bash
# In .env file
VLLM_SERVER_PORT=8001
VLLM_SERVER_HOST=0.0.0.0
```

### Using Docker

```bash
docker run --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768
```

### Using vLLM CLI

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --port 8000
```

### Verification

Test the vLLM server is running:

```bash
curl http://localhost:8000/v1/models
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `local_vllm` | Backend selection: `local_vllm` or `vllm_server` |
| `LLM_SERVER__BASE_URL` | `http://localhost:8000/v1` | vLLM server OpenAI-compatible API endpoint |
| `LLM_SERVER__API_KEY` | `EMPTY` | API key for vLLM server (use `EMPTY` if no auth) |
| `LLM_SERVER__MODEL_ENHANCE` | - | Model name for text enhancement |
| `LLM_SERVER__MODEL_SUMMARY` | - | Model name for summarization |
| `LLM_SERVER__REQUEST_TIMEOUT_SECONDS` | `120` | HTTP request timeout |

### Model Configuration

When using `vllm_server` backend:
- `LLM_ENHANCE_MODEL` and `LLM_SUM_MODEL` are ignored
- Use `LLM_SERVER__MODEL_ENHANCE` and `LLM_SERVER__MODEL_SUMMARY` instead
- Sampling parameters (temperature, top_p, etc.) are still used from `LLM_ENHANCE_*` and `LLM_SUM_*`

## Migration Guide

### From Local vLLM to vLLM Server

1. **Deploy vLLM Server:**
   ```bash
   docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
     --model cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit
   ```

2. **Update Configuration:**
   ```bash
export LLM_BACKEND=vllm_server

# Enhancement endpoint (required)
export LLM_SERVER__ENHANCE_BASE_URL=http://localhost:8001/v1
export LLM_SERVER__ENHANCE_MODEL_NAME=data/models/qwen3-4b-instruct-2507-awq

# Summary endpoint (optional - defaults to enhancement endpoint if not specified)
export LLM_SERVER__SUMMARY_BASE_URL=http://localhost:8002/v1
export LLM_SERVER__SUMMARY_MODEL_NAME=data/models/qwen3-1.5b-instruct-awq
```

### Single Server Configuration

Use one vLLM server for both enhancement and summary tasks:

```bash
export LLM_BACKEND=vllm_server
export LLM_SERVER__ENHANCE_BASE_URL=http://localhost:8001/v1
# Summary automatically uses enhancement endpoint
```

### Dual Server Configuration

Run separate servers for task-specific optimization:

```bash
export LLM_BACKEND=vllm_server

# Enhancement: larger model, more GPU memory
export LLM_SERVER__ENHANCE_BASE_URL=http://localhost:8001/v1
export LLM_SERVER__ENHANCE_MODEL_NAME=Qwen2.5-14B-Instruct-AWQ

# Summary: smaller/faster model
export LLM_SERVER__SUMMARY_BASE_URL=http://localhost:8002/v1
export LLM_SERVER__SUMMARY_MODEL_NAME=Qwen2.5-1.5B-Instruct
```
   ```

3. **Restart Workers:**
   ```bash
   docker-compose restart worker
   ```

4. **Verify:**
   ```bash
   # Submit a test job
   curl -X POST http://localhost:8000/v1/process \
     -H "X-API-Key: your-api-key" \
     -F "file=@test.wav" \
     -F "features=summary" \
     -F "template_id=meeting_notes_v1"
   ```

## Performance Considerations

### Local vLLM
- **First job latency**: 10-30 seconds (model loading)
- **Subsequent jobs**: 5-15 seconds (if model cached)
- **GPU memory**: Varies (loading/unloading)

### vLLM Server
- **First job latency**: 2-5 seconds (no loading)
- **Subsequent jobs**: 2-5 seconds (consistent)
- **GPU memory**: Stable (always loaded)

## Troubleshooting

### vLLM Server Startup Script Errors

**Error:** `api_server.py: error: unrecognized arguments: 15:48:30.397 | INFO | ...`

**Cause:** Log messages from the configuration script are being mixed with vLLM command-line arguments.

**Solution:** This was fixed in the latest version of `scripts/vllm_server_config.py`. If you encounter this error:
1. Update to the latest version of the script
2. The fix ensures log messages go to stderr while vLLM arguments go to stdout
3. If you modified the script, ensure `configure_logging()` is not called in `vllm_server_config.py`

**Verification:**
```bash
# Test the config script output (should only show vLLM arguments)
python scripts/vllm_server_config.py --model-type enhance --show-config 2>/dev/null

# Expected output (only vLLM args, no log messages):
# --host 0.0.0.0 --port 8001 --model data/models/... --gpu-memory-utilization 0.9 ...
```

### Connection Errors

**Error:** `Failed to connect to vLLM server`

**Solution:**
1. Verify vLLM server is running: `curl http://localhost:8000/v1/models`
2. Check `LLM_SERVER__BASE_URL` is correct
3. Ensure network connectivity between worker and server

### Model Not Found

**Error:** `Model not found on vLLM server`

**Solution:**
1. Verify model name matches server deployment
2. Check `LLM_SERVER__MODEL_ENHANCE` and `LLM_SERVER__MODEL_SUMMARY`
3. Restart vLLM server with correct model

### Timeout Errors

**Error:** `Request timeout`

**Solution:**
1. Increase `LLM_SERVER__REQUEST_TIMEOUT_SECONDS`
2. Check vLLM server performance
3. Verify GPU resources are sufficient

## Best Practices

1. **Production Deployment:**
   - Use `vllm_server` for production workloads
   - Deploy vLLM server on dedicated GPU instance
   - Use load balancer for multiple vLLM servers

2. **Development:**
   - Use `local_vllm` for development/testing
   - Simpler setup, easier debugging

3. **Monitoring:**
   - Monitor vLLM server health endpoint
   - Track request latency and throughput
   - Set up alerts for server downtime

4. **Security:**
   - Use API keys for vLLM server authentication
   - Restrict network access to vLLM server
   - Use HTTPS for production deployments

## Architecture

### Local vLLM Flow
```
Worker → Load vLLM → Process → Unload vLLM → Clear GPU Cache
```

### vLLM Server Flow
```
Worker → HTTP Request → vLLM Server (persistent) → HTTP Response
```

## See Also

- [LLM_PERSISTENCE_VLLM_SERVER_PLAN.md](LLM_PERSISTENCE_VLLM_SERVER_PLAN.md) - Implementation plan
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [README.md](../README.md) - Main documentation

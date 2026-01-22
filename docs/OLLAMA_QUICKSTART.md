# Ollama Quick Start Guide

## TL;DR - Get Running in 5 Steps

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Pull a Model (in another terminal)
```bash
ollama pull ministral-3:3b
```

### 3. Configure MAIE Environment
```bash
# Add to your .env or environment
export APP_LLM_BACKEND=VLLM_SERVER
export APP_LLM_SERVER__ENHANCE_BASE_URL=http://localhost:11434/v1
export APP_LLM_SERVER__ENHANCE_MODEL_NAME=ministral-3:3b
export APP_LLM_SERVER__SUMMARY_URL=http://localhost:11434/v1
export APP_LLM_SERVER__SUMMARY_MODEL_NAME=ministral-3:3b
export APP_LLM_SERVER__KEEP_ALIVE=0
export APP_LLM_SUM__STRUCTURED_OUTPUTS_ENABLED=false
```

### 4. Run MAIE
```bash
python main.py
```

### 5. Verify It Works
Check logs for:
- "Initializing vLLM server clients"
- "Successfully enhanced transcription"
- No OOM errors

## Configuration Deep Dive

| Setting | Value | Why |
|---------|-------|-----|
| `APP_LLM_BACKEND` | `VLLM_SERVER` | Routes to OpenAI-compatible API |
| `APP_LLM_SERVER__ENHANCE_BASE_URL` | `http://localhost:11434/v1` | Ollama server endpoint |
| `APP_LLM_SERVER__KEEP_ALIVE` | `0` | **Unload model immediately (sequential execution)** |
| `APP_LLM_SUM__STRUCTURED_OUTPUTS_ENABLED` | `false` | Ollama doesn't support structured outputs |

## For Jetson Nano (8GB VRAM)

### Expected Memory Usage
- Idle: ~500MB
- During ASR: ~2.5GB
- During LLM: ~3.5GB
- **Peak total**: 3.5GB (sequential, never both)

### Recommended Models
- ASR: `openai/whisper-small` (or smaller)
- LLM: `ministral-3:3b` or `qwen2.5-3b`

### Environment Setup
```bash
# Jetson Nano specific
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_NUM_GPU=1
export APP_LLM_SERVER__KEEP_ALIVE=0  # Critical for sequential execution
```

## What's Different from vLLM?

| Feature | vLLM Server | Ollama |
|---------|-------------|--------|
| API | OpenAI-compatible | OpenAI-compatible ✓ |
| Models | Full precision by default | Auto-quantized |
| Memory | Keeps models loaded | Can unload (keep_alive) |
| Structured outputs | ✓ Supported | ✗ Not supported |
| Setup complexity | Medium | Simple |
| Edge devices | OK | **Excellent** |

## Troubleshooting

### "Connection refused on localhost:11434"
- Check Ollama is running: `ollama serve`
- Check endpoint: Should be `http://localhost:11434/v1`

### "Model not found: ministral-3:3b"
- Pull the model: `ollama pull ministral-3:3b`
- List models: `ollama list`

### OOM Errors on Jetson
- Set `APP_LLM_SERVER__KEEP_ALIVE=0`
- Use smaller model: `ollama pull qwen2.5-1.5b`
- Disable structured outputs: `APP_LLM_SUM__STRUCTURED_OUTPUTS_ENABLED=false`

### Slow Response Times
- Verify GPU is being used: `nvidia-smi` (watch VRAM)
- Increase `OLLAMA_NUM_PARALLEL` if you have space
- Use a quantized model

## Performance Tips

1. **Sequential Execution (Jetson Nano)**
   ```bash
   export APP_LLM_SERVER__KEEP_ALIVE=0  # Unload after each use
   ```

2. **Faster Responses (Higher VRAM)**
   ```bash
   export APP_LLM_SERVER__KEEP_ALIVE=60  # Keep loaded for 60 seconds
   ```

3. **Model Optimization**
   - Ollama auto-quantizes to fit available VRAM
   - Smaller quantization = faster but lower quality
   - Check model details: `ollama show ministral-3:3b`

## Testing Your Setup

```bash
# Test Ollama directly
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ministral-3:3b",
    "messages": [{"role": "user", "content": "test"}],
    "keep_alive": 0
  }'

# Test MAIE integration
python -m pytest tests/integration/test_ollama_integration.py -v
```

## Next Steps

1. Review [OLLAMA_INTEGRATION.md](OLLAMA_INTEGRATION.md) for detailed deployment
2. Check [OLLAMA_IMPLEMENTATION_SUMMARY.md](OLLAMA_IMPLEMENTATION_SUMMARY.md) for architecture
3. Run e2e tests: `./scripts/validate-e2e-results.py`

# Ollama Integration Guide for MAIE

This guide explains how to use [Ollama](https://ollama.ai/) as the LLM backend for MAIE (Modular Audio Intelligence Engine).

## Overview

Ollama provides a simple way to run LLMs locally. MAIE can use Ollama through its OpenAI-compatible API, making it a lightweight alternative to vLLM—especially useful on edge devices like Jetson.

| Feature | Ollama | vLLM |
|---------|--------|------|
| Setup complexity | Low | Medium-High |
| Memory efficiency | Good | Better |
| Model variety | Excellent | Good |
| Jetson support | Native | Build from source |
| API compatibility | OpenAI-compatible | OpenAI-compatible |

## Quick Start

### 1. Install Ollama

```bash
# Linux/Jetson
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version
```

### 2. Pull the Default Model

```bash
# Pull the recommended model for MAIE
ollama pull ministral-3:3b
```

### 3. Start Ollama Server

```bash
# Option A: Use MAIE's launcher script (recommended)
./scripts/start-ollama.sh

# Option B: Start manually
ollama serve
```

### 4. Configure MAIE

Add to your `.env` file:

```bash
APP_LLM_BACKEND=vllm_server
APP_LLM_SERVER__ENHANCE_BASE_URL=http://localhost:11434/v1
APP_LLM_SERVER__SUMMARY_BASE_URL=http://localhost:11434/v1
APP_LLM_SERVER__ENHANCE_MODEL_NAME=ministral-3:3b
APP_LLM_SERVER__SUMMARY_MODEL_NAME=ministral-3:3b
APP_LLM_SUM__STRUCTURED_OUTPUTS_ENABLED=false
```

### 5. Start MAIE

```bash
./scripts/dev.sh
```

---

## Helper Scripts

MAIE includes two helper scripts for Ollama management:

### `scripts/ollama_config.py`

Python utility for checking Ollama status and generating configuration.

```bash
# Show full status
python3 scripts/ollama_config.py --status

# Check if Ollama is running (exit code 0/1)
python3 scripts/ollama_config.py --check

# List available models
python3 scripts/ollama_config.py --list-models

# Show MAIE configuration
python3 scripts/ollama_config.py --show-config

# Pull required models
python3 scripts/ollama_config.py --pull

# Ensure Ollama is ready (check + pull)
python3 scripts/ollama_config.py --ensure
```

### `scripts/start-ollama.sh`

Shell script for launching and configuring Ollama.

```bash
# Start Ollama and ensure models are ready
./scripts/start-ollama.sh

# Show status only
./scripts/start-ollama.sh --status

# Generate systemd service file
./scripts/start-ollama.sh --systemd
```

---

## Production Setup (Systemd)

For production deployments, run Ollama as a systemd service:

```bash
# Generate and install the service file
./scripts/start-ollama.sh --systemd | sudo tee /etc/systemd/system/ollama.service

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Check status
sudo systemctl status ollama
```

---

## Model Selection

### Recommended Models for MAIE

| Model | Size | Use Case | Command |
|-------|------|----------|---------|
| `ministral-3:3b` | 3GB | Default, edge devices | `ollama pull ministral-3:3b` |
| `qwen2.5:7b` | 4.4GB | Better quality | `ollama pull qwen2.5:7b` |
| `llama3.1:8b` | 4.7GB | General purpose | `ollama pull llama3.1:8b` |
| `qwen2.5:14b` | 9GB | High quality (needs 16GB+ RAM) | `ollama pull qwen2.5:14b` |

### Using Different Models

```bash
# Set via environment variable
export OLLAMA_MODEL=qwen2.5:7b
./scripts/start-ollama.sh

# Or configure in .env
APP_LLM_SERVER__ENHANCE_MODEL_NAME=qwen2.5:7b
APP_LLM_SERVER__SUMMARY_MODEL_NAME=qwen2.5:7b
```

---

## Jetson-Specific Notes

### Sequential Execution (Low Memory Mode)

For devices with limited VRAM (e.g., Jetson Nano 8GB), it is critical to run ASR and LLM sequentially to avoid Out-Of-Memory (OOM) errors.

1.  **ASR Unloading**: MAIE automatically unloads the Whisper model and clears GPU cache after transcription.
2.  **LLM Unloading**: Configure Ollama to unload the model immediately after generation by setting `APP_LLM_SERVER__KEEP_ALIVE=0`.

Add this to your `.env`:
```bash
APP_LLM_SERVER__KEEP_ALIVE=0
```

### Memory Considerations

On Jetson Orin Nano Super (8GB RAM):
- Use models ≤4GB (e.g., `ministral-3:3b`, `qwen2.5:3b`)
- Close unnecessary applications
- Consider swap if needed

```bash
# Add swap (if using microSD)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Mode

Enable Super Mode for best performance:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## Troubleshooting

### Ollama not responding

```bash
# Check if running
curl http://localhost:11434/api/version

# Check logs
journalctl -u ollama -f

# Restart service
sudo systemctl restart ollama
```

### Model loading fails

```bash
# Check available memory
free -h

# Check GPU memory (Jetson)
tegrastats

# Try a smaller model
ollama pull qwen2.5:1.5b
```

### Connection refused from Docker

If MAIE runs in Docker and Ollama runs on the host:

```bash
# Use host.docker.internal (if Docker supports it)
APP_LLM_SERVER__ENHANCE_BASE_URL=http://host.docker.internal:11434/v1

# Or use the host's IP address
APP_LLM_SERVER__ENHANCE_BASE_URL=http://192.168.1.100:11434/v1

# Or run Ollama on 0.0.0.0
OLLAMA_HOST=0.0.0.0 ollama serve
```

### Slow inference

- Enable GPU acceleration (default on supported hardware)
- Use quantized models (default in Ollama)
- Reduce context length if needed

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `localhost` | Ollama server host |
| `OLLAMA_PORT` | `11434` | Ollama server port |
| `OLLAMA_MODEL` | `ministral-3:3b` | Default model |
| `APP_LLM_BACKEND` | - | Set to `vllm_server` for Ollama |
| `APP_LLM_SERVER__ENHANCE_BASE_URL` | - | Ollama API URL + `/v1` |
| `APP_LLM_SERVER__ENHANCE_MODEL_NAME` | - | Model for enhancement |
| `APP_LLM_SERVER__SUMMARY_MODEL_NAME` | - | Model for summarization |
| `APP_LLM_SUM__STRUCTURED_OUTPUTS_ENABLED` | - | Set to `false` for Ollama |

---

## API Compatibility

Ollama's OpenAI-compatible endpoint supports:

- ✅ `/v1/chat/completions` - Used by MAIE
- ✅ `/v1/models` - List models
- ❌ Structured outputs (JSON schema enforcement) - Not supported

> [!IMPORTANT]
> Set `APP_LLM_SUM__STRUCTURED_OUTPUTS_ENABLED=false` when using Ollama, as it doesn't support vLLM's structured output format.

---

## See Also

- [Ollama Documentation](https://ollama.ai/)
- [Ollama Model Library](https://ollama.ai/library)
- [MAIE Jetson Deployment Guide](JETSON_DEPLOYMENT_GUIDE.md)
- [LLM Backend Configuration](LLM_BACKEND_CONFIGURATION.md)

# MAIE on Jetson Orin Nano Super — Complete Deployment Guide

> **One document to rule them all**: This guide consolidates all Jetson deployment documentation into a single, detailed, and easy-to-follow reference.

| Field               | Value                                       |
| ------------------- | ------------------------------------------- |
| **Target Platform** | NVIDIA Jetson Orin Nano Super Developer Kit |
| **JetPack SDK**     | 6.2                                         |
| **Architecture**    | ARM64 (aarch64)                             |
| **CUDA**            | 12.6                                        |
| **Status**          | ✅ Ready for Implementation                 |
| **Last Updated**    | December 2025                               |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Hardware & Software Requirements](#2-hardware--software-requirements)
3. [Quick Start (TL;DR)](#3-quick-start-tldr)
4. [Dependency Compatibility](#4-dependency-compatibility)
5. [Step-by-Step Migration](#5-step-by-step-migration)
6. [Docker Deployment](#6-docker-deployment)
7. [Non-Container (Pixi) Deployment](#7-non-container-pixi-deployment)
8. [Configuration & Tuning](#8-configuration--tuning)
9. [Performance Expectations](#9-performance-expectations)
10. [Monitoring & Troubleshooting](#10-monitoring--troubleshooting)
11. [Testing & Validation](#11-testing--validation)
12. [Rollback & Recovery](#12-rollback--recovery)
13. [Resources & References](#13-resources--references)

---

## 1. Executive Summary

### Is This Migration Right for You?

| Use Case                           | Recommendation       |
| ---------------------------------- | -------------------- |
| Edge/on-premises deployment        | ✅ **Yes**           |
| Data privacy / offline operation   | ✅ **Yes**           |
| Low power / portable               | ✅ **Yes**           |
| Budget-conscious (<$500/device)    | ✅ **Yes**           |
| Maximum throughput required        | ❌ Use x86_64 server |
| Processing >60 min audio regularly | ⚠️ Consider x86_64   |

### Feasibility Verdict

**MAIE runs on Jetson Orin Nano Super** with:

- **85% native compatibility** — most packages work out of the box
- **15% requiring workarounds** — all have documented solutions
- **70-85% performance** vs. x86_64 — acceptable for edge use
- **90% power reduction** — 25W vs 300W
- **$249 hardware cost** — vs $1,600+ for a server GPU

### Key Changes Required

| Change                        | Impact              | Solution                     |
| ----------------------------- | ------------------- | ---------------------------- |
| Remove `flashinfer`           | 10-15% LLM slowdown | vLLM fallback attention      |
| Build vLLM from source        | ~30 min build       | ARM64 build script provided  |
| Build CTranslate2 from source | ~20 min build       | CMake script provided        |
| Use L4T base image            | Docker rebuild      | `Dockerfile.jetson` provided |
| Tune for 8 GB RAM             | Config changes      | Jetson profile included      |

---

## 2. Hardware & Software Requirements

### Hardware

| Component   | Requirement                    | Notes                                          |
| ----------- | ------------------------------ | ---------------------------------------------- |
| **Board**   | Jetson Orin Nano Super Dev Kit | 8 GB LPDDR5 @ 102 GB/s                         |
| **Storage** | 128 GB+ NVMe SSD (recommended) | Or 64 GB+ microSD (slower)                     |
| **Power**   | 65 W USB-C supply              | Included with dev kit                          |
| **Cooling** | Stock heatsink + fan           | Active cooling recommended for sustained loads |
| **Network** | Ethernet or Wi-Fi              | Required for model downloads                   |

### Jetson Orin Nano Super Specs

| Spec           | Value                                            |
| -------------- | ------------------------------------------------ |
| GPU            | NVIDIA Ampere — 1024 CUDA cores, 32 Tensor cores |
| CPU            | 6-core ARM Cortex-A78AE @ 1.5 GHz                |
| Memory         | 8 GB LPDDR5 @ 102 GB/s                           |
| AI Performance | 67 TOPS (Super Mode) / 40 TOPS (standard)        |
| TDP            | 25 W (Super Mode) / 15 W (standard)              |
| CUDA           | 12.6                                             |
| TensorRT       | 10.3                                             |
| cuDNN          | 9.3                                              |
| Price          | $249 USD                                         |

### Software (Pre-installed with JetPack 6.2)

- Ubuntu 22.04
- CUDA 12.6
- Python 3.10+
- Docker with NVIDIA runtime

---

## 3. Quick Start (TL;DR)

For experienced users who want to get running fast. Detailed steps follow in later sections.

### Option A: Non-Container (Pixi) — Recommended

```bash
# 1. Enable Super Mode
sudo nvpmodel -m 0 && sudo jetson_clocks

# 2. Clone and enter repo
git clone <repo-url> && cd maie

# 3. Add swap if using microSD
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

# 4. Install Pixi (https://pixi.sh)
curl -fsSL https://pixi.sh/install.sh | bash

# 5. Configure Jetson PyPI in pyproject.toml (see Section 7)

# 6. Install dependencies
pixi install

# 7. Download models
pixi run download-models

# 8. Start API + Worker (two terminals)
pixi run api        # Terminal 1
pixi run worker     # Terminal 2

# 9. Test
curl http://localhost:8000/health
```

### Option B: Docker

```bash
# 1. Enable Super Mode
sudo nvpmodel -m 0 && sudo jetson_clocks

# 2. Clone repo
git clone <repo-url> && cd maie

# 3. Build image (~60-90 min first time)
docker build -f Dockerfile.jetson -t maie:jetson-latest .

# 4. Start services
docker-compose -f docker-compose.jetson.yml up -d

# 5. Test
curl http://localhost:8000/health
```

---

## 4. Dependency Compatibility

### Summary Table

| Dependency     | Status               | Action                            | Build Time |
| -------------- | -------------------- | --------------------------------- | ---------- |
| Python 3.12    | ✅ Compatible        | None                              | —          |
| PyTorch 2.5+   | ✅ Compatible        | Use L4T container or Jetson PyPI  | —          |
| TorchAudio     | ✅ Compatible        | Included with PyTorch             | —          |
| vLLM 0.10+     | ⚠️ Build from source | See script                        | 25-30 min  |
| flashinfer     | ⚠️ Check Jetson PyPI | Install if available, else remove | —          |
| faster-whisper | ⚠️ Build CTranslate2 | See script                        | 15-20 min  |
| pyannote-audio | ✅ Compatible        | pip install                       | —          |
| silero-vad     | ✅ Compatible        | pip install                       | —          |
| onnxruntime    | ✅ Compatible        | pip install                       | —          |
| transformers   | ✅ Compatible        | pip install                       | —          |
| rq / redis     | ✅ Compatible        | pip install                       | —          |

### Critical Notes

1. **flashinfer**: The Jetson AI Lab PyPI (`https://pypi.jetson-ai-lab.io/jp6/cu126`) may have ARM64 wheels. Check first; remove dependency only if unavailable.
2. **vLLM**: Jetson PyPI hosts `vllm-0.10.2+cu126`. If MAIE needs `>=0.11`, build from source.
3. **CTranslate2**: Required by `faster-whisper`. Build from source or check Jetson PyPI for wheels.

---

## 5. Step-by-Step Migration

### Phase 1: Dependency Adaptation (Week 1-2)

#### 1.1 Remove or Adjust flashinfer

Check if flashinfer is available on Jetson PyPI:

```bash
pip index versions flashinfer-python --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
```

If unavailable, remove from `pyproject.toml`:

```diff
- flashinfer-python = ">=0.5.2,<0.6"
- flashinfer-cubin = ">=0.5.2,<0.6"
- flashinfer-jit-cache = { url = "..." }
```

vLLM will automatically use fallback attention kernels (10-15% slower but fully functional).

#### 1.2 Adjust PyTorch Version Bounds

```diff
- "torch>=2.8.0,<3"
- "torchaudio>=2.8.0,<3"
+ "torch>=2.5.0,<3"
+ "torchaudio>=2.5.0,<3"
```

PyTorch 2.5+ has official ARM64 wheels. For newer versions, use Jetson PyPI or build nightly.

#### 1.3 Create vLLM Build Script

Save as `scripts/build-vllm-jetson.sh`:

```bash
#!/bin/bash
set -euo pipefail

echo "Building vLLM for Jetson Orin Nano (ARM64)..."

# Prerequisites check
[[ -f /usr/local/cuda-12.6/bin/nvcc ]] || { echo "CUDA 12.6 not found"; exit 1; }

# Clone
[[ -d vllm ]] || git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout v0.11.0

# Prepare existing PyTorch
python3 use_existing_torch.py

# Build settings for Jetson Orin (Ampere SM 8.7)
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=6
export NVCC_THREADS=2
export VLLM_INSTALL_PUNICA_KERNELS=0

pip install --upgrade pip setuptools wheel
pip install -r requirements/build.txt
pip install -e . --no-build-isolation

echo "✅ vLLM installed. Verify: python -c 'import vllm; print(vllm.__version__)'"
```

#### 1.4 Create CTranslate2 Build Script

Save as `scripts/build-ctranslate2-jetson.sh`:

```bash
#!/bin/bash
set -euo pipefail

echo "Building CTranslate2 for Jetson Orin Nano (ARM64)..."

apt-get update && apt-get install -y cmake build-essential git
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

[[ -d CTranslate2 ]] || git clone https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

make -j6 && make install

cd ../python && pip install -e .

echo "✅ CTranslate2 installed. Verify: python -c 'import ctranslate2; print(ctranslate2.__version__)'"
```

---

### Phase 2: Deployment & Packaging (Week 2-3)

Choose **Docker** or **Pixi** (non-container). Pixi is recommended for Jetson to reduce overhead.

See [Section 6](#6-docker-deployment) or [Section 7](#7-non-container-pixi-deployment).

---

### Phase 3: Functional Testing (Week 3-4)

See [Section 11](#11-testing--validation).

---

### Phase 4: Performance Optimization (Week 4-6)

See [Section 8](#8-configuration--tuning) and [Section 9](#9-performance-expectations).

---

## 6. Docker Deployment

### 6.1 Dockerfile.jetson

```dockerfile
# Multi-stage Dockerfile for MAIE on Jetson Orin Nano Super
FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git wget curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# === BUILDER ===
FROM base AS builder

COPY scripts/build-ctranslate2-jetson.sh /tmp/
RUN bash /tmp/build-ctranslate2-jetson.sh

COPY scripts/build-vllm-jetson.sh /tmp/
RUN bash /tmp/build-vllm-jetson.sh

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    soxr transformers rq-dashboard \
    "pyannote-audio>=4.0.0,<5" "silero-vad>=6.2.0,<7" \
    "onnxruntime>=1.23.2,<2" faster-whisper

COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# === PRODUCTION ===
FROM base AS production

COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/lib/python3.*/dist-packages /usr/lib/python3.*/dist-packages

RUN groupadd -r -g 1000 maie && useradd -r -u 1000 -g maie -m maie

COPY --chown=maie:maie src/ ./src/
COPY --chown=maie:maie templates/ ./templates/
COPY --chown=maie:maie main.py pyproject.toml ./

RUN mkdir -p /app/data/audio /app/data/models /app/logs && chown -R maie:maie /app

USER maie
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "src.api.main"]
```

### 6.2 docker-compose.jetson.yml

```yaml
version: "3.8"

networks:
  maie-internal:
    driver: bridge

volumes:
  redis_data:
  audio_data:
  model_data:

services:
  redis:
    image: redis:8-alpine
    container_name: maie-redis-jetson
    restart: unless-stopped
    networks: [maie-internal]
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes: [redis_data:/data]
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build:
      context: .
      dockerfile: Dockerfile.jetson
      target: production
    image: maie:jetson-latest
    container_name: maie-api-jetson
    restart: unless-stopped
    networks: [maie-internal]
    depends_on:
      redis: { condition: service_healthy }
    ports: ["8000:8000"]
    volumes:
      - audio_data:/data/audio
      - ./templates:/app/templates:ro
    environment:
      REDIS_URL: redis://redis:6379/0
      GPU_MEMORY_UTILIZATION: "0.85"
      LLM_MAX_MODEL_LEN: "8192"
      MAX_NUM_SEQS: "1"
      WHISPER_COMPUTE_TYPE: int8_float16
      SECRET_API_KEY: ${SECRET_API_KEY}
    deploy:
      resources:
        limits: { memory: 6G }
    command: ["python", "-m", "src.api.main"]

  worker:
    image: maie:jetson-latest
    container_name: maie-worker-jetson
    restart: unless-stopped
    networks: [maie-internal]
    depends_on:
      redis: { condition: service_healthy }
    volumes:
      - audio_data:/data/audio:ro
      - model_data:/data/models:ro
      - ./templates:/app/templates:ro
    environment:
      REDIS_URL: redis://redis:6379/0
      GPU_MEMORY_UTILIZATION: "0.85"
      WHISPER_COMPUTE_TYPE: int8_float16
      LLM_MAX_MODEL_LEN: "8192"
      MAX_NUM_BATCHED_TOKENS: "4096"
      MAX_NUM_SEQS: "1"
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
        limits: { memory: 7G }
    command: ["python", "-m", "src.worker.main"]
```

### 6.3 Build & Run

```bash
# Build (first time ~60-90 min)
docker-compose -f docker-compose.jetson.yml build

# Start
docker-compose -f docker-compose.jetson.yml up -d

# Logs
docker-compose -f docker-compose.jetson.yml logs -f

# Stop
docker-compose -f docker-compose.jetson.yml down
```

---

## 7. Non-Container (Pixi) Deployment

Pixi offers a lighter footprint and avoids Docker overhead—ideal for Jetson.

### 7.1 Configure Jetson PyPI

Add to `pyproject.toml`:

```toml
[tool.pixi.pypi-options]
extra-index-url = ["https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"]

[tool.pixi.workspace]
platforms = ["linux-64", "linux-aarch64"]
```

### 7.2 Install & Run

```bash
# 1. Prepare system
sudo nvpmodel -m 0 && sudo jetson_clocks

# 2. Optional swap (microSD users)
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

# 3. Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# 4. Clone and install
git clone <repo-url> && cd maie
pixi install

# 5. Download models
pixi run download-models

# 6. Start services
pixi run api      # Terminal 1
pixi run worker   # Terminal 2

# 7. Verify
curl http://localhost:8000/health
```

---

## 8. Configuration & Tuning

### 8.1 Environment Variables

| Variable                 | Default | Jetson Recommended | Notes                    |
| ------------------------ | ------- | ------------------ | ------------------------ |
| `GPU_MEMORY_UTILIZATION` | 0.90    | **0.85**           | Leave headroom for 8 GB  |
| `LLM_MAX_MODEL_LEN`      | 32768   | **8192**           | Reduced context window   |
| `MAX_NUM_SEQS`           | 4       | **1**              | Single request at a time |
| `MAX_NUM_BATCHED_TOKENS` | 8192    | **4096**           | Lower batch size         |
| `WHISPER_COMPUTE_TYPE`   | float16 | **int8_float16**   | Quantized for memory     |
| `WHISPER_BEAM_SIZE`      | 5       | **3**              | Faster decoding          |

### 8.2 Performance Profiles

**Maximum Performance** (may hit memory limits):

```bash
GPU_MEMORY_UTILIZATION=0.90
WHISPER_COMPUTE_TYPE=float16
WHISPER_BEAM_SIZE=5
```

**Balanced (Recommended)**:

```bash
GPU_MEMORY_UTILIZATION=0.85
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_BEAM_SIZE=3
```

**Maximum Stability** (for very long runs):

```bash
GPU_MEMORY_UTILIZATION=0.75
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_BEAM_SIZE=3
MAX_QUEUE_DEPTH=3
```

**Low Power Mode** (15 W TDP):

```bash
sudo nvpmodel -m 1
GPU_MEMORY_UTILIZATION=0.70
MAX_NUM_BATCHED_TOKENS=2048
```

### 8.3 Auto-Detect Jetson in Code

```python
def is_jetson() -> bool:
    try:
        with open('/etc/nv_tegra_release') as f:
            return 'Tegra' in f.read()
    except FileNotFoundError:
        return False

if is_jetson():
    # Apply Jetson-specific settings
    settings.gpu_memory_utilization = 0.85
    settings.max_model_len = 8192
    settings.max_num_seqs = 1
```

---

## 9. Performance Expectations

### 9.1 Benchmark Comparison

| Metric            | x86_64 (RTX 4090) | Jetson Target | Delta    |
| ----------------- | ----------------- | ------------- | -------- |
| ASR RTF           | 0.12              | 0.15-0.18     | +25-50%  |
| LLM Enhancement   | 78-80 tok/s       | 55-65 tok/s   | -20-30%  |
| LLM Summary       | 40 tok/s          | 30-35 tok/s   | -15-25%  |
| E2E (2 min audio) | 18 s              | 22-28 s       | +20-55%  |
| Power             | 300 W             | 15-25 W       | **-90%** |

### 9.2 Processing Time Estimates

| Audio Length | Processing Time | Real-Time Factor |
| ------------ | --------------- | ---------------- |
| 30 s         | 5-7 s           | 0.16-0.23        |
| 2 min        | 22-28 s         | 0.18-0.23        |
| 5 min        | 60-75 s         | 0.20-0.25        |
| 10 min       | 2.5-3 min       | 0.25-0.30        |

### 9.3 Resource Utilization

| Resource   | Idle  | Processing | Target Max |
| ---------- | ----- | ---------- | ---------- |
| GPU Memory | ~2 GB | 6-7 GB     | <7.5 GB    |
| CPU Memory | ~1 GB | 4-5 GB     | <6 GB      |
| GPU Util   | 0%    | 80-95%     | 100% OK    |
| Power      | 8 W   | 20-24 W    | 25 W       |

---

## 10. Monitoring & Troubleshooting

### 10.1 Essential Commands

```bash
# System info
cat /etc/nv_tegra_release

# Real-time monitoring (install: sudo pip3 install -U jetson-stats)
jtop

# Power mode
sudo nvpmodel -q

# Enable Super Mode
sudo nvpmodel -m 0 && sudo jetson_clocks

# GPU info (note: nvidia-smi may not work on Jetson; use jtop)
tegrastats
```

### 10.2 Common Issues

| Issue                       | Cause                            | Solution                                                                     |
| --------------------------- | -------------------------------- | ---------------------------------------------------------------------------- |
| **Build OOM**               | Not enough memory during compile | `export MAX_JOBS=2 NVCC_THREADS=1`                                           |
| **GPU not detected**        | CUDA not loaded                  | Verify `python -c "import torch; print(torch.cuda.is_available())"`          |
| **vLLM model fails**        | Out of VRAM                      | Lower `GPU_MEMORY_UTILIZATION` and `LLM_MAX_MODEL_LEN`                       |
| **Slow performance**        | Not in Super Mode                | `sudo nvpmodel -m 0 && sudo jetson_clocks`                                   |
| **Container can't use GPU** | Missing runtime                  | `sudo apt install nvidia-container-runtime && sudo systemctl restart docker` |

### 10.3 Log Locations

- **API**: `logs/api.log` or `docker logs maie-api-jetson`
- **Worker**: `logs/worker.log` or `docker logs maie-worker-jetson`
- **Redis**: `docker logs maie-redis-jetson`

---

## 11. Testing & Validation

### 11.1 Health Check

```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", ...}
```

### 11.2 Process Test Audio

```bash
# Upload
curl -X POST "http://localhost:8000/v1/process" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@test_audio.wav" \
  -F "features=clean_transcript"

# Check status (use task_id from response)
curl "http://localhost:8000/v1/status/${TASK_ID}" \
  -H "X-API-Key: ${API_KEY}"
```

### 11.3 Run Test Suite

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# E2E (skip real LLM if no API key)
pytest tests/e2e -v -m "not real_llm"
```

### 11.4 Success Criteria

| Criterion         | Target     |
| ----------------- | ---------- |
| API health check  | ✅ 200 OK  |
| ASR RTF           | < 0.25     |
| LLM tokens/sec    | > 25       |
| E2E (2 min audio) | < 30 s     |
| GPU memory        | < 7 GB     |
| 24-hour stability | No crashes |

---

## 12. Rollback & Recovery

### 12.1 Pre-Migration Backup

```bash
git checkout -b backup-pre-jetson-migration
git push origin backup-pre-jetson-migration

cp pyproject.toml backup/pyproject-x86.toml
cp Dockerfile backup/Dockerfile-x86
docker save maie:latest > backup/maie-x86.tar
```

### 12.2 Rollback Procedure

```bash
# Stop Jetson services
docker-compose -f docker-compose.jetson.yml down

# Restore x86_64 branch
git checkout backup-pre-jetson-migration

# Rebuild and restart
docker-compose build && docker-compose up -d

# Verify
curl http://localhost:8000/health
```

### 12.3 Contingency Plans

| Problem             | Alternative                                                        |
| ------------------- | ------------------------------------------------------------------ |
| vLLM too slow       | Use [NanoLLM](https://github.com/dusty-nv/nanollm) or TensorRT-LLM |
| Memory insufficient | Use lighter models (Whisper small, Qwen-1.5B)                      |
| ASR too slow        | Use [whisper.cpp](https://github.com/ggml-org/whisper.cpp)         |

---

## 13. Resources & References

### Official Documentation

- [JetPack SDK 6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62)
- [Jetson Orin Nano Super Product Page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- [L4T PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)
- [Jetson AI Lab PyPI (cu126)](https://pypi.jetson-ai-lab.io/jp6/cu126)
- [vLLM ARM64 Build Docs](https://docs.vllm.ai/en/latest/deployment/docker)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)

### Community Resources

- [Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [NVIDIA Developer Forums (Jetson)](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/632)
- [Jetson Projects (dusty-nv)](https://github.com/dusty-nv/jetson-inference)
- [NanoLLM](https://github.com/dusty-nv/nanollm)

### Support

- **Hardware/Platform**: NVIDIA Developer Forums
- **vLLM**: GitHub Issues / Discussions
- **MAIE**: Repository Issues

---

## Appendix: Quick Reference

### Command Cheat Sheet

```bash
# System
cat /etc/nv_tegra_release          # JetPack version
jtop                                # Real-time monitoring
sudo nvpmodel -m 0 && sudo jetson_clocks  # Super Mode

# Docker
docker-compose -f docker-compose.jetson.yml build
docker-compose -f docker-compose.jetson.yml up -d
docker-compose -f docker-compose.jetson.yml logs -f
docker stats maie-worker-jetson

# Pixi
pixi install
pixi run api
pixi run worker
pixi run download-models

# Verification
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python -c "import vllm; print(vllm.__version__)"
python -c "import ctranslate2; print(ctranslate2.__version__)"
curl http://localhost:8000/health
```

### Key Configuration (`.env.jetson`)

```bash
SECRET_API_KEY=your-key-here
GPU_MEMORY_UTILIZATION=0.85
LLM_MAX_MODEL_LEN=8192
MAX_NUM_SEQS=1
WHISPER_COMPUTE_TYPE=int8_float16
```

---

**Document Version**: 2.0 (Consolidated)  
**Supersedes**: `JETSON_README.md`, `JETSON_QUICK_START.md`, `JETSON_COMPATIBILITY_MATRIX.md`, `JETSON_EXECUTIVE_SUMMARY.md`, `JETSON_MIGRATION_PLAN.md`  
**Last Updated**: December 2025  
**Status**: ✅ Ready for Implementation

---

Good luck with your Jetson deployment! 🚀

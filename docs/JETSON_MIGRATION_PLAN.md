# MAIE Jetson Orin Nano Super Migration Plan

**Target Platform**: NVIDIA Jetson Orin Nano Super Developer Kit  
**JetPack SDK**: 6.2  
**Project**: MAIE (Modular Audio Intelligence Engine)  
**Migration Type**: x86_64 → ARM64 (aarch64)  
**Date**: November 25, 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Prerequisites](#prerequisites)
3. [Migration Phases](#migration-phases)
4. [Detailed Implementation](#detailed-implementation)
5. [Testing Strategy](#testing-strategy)
6. [Performance Expectations](#performance-expectations)
7. [Rollback Plan](#rollback-plan)
8. [Timeline & Resources](#timeline--resources)
9. [Risk Mitigation](#risk-mitigation)
10. [Success Criteria](#success-criteria)

---

## Executive Summary

This document outlines a comprehensive, phased approach to migrate MAIE from x86_64 architecture to NVIDIA Jetson Orin Nano Super (ARM64/aarch64). The migration addresses critical compatibility issues while maintaining system functionality and performance.

### Key Objectives
1. ✅ Enable MAIE to run on Jetson Orin Nano Super with JetPack SDK 6.2
2. ✅ Maintain core functionality (ASR, LLM, diarization)
3. ✅ Achieve acceptable performance (70-85% of x86_64 throughput)
4. ✅ Minimize code changes to core application logic
5. ✅ Provide clear deployment path via Docker

### Migration Approach
- **Strategy**: Phased migration with incremental validation
- **Duration**: 4-6 weeks (with hardware access)
- **Risk Level**: Medium-High (requires source builds)
- **Reversibility**: High (Docker-based, isolated)

### Critical Changes Required
1. **Remove flashinfer dependencies** (x86_64-only)
2. **Build vLLM from source** for ARM64
3. **Build CTranslate2 from source** for faster-whisper
4. **Replace Docker base image** with NVIDIA L4T
5. **Adjust memory/performance configurations** for 8GB constraint

---

## Prerequisites

### Hardware Requirements
- ✅ NVIDIA Jetson Orin Nano Super Developer Kit (8GB)
- ✅ JetPack SDK 6.2 installed (with Super Mode enabled)
- ✅ MicroSD card (64GB+) or NVMe SSD (128GB+ recommended)
- ✅ 65W USB-C power supply
- ✅ Network connectivity (for model downloads)
- ✅ Keyboard, mouse, monitor (for initial setup)

### Development Environment
- ✅ Access to Jetson hardware (for testing and validation)
- ✅ Docker installed on Jetson (comes with JetPack)
- ✅ GitHub repository access
- ✅ Storage for models: 50GB+ free space

### Knowledge Prerequisites
- Docker multi-stage builds
- ARM architecture considerations
- CUDA cross-compilation
- Python package management (pip, uv)
- Basic Jetson platform knowledge

### Pre-Migration Backup
```bash
# Backup current codebase
git checkout -b backup-pre-jetson-migration
git push origin backup-pre-jetson-migration

# Document current configuration
docker-compose config > backup/docker-compose-x86.yml
cp pyproject.toml backup/pyproject-x86.toml
cp Dockerfile backup/Dockerfile-x86
```

---

## Migration Phases

### Phase 1: Dependency Adaptation (Week 1-2)
**Goal**: Modify dependencies for ARM64 compatibility

**Tasks**:
1. Remove flashinfer dependencies
2. Update pyproject.toml for ARM64
3. Create ARM64-specific build scripts
4. Document dependency changes

**Deliverables**:
- ✅ Updated `pyproject.toml`
- ✅ Build script for CTranslate2
- ✅ Build script for vLLM
- ✅ Updated documentation

**Risk**: Medium  
**Validation**: Dependency resolution succeeds on ARM64

---

### Phase 2: Docker Containerization (Week 2-3)
**Goal**: Create Jetson-compatible Docker image

**Tasks**:
1. Create new Dockerfile for Jetson
2. Multi-stage build with L4T base
3. Build and test locally on Jetson
4. Optimize image size

**Deliverables**:
- ✅ `Dockerfile.jetson`
- ✅ `docker-compose.jetson.yml`
- ✅ Build scripts and CI/CD updates
- ✅ Image size optimization

**Risk**: High  
**Validation**: Docker image builds successfully and starts

---

### Phase 3: Functional Testing (Week 3-4)
**Goal**: Validate core functionality on Jetson

**Tasks**:
1. Unit test execution
2. Integration testing
3. E2E workflow validation
4. Identify and fix issues

**Deliverables**:
- ✅ Test results report
- ✅ Bug fixes
- ✅ Performance baseline

**Risk**: Medium  
**Validation**: 80%+ tests passing

---

### Phase 4: Performance Optimization (Week 4-6)
**Goal**: Optimize for Jetson constraints

**Tasks**:
1. Memory optimization
2. Model quantization tuning
3. vLLM configuration optimization
4. Benchmark against targets

**Deliverables**:
- ✅ Performance report
- ✅ Optimization recommendations
- ✅ Production configuration

**Risk**: Low  
**Validation**: Performance meets targets (70-85%)

---

## Detailed Implementation

### Phase 1: Dependency Adaptation

#### Step 1.1: Remove flashinfer Dependencies

**File**: `pyproject.toml`

```diff
[project]
dependencies = [
    "soxr",
    "transformers",
-   "flashinfer-cubin>=0.5.2,<0.6",
    "rq-dashboard",
    "pyannote-audio>=4.0.0,<5",
    "torch>=2.8.0,<3",
    "torchaudio>=2.8.0,<3",
    "silero-vad>=6.2.0,<7",
    "onnxruntime>=1.23.2,<2",
    "torchcodec==0.7.0"
]

[tool.pixi.pypi-dependencies]
-flashinfer-python = ">=0.5.2,<0.6"
-flashinfer-jit-cache = { url = "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.5.2/flashinfer_jit_cache-0.5.2%2Bcu128-cp39-abi3-manylinux_2_28_x86_64.whl" }
```

**Verification**:
```bash
# Check for flashinfer imports in codebase
rg "from flashinfer|import flashinfer" src/

# Expected: No results (vLLM handles this internally)
```

#### Step 1.2: Update PyTorch Dependencies for ARM64

**File**: `pyproject.toml`

```diff
[project]
dependencies = [
-   "torch>=2.8.0,<3",
-   "torchaudio>=2.8.0,<3",
+   "torch>=2.5.0,<3",  # ARM64 wheels available from 2.5+
+   "torchaudio>=2.5.0,<3",
]
```

**Note**: PyTorch 2.5+ has official ARM64 wheels. For 2.8+, use nightly builds or NVIDIA's L4T containers.

#### Step 1.3: Create vLLM Build Script

**File**: `scripts/build-vllm-jetson.sh`

```bash
#!/bin/bash
# Build vLLM for Jetson Orin Nano Super
# Requires: JetPack SDK 6.2, CUDA 12.6, PyTorch installed

set -euo pipefail

echo "Building vLLM for Jetson Orin Nano (ARM64)..."

# Check prerequisites
if [ ! -f "/usr/local/cuda-12.6/bin/nvcc" ]; then
    echo "ERROR: CUDA 12.6 not found. Install JetPack SDK 6.2"
    exit 1
fi

# Clone vLLM if not exists
if [ ! -d "vllm" ]; then
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout v0.11.0  # Or latest stable
else
    cd vllm
    git pull
fi

# Prepare PyTorch for existing installation
python3 use_existing_torch.py

# Set build variables for Jetson Orin (Ampere 8.7)
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=6  # Jetson has 6 cores
export NVCC_THREADS=2  # Limit parallel NVCC compilations
export VLLM_INSTALL_PUNICA_KERNELS=0  # Disable punica (x86 only)

# Install build dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements/build.txt

# Build and install
pip install -e . --no-build-isolation

echo "✅ vLLM built successfully for Jetson!"
echo "Verify installation:"
echo "  python -c 'import vllm; print(vllm.__version__)'"
```

**Execution**:
```bash
chmod +x scripts/build-vllm-jetson.sh
# Run on Jetson device
./scripts/build-vllm-jetson.sh
```

#### Step 1.4: Create CTranslate2 Build Script

**File**: `scripts/build-ctranslate2-jetson.sh`

```bash
#!/bin/bash
# Build CTranslate2 for Jetson Orin Nano Super
# Required for faster-whisper

set -euo pipefail

echo "Building CTranslate2 for Jetson Orin Nano (ARM64)..."

# Install build dependencies
apt-get update
apt-get install -y cmake build-essential git

# Install NVIDIA Python libraries
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

# Set library paths
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')

# Clone CTranslate2
if [ ! -d "CTranslate2" ]; then
    git clone https://github.com/OpenNMT/CTranslate2.git
    cd CTranslate2
else
    cd CTranslate2
    git pull
fi

# Build with CUDA support
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

make -j6
make install

# Install Python bindings
cd ../python
pip install -e .

echo "✅ CTranslate2 built successfully!"
echo "Verify installation:"
echo "  python -c 'import ctranslate2; print(ctranslate2.__version__)'"
```

#### Step 1.5: Update Pixi Configuration

**File**: `pyproject.toml`

```diff
[tool.pixi.workspace]
-channels = ["conda-forge", "pytorch", "nvidia"]
+channels = ["conda-forge", "pytorch", "nvidia/label/cuda-12.6.0"]
-platforms = ["linux-64"]
+platforms = ["linux-64", "linux-aarch64"]
```

---

### Phase 2: Docker Containerization

#### Step 2.1: Create Jetson-Specific Dockerfile

**File**: `Dockerfile.jetson`

```dockerfile
# Multi-stage Dockerfile for MAIE on Jetson Orin Nano Super
# Base: NVIDIA L4T PyTorch container with JetPack SDK 6.2
# Architecture: ARM64 (aarch64)
# CUDA: 12.6

# ============================================================================
# BASE STAGE - NVIDIA L4T PyTorch with JetPack 6.2
# ============================================================================
FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3 AS base

# Set build arguments for versioning
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0-jetson

# Labels for metadata (OCI standard)
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="MAIE-Jetson" \
      org.opencontainers.image.description="MAIE for Jetson Orin Nano Super" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="MAIE Team" \
      org.opencontainers.image.platform="linux/arm64"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================================================
# BUILDER STAGE - Compile ARM64-specific dependencies
# ============================================================================
FROM base AS builder

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Python libraries for CTranslate2
RUN pip install --upgrade pip setuptools wheel && \
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

# Build CTranslate2 from source
COPY scripts/build-ctranslate2-jetson.sh /tmp/
RUN cd /tmp && \
    bash build-ctranslate2-jetson.sh

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies (without flashinfer)
RUN pip install --no-cache-dir \
    soxr \
    transformers \
    rq-dashboard \
    "pyannote-audio>=4.0.0,<5" \
    "silero-vad>=6.2.0,<7" \
    "onnxruntime>=1.23.2,<2"

# Install faster-whisper (now that CTranslate2 is built)
RUN pip install --no-cache-dir faster-whisper

# Build vLLM from source
COPY scripts/build-vllm-jetson.sh /tmp/
RUN cd /tmp && \
    bash build-vllm-jetson.sh

# Install MAIE package
RUN pip install --no-cache-dir -e .

# ============================================================================
# PRODUCTION STAGE - Minimal runtime image
# ============================================================================
FROM base AS production

# Copy compiled binaries and Python packages from builder
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/lib/python3.*/dist-packages /usr/lib/python3.*/dist-packages

# Create non-root user
RUN groupadd -r -g 1000 maie && \
    useradd -r -u 1000 -g maie -m -s /bin/bash maie

# Copy application code
COPY --chown=maie:maie src/ ./src/
COPY --chown=maie:maie templates/ ./templates/
COPY --chown=maie:maie main.py ./
COPY --chown=maie:maie pyproject.toml ./

# Create necessary directories
RUN mkdir -p \
    /app/data/audio \
    /app/data/models \
    /app/data/redis \
    /app/logs \
    && chown -R maie:maie /app

# Switch to non-root user
USER maie

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (overridden by docker-compose)
CMD ["python", "-m", "src.api.main"]
```

**Build Instructions**:
```bash
# On Jetson device
docker build -f Dockerfile.jetson -t maie:jetson-latest .

# Build time: 60-90 minutes (first build)
# Image size: ~8-10GB
```

#### Step 2.2: Create Jetson Docker Compose

**File**: `docker-compose.jetson.yml`

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
    image: redis:8.2-alpine
    container_name: maie-redis-jetson
    restart: unless-stopped
    networks:
      - maie-internal
    command: >
      redis-server
      --appendonly yes
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
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
      args:
        BUILD_DATE: ${BUILD_DATE:-}
        VCS_REF: ${VCS_REF:-}
        VERSION: ${VERSION:-1.0.0-jetson}
    image: maie:jetson-${VERSION:-1.0.0}
    container_name: maie-api-jetson
    restart: unless-stopped
    networks:
      - maie-internal
    depends_on:
      redis:
        condition: service_healthy
    ports:
      - "8000:8000"
    volumes:
      - audio_data:/data/audio
      - ./templates:/app/templates:ro
    environment:
      # Jetson-optimized configuration
      - REDIS_URL=redis://redis:6379/0
      - GPU_MEMORY_UTILIZATION=0.85  # Conservative for 8GB
      - MAX_QUEUE_DEPTH=5  # Lower for memory constraints
      - LLM_MAX_MODEL_LEN=8192  # Reduced context window
      - MAX_NUM_SEQS=1  # Single request processing
      - SECRET_API_KEY=${SECRET_API_KEY}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - WHISPER_COMPUTE_TYPE=int8_float16  # Quantized for memory
      - LLM_BACKEND=vllm_server
    deploy:
      resources:
        limits:
          memory: 6G  # Reserve 2GB for system
    command: ["python", "-m", "src.api.main"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

  worker:
    image: maie:jetson-${VERSION:-1.0.0}
    container_name: maie-worker-jetson
    restart: unless-stopped
    networks:
      - maie-internal
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - audio_data:/data/audio:ro
      - model_data:/data/models:ro
      - ./templates:/app/templates:ro
    environment:
      # Jetson-optimized worker config
      - REDIS_URL=redis://redis:6379/0
      - GPU_MEMORY_UTILIZATION=0.85
      - WHISPER_COMPUTE_TYPE=int8_float16
      - LLM_MAX_MODEL_LEN=8192
      - MAX_NUM_BATCHED_TOKENS=4096  # Reduced
      - MAX_NUM_SEQS=1
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    runtime: nvidia  # Required for GPU access
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
        limits:
          memory: 7G  # Most of available memory
    command: ["python", "-m", "src.worker.main"]
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; exit(0 if torch.cuda.is_available() else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

  rq-dashboard:
    image: eoranged/rq-dashboard:latest
    container_name: maie-rq-dashboard-jetson
    restart: unless-stopped
    networks:
      - maie-internal
    depends_on:
      redis:
        condition: service_healthy
    ports:
      - "9181:9181"
    environment:
      - RQ_DASHBOARD_REDIS_URL=redis://redis:6379/0
```

---

### Phase 3: Functional Testing

#### Step 3.1: Prepare Test Environment

```bash
# On Jetson device
cd /path/to/maie

# Build Docker images
docker-compose -f docker-compose.jetson.yml build

# Start services
docker-compose -f docker-compose.jetson.yml up -d

# Monitor startup
docker-compose -f docker-compose.jetson.yml logs -f
```

#### Step 3.2: Run Test Suite

```bash
# Unit tests
docker exec maie-worker-jetson pytest tests/unit -v

# Integration tests
docker exec maie-worker-jetson pytest tests/integration -v

# E2E tests (exclude real_llm if no API keys)
docker exec maie-worker-jetson pytest tests/e2e -v -m "not real_llm"
```

#### Step 3.3: Manual Functional Tests

**Test 1: Basic Transcription**
```bash
# Upload test audio file
curl -X POST "http://localhost:8000/v1/process" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@tests/fixtures/audio/test_sample.wav" \
  -F "features=clean_transcript"

# Note task_id from response
# Check status
curl "http://localhost:8000/v1/status/${TASK_ID}" \
  -H "X-API-Key: ${API_KEY}"
```

**Test 2: Transcription + Summary**
```bash
curl -X POST "http://localhost:8000/v1/process" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@tests/fixtures/audio/meeting_sample.wav" \
  -F "features=clean_transcript" \
  -F "features=summary" \
  -F "template_id=meeting_notes_v1"
```

**Test 3: Diarization**
```bash
curl -X POST "http://localhost:8000/v1/process" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@tests/fixtures/audio/multi_speaker.wav" \
  -F "features=clean_transcript" \
  -F "enable_diarization=true"
```

#### Step 3.4: Performance Baseline

```bash
# Run benchmark
python scripts/benchmark_maie_llm.py \
  --task both \
  --iterations 5 \
  --template-id generic_summary_v1

# Capture baseline metrics:
# - Transcription RTF (Real-Time Factor)
# - LLM tokens/second
# - Total processing time
# - Memory usage
# - GPU utilization
```

---

### Phase 4: Performance Optimization

#### Step 4.1: Memory Optimization

**Enable Super Mode** (if not already enabled):
```bash
# On Jetson device
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks   # Lock clocks to maximum
```

**Monitor Memory Usage**:
```bash
# Use jtop (Jetson monitoring tool)
sudo pip3 install -U jetson-stats
jtop

# Watch for:
# - GPU memory usage (should stay < 7GB)
# - CPU memory (should stay < 6GB)
# - Swap usage (should be minimal)
```

**Configuration Tuning**:
```python
# In src/config/model.py - Jetson-specific overrides
class LlmEnhanceSettings(BaseModel):
    # Optimized for Jetson Orin Nano 8GB
    gpu_memory_utilization: float = 0.85  # More conservative
    max_model_len: int = 8192  # Reduced from 32768
    max_num_seqs: int = 1  # Single sequence only
    max_num_batched_tokens: int = 4096  # Reduced from 8192
    
class WhisperSettings(BaseModel):
    # Memory-efficient settings
    compute_type: str = "int8_float16"  # Quantized
    beam_size: int = 3  # Reduced from 5 for speed
```

#### Step 4.2: Model Quantization Validation

**Whisper Model**:
```bash
# Verify INT8 quantization is working
docker exec maie-worker-jetson python -c "
from faster_whisper import WhisperModel
model = WhisperModel('erax-wow-turbo', device='cuda', compute_type='int8_float16')
print(f'Model loaded successfully with INT8 quantization')
"
```

**LLM Model**:
```bash
# Verify AWQ 4-bit quantization
docker exec maie-worker-jetson python -c "
from vllm import LLM
model = LLM('cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit')
print(f'LLM loaded successfully with AWQ quantization')
"
```

#### Step 4.3: vLLM Configuration Optimization

**File**: `src/config/model.py` (Jetson overrides)

```python
# Add Jetson-specific configuration profile
class JetsonProfile(ConfigurationProfile):
    """Optimized for Jetson Orin Nano 8GB"""
    
    @classmethod
    def apply_overrides(cls, settings: BaseSettings) -> None:
        if hasattr(settings, 'gpu_memory_utilization'):
            settings.gpu_memory_utilization = 0.85
        if hasattr(settings, 'max_model_len'):
            settings.max_model_len = 8192
        if hasattr(settings, 'max_num_seqs'):
            settings.max_num_seqs = 1
        if hasattr(settings, 'max_num_batched_tokens'):
            settings.max_num_batched_tokens = 4096

# Auto-detect Jetson platform
def is_jetson() -> bool:
    """Detect if running on Jetson platform"""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            return 'Tegra' in f.read()
    except FileNotFoundError:
        return False

# Apply Jetson profile automatically
if is_jetson():
    JetsonProfile.apply_overrides(llm_enhance_settings)
```

#### Step 4.4: Benchmark and Compare

**Run comprehensive benchmarks**:
```bash
# Create benchmark script for Jetson
cat > scripts/benchmark_jetson.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "=== MAIE Jetson Orin Nano Benchmark ==="
echo ""

# System info
echo "System Information:"
cat /etc/nv_tegra_release
nvidia-smi
echo ""

# Benchmark ASR
echo "Benchmarking ASR (Whisper)..."
python scripts/benchmark_maie_llm.py \
  --task enhancement \
  --iterations 10 \
  --output-file benchmark_asr_jetson.json

# Benchmark LLM
echo "Benchmarking LLM (vLLM)..."
python scripts/benchmark_maie_llm.py \
  --task summary \
  --template-id generic_summary_v1 \
  --iterations 10 \
  --output-file benchmark_llm_jetson.json

# E2E workflow
echo "Benchmarking E2E workflow..."
python scripts/benchmark_maie_llm.py \
  --task both \
  --iterations 5 \
  --output-file benchmark_e2e_jetson.json

echo "✅ Benchmarks complete!"
echo "Results saved to benchmark_*_jetson.json"
EOF

chmod +x scripts/benchmark_jetson.sh
./scripts/benchmark_jetson.sh
```

**Compare against targets**:
```python
# Expected performance (vs x86_64 baseline):
targets = {
    "asr_rtf": 0.15,  # Target: <0.2 (vs 0.12 on x86)
    "llm_tokens_per_sec": 40,  # Target: >35 (vs 50-60 on x86)
    "e2e_time_2min_audio": 25,  # Target: <30s (vs 18s on x86)
}
```

---

## Testing Strategy

### Test Levels

#### 1. Unit Tests
- **Scope**: Individual components (processors, utilities)
- **Environment**: Mocked dependencies
- **Execution**: `pytest tests/unit -v`
- **Success Criteria**: 100% pass rate

#### 2. Integration Tests
- **Scope**: Component interactions (API ↔ Worker, Worker ↔ Redis)
- **Environment**: Real dependencies (Redis, file system)
- **Execution**: `pytest tests/integration -v`
- **Success Criteria**: 95%+ pass rate

#### 3. End-to-End Tests
- **Scope**: Full workflows (upload → process → retrieve)
- **Environment**: Full system stack
- **Execution**: `pytest tests/e2e -v -m "not real_llm"`
- **Success Criteria**: 90%+ pass rate

#### 4. Performance Tests
- **Scope**: Throughput, latency, resource usage
- **Environment**: Production-like configuration
- **Execution**: Custom benchmark scripts
- **Success Criteria**: Meet performance targets

### Test Matrix

| Test Category | x86_64 Baseline | Jetson Target | Critical? |
|---------------|----------------|---------------|-----------|
| API Health | ✅ Pass | ✅ Pass | Yes |
| Audio Upload | ✅ Pass | ✅ Pass | Yes |
| ASR Processing | ✅ Pass | ✅ Pass | Yes |
| LLM Enhancement | ✅ Pass | ⚠️ Slower acceptable | Yes |
| LLM Summary | ✅ Pass | ⚠️ Slower acceptable | Yes |
| Diarization | ✅ Pass | ✅ Pass | Yes |
| Queue Management | ✅ Pass | ✅ Pass | No |
| Error Handling | ✅ Pass | ✅ Pass | Yes |

### Known Limitations on Jetson

1. **Memory Constraints**
   - Limited to 8GB total memory
   - Cannot process extremely long audio (>60 min)
   - Single concurrent request only

2. **Performance Degradation**
   - 15-30% slower than x86_64 (acceptable)
   - LLM inference: 40 tok/s vs 60-80 tok/s

3. **Feature Limitations**
   - Reduced context window (8K vs 32K)
   - Lower batch sizes
   - Simplified attention mechanisms (no flashinfer)

---

## Performance Expectations

### Baseline Comparison

| Metric | x86_64 (RTX 4090) | Jetson Orin Nano | Delta |
|--------|-------------------|------------------|-------|
| **ASR RTF** | 0.12 | 0.15-0.18 | +25-50% |
| **LLM (Enhancement)** | 78-80 tok/s | 55-65 tok/s | -20-30% |
| **LLM (Summary)** | 40 tok/s | 30-35 tok/s | -15-25% |
| **E2E (2min audio)** | 18s | 22-28s | +20-55% |
| **Memory Usage** | 16GB VRAM | 6-7GB VRAM | N/A |
| **Power Consumption** | 300W | 15-25W | -90% |

### Target Performance Goals

✅ **Acceptable Performance**:
- ASR RTF < 0.20 (real-time capable)
- LLM > 30 tokens/second (usable)
- E2E processing < 30 seconds for 2-minute audio
- Stable operation for 24/7 deployment

⚠️ **Minimum Performance**:
- ASR RTF < 0.30 (still faster than real-time)
- LLM > 20 tokens/second
- E2E processing < 45 seconds for 2-minute audio

❌ **Unacceptable Performance**:
- ASR RTF > 0.30 (slower than real-time)
- LLM < 15 tokens/second (too slow)
- Frequent OOM errors
- System instability

### Resource Utilization Targets

| Resource | Target | Max Acceptable |
|----------|--------|---------------|
| GPU Memory | 70-85% | 90% |
| CPU Memory | 60-75% | 80% |
| GPU Utilization | 80-95% | 100% |
| CPU Utilization | 50-70% | 85% |
| Swap Usage | 0% | 5% |

---

## Rollback Plan

### Pre-Migration Backup

1. ✅ Git branch: `backup-pre-jetson-migration`
2. ✅ Docker images saved: `docker save maie:latest > maie-x86.tar`
3. ✅ Configuration backup: `backup/` directory

### Rollback Triggers

Execute rollback if:
- Critical functionality broken (ASR or LLM fails)
- Performance <50% of targets
- Stability issues (crashes, OOM errors)
- Cannot complete Phase 3 within timeline
- Hardware limitations insurmountable

### Rollback Procedure

```bash
# Step 1: Stop Jetson deployment
docker-compose -f docker-compose.jetson.yml down

# Step 2: Restore x86_64 codebase
git checkout backup-pre-jetson-migration

# Step 3: Rebuild x86_64 images
docker-compose build

# Step 4: Restart x86_64 deployment
docker-compose up -d

# Step 5: Verify restoration
curl http://localhost:8000/health
```

### Partial Rollback (Hybrid Approach)

If Jetson is partially functional:
- Deploy x86_64 for production
- Use Jetson for development/testing
- Continue migration in parallel

---

## Timeline & Resources

### Phase Breakdown

| Phase | Duration | Key Activities | Dependencies |
|-------|----------|----------------|--------------|
| **Phase 1** | 1-2 weeks | Dependency adaptation, build scripts | Access to Jetson hardware |
| **Phase 2** | 1 week | Docker image creation | Phase 1 complete |
| **Phase 3** | 1-2 weeks | Testing and validation | Phase 2 complete |
| **Phase 4** | 1-2 weeks | Performance optimization | Phase 3 complete |
| **Buffer** | 1 week | Contingency for issues | - |

**Total Duration**: 4-6 weeks (with hardware access)

### Resource Requirements

#### Hardware
- ✅ 1x Jetson Orin Nano Super Developer Kit
- ✅ MicroSD card (64GB+) or NVMe SSD (128GB+)
- ✅ Development workstation (for Docker builds)
- ✅ Network connectivity (stable, for model downloads)

#### Personnel
- 1x Full-stack engineer (primary)
- 0.5x DevOps engineer (Docker, CI/CD)
- 0.25x QA engineer (testing)

**Total Effort**: ~120-160 hours

#### Software/Licenses
- ✅ JetPack SDK 6.2 (free)
- ✅ Docker (free)
- ✅ All dependencies (open source)
- No additional licenses required

---

## Risk Mitigation

### Risk Matrix

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **vLLM build fails** | Medium | High | Use documented GH200 process, allocate debug time |
| **Out of memory** | High | High | Aggressive quantization, sequential processing, monitoring |
| **Performance insufficient** | Medium | Medium | Accept 70-85% target, optimize iteratively |
| **flashinfer required** | Low | High | Workaround: remove dependency, use fallback attention |
| **CTranslate2 build issues** | Low | Medium | Alternative: use whisper.cpp |
| **Timeline overrun** | Medium | Medium | Buffer week included, partial rollback option |
| **Hardware failure** | Low | High | Backup Jetson device, warranty coverage |

### Contingency Plans

#### If vLLM Performance is Inadequate
1. **Alternative**: Use [NanoLLM](https://github.com/dusty-nv/nanollm) (Jetson-optimized)
2. **Alternative**: Use TensorRT-LLM for Jetson
3. **Alternative**: Use lighter models (Phi-3, Qwen-1.5B)
4. **Alternative**: Offload LLM to cloud API (hybrid)

#### If Memory is Insufficient
1. Reduce model sizes (Whisper small/base instead of large)
2. Implement model swapping (load/unload between tasks)
3. Use external swap (USB SSD swap space)
4. Split workloads (ASR on Jetson, LLM on cloud)

#### If ASR Performance is Poor
1. **Alternative**: Use [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (optimized C++)
2. **Alternative**: Use lighter Whisper models
3. **Alternative**: NVIDIA Riva ASR service
4. Disable VAD filtering (trade accuracy for speed)

---

## Success Criteria

### Minimum Viable Product (MVP)

✅ **Core Functionality**:
- [ ] API server starts and responds to health checks
- [ ] Audio upload and preprocessing works
- [ ] ASR transcription completes successfully
- [ ] LLM enhancement produces valid output
- [ ] LLM summary generates structured JSON
- [ ] E2E workflow completes without errors

✅ **Performance**:
- [ ] ASR RTF < 0.25 (faster than real-time)
- [ ] LLM > 25 tokens/second (minimum usable)
- [ ] E2E processing < 40 seconds for 2-minute audio
- [ ] No OOM errors during normal operation

✅ **Stability**:
- [ ] 24-hour continuous operation without crashes
- [ ] Queue processing handles backlog
- [ ] Memory usage stays within limits
- [ ] GPU utilization optimized

### Production-Ready

✅ **All MVP criteria** +
- [ ] 80%+ test suite passing
- [ ] Performance within 70-85% of x86_64
- [ ] Full documentation updated
- [ ] Deployment scripts finalized
- [ ] Monitoring and logging functional
- [ ] Backup and recovery tested

---

## Appendix

### A. Useful Commands

#### Jetson-Specific

```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Monitor GPU/CPU/Memory in real-time
jtop

# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check CUDA version
nvcc --version

# Check available memory
free -h
```

#### Docker on Jetson

```bash
# Build for ARM64
docker build --platform linux/arm64 -f Dockerfile.jetson -t maie:jetson .

# Check GPU access in container
docker run --rm --runtime nvidia nvidia/cuda:12.6-base-ubuntu22.04 nvidia-smi

# Monitor container resources
docker stats maie-worker-jetson
```

#### Debugging

```bash
# Check vLLM installation
python -c "import vllm; print(vllm.__version__)"

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# Check CTranslate2
python -c "import ctranslate2; print(ctranslate2.__version__)"

# Test GPU in container
docker exec maie-worker-jetson python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### B. Reference Links

- [NVIDIA JetPack SDK 6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62)
- [Jetson Orin Nano Super Product Page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- [L4T PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)
- [vLLM ARM64 Build](https://docs.vllm.ai/en/latest/deployment/docker)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)
- [Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [NanoLLM Project](https://github.com/dusty-nv/nanollm)

### C. Troubleshooting Guide

#### Issue: Docker build fails with "out of memory"
**Solution**: Increase swap space:
```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Issue: vLLM compilation takes too long
**Solution**: Reduce parallelism:
```bash
export MAX_JOBS=2
export NVCC_THREADS=1
```

#### Issue: Container cannot access GPU
**Solution**: Install nvidia-container-runtime:
```bash
sudo apt-get install nvidia-container-runtime
sudo systemctl restart docker
```

#### Issue: Model download fails
**Solution**: Pre-download models before building:
```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('erax-ai/EraX-WoW-Turbo-V1.1-CT2'); \
  snapshot_download('cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit')"
```

---

**Document Version**: 1.0  
**Last Updated**: November 25, 2025  
**Status**: Ready for Implementation  
**Next Review**: After Phase 1 completion

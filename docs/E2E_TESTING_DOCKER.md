# E2E Testing with Docker - Complete Guide

## Overview

This guide provides step-by-step instructions for running End-to-End (E2E) tests using Docker and Docker Compose. This approach provides isolation, reproducibility, and simplified deployment compared to local installation.

**Benefits of Docker approach**:

- ✅ Isolated environment (no conflicts with system packages)
- ✅ Reproducible builds (consistent across machines)
- ✅ Easy cleanup (remove containers when done)
- ✅ Production-like environment
- ✅ GPU support via NVIDIA Container Toolkit

## Prerequisites

### Required Software

1. **Docker Engine** 24.0+ with Compose V2
2. **NVIDIA Container Toolkit** (for GPU support)
3. **NVIDIA GPU** with ≥16GB VRAM
4. **NVIDIA Drivers** 525.60.13+ (CUDA 12.1 compatible)

### Installation

#### Install Docker (Ubuntu/Debian)

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine and Compose
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

#### Install NVIDIA Container Toolkit

```bash
# Configure the repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

**Expected output**: You should see your GPU information displayed.

#### Add User to Docker Group (Optional - avoid sudo)

```bash
# Add current user to docker group
sudo usermod -aG docker $USER

# Log out and log back in for changes to take effect
# Or run: newgrp docker

# Verify you can run docker without sudo
docker ps
```

## Project Setup

### 1. Clone and Configure

```bash
# Clone repository
git clone <repository-url>
cd maie

# Copy environment template
cp .env.template .env

# Edit .env file with your settings
nano .env  # or use your preferred editor
```

**Required .env variables**:

```bash
SECRET_API_KEY=your-secret-key-here
WHISPER_MODEL_VARIANT=erax-wow-turbo
LLM_MODEL_NAME=cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit
GPU_MEMORY_UTILIZATION=0.9
```

### 2. Download AI Models

```bash
# Create models directory
mkdir -p data/models

# Download models (this may take 10-30 minutes)
./scripts/download-models.sh
```

**What this does**:

- Downloads Whisper ASR models (~1-3GB)
- Downloads LLM models (~2-8GB)
- Stores in `data/models/` for container mounting

### 3. Create Required Directories

```bash
# Create all necessary directories
mkdir -p data/audio data/redis data/models templates/prompts

# Set appropriate permissions (important for container access)
chmod -R 755 data/
```

## Docker Commands Explained

### Core Docker Compose Commands

#### 1. Build Docker Images

```bash
docker compose build
```

**What it does**:

- Reads `docker-compose.yml` and `Dockerfile`
- Builds container images for API and worker services
- Downloads base images (nvidia/cuda, redis, etc.)
- Installs dependencies inside containers
- Creates optimized production images

**When to use**: First time setup, after Dockerfile changes, or after dependency updates.

**Flags**:

- `--no-cache`: Force rebuild without using cache (useful after major changes)
- `--pull`: Always pull newer versions of base images
- `--progress plain`: Show detailed build output

```bash
# Full rebuild without cache
docker compose build --no-cache --pull
```

#### 2. Start Services

```bash
docker compose up -d
```

**What it does**:

- **`docker compose`**: Uses Docker Compose V2 (plugin-based)
- **`up`**: Creates and starts containers defined in docker-compose.yml
- **`-d`**: Detached mode (runs in background)

**Services started**:

1. **redis**: Job queue and results storage
2. **api**: HTTP API server (port 8000)
3. **worker**: GPU-accelerated processing worker
4. **rq-dashboard**: Queue monitoring UI (port 9181)
5. **jaeger**: Distributed tracing (optional, port 16686)

**Alternative options**:

```bash
# Start in foreground (see logs in terminal)
docker compose up

# Start specific services only
docker compose up -d redis api

# Recreate containers (useful after config changes)
docker compose up -d --force-recreate
```

#### 3. Check Service Status

```bash
docker compose ps
```

**What it does**:

- Lists all containers defined in docker-compose.yml
- Shows container status (running, exited, restarting)
- Displays mapped ports
- Shows health check status

**Sample output**:

```
NAME                IMAGE               STATUS              PORTS
maie-api            maie-api:latest     Up 2 minutes        0.0.0.0:8000->8000/tcp (healthy)
maie-worker         maie-worker:latest  Up 2 minutes        (healthy)
maie-redis          redis:7-alpine      Up 2 minutes        0.0.0.0:6379->6379/tcp (healthy)
maie-rq-dashboard   eoranged/rq-dash    Up 2 minutes        0.0.0.0:9181->9181/tcp
```

#### 4. View Logs

```bash
# View logs from all services
docker compose logs

# Follow logs in real-time
docker compose logs -f

# View logs from specific service
docker compose logs -f worker

# View last 100 lines
docker compose logs --tail=100

# View logs with timestamps
docker compose logs -f -t
```

**What it does**:

- Displays stdout/stderr from containers
- **`-f`**: Follow mode (continuous stream)
- **`--tail=N`**: Show last N lines
- **`-t`**: Include timestamps

**Useful for**:

- Debugging processing issues
- Monitoring GPU model loading
- Tracking API requests
- Identifying errors

#### 5. Execute Commands Inside Containers

```bash
# Check Redis queue depth
docker compose exec redis redis-cli LLEN rq:queue:default

# Check GPU in worker container
docker compose exec worker nvidia-smi

# Access container shell
docker compose exec worker bash

# Run Python command in worker
docker compose exec worker python -c "import torch; print(torch.cuda.is_available())"
```

**What it does**:

- **`exec`**: Runs command in running container
- **`<service-name>`**: Target service from docker-compose.yml
- **`<command>`**: Command to execute

#### 6. Stop Services

```bash
# Stop all services (preserves containers)
docker compose stop

# Stop specific service
docker compose stop worker

# Stop and remove containers (keeps volumes/data)
docker compose down

# Stop and remove everything including volumes (⚠️ deletes data)
docker compose down -v
```

**What it does**:

- **`stop`**: Stops containers but keeps them
- **`down`**: Stops and removes containers, networks
- **`-v`**: Also removes volumes (Redis data, uploaded audio)

#### 7. Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart worker

# Restart with rebuild
docker compose up -d --build --force-recreate
```

**When to use**:

- After changing environment variables in `.env`
- After code changes (with `--build`)
- After GPU/memory issues (restart worker)

## E2E Testing Workflow

### Complete Testing Process

#### Step 1: Start the Stack

```bash
# Start all services
docker compose up -d

# Wait for services to be healthy (30-120 seconds for GPU model loading)
echo "Waiting for services to be ready..."
timeout 300 bash -c 'until curl -f http://localhost:8000/health; do echo "Waiting..."; sleep 5; done'
```

**What happens**:

1. Redis starts (1-2 seconds)
2. API starts, connects to Redis (5-10 seconds)
3. Worker starts, loads GPU models (30-120 seconds)
   - Loads Whisper model into GPU memory
   - Initializes vLLM with LLM model
4. Health checks pass
5. System ready for requests

#### Step 2: Verify Services

```bash
# Check all containers are running
docker compose ps

# Check API health
curl http://localhost:8000/health

# Check worker can access GPU
docker compose exec worker nvidia-smi

# Check Redis connectivity
docker compose exec redis redis-cli ping

# View service logs
docker compose logs --tail=50 worker
```

#### Step 3: Run Manual Test

```bash
# Submit a test audio file
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'features=["clean_transcript","summary"]' \
  -F "template_id=meeting_notes_v1" \
  http://localhost:8000/v1/process
```

**Expected response**:

```json
{
  "task_id": "abc123def456",
  "status": "QUEUED",
  "message": "Processing started"
}
```

**What happens**:

1. API validates request and saves audio file
2. Task stored in Redis
3. Job enqueued for worker
4. Worker picks up job from queue
5. Audio preprocessed (16kHz mono conversion)
6. ASR processes audio on GPU
7. LLM generates summary on GPU
8. Results stored in Redis
9. Status becomes "COMPLETE"

#### Step 4: Monitor Processing

```bash
# Get task ID from previous response
TASK_ID="abc123def456"

# Poll for status
while true; do
  STATUS=$(curl -s -H "X-API-Key: your-secret-key" \
    http://localhost:8000/v1/status/$TASK_ID | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" = "COMPLETE" ] || [ "$STATUS" = "FAILED" ]; then
    break
  fi

  sleep 5
done

# Get final results
curl -s -H "X-API-Key: your-secret-key" \
  http://localhost:8000/v1/status/$TASK_ID | jq '.'
```

**Monitor worker in real-time**:

```bash
# Open separate terminal and watch worker logs
docker compose logs -f worker
```

#### Step 5: Run Automated Tests

```bash
# Set environment variables
export API_BASE_URL=http://localhost:8000
export SECRET_API_KEY=your-secret-key

# Run E2E tests from host (requires Python)
pytest tests/e2e/ -v

# Or run tests inside a container
docker compose run --rm api pytest tests/e2e/ -v
```

## Monitoring and Debugging

### RQ Dashboard (Queue Monitoring)

```bash
# Access RQ Dashboard in browser
open http://localhost:9181
```

**What you see**:

- Active workers and their status
- Queue depth (pending jobs)
- Failed jobs with error details
- Processing statistics

### Jaeger Tracing (Optional)

```bash
# Access Jaeger UI
open http://localhost:16686
```

**What it provides**:

- Distributed tracing across services
- Request flow visualization
- Performance bottleneck identification

### GPU Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 docker compose exec worker nvidia-smi

# GPU utilization log
docker compose exec worker nvidia-smi dmon -s u -d 1
```

### Redis Queue Inspection

```bash
# Check queue depth
docker compose exec redis redis-cli LLEN rq:queue:default

# List queued job IDs (first 10)
docker compose exec redis redis-cli LRANGE rq:queue:default 0 9

# Check failed queue
docker compose exec redis redis-cli LLEN rq:queue:failed

# Clear all queues (⚠️ destructive)
docker compose exec redis redis-cli FLUSHDB
```

### Container Resource Usage

```bash
# View resource usage (CPU, memory, network)
docker stats

# View resource usage for specific service
docker stats maie-worker
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Container Won't Start

**Symptom**: Service exits immediately after starting

```bash
# Check logs for errors
docker compose logs api
docker compose logs worker

# Check if port is already in use
sudo lsof -i :8000

# Recreate containers
docker compose down
docker compose up -d --force-recreate
```

#### 2. GPU Not Accessible

**Symptom**: Worker can't find GPU

```bash
# Verify NVIDIA Container Toolkit is installed
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Check Docker daemon configuration
cat /etc/docker/daemon.json

# Restart Docker daemon
sudo systemctl restart docker

# Rebuild worker with GPU support
docker compose build --no-cache worker
docker compose up -d worker
```

#### 3. Models Not Found

**Symptom**: "Model not found" errors in worker logs

```bash
# Verify models exist on host
ls -lh data/models/

# Check volume mounting
docker compose exec worker ls -l /data/models/

# Re-download models
./scripts/download-models.sh

# Recreate worker to remount volumes
docker compose up -d --force-recreate worker
```

#### 4. Out of Memory (GPU)

**Symptom**: CUDA out of memory errors

```bash
# Check GPU memory usage
docker compose exec worker nvidia-smi

# Reduce GPU memory utilization in .env
# Edit: GPU_MEMORY_UTILIZATION=0.8 (from 0.9)

# Restart worker to apply changes
docker compose restart worker

# Ensure only one worker is running
docker compose ps | grep worker
```

#### 5. Redis Connection Failed

**Symptom**: "Connection refused" to Redis

```bash
# Check Redis is running
docker compose ps redis

# Check Redis logs
docker compose logs redis

# Test Redis connectivity
docker compose exec redis redis-cli ping

# Restart Redis
docker compose restart redis
```

#### 6. Slow Processing

**Symptom**: Jobs take too long

```bash
# Monitor GPU utilization
docker compose exec worker nvidia-smi

# Check if GPU is actually being used
docker compose logs worker | grep -i "gpu\|cuda"

# Check CPU usage
docker stats

# Verify worker is processing jobs
docker compose exec redis redis-cli LLEN rq:queue:default

# Check for model loading issues
docker compose logs worker | grep -i "model"
```

### Debug Commands

```bash
# Interactive shell in worker container
docker compose exec worker bash

# Test imports inside container
docker compose exec worker python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Test ASR processor
docker compose exec worker python -c "
from src.processors.asr.whisper import WhisperProcessor
processor = WhisperProcessor()
print('Whisper processor loaded successfully')
"

# Check environment variables in container
docker compose exec worker env | grep -E "WHISPER|LLM|GPU"
```

## Cleanup and Maintenance

### Stop and Remove Everything

```bash
# Stop services
docker compose stop

# Remove containers (keeps volumes)
docker compose down

# Remove containers AND volumes (⚠️ deletes all data)
docker compose down -v

# Remove images (frees disk space)
docker rmi maie-api maie-worker

# Full cleanup (everything)
docker compose down -v --rmi all
```

### Prune Docker Resources

```bash
# Remove unused containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused (⚠️ careful!)
docker system prune -a --volumes
```

### Update Images

```bash
# Pull latest base images
docker compose pull

# Rebuild with latest dependencies
docker compose build --pull --no-cache

# Restart with new images
docker compose up -d --force-recreate
```

## Performance Optimization

### Resource Limits

Edit `docker-compose.yml` to add resource constraints:

```yaml
services:
  worker:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: "8"
          memory: 16G
        reservations:
          memory: 8G
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

### Multiple Workers (Advanced)

```bash
# Scale workers (use with caution - GPU memory!)
docker compose up -d --scale worker=2

# Note: Ensure sufficient GPU memory for multiple models
```

## Quick Reference

### Essential Commands

```bash
# Start everything
docker compose up -d

# Stop everything
docker compose down

# View logs
docker compose logs -f

# Restart a service
docker compose restart worker

# Execute command
docker compose exec worker <command>

# Check status
docker compose ps

# Health check
curl http://localhost:8000/health

# Monitor GPU
docker compose exec worker nvidia-smi

# Check queue
docker compose exec redis redis-cli LLEN rq:queue:default
```

### Environment Variables

```bash
# Required in .env file
SECRET_API_KEY=your-key
WHISPER_MODEL_VARIANT=erax-wow-turbo
LLM_MODEL_NAME=cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit

# Optional tuning
GPU_MEMORY_UTILIZATION=0.9  # 0.7-0.95
JOB_TIMEOUT=600             # seconds
LOG_LEVEL=INFO              # DEBUG|INFO|WARNING|ERROR
```

### Useful Ports

- **8000**: API HTTP server
- **6379**: Redis (internal)
- **9181**: RQ Dashboard
- **16686**: Jaeger UI

---

**This Docker-based approach provides a production-ready, isolated environment for E2E testing with full GPU support and comprehensive monitoring capabilities.**

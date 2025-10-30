# E2E Testing - Hybrid Approach Quick Start

## Overview

This guide shows you how to run E2E tests with the **recommended hybrid approach**:

- ✅ **Redis in Docker** (no system installation, easy management)
- ✅ **API and Worker run locally** (easy debugging, fast iteration)

## Prerequisites

- **Pixi** installed ([https://pixi.sh](https://pixi.sh))
- **Docker** installed (for Redis only)
- **NVIDIA GPU** with drivers installed
- **Models downloaded** (run once)

## Complete Setup (First Time)

```bash
# 1. Clone and setup
git clone <repository-url>
cd maie
cp .env.template .env

# 2. Install dependencies
pixi install

# 3. Download AI models (10-30 minutes, run once)
./scripts/download-models.sh

# 4. Edit .env file
nano .env
# Set: SECRET_API_KEY=your-key-here
# Set: REDIS_URL=redis://localhost:6379/0
```

## Daily Development Workflow

### Step 1: Start Redis (Once per session)

```bash
# Start Redis in Docker
docker run -d --name maie-redis -p 6379:6379 redis:latest

# Verify it's running
docker exec maie-redis redis-cli ping
# Expected output: PONG
```

**What this does**:

- `-d`: Runs in background
- `--name maie-redis`: Names container for easy reference
- `-p 6379:6379`: Exposes Redis on localhost:6379
- `redis:latest`: Latest Redis version with Debian base (~100MB)

### Step 2: Start API Server (Terminal 1)

```bash
# Start API with auto-reload
./scripts/dev.sh --api-only --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Test it**:

```bash
# In another terminal
curl http://localhost:8000/health
# Expected: {"status":"healthy"}
```

### Step 3: Start Worker (Terminal 2)

```bash
# Start worker process
./scripts/dev.sh --worker-only
```

You should see:

```
Worker starting...
Loading Whisper model...
Loading LLM model...
Worker ready!
```

### Step 4: Run E2E Tests (Terminal 3)

```bash
# Set environment variables
export API_BASE_URL=http://localhost:8000
export SECRET_API_KEY=your-secret-key  # Match your .env

# Run all E2E tests
pixi run pytest tests/e2e/ -v

# Or run specific test
pixi run pytest tests/e2e/test_core_workflow.py::TestCoreWorkflow::test_happy_path_whisper -v
```

## Stopping and Cleanup

```bash
# Stop API and Worker
# Press Ctrl+C in their respective terminals

# Stop Redis (keeps data)
docker stop maie-redis

# Or remove Redis completely (deletes data)
docker rm -f maie-redis
```

## Restarting Later

```bash
# If Redis container exists but stopped
docker start maie-redis

# Then start API and Worker again as in Steps 2-3
```

## Troubleshooting

### Redis Not Running

```bash
# Check if Redis is running
docker ps | grep maie-redis

# If not found, start it
docker run -d --name maie-redis -p 6379:6379 redis:latest

# If "container name already in use"
docker rm maie-redis  # Remove old container
docker run -d --name maie-redis -p 6379:6379 redis:latest
```

### API/Worker Can't Connect to Redis

```bash
# Check Redis is accessible
docker exec maie-redis redis-cli ping
# Should return: PONG

# Check .env has correct URL
grep REDIS_URL .env
# Should be: REDIS_URL=redis://localhost:6379/0
```

### GPU Not Found

```bash
# Check GPU is available
nvidia-smi

# Check CUDA in Pixi environment
pixi run python -c "import torch; print(torch.cuda.is_available())"
# Should return: True
```

### Worker Shows Model Errors

```bash
# Check models exist
ls -lh data/models/

# If missing, download them
./scripts/download-models.sh
```

## Redis Persistence (Optional - Not Recommended for E2E)

**For E2E testing, AOF is NOT needed** - it adds startup overhead and complicates cleanup.

If you want to persist Redis data across container restarts (rarely needed for E2E):

```bash
# Create volume for data persistence
docker volume create maie-redis-data

# Run Redis with AOF persistence (adds ~10-20% overhead)
docker run -d \
  --name maie-redis \
  -p 6379:6379 \
  -v maie-redis-data:/data \
  redis:latest redis-server --appendonly yes --appendfsync everysec

# Data survives container removal
docker rm -f maie-redis
docker run -d --name maie-redis -p 6379:6379 -v maie-redis-data:/data redis:latest
```

**Note**: AOF slows down Redis and makes container startup slower. For E2E testing, use the basic command without AOF.

## Useful Commands

### Redis Management

```bash
# View Redis logs
docker logs -f maie-redis

# Check queue depth
docker exec maie-redis redis-cli LLEN rq:queue:default

# Clear queue (⚠️ destructive)
docker exec maie-redis redis-cli FLUSHDB

# Interactive Redis CLI
docker exec -it maie-redis redis-cli
```

### Monitor GPU

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check GPU from Pixi
pixi run python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

### Run Specific Tests

```bash
# All E2E tests
pixi run pytest tests/e2e/ -v

# Specific test file
pixi run pytest tests/e2e/test_core_workflow.py -v

# Specific test function
pixi run pytest tests/e2e/test_core_workflow.py::TestCoreWorkflow::test_happy_path_whisper -v

# With debug logging
pixi run pytest tests/e2e/ -vv --log-cli-level=DEBUG

# Test suite breakdown:
# - Unit tests (fast): pytest -m "not integration" --ignore=tests/e2e -q (~1.8 min)
# - Integration tests: pytest -m "integration" -v (~1.2 min)
# - Full suite: pytest --ignore=tests/e2e -q (~2.8 min)
#
# Current Status: 836 unit tests + 34 integration tests passing (100% pass rate)
# See docs/TDD.md section 6.2-6.3 for complete testing strategy
```

## Manual Testing

### Submit Test Request

```bash
# Submit audio file
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'features=["clean_transcript","summary"]' \
  -F "template_id=meeting_notes_v1" \
  http://localhost:8000/v1/process
```

Response:

```json
{
  "task_id": "abc123...",
  "status": "PENDING"
}
```

### Check Status

```bash
# Replace with your task_id
curl -H "X-API-Key: your-secret-key" \
  http://localhost:8000/v1/status/abc123...
```

### Monitor Processing

```bash
# Watch worker terminal for processing logs
# Watch queue depth
watch -n 1 docker exec maie-redis redis-cli LLEN rq:queue:default

# Watch GPU usage
watch -n 1 nvidia-smi
```

## Advantages of This Approach

### Why Redis in Docker?

✅ **No system installation**: Don't need to install Redis on your system  
✅ **Easy start/stop**: Single command to start/stop  
✅ **Clean isolation**: Doesn't interfere with other Redis instances  
✅ **Easy reset**: `docker rm -f maie-redis` completely removes it  
✅ **Version control**: Always use same Redis version

### Why API/Worker Local?

✅ **Fast iteration**: No container rebuilds after code changes  
✅ **Easy debugging**: Use Python debugger, breakpoints, print statements  
✅ **Direct logs**: See output directly in terminal  
✅ **GPU access**: No Docker GPU setup needed  
✅ **Quick restarts**: Just Ctrl+C and restart

## Pro Tips

1. **Keep Redis running between sessions**

   ```bash
   # Just stop it, don't remove
   docker stop maie-redis
   # Next time: docker start maie-redis
   ```

2. **Monitor everything in one terminal**

   ```bash
   # Use tmux or screen to split terminals
   tmux new-session \; split-window -v \; split-window -h
   ```

3. **Create aliases for common commands**

````bash
# Add to ~/.zshrc
alias redis-start='docker start maie-redis || docker run -d --name maie-redis -p 6379:6379 redis:latest'
alias redis-stop='docker stop maie-redis'
alias redis-clean='docker rm -f maie-redis'
alias maie-api='cd ~/maie && ./scripts/dev.sh --api-only --port 8000'
alias maie-worker='cd ~/maie && ./scripts/dev.sh --worker-only'
```4. **Run tests while developing**
   ```bash
   # In separate terminal, run tests after each change
   pixi run pytest tests/e2e/test_core_workflow.py -v
````

## Summary

**Hybrid approach = Best of both worlds!**

🐳 Redis in Docker = Easy management  
💻 API/Worker local = Easy debugging  
🚀 Fast iteration = Happy developer

For full documentation:

- **Hybrid/Local Guide**: `docs/E2E_TESTING_GUIDE.md`
- **Full Docker Guide**: `docs/E2E_TESTING_DOCKER.md`
- **Test Directory**: `tests/e2e/README.md`
- **Testing Strategy & Implementation**: `docs/TDD.md` sections 6.1-6.4
  - Unit tests with mock factories (836 tests, 100% pass rate)
  - Integration tests with real components (34 tests, 100% pass rate)
  - Optional E2E tests for LLM, API, GPU validation
  - CI/CD pipeline recommendations

**Recent Improvements:**

✅ 100% test pass rate (859 tests → cleanup → 836 tests)  
✅ Comprehensive mock factory pattern for reusable fixtures  
✅ Integration tests properly separated for fast unit test execution  
✅ 23 deprecated IoU tests removed  
✅ 11 intentional skipped tests documented with setup instructions  
✅ Complete testing guide in `docs/TDD.md`

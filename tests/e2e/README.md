````markdown
# E2E Tests for MAIE

This directory contains End-to-End (E2E) tests for the Modular Audio Intelligence Engine.

## Testing Approaches

MAIE supports three approaches for E2E testing:

### 1. **Hybrid Approach** (⭐ Recommended for Development)

- ✅ **Redis in Docker** (easy management, no system installation)
- ✅ **API and Worker run locally** (direct debugging, faster iteration)
- ✅ Best of both worlds: isolation + debugging ease
- ✅ Quick setup with single command

**Quick Start**:

```bash
# Start Redis in Docker
docker run -d --name maie-redis -p 6379:6379 redis:latest

# Start API and Worker locally (see below)
./scripts/dev.sh --api-only --port 8000    # Terminal 1
./scripts/dev.sh --worker-only              # Terminal 2
```

### 2. **Full Docker-Based Testing** (Production-Like)

- ✅ Complete isolation with all services in containers
- ✅ Production-like setup with Docker Compose
- ✅ GPU support via NVIDIA Container Toolkit
- ✅ Complete guide: **[`docs/E2E_TESTING_DOCKER.md`](../docs/E2E_TESTING_DOCKER.md)**

### 3. **Fully Local Testing** (All Services on Host)

- ✅ Everything runs directly on host system
- ✅ Requires Redis system installation
- ✅ Complete guide: **[`docs/E2E_TESTING_GUIDE.md`](../docs/E2E_TESTING_GUIDE.md)**

## Quick Start (Hybrid - Recommended) ⭐

```bash
# 1. Install dependencies
pixi install

# 2. Download models
pixi run download-models

# 3. Setup environment
cp .env.template .env
# Edit .env with REDIS_URL=redis://localhost:6379/0

# 4. Start Redis in Docker (single command!)
docker run -d --name maie-redis -p 6379:6379 redis:latest

# Verify Redis is running
docker ps | grep maie-redis
docker exec maie-redis redis-cli ping  # Should return "PONG"

# 5. Start API (Terminal 1)
./scripts/dev.sh --api-only --host 0.0.0.0 --port 8000

# 6. Start Worker (Terminal 2)
./scripts/dev.sh --worker-only

# 7. Run E2E tests (Terminal 3)
export API_BASE_URL=http://localhost:8000
export SECRET_API_KEY=your-secret-key
pixi run pytest tests/e2e/ -v
```

**Cleanup**:

```bash
# Stop and remove Redis container
docker stop maie-redis && docker rm maie-redis

# Or just stop (keeps data for next time)
docker stop maie-redis

# Restart Redis later
docker start maie-redis
```

## Quick Start (Docker)

```bash
# 1. Setup environment
cp .env.template .env
# Edit .env with your configuration

# 2. Download models
./scripts/download-models.sh

# 3. Start services with Docker Compose
docker compose up -d

# 4. Wait for services to be ready
timeout 300 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'

# 5. Run E2E tests
export API_BASE_URL=http://localhost:8000
export SECRET_API_KEY=your-secret-key
pytest tests/e2e/ -v

# Or use test script
./scripts/test.sh --e2e
```

## Quick Start (Local)

```bash
# 1. Install dependencies
pixi install

# 2. Download models
pixi run download-models

# 3. Start Redis
sudo systemctl start redis-server

# 4. Start API (Terminal 1)
./scripts/dev.sh --api-only --host 0.0.0.0 --port 8000

# 5. Start worker (Terminal 2)
./scripts/dev.sh --worker-only

# 6. Run E2E tests (Terminal 3)
export API_BASE_URL=http://localhost:8000
export SECRET_API_KEY=your-secret-key
pixi run pytest tests/e2e/ -v
```

## Test Structure

- `conftest.py`: Shared fixtures and API client
- `test_core_workflow.py`: Core happy path tests
- `assets/`: Test audio files and data
- `golden/`: Expected result baselines

## Redis Docker Commands Explained

### Start Redis Container

```bash
docker run -d --name maie-redis -p 6379:6379 redis:latest
```

**What each part does**:

- `docker run`: Creates and starts a new container
- `-d`: Detached mode (runs in background)
- `--name maie-redis`: Names the container "maie-redis" for easy reference
- `-p 6379:6379`: Maps port 6379 from container to host (Redis default port)
- `redis:latest`: Uses the latest Redis version with Debian base (~100MB)

### Check Redis Status

```bash
# List running containers
docker ps

# Check Redis logs
docker logs maie-redis

# Follow logs in real-time
docker logs -f maie-redis

# Test Redis connection
docker exec maie-redis redis-cli ping  # Returns "PONG"
```

### Redis Management

```bash
# Stop Redis (keeps data)
docker stop maie-redis

# Start stopped Redis
docker start maie-redis

# Restart Redis
docker restart maie-redis

# Remove Redis container (⚠️ deletes data)
docker rm -f maie-redis

# View Redis memory/CPU usage
docker stats maie-redis
```

### Redis Operations

```bash
# Check queue depth
docker exec maie-redis redis-cli LLEN rq:queue:default

# View queued items (first 10)
docker exec maie-redis redis-cli LRANGE rq:queue:default 0 9

# Clear all Redis data (⚠️ destructive)
docker exec maie-redis redis-cli FLUSHDB

# Get Redis info
docker exec maie-redis redis-cli INFO

# Interactive Redis CLI
docker exec -it maie-redis redis-cli
```

### Redis with Persistence (Optional - Not Recommended for E2E)

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

## Test Categories

### Core Workflow Tests

- Happy path processing with Whisper backend
- Happy path processing with ChunkFormer backend
- Feature combination testing
- Result structure validation

### Error Handling Tests

- Invalid file formats
- Missing required parameters
- Queue backpressure testing

### Performance Tests

- Processing time validation
- Concurrent request handling
- Resource usage monitoring

## Manual Testing

For manual E2E validation:

```bash
# Submit audio for processing
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'features=["clean_transcript","summary"]' \
  -F "template_id=meeting_notes_v1" \
  http://localhost:8000/v1/process

# Monitor progress
curl -H "X-API-Key: your-secret-key" \
  http://localhost:8000/v1/status/{task_id}
```

## Validation

Use the validation script to check results:

```bash
python scripts/validate-e2e-results.py result.json
```

## Troubleshooting

### Common Issues

**Services not ready**:

- Check API: `curl http://localhost:8000/health`
- Check Redis: `redis-cli ping`
- Check worker: `ps aux | grep worker`

**GPU memory errors**:

- Monitor with `nvidia-smi`
- Ensure only one worker is running
- Restart worker to clear GPU memory

**Model not found**:

- Run `./scripts/download-models.sh`
- Verify files exist: `ls -lh data/models/`

**Test timeouts**:

- Increase timeout values for longer audio files
- Check worker logs for bottlenecks
- Monitor GPU utilization: `watch -n 1 nvidia-smi`

### Debug Commands

```bash
# Check service status
ps aux | grep -E "uvicorn|worker"

# View API logs (in API terminal)

# View worker logs (in worker terminal)

# Monitor GPU
watch -n 1 nvidia-smi

# Check queue status
redis-cli LLEN rq:queue:default

# Clear Redis queue
redis-cli FLUSHDB
```

## Running Specific Tests

```bash
# Run all E2E tests
pixi run pytest tests/e2e/ -v

# Run specific test file
pixi run pytest tests/e2e/test_core_workflow.py -v

# Run specific test
pixi run pytest tests/e2e/test_core_workflow.py::TestCoreWorkflow::test_happy_path_whisper -v

# Run with detailed output
pixi run pytest tests/e2e/ -vv --tb=long --log-cli-level=DEBUG

# Run with coverage
pixi run pytest tests/e2e/ --cov=src --cov-report=html
```

## CI/CD Integration

E2E tests can run in CI/CD pipelines with GPU-enabled runners. See `docs/E2E_TESTING_GUIDE.md` for the complete pipeline configuration.

## Further Documentation

For comprehensive E2E testing instructions, troubleshooting, and advanced scenarios, see `docs/E2E_TESTING_GUIDE.md`.
````

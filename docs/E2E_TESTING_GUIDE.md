# End-to-End Testing Guide for MAIE

## Overview

This guide provides comprehensive instructions for performing End-to-End (E2E) tests and validation of the Modular Audio Intelligence Engine (MAIE). E2E testing validates the complete system workflow from API request to final results, ensuring all components work together correctly.

**Testing Approach**: This guide focuses on local development testing with a **hybrid approach** - running API and Worker locally for easy debugging, while using **Redis in Docker** for simplified management.

## Why Hybrid Approach? ⭐

The recommended setup combines the best of both worlds:

✅ **Redis in Docker**:

- No system Redis installation needed
- Easy to start/stop/reset
- Isolated from other Redis instances
- Single command setup

✅ **API & Worker Local**:

- Direct Python debugging (breakpoints, print statements)
- Faster code iteration (no container rebuilds)
- See logs directly in terminal
- GPU access without Docker GPU setup

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with ≥16GB VRAM (recommended: 24GB+ for concurrent testing)
- **CPU**: 4+ cores with AVX2 support
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space for models and test data

### Software Requirements

- **Python**: 3.12+ (managed via Pixi)
- **Pixi**: Package and environment manager ([installation guide](https://pixi.sh))
- **Redis**: Local Redis server for task queue (install via system package manager)
- **CUDA**: NVIDIA CUDA 12.1+ drivers and toolkit
- **curl**: For API testing
- **ffmpeg**: For audio validation (automatically managed by Pixi)

### Environment Setup

1. **Clone and Setup Project**:

```bash
git clone <repository-url>
cd maie
cp .env.template .env
# Edit .env with your configuration
```

2. **Install Pixi** (if not already installed):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
# Restart your shell or source the profile
```

3. **Install Dependencies**:

```bash
# Pixi will automatically create an isolated environment and install all dependencies
pixi install
```

4. **Download AI Models** (Required for E2E):

```bash
./scripts/download-models.sh
# Or use Pixi task
pixi run download-models
```

5. **Verify GPU Setup**:

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA is accessible in Pixi environment
pixi run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

6. **Start Redis in Docker** (Recommended - Hybrid Approach):

```bash
# Start Redis container (single command!)
docker run -d --name maie-redis -p 6379:6379 redis:latest

# Verify Redis is running
docker ps | grep maie-redis
docker exec maie-redis redis-cli ping  # Should return "PONG"
```

**What this command does**:

- `docker run`: Creates and starts a container
- `-d`: Detached mode (runs in background)
- `--name maie-redis`: Names container for easy reference
- `-p 6379:6379`: Exposes Redis port to localhost
- `redis:latest`: Uses the latest Redis version with Debian base (~100MB image)

**Note**: This basic Redis setup uses no persistence (no AOF/RDB) for faster startup and simpler cleanup during E2E testing. For production, use AOF with `--appendonly yes --appendfsync everysec`.

**Alternative: System Redis Installation**:

If you prefer not to use Docker for Redis:

```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server

# Start Redis service
sudo systemctl start redis-server

# Verify Redis is running
redis-cli ping  # Should return "PONG"
```

7. **Configure Environment**:

Ensure your `.env` file has the correct Redis URL:

```bash
# For Docker Redis (default)
REDIS_URL=redis://localhost:6379/0

# Or for system Redis (same URL)
REDIS_URL=redis://localhost:6379/0
```

## Test Data Preparation

### Audio Test Files

Create a `tests/e2e/assets/` directory with diverse test audio:

```bash
mkdir -p tests/e2e/assets
cd tests/e2e/assets
```

**Recommended Test Files**:

- `sample_30s.wav`: 30-second clean speech (English)
- `sample_2min.wav`: 2-minute meeting recording
- `sample_vietnamese.wav`: Vietnamese speech sample
- `sample_music.wav`: Music with speech (VAD test)
- `sample_corrupted.wav`: Invalid/corrupted file (error handling)

**Generate Test Audio** (if needed):

```bash
# Using ffmpeg to create test files
ffmpeg -f lavfi -i "sine=frequency=1000:duration=30" -ac 1 -ar 16000 sample_30s.wav
ffmpeg -f lavfi -i "sine=frequency=1000:duration=120" -ac 1 -ar 16000 sample_2min.wav
```

### Expected Results Baselines

Create `tests/e2e/golden/` with expected outputs:

```bash
mkdir -p tests/e2e/golden
```

Store JSON files with expected results for each test case, including:

- Transcript content validation
- Summary structure validation
- Metrics ranges (RTF, confidence, etc.)
- Version metadata validation

## Manual E2E Testing

### 1. System Startup (Hybrid Approach)

**Start Redis in Docker**:

```bash
# Start Redis container
docker run -d --name maie-redis -p 6379:6379 redis:latest

# Verify it's running
docker ps | grep maie-redis
docker exec maie-redis redis-cli ping  # Should return "PONG"
```

**Start API Server** (in separate terminal):

```bash
# Start API server with auto-reload (development mode)
./scripts/dev.sh --api-only --host 0.0.0.0 --port 8000

# Alternative: Start without dev.sh wrapper
pixi run api --host 0.0.0.0 --port 8000 --reload
```

**Start Worker Process** (in separate terminal):

```bash
# Start worker process
./scripts/dev.sh --worker-only

# Alternative: Start without dev.sh wrapper
pixi run worker
```

**Verify Services**:

```bash
# Check API health
curl -f http://localhost:8000/health

# Check Redis connectivity (Docker)
docker exec maie-redis redis-cli ping  # Should return "PONG"

# Check worker is running (view logs in worker terminal)
# Worker output shows in the terminal where you started it
```

**Redis Management Commands**:

```bash
# Stop Redis
docker stop maie-redis

# Start Redis (after stopping)
docker start maie-redis

# Restart Redis
docker restart maie-redis

# View Redis logs
docker logs -f maie-redis

# Remove Redis (⚠️ deletes data)
docker rm -f maie-redis
```

### 2. Basic API Validation

**Test Discovery Endpoints**:

```bash
# List available models
curl -H "X-API-Key: your-secret-key" http://localhost:8000/v1/models

# List available templates
curl -H "X-API-Key: your-secret-key" http://localhost:8000/v1/templates
```

**Test File Upload Limits**:

```bash
# Test with oversized file (should return 413)
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@large_file.wav" \
  http://localhost:8000/v1/process

# Test with invalid format (should return 415)
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@invalid.txt" \
  http://localhost:8000/v1/process
```

### 3. Core Processing Tests

**Happy Path Test**:

```bash
# Submit audio for processing
TASK_ID=$(curl -s -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample_30s.wav" \
  -F 'features=["clean_transcript","summary"]' \
  -F "template_id=meeting_notes_v1" \
  http://localhost:8000/v1/process | jq -r '.task_id')

echo "Task ID: $TASK_ID"
```

**Monitor Processing**:

```bash
# Poll for completion (adjust sleep time based on audio length)
for i in {1..60}; do
  STATUS=$(curl -s -H "X-API-Key: your-secret-key" \
    http://localhost:8000/v1/status/$TASK_ID | jq -r '.status')

  echo "Status: $STATUS"

  if [ "$STATUS" = "COMPLETE" ]; then
    break
  elif [ "$STATUS" = "FAILED" ]; then
    echo "Task failed!"
    exit 1
  fi

  sleep 10
done
```

**Validate Results**:

```bash
# Get final results
curl -s -H "X-API-Key: your-secret-key" \
  http://localhost:8000/v1/status/$TASK_ID | jq '.'
```

### 4. Feature-Specific Tests

**ASR Backend Testing**:

```bash
# Test Whisper backend
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'features=["raw_transcript"]' \
  -F 'asr_backend="whisper"' \
  http://localhost:8000/v1/process

# Test ChunkFormer backend
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'features=["raw_transcript"]' \
  -F 'asr_backend="chunkformer"' \
  http://localhost:8000/v1/process
```

**Feature Combination Testing**:

```bash
# Test all features
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'features=["raw_transcript","clean_transcript","summary","enhancement_metrics"]' \
  -F "template_id=meeting_notes_v1" \
  http://localhost:8000/v1/process
```

### 5. Error Handling Tests

**Invalid Requests**:

```bash
# Missing template_id when summary requested
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'features=["summary"]' \
  http://localhost:8000/v1/process
# Should return 422

# Invalid ASR backend
curl -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'asr_backend="invalid"' \
  http://localhost:8000/v1/process
# Should return 422
```

**Queue Backpressure Test**:

```bash
# Submit multiple concurrent requests
for i in {1..10}; do
  curl -X POST \
    -H "X-API-Key: your-secret-key" \
    -F "file=@tests/e2e/assets/sample.wav" \
    -F 'features=["summary"]' \
    -F "template_id=meeting_notes_v1" \
    http://localhost:8000/v1/process &
done
wait

# Check queue depth via Docker Redis
docker exec maie-redis redis-cli LLEN rq:queue:default
```

## Automated E2E Test Scripts

### E2E Test Framework Setup

The E2E test framework is already configured in `tests/e2e/conftest.py`. To run tests against your local instance:

**Configure Environment Variables**:

```bash
# Set API endpoint and credentials
export API_BASE_URL=http://localhost:8000
export SECRET_API_KEY=your-secret-key  # Must match .env file

# Optionally set Redis URL if using non-default
export REDIS_URL=redis://localhost:6379/0
```

### Running E2E Tests

**Prerequisites**:

1. Redis running in Docker: `docker run -d --name maie-redis -p 6379:6379 redis:latest`
2. API server running: `./scripts/dev.sh --api-only` (in terminal 1)
3. Worker running: `./scripts/dev.sh --worker-only` (in terminal 2)

**Run Tests**:

```bash
# Run all E2E tests
pixi run pytest tests/e2e/ -v

# Run specific test file
pixi run pytest tests/e2e/test_core_workflow.py -v

# Run specific test case
pixi run pytest tests/e2e/test_core_workflow.py::TestCoreWorkflow::test_happy_path_whisper -v

# Run with detailed output and logging
pixi run pytest tests/e2e/ -vv --tb=long --log-cli-level=DEBUG

# Run using the test script wrapper
./scripts/test.sh --e2e
```

**Parallel Testing** (with caution - GPU memory constraints):

```bash
# Run tests with 2 workers (monitor GPU memory)
pixi run pytest tests/e2e/ -v -n 2
```

```python
import pytest
import requests
import time
import os
from pathlib import Path

@pytest.fixture(scope="session")
def api_base_url():
    return os.getenv("API_BASE_URL", "http://localhost:8000")

@pytest.fixture(scope="session")
def api_key():
    return os.getenv("SECRET_API_KEY", "test-key")

@pytest.fixture(scope="session")
def test_assets_dir():
    return Path(__file__).parent / "assets"

@pytest.fixture
def api_client(api_base_url, api_key):
    """API client fixture for E2E tests"""
    class APIClient:
        def __init__(self, base_url, key):
            self.base_url = base_url
            self.session = requests.Session()
            self.session.headers.update({"X-API-Key": key})

        def submit_audio(self, file_path, features=None, template_id=None, asr_backend="whisper"):
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {}
                if features:
                    data['features'] = str(features).replace("'", '"')
                if template_id:
                    data['template_id'] = template_id
                if asr_backend:
                    data['asr_backend'] = asr_backend

                response = self.session.post(f"{self.base_url}/v1/process",
                                           files=files, data=data)
                return response

        def get_status(self, task_id):
            return self.session.get(f"{self.base_url}/v1/status/{task_id}")

        def wait_for_completion(self, task_id, timeout=300, poll_interval=5):
            start_time = time.time()
            while time.time() - start_time < timeout:
                response = self.get_status(task_id)
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] in ['COMPLETE', 'FAILED']:
                        return data
                time.sleep(poll_interval)
            raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    return APIClient(api_base_url, api_key)
```

### Core E2E Test Cases

The core E2E tests are located in `tests/e2e/test_core_workflow.py`. Key test cases include:

```python
import pytest
import json
from pathlib import Path

class TestCoreWorkflow:
    """Core E2E workflow tests"""

    def test_happy_path_whisper(self, api_client, test_assets_dir):
        """Test complete workflow with Whisper backend"""
        audio_file = test_assets_dir / "sample_30s.wav"
        assert audio_file.exists()

        # Submit job
        response = api_client.submit_audio(
            audio_file,
            features=["clean_transcript", "summary"],
            template_id="meeting_notes_v1"
        )
        assert response.status_code == 202
        task_id = response.json()['task_id']

        # Wait for completion
        result = api_client.wait_for_completion(task_id, timeout=180)

        # Validate result structure
        self._validate_complete_result(result)

    def test_happy_path_chunkformer(self, api_client, test_assets_dir):
        """Test complete workflow with ChunkFormer backend"""
        audio_file = test_assets_dir / "sample_30s.wav"

        response = api_client.submit_audio(
            audio_file,
            features=["clean_transcript", "summary"],
            template_id="meeting_notes_v1",
            asr_backend="chunkformer"
        )
        assert response.status_code == 202
        task_id = response.json()['task_id']

        result = api_client.wait_for_completion(task_id, timeout=120)
        self._validate_complete_result(result)

    def test_feature_combinations(self, api_client, test_assets_dir):
        """Test different feature combinations"""
        audio_file = test_assets_dir / "sample_30s.wav"
        test_cases = [
            ["raw_transcript"],
            ["clean_transcript"],
            ["summary"],
            ["raw_transcript", "clean_transcript", "summary"]
        ]

        for features in test_cases:
            response = api_client.submit_audio(
                audio_file,
                features=features,
                template_id="meeting_notes_v1" if "summary" in features else None
            )
            assert response.status_code == 202
            task_id = response.json()['task_id']

            result = api_client.wait_for_completion(task_id)
            self._validate_features_present(result, features)

    def _validate_complete_result(self, result):
        """Validate complete processing result"""
        assert result['status'] == 'COMPLETE'
        assert 'task_id' in result
        assert 'versions' in result
        assert 'metrics' in result
        assert 'results' in result

        # Validate versions
        versions = result['versions']
        assert 'pipeline_version' in versions
        assert 'asr_backend' in versions
        assert 'summarization_llm' in versions

        # Validate metrics
        metrics = result['metrics']
        required_metrics = ['input_duration_seconds', 'processing_time_seconds',
                          'rtf', 'vad_coverage', 'asr_confidence_avg', 'edit_rate_cleaning']
        for metric in required_metrics:
            assert metric in metrics

        # Validate results
        results = result['results']
        assert 'raw_transcript' in results
        assert 'clean_transcript' in results
        assert 'summary' in results

        # Validate summary structure
        summary = results['summary']
        assert 'title' in summary
        assert 'abstract' in summary
        assert 'main_points' in summary
        assert 'tags' in summary
        assert isinstance(summary['tags'], list)

    def _validate_features_present(self, result, requested_features):
        """Validate that requested features are present in results"""
        results = result['results']
        feature_mapping = {
            'raw_transcript': 'raw_transcript',
            'clean_transcript': 'clean_transcript',
            'summary': 'summary'
        }

        for feature in requested_features:
            if feature in feature_mapping:
                assert feature_mapping[feature] in results
```

### Error Handling Tests

Error handling tests are in `tests/e2e/test_error_handling.py`. To create or extend:

```python
import pytest
from pathlib import Path

class TestErrorHandling:
    """Error handling and edge case tests"""

    def test_invalid_file_format(self, api_client, tmp_path):
        """Test rejection of invalid file formats"""
        # Create invalid file
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("not audio")

        response = api_client.submit_audio(invalid_file)
        assert response.status_code == 415

    def test_missing_template_for_summary(self, api_client, test_assets_dir):
        """Test rejection when summary requested without template_id"""
        audio_file = test_assets_dir / "sample_30s.wav"

        response = api_client.submit_audio(
            audio_file,
            features=["summary"]
            # Missing template_id
        )
        assert response.status_code == 422

    def test_invalid_asr_backend(self, api_client, test_assets_dir):
        """Test rejection of invalid ASR backend"""
        audio_file = test_assets_dir / "sample_30s.wav"

        response = api_client.submit_audio(
            audio_file,
            asr_backend="invalid_backend"
        )
        assert response.status_code == 422

    def test_queue_backpressure(self, api_client, test_assets_dir):
        """Test queue depth limits"""
        audio_file = test_assets_dir / "sample_30s.wav"

        # Submit multiple requests rapidly
        responses = []
        for i in range(10):
            response = api_client.submit_audio(audio_file)
            responses.append(response)

        # At least one should succeed, some may be queued
        success_count = sum(1 for r in responses if r.status_code == 202)
        assert success_count > 0

        # Check that we don't get 429 (backpressure) immediately
        # Note: This depends on MAX_QUEUE_DEPTH setting
```

### Performance Tests

Performance tests are in `tests/e2e/test_performance.py`. To create or extend:

```python
import pytest
import time
from statistics import mean, stdev

class TestPerformance:
    """Performance and scalability tests"""

    def test_processing_time_bounds(self, api_client, test_assets_dir):
        """Test that processing times are within expected bounds"""
        audio_file = test_assets_dir / "sample_30s.wav"

        start_time = time.time()
        response = api_client.submit_audio(
            audio_file,
            features=["clean_transcript", "summary"],
            template_id="meeting_notes_v1"
        )
        submit_time = time.time() - start_time
        assert submit_time < 5.0  # Submit should be fast

        task_id = response.json()['task_id']
        result = api_client.wait_for_completion(task_id, timeout=300)

        # Validate processing time
        processing_time = result['metrics']['processing_time_seconds']
        rtf = result['metrics']['rtf']

        # For 30s audio, expect reasonable bounds
        assert 10 < processing_time < 120  # 10s to 2min
        assert 0.1 < rtf < 5.0  # Reasonable RTF range

    def test_concurrent_processing(self, api_client, test_assets_dir):
        """Test concurrent request handling"""
        audio_file = test_assets_dir / "sample_30s.wav"
        num_concurrent = 3

        # Submit concurrent requests
        task_ids = []
        for i in range(num_concurrent):
            response = api_client.submit_audio(audio_file)
            assert response.status_code == 202
            task_ids.append(response.json()['task_id'])

        # Wait for all to complete
        results = []
        for task_id in task_ids:
            result = api_client.wait_for_completion(task_id, timeout=600)
            results.append(result)

        # Validate all completed successfully
        for result in results:
            assert result['status'] == 'COMPLETE'

        # Check that processing times are reasonable
        processing_times = [r['metrics']['processing_time_seconds'] for r in results]
        avg_time = mean(processing_times)

        # Concurrent processing should not be dramatically slower
        assert avg_time < 180  # Less than 3 minutes average
```

## Validation Criteria

### Success Criteria

**Functional Validation**:

- ✅ All API endpoints return correct status codes
- ✅ Audio processing completes without errors
- ✅ Results contain all requested features
- ✅ JSON schemas are validated
- ✅ Version metadata is complete and accurate

**Quality Validation**:

- ✅ Transcripts are accurate (>80% WER for clean audio)
- ✅ Summaries are coherent and structured
- ✅ Tags are relevant and properly categorized
- ✅ Metrics are within expected ranges

**Performance Validation**:

- ✅ Processing time < 3x real-time for typical audio
- ✅ Memory usage stays within GPU limits
- ✅ No crashes or hangs during processing

### Automated Validation Scripts

The validation script `scripts/validate-e2e-results.py` validates E2E test results against expected criteria.

**Usage**:

```bash
# Submit a test request and save the result
TASK_ID=$(curl -s -X POST \
  -H "X-API-Key: your-secret-key" \
  -F "file=@tests/e2e/assets/sample.wav" \
  -F 'features=["clean_transcript","summary"]' \
  -F "template_id=meeting_notes_v1" \
  http://localhost:8000/v1/process | jq -r '.task_id')

# Wait for completion and save result
sleep 30  # Adjust based on audio length
curl -s -H "X-API-Key: your-secret-key" \
  http://localhost:8000/v1/status/$TASK_ID > result.json

# Validate the result
pixi run python scripts/validate-e2e-results.py result.json
```

**Script Implementation** (`scripts/validate-e2e-results.py`):

```python
#!/usr/bin/env python3
"""
E2E Results Validation Script

Validates E2E test results against expected criteria.
Usage: python scripts/validate-e2e-results.py <result_json_file>
"""

import json
import sys
from pathlib import Path
from jsonschema import validate, ValidationError

def validate_result_structure(result: dict) -> bool:
    """Validate basic result structure"""
    required_fields = ['task_id', 'status', 'versions', 'metrics', 'results']

    for field in required_fields:
        if field not in result:
            print(f"❌ Missing required field: {field}")
            return False

    if result['status'] != 'COMPLETE':
        print(f"❌ Status not COMPLETE: {result['status']}")
        return False

    return True

def validate_versions(result: dict) -> bool:
    """Validate version metadata completeness"""
    versions = result['versions']

    required_version_fields = [
        'pipeline_version',
        'asr_backend.name',
        'asr_backend.model_variant',
        'summarization_llm.name'
    ]

    for field_path in required_version_fields:
        keys = field_path.split('.')
        value = versions
        try:
            for key in keys:
                value = value[key]
            if not value:
                print(f"❌ Empty version field: {field_path}")
                return False
        except KeyError:
            print(f"❌ Missing version field: {field_path}")
            return False

    return True

def validate_metrics(result: dict) -> bool:
    """Validate metrics are within reasonable bounds"""
    metrics = result['metrics']

    # Define reasonable ranges
    ranges = {
        'rtf': (0.01, 10.0),
        'vad_coverage': (0.0, 1.0),
        'asr_confidence_avg': (0.0, 1.0),
        'edit_rate_cleaning': (0.0, 1.0)
    }

    for metric, (min_val, max_val) in ranges.items():
        if metric in metrics:
            value = metrics[metric]
            if not (min_val <= value <= max_val):
                print(f"❌ Metric {metric} out of range: {value} (expected {min_val}-{max_val})")
                return False

    return True

def validate_summary_schema(result: dict) -> bool:
    """Validate summary against JSON schema"""
    if 'summary' not in result['results']:
        return True  # Summary not requested

    summary = result['results']['summary']

    # Basic schema check
    required_summary_fields = ['title', 'abstract', 'main_points', 'tags']

    for field in required_summary_fields:
        if field not in summary:
            print(f"❌ Missing summary field: {field}")
            return False

    # Validate tags
    tags = summary['tags']
    if not isinstance(tags, list) or len(tags) < 1 or len(tags) > 5:
        print(f"❌ Invalid tags: {tags} (must be list of 1-5 items)")
        return False

    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate-e2e-results.py <result_json_file>")
        sys.exit(1)

    result_file = Path(sys.argv[1])
    if not result_file.exists():
        print(f"❌ Result file not found: {result_file}")
        sys.exit(1)

    with open(result_file) as f:
        result = json.load(f)

    print(f"🔍 Validating E2E result: {result_file}")

    validations = [
        ("Result Structure", validate_result_structure),
        ("Version Metadata", validate_versions),
        ("Metrics Ranges", validate_metrics),
        ("Summary Schema", validate_summary_schema),
    ]

    all_passed = True
    for name, validator in validations:
        print(f"  Checking {name}...")
        if validator(result):
            print(f"  ✅ {name} passed")
        else:
            print(f"  ❌ {name} failed")
            all_passed = False

    if all_passed:
        print("🎉 All validations passed!")
        sys.exit(0)
    else:
        print("💥 Some validations failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Running E2E Tests

### Local Development Testing

**Start Services** (required before running tests):

```bash
# Terminal 1: Start Redis (if not running as service)
sudo systemctl start redis-server
# Or: docker run -d --name maie-redis -p 6379:6379 redis:latest

# Terminal 2: Start API server
./scripts/dev.sh --api-only --host 0.0.0.0 --port 8000

# Terminal 3: Start worker
./scripts/dev.sh --worker-only

# Terminal 4: Run tests
export API_BASE_URL=http://localhost:8000
export SECRET_API_KEY=your-secret-key  # Match your .env file
```

**Run E2E Tests**:

```bash
# Run specific E2E test
pixi run pytest tests/e2e/test_core_workflow.py::TestCoreWorkflow::test_happy_path_whisper -v

# Run all E2E tests
pixi run pytest tests/e2e/ -v --tb=short

# Run with coverage
pixi run pytest tests/e2e/ --cov=src --cov-report=html

# Run using the test script
./scripts/test.sh --e2e
```

**Cleanup After Testing**:

```bash
# Stop API and worker (Ctrl+C in their respective terminals)

# Clear Redis queue (optional)
redis-cli FLUSHDB

# Stop Redis if running in Docker
docker stop maie-redis && docker rm maie-redis
```

### CI/CD Integration

For CI/CD pipelines with GPU runners:

```yaml
# .github/workflows/e2e.yml
name: E2E Tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  e2e:
    runs-on: self-hosted-gpu
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - name: Install Pixi
        run: curl -fsSL https://pixi.sh/install.sh | bash

      - name: Setup Environment
        run: |
          pixi install
          pixi run download-models

      - name: Start Redis
        run: |
          sudo systemctl start redis-server || docker run -d --name maie-redis -p 6379:6379 redis:latest

      - name: Start API Server
        run: |
          pixi run api --host 0.0.0.0 --port 8000 &
          echo $! > api.pid
          # Wait for API to be ready
          timeout 60 bash -c 'until curl -f http://localhost:8000/health; do sleep 2; done'

      - name: Start Worker
        run: |
          pixi run worker &
          echo $! > worker.pid
          sleep 5  # Give worker time to initialize

      - name: Run E2E Tests
        env:
          API_BASE_URL: http://localhost:8000
          SECRET_API_KEY: ${{ secrets.E2E_API_KEY }}
        run: |
          pixi run pytest tests/e2e/ -v --tb=short --junitxml=e2e-results.xml

      - name: Generate Performance Report
        if: always()
        run: |
          python scripts/benchmark-e2e.py

      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: e2e-results
          path: |
            e2e-results.xml
            e2e-benchmark-results.json

      - name: Cleanup
        if: always()
        run: |
          # Stop services
          [ -f api.pid ] && kill $(cat api.pid) || true
          [ -f worker.pid ] && kill $(cat worker.pid) || true
          # Stop Redis if running in Docker
          docker stop maie-redis || true
          docker rm maie-redis || true
```

### Performance Benchmarking

Create `scripts/benchmark-e2e.py`:

```python
#!/usr/bin/env python3
"""
E2E Performance Benchmarking Script

Runs performance tests and generates reports.
"""

import time
import statistics
import json
from pathlib import Path
import requests

def benchmark_processing(audio_files, api_url, api_key, iterations=3):
    """Benchmark processing performance across multiple files"""

    results = []

    for audio_file in audio_files:
        file_results = {
            'file': str(audio_file),
            'duration_seconds': get_audio_duration(audio_file),
            'runs': []
        }

        for i in range(iterations):
            print(f"Benchmarking {audio_file.name} - run {i+1}/{iterations}")

            # Submit job
            start_time = time.time()
            task_id = submit_job(audio_file, api_url, api_key)
            submit_time = time.time() - start_time

            # Wait for completion
            result = wait_for_completion(task_id, api_url, api_key)
            total_time = time.time() - start_time

            run_result = {
                'iteration': i + 1,
                'submit_time': submit_time,
                'total_time': total_time,
                'processing_time': result['metrics']['processing_time_seconds'],
                'rtf': result['metrics']['rtf']
            }

            file_results['runs'].append(run_result)

        # Calculate statistics
        processing_times = [r['processing_time'] for r in file_results['runs']]
        rtfs = [r['rtf'] for r in file_results['runs']]

        file_results['stats'] = {
            'processing_time_avg': statistics.mean(processing_times),
            'processing_time_std': statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
            'rtf_avg': statistics.mean(rtfs),
            'rtf_std': statistics.stdev(rtfs) if len(rtfs) > 1 else 0
        }

        results.append(file_results)

    return results

def generate_report(results, output_file):
    """Generate performance report"""

    report = {
        'timestamp': time.time(),
        'summary': {
            'total_files': len(results),
            'avg_rtf': statistics.mean([r['stats']['rtf_avg'] for r in results])
        },
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📊 Benchmark report saved to {output_file}")

if __name__ == "__main__":
    # Configuration
    API_URL = "http://localhost:8000"
    API_KEY = "your-secret-key"
    AUDIO_FILES = [
        Path("tests/e2e/assets/sample_30s.wav"),
        Path("tests/e2e/assets/sample_2min.wav")
    ]

    # Run benchmarks
    results = benchmark_processing(AUDIO_FILES, API_URL, API_KEY)

    # Generate report
    generate_report(results, "e2e-benchmark-results.json")
```

## Troubleshooting

### Common Issues

**GPU Memory Errors**:

```
CUDA out of memory
```

**Solution**:

- Check current GPU memory usage: `nvidia-smi`
- Reduce concurrent processing (ensure only one worker is running)
- Lower `GPU_MEMORY_UTILIZATION` in `.env` file (default: 0.9)
- Restart worker to clear GPU memory

**Model Loading Failures**:

```
Model not found: /data/models/whisper/erax-wow-turbo
```

**Solution**:

```bash
# Download models
./scripts/download-models.sh
# Or using Pixi
pixi run download-models

# Verify models exist
ls -lh data/models/
```

**Redis Connection Errors**:

```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution**:

```bash
# If using Docker Redis (recommended):

# Check if Redis container is running
docker ps | grep maie-redis

# Start Redis container if not running
docker start maie-redis

# Or create new Redis container
docker run -d --name maie-redis -p 6379:6379 redis:latest

# Test connection
docker exec maie-redis redis-cli ping  # Should return "PONG"

# Check Redis logs
docker logs maie-redis

# If using system Redis:

# Check if Redis service is running
redis-cli ping

# Start Redis service
sudo systemctl start redis-server

# Check Redis logs
sudo journalctl -u redis-server -n 50
```

**Worker Not Processing Jobs**:

```
No worker output, jobs stuck in queue
```

**Solution**:

```bash
# Check if worker is running
ps aux | grep "python.*worker"

# Check queue status (Docker Redis)
docker exec maie-redis redis-cli LLEN rq:queue:default

# Or with system Redis
redis-cli LLEN rq:queue:default

# Restart worker
# Stop: Ctrl+C in worker terminal or kill <worker_pid>
# Start: ./scripts/dev.sh --worker-only

# Check worker logs for errors
# Worker outputs to stdout/stderr
```

**API Not Responding**:

```
curl: (7) Failed to connect to localhost port 8000
```

**Solution**:

```bash
# Check if API is running
ps aux | grep uvicorn

# Check if port is in use
lsof -i :8000

# Restart API
# Stop: Ctrl+C in API terminal or kill <api_pid>
# Start: ./scripts/dev.sh --api-only --port 8000

# Check for port conflicts
netstat -tuln | grep 8000
```

**Queue Timeouts**:

```
Job timeout after 600 seconds
```

**Solution**:

- Increase `JOB_TIMEOUT` in `.env` for longer audio files
- Check worker logs for bottlenecks
- Monitor GPU utilization during processing: `watch -n 1 nvidia-smi`

**Schema Validation Errors**:

```
LLM output does not match JSON schema
```

**Solution**:

- Check LLM temperature settings in `.env` (lower = more consistent)
- Verify prompt templates in `templates/prompts/`
- Review worker logs for raw LLM output
- Test with simpler audio samples first

**cuDNN Library Errors**:

```
Unable to load libcudnn_ops.so
```

**Solution**:

```bash
# Source the cuDNN environment helper
source scripts/use-cudnn-env.sh

# Then start worker
./scripts/dev.sh --worker-only

# Or set LD_LIBRARY_PATH permanently in .env
export LD_LIBRARY_PATH=/path/to/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

### Debugging Steps

1. **Check Service Health**:

```bash
# API health
curl http://localhost:8000/health

# Redis connectivity (Docker)
docker exec maie-redis redis-cli ping

# Or system Redis
redis-cli ping

# GPU availability
nvidia-smi

# Worker process
ps aux | grep "worker"
```

2. **Monitor GPU Usage**:

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or continuous log
nvidia-smi dmon -s u -d 1
```

3. **Check Queue Status**:

```bash
# Using Docker Redis (recommended)
docker exec maie-redis redis-cli LLEN rq:queue:default

# View queue contents (first 10 items)
docker exec maie-redis redis-cli LRANGE rq:queue:default 0 9

# Check failed queue
docker exec maie-redis redis-cli LLEN rq:queue:failed

# Clear queues (⚠️ if needed)
docker exec maie-redis redis-cli FLUSHDB

# Or using system Redis
redis-cli LLEN rq:queue:default
redis-cli LRANGE rq:queue:default 0 9
redis-cli LLEN rq:queue:failed
redis-cli FLUSHDB
```

4. **Inspect Worker Output**:

```bash
# Worker outputs to stdout/stderr in the terminal where it's running
# For background workers, redirect output to a log file:
pixi run worker > worker.log 2>&1 &

# Then tail the log
tail -f worker.log
```

5. **Validate Test Data**:

```bash
# Check audio file integrity
ffmpeg -i tests/e2e/assets/test_file.wav -f null -

# Get audio file info
ffprobe tests/e2e/assets/test_file.wav

# Convert audio to supported format if needed
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

6. **Test Individual Components**:

```bash
# Test ASR in isolation
pixi run python -c "
from src.processors.asr.whisper import WhisperProcessor
processor = WhisperProcessor()
result = processor.process('tests/e2e/assets/sample.wav')
print(result)
"

# Test LLM in isolation
pixi run python -c "
from src.processors.llm.vllm_processor import VLLMProcessor
processor = VLLMProcessor()
result = processor.process('Test prompt', template_id='meeting_notes_v1')
print(result)
"
```

### Performance Optimization

**GPU Memory Management**:

- Monitor VRAM usage during processing
- Adjust `GPU_MEMORY_UTILIZATION` (0.8-0.95)
- Use appropriate compute types (`int8_float16` vs `float16`)

**Processing Optimization**:

- Enable VAD filtering for better performance
- Choose appropriate ASR backend (Whisper for quality, ChunkFormer for speed)
- Adjust LLM parameters for faster inference

**System Tuning**:

- Increase `OMP_NUM_THREADS` for CPU-bound operations
- Adjust Redis persistence settings for performance vs durability trade-off

## Continuous Integration

### E2E Test Pipeline

```yaml
# .github/workflows/e2e.yml
name: E2E Tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  e2e:
    runs-on: self-hosted-gpu
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - name: Setup Environment
        run: |
          ./scripts/download-models.sh

      - name: Start Services
        run: docker-compose up -d

      - name: Health Check
        run: |
          timeout 300 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'

      - name: Run E2E Tests
        run: |
          export API_BASE_URL=http://localhost:8000
          export SECRET_API_KEY=test-key
          pytest tests/e2e/ -v --tb=short --junitxml=e2e-results.xml

      - name: Generate Performance Report
        run: |
          python scripts/benchmark-e2e.py

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: e2e-results
          path: |
            e2e-results.xml
            e2e-benchmark-results.json

      - name: Cleanup
        run: docker-compose down -v
```

## Quick Reference Cheat Sheet

### Starting Services for E2E Testing (Hybrid Approach) ⭐

```bash
# 1. Start Redis in Docker (single command!)
docker run -d --name maie-redis -p 6379:6379 redis:latest

# 2. Start API (Terminal 1)
./scripts/dev.sh --api-only --port 8000

# 3. Start Worker (Terminal 2)
./scripts/dev.sh --worker-only

# 4. Run Tests (Terminal 3)
export API_BASE_URL=http://localhost:8000
export SECRET_API_KEY=your-secret-key
pixi run pytest tests/e2e/ -v
```

### Common Commands

```bash
# Quick health check
curl http://localhost:8000/health

# Check Redis connectivity (Docker)
docker exec maie-redis redis-cli ping

# Check queue depth (Docker Redis)
docker exec maie-redis redis-cli LLEN rq:queue:default

# Monitor GPU usage
watch -n 1 nvidia-smi

# View worker logs (if running in background)
tail -f worker.log

# Clear Redis queue (Docker)
docker exec maie-redis redis-cli FLUSHDB

# Run single test
pixi run pytest tests/e2e/test_core_workflow.py::TestCoreWorkflow::test_happy_path_whisper -v
```

### Redis Docker Management

```bash
# Start Redis
docker start maie-redis

# Stop Redis (keeps data)
docker stop maie-redis

# Remove Redis (deletes data)
docker rm -f maie-redis

# View Redis logs
docker logs -f maie-redis

# Check Redis status
docker ps | grep maie-redis
```

### Environment Variables

```bash
# Required for E2E tests
export API_BASE_URL=http://localhost:8000
export SECRET_API_KEY=your-secret-key

# Redis URL (same for Docker or system Redis)
export REDIS_URL=redis://localhost:6379/0

# Optional logging
export LOG_LEVEL=DEBUG
```

---

This comprehensive E2E testing guide ensures the MAIE system works correctly end-to-end, from API submission through GPU processing to final results delivery, all tested directly on the host system without containerization.</content>
<parameter name="filePath">/home/cetech/maie/docs/E2E_TESTING_GUIDE.md

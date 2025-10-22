# MAIE — Modular Audio Intelligence Engine

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.12+-green.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-70%2F70-green.svg)]()

A production-ready, on-premises audio processing system that provides comprehensive audio intelligence capabilities through a well-architected API designed for enterprise deployment.

## 🚀 Project Overview

MAIE (Modular Audio Intelligence Engine) is a sophisticated audio processing platform that combines state-of-the-art Automatic Speech Recognition (ASR) and Large Language Model (LLM) capabilities to deliver accurate transcription, summary, and content enhancement services.

**Key Capabilities:**

- **Multi-format Audio Support**: WAV, MP3, M4A, FLAC processing
- **Dual ASR Backends**: Whisper and ChunkFormer for optimal accuracy
- **Intelligent Summary**: Template-driven content summary
- **GPU Acceleration**: Optimized for 16-24GB VRAM deployments
- **Enterprise Ready**: Comprehensive monitoring, logging, and health checks
- **Production Tested**: 70/70 test suite passing

## ⚡ Quick Start

### Using Docker Compose (Recommended)

1. **Clone and configure:**

```bash
git clone <repository-url>
cd maie
cp .env.template .env
# Edit .env with your configuration
```

2. **Start the complete system:**

```bash
docker-compose up -d
```

3. **Verify deployment:**

```bash
curl -f http://localhost:8000/health
```

4. **Process your first audio file:**

```bash
curl -X POST "http://localhost:8000/v1/process" \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@/path/to/your/audio.wav" \
  -F "features=clean_transcript" \
  -F "features=summary" \
  -F "template_id=meeting_notes_v1"
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client Applications                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP/REST API
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Server (Litestar)                        │
│  • Request validation & routing                                 │
│  • File upload handling                                         │
│  • Task queue management                                        │
│  • Health monitoring                                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Redis Queue
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GPU Worker (RQ + PyTorch)                     │
│  • Audio preprocessing (16kHz mono WAV)                         │
│  • ASR processing (Whisper/ChunkFormer)                          │
│  • LLM enhancement & summary                              │
│  • Results storage & metrics calculation                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼───────┐ ┌─────▼──────┐ ┌─────▼──────┐
        │   Redis       │ │  Templates │ │   Models   │
        │   (Queue)     │ │  (JSON)    │ │   (Hugging │
        │               │ │            │ │    Face)   │
        └───────────────┘ └────────────┘ └────────────┘
```

**Architecture Components:**

- **API Layer**: Stateless Litestar server handling HTTP requests
- **Worker Layer**: GPU-accelerated RQ workers for audio processing
- **Storage Layer**: Redis for queuing and results, filesystem for audio/models
- **Monitoring**: Built-in health checks, Jaeger tracing, RQ Dashboard

## ✨ Key Features

### 🎵 Audio Processing

- **Multi-format Support**: WAV, MP3, M4A, FLAC with automatic format detection
- **Streaming Upload**: Memory-efficient file handling for large audio files (up to 500MB)
- **Audio Normalization**: Automatic 16kHz mono WAV conversion for optimal processing

### 🧠 AI Capabilities

- **Dual ASR Backends**:
  - Whisper: Industry-standard speech recognition
  - ChunkFormer: Advanced long-form audio processing
- **LLM Integration**: VLLM-powered text enhancement and summary
- **Template System**: Customizable output formats (meeting notes, interview transcripts, etc.)

### 🔧 Enterprise Features

- **Queue Management**: Redis-based job queuing with backpressure handling
- **Real-time Metrics**: RTF calculation, confidence scoring, processing statistics
- **Health Monitoring**: Comprehensive system health checks and status reporting
- **Security**: API key authentication, file validation, path traversal protection

### 📊 Monitoring & Observability

- **RQ Dashboard**: Queue monitoring and job management interface
- **Jaeger Tracing**: Distributed tracing for request flow analysis
- **Structured Logging**: Loguru-based logging with configurable levels
- **Performance Metrics**: Processing speed, accuracy, and resource utilization tracking

## 🛠️ Installation

### System Requirements

- **GPU**: NVIDIA GPU with 16-24GB VRAM (RTX 4090, A6000, or similar)
- **OS**: Linux with Docker support
- **Storage**: 100GB+ SSD for models and audio files
- **Memory**: 32GB+ system RAM
- **Network**: Stable network connection for model downloads

### Production Deployment

1. **Environment Setup:**

```bash
# Copy and configure environment
cp .env.template .env
# Edit .env with production values:
# - SECRET_API_KEY: Generate secure API key
# - GPU device configuration
# - Model paths and versions
```

2. **Docker Deployment:**

```bash
# Start all services
docker-compose up -d

# Monitor startup
docker-compose logs -f api

# Verify all services are healthy
curl -f http://localhost:8000/health
```

3. **Model Setup:**

```bash
# Download required models (first run)
./scripts/download-models.sh
```

### Manual Installation (Development)

```bash
# Using Pixi (recommended)
pixi install

# Or using uv/venv
uv venv
source .venv/bin/activate
uv pip install -e .
```

## 📖 Usage Examples

### Basic Audio Processing

```bash
# Process audio file with default settings
curl -X POST "http://localhost:8000/v1/process" \
  -H "X-API-Key: your-api-key" \
  -F "file=@meeting.wav" \
  -F "features=clean_transcript" \
  -F "features=summary" \
  -F "template_id=meeting_notes_v1"
```

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING"
}
```

### Check Processing Status

```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/v1/status/550e8400-e29b-41d4-a716-446655440000
```

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "COMPLETED",
  "submitted_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:45Z",
  "results": {
    "transcript": "Meeting discussion content...",
    "summary": "Key points from the meeting...",
    "metrics": {
      "processing_time": 45.2,
      "confidence": 0.94,
      "rtf": 0.12
    }
  }
}
```

### Advanced Processing Options

```bash
# Use ChunkFormer backend for long audio
curl -X POST "http://localhost:8000/v1/process" \
  -H "X-API-Key: your-api-key" \
  -F "file=@long-presentation.m4a" \
  -F "features=raw_transcript" \
  -F "features=clean_transcript" \
  -F "features=summary" \
  -F "features=enhancement_metrics" \
  -F "template_id=interview_transcript_v1" \
  -F "asr_backend=chunkformer"
```

### Batch Processing

```bash
# Process multiple files
for file in *.wav; do
  curl -X POST "http://localhost:8000/v1/process" \
    -H "X-API-Key: your-api-key" \
    -F "file=@$file" \
    -F "features=clean_transcript"
done
```

## 📋 API Documentation

### Core Endpoints

| Endpoint               | Method | Description                             |
| ---------------------- | ------ | --------------------------------------- |
| `/health`              | GET    | System health check with service status |
| `/v1/process`          | POST   | Submit audio file for processing        |
| `/v1/status/{task_id}` | GET    | Check processing task status            |
| `/v1/models`           | GET    | List available ASR models/backends      |
| `/v1/templates`        | GET    | List available processing templates     |

### Request Parameters

**Audio Processing (`/v1/process`):**

- `file` (required): Audio file (WAV, MP3, M4A, FLAC)
- `features` (optional): Processing features array
  - `raw_transcript`: Raw speech-to-text output
  - `clean_transcript`: Cleaned and formatted transcript
  - `summary`: Template-based summary
  - `enhancement_metrics`: Processing quality metrics
- `template_id` (optional): Summary format template
- `asr_backend` (optional): ASR engine (`whisper` or `chunkformer`)

## ⚙️ Configuration

### Environment Variables

| Variable                 | Default                     | Description              |
| ------------------------ | --------------------------- | ------------------------ |
| `SECRET_API_KEY`         | -                           | API authentication key   |
| `REDIS_URL`              | `redis://localhost:6379/0`  | Redis connection URL     |
| `MAX_FILE_SIZE_MB`       | `500`                       | Maximum upload file size |
| `WHISPER_MODEL_VARIANT`  | `erax-wow-turbo`            | Whisper model variant    |
| `LLM_MODEL_NAME`         | `cpatonn/Qwen3-4B-Instruct` | VLLM model name          |
| `GPU_MEMORY_UTILIZATION` | `0.9`                       | GPU memory usage limit   |
| `LOG_LEVEL`              | `INFO`                      | Logging level            |
| `DEBUG`                  | `false`                     | Debug mode enable        |

### GPU Configuration

```bash
# For multi-GPU setups
export CUDA_VISIBLE_DEVICES=0,1

# Memory optimization
export GPU_MEMORY_UTILIZATION=0.85
export LLM_MAX_MODEL_LEN=32768
```

### Performance Tuning

```bash
# Worker configuration
export OMP_NUM_THREADS=4
export WHISPER_BEAM_SIZE=5
export LLM_TEMPERATURE=0.3
```

## 🔧 Development

### Development Setup

1. **Environment Setup:**

```bash
# Install development dependencies
pixi install --feature dev

# Or using uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

2. **Start Development Services:**

```bash
# Start only API server
pixi run api

# Start only worker
pixi run worker

# For more complex scenarios, you can use the dev.sh script
./scripts/dev.sh
```

For running the application directly without any external tools, you can use the `main.py` script:
```bash
python main.py
```

3. **Code Quality:**

```bash
# Format code
pixi run format

# Lint code
pixi run lint

# Type checking (if mypy configured)
mypy src/
```

### Project Structure

```
maie/
├── src/                 # Source code
│   ├── api/            # Litestar API server
│   ├── worker/         # RQ worker implementation
│   ├── processors/     # Audio processing modules
│   ├── config/         # Configuration management
│   └── core/           # Core utilities
├── tests/              # Test suite (70/70 passing)
├── templates/          # Processing templates
├── scripts/            # Development utilities
├── docs/               # Documentation
└── examples/           # Usage examples
```

## 🧪 Testing

### Test Execution

```bash
# Run all tests
pixi run test

# Run specific test categories by using pytest markers
pixi run test -m "not real_llm"

# Run with coverage
pixi run test --cov=src
```

### Test Categories

- **Unit Tests**: Fast, isolated component tests (pytest markers: `unit`)
- **Integration Tests**: Multi-component integration tests (`integration`)
- **E2E Tests**: Full system workflow tests (`e2e`)
- **GPU Tests**: Hardware-accelerated tests (`gpu`)
- **LLM Tests**: Real LLM API calls (`real_llm`)

### Test Coverage

- **70/70 tests passing** ✅
- **Comprehensive coverage** of API endpoints, processors, and utilities
- **Performance benchmarks** for processing speed validation
- **Error handling tests** for robust failure scenarios

## 🚢 Deployment

### Production Deployment

1. **Infrastructure Requirements:**

```bash
# GPU server with Docker support
nvidia-docker available
32GB+ RAM
100GB+ SSD storage
Network connectivity for model downloads
```

2. **Docker Compose Deployment:**

```bash
# Production configuration
cp .env.template .env.production
# Edit with production values

# Deploy with production settings
docker-compose --env-file .env.production up -d

# Scale workers if needed
docker-compose up -d --scale worker=2
```

3. **Monitoring Setup:**

```bash
# Access monitoring interfaces
open http://localhost:9181  # RQ Dashboard
open http://localhost:16686 # Jaeger UI

# Health check script
curl -f http://localhost:8000/health || exit 1
```

### Scaling Considerations

- **Worker Scaling**: Add more worker containers for higher throughput
- **Redis Clustering**: For multi-node deployments
- **Load Balancing**: Use reverse proxy for API scaling
- **Storage**: NFS or distributed filesystem for shared audio storage

### Backup & Recovery

```bash
# Backup Redis data
docker exec maie-redis redis-cli SAVE

# Backup audio and model data
tar -czf backup.tar.gz data/

# Restore process
docker-compose down
# Restore data files
docker-compose up -d
```

## Utilities

### Clean Logs

The `clean-logs.sh` script removes log files older than a specified number of days.

```bash
# Clean logs older than 7 days (default)
./scripts/clean-logs.sh

# Clean logs older than 30 days
DAYS_TO_KEEP=30 ./scripts/clean-logs.sh

# Dry run to see which files would be deleted
DRY_RUN=true ./scripts/clean-logs.sh
```

## 🤝 Contributing

### Development Workflow

1. **Fork and clone** the repository
2. **Create feature branch** from `main`
3. **Write tests** following TDD methodology
4. **Implement features** with comprehensive documentation
5. **Run full test suite** to ensure no regressions
6. **Submit pull request** with clear description

### Code Standards

- **TDD First**: Write failing tests before implementation
- **Type Hints**: Complete type annotations for all functions
- **Documentation**: Docstrings for all public functions
- **Testing**: 100% test coverage for new features
- **Code Style**: Black formatting, ruff linting

### Contribution Process

```bash
# Development workflow
git checkout -b feature/your-feature-name
# Write tests first (TDD)
pixi run test # Should fail
# Implement feature
pixi run test  # Should pass
pixi run lint  # Code quality
git push origin feature/your-feature-name
```

## 📄 License

This project is proprietary software. All rights reserved.

For licensing inquiries, please contact the development team.

---

**MAIE v1.0** - Production-ready audio intelligence platform for enterprise deployment.

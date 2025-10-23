"""
Worker-specific test fixtures.

Provides mocks for:
- Redis clients (sync and async)
- ASR backends
- LLM processors
- Audio metadata
- Task data structures
"""

from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest

# =============================================================================
# Redis Mocks
# =============================================================================


@pytest.fixture
def mock_redis_sync():
    """Mock synchronous Redis client for worker pipeline."""
    redis = MagicMock()
    redis.from_url = MagicMock(return_value=redis)
    redis.ping = MagicMock(return_value=True)
    redis.hgetall = MagicMock(return_value={})
    redis.hset = MagicMock(return_value=1)
    redis.get = MagicMock(return_value=None)
    redis.set = MagicMock(return_value=True)
    redis.close = MagicMock()
    redis.decode_responses = True

    # Store for testing
    redis._data = {}

    def hset_impl(key, mapping=None, **kwargs):
        if key not in redis._data:
            redis._data[key] = {}
        if mapping:
            redis._data[key].update(mapping)
        if kwargs:
            redis._data[key].update(kwargs)
        return len(mapping or kwargs)

    def hgetall_impl(key):
        return redis._data.get(key, {})

    def get_impl(key):
        # Simple key-value store
        return redis._data.get(key)

    def set_impl(key, value, **kwargs):
        redis._data[key] = value
        return True

    redis.hset = Mock(side_effect=hset_impl)
    redis.hgetall = Mock(side_effect=hgetall_impl)
    redis.get = Mock(side_effect=get_impl)
    redis.set = Mock(side_effect=set_impl)

    return redis


@pytest.fixture
def mock_redis_async():
    """Mock asynchronous Redis client for API layer."""
    redis = AsyncMock()
    redis.from_url = AsyncMock(return_value=redis)
    redis.ping = AsyncMock(return_value=True)
    redis.hgetall = AsyncMock(return_value={})
    redis.hset = AsyncMock(return_value=1)
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.close = AsyncMock()
    redis.decode_responses = True

    # Store for testing
    redis._data = {}

    async def hset_impl(key, mapping=None, **kwargs):
        if key not in redis._data:
            redis._data[key] = {}
        if mapping:
            redis._data[key].update(mapping)
        if kwargs:
            redis._data[key].update(kwargs)
        return len(mapping or kwargs)

    async def hgetall_impl(key):
        return redis._data.get(key, {})

    async def get_impl(key):
        return redis._data.get(key)

    async def set_impl(key, value, **kwargs):
        redis._data[key] = value
        return True

    redis.hset = AsyncMock(side_effect=hset_impl)
    redis.hgetall = AsyncMock(side_effect=hgetall_impl)
    redis.get = AsyncMock(side_effect=get_impl)
    redis.set = AsyncMock(side_effect=set_impl)

    return redis


# =============================================================================
# ASR Backend Mocks
# =============================================================================


@pytest.fixture
def mock_asr_result():
    """Mock ASR result object."""

    class MockASRResult:
        def __init__(self):
            self.text = "This is a test transcription from the mock ASR backend."
            self.transcript = self.text  # Alias
            self.segments = [
                {"start": 0.0, "end": 2.5, "text": "This is a test"},
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": "transcription from the mock ASR backend.",
                },
            ]
            self.language = "en"
            self.language_probability = 0.95
            self.confidence = 0.92
            self.confidence_avg = 0.92
            self.vad_coverage = 0.85
            self.duration_ms = 5000
            self.model_name = "mock-whisper-v1"
            self.checkpoint_hash = "abc123def456"
            self.decoding_params = {
                "beam_size": 5,
                "vad_filter": True,
                "temperature": 0.0,
            }

    return MockASRResult()


@pytest.fixture
def mock_asr_backend(mock_asr_result):
    """Mock ASR backend processor."""
    backend = MagicMock()
    backend.execute = MagicMock(return_value=mock_asr_result)
    backend.unload = MagicMock()
    backend.get_version_info = MagicMock(
        return_value={
            "backend": "whisper",
            "model_variant": "mock-v1",
            "model_path": "/mock/path/to/model",
            "checkpoint_hash": "abc123def456",
            "compute_type": "float16",
            "decoding_params": {"beam_size": 5, "vad_filter": True},
        }
    )

    return backend


@pytest.fixture
def mock_asr_factory(mock_asr_backend, monkeypatch):
    """Mock ASRFactory to return mock backend."""
    from src.processors.asr import factory

    def mock_create(backend_id: str, **kwargs):
        return mock_asr_backend

    monkeypatch.setattr(factory.ASRFactory, "create", mock_create)

    return factory.ASRFactory


# =============================================================================
# LLM Processor Mocks
# =============================================================================


@pytest.fixture
def mock_llm_processor():
    """Mock LLM processor."""
    processor = MagicMock()

    # Enhancement result
    processor.enhance_text = MagicMock(
        return_value={
            "enhanced_text": "This is a test transcription from the mock ASR backend.",
            "enhancement_applied": False,  # No changes needed
            "edit_rate": 0.0,
            "edit_distance": 0,
        }
    )

    # Summary result
    processor.generate_summary = MagicMock(
        return_value={
            "summary": {
                "title": "Test Meeting Summary",
                "abstract": "A brief summary of the test meeting.",
                "main_points": ["First main point", "Second main point"],
                "tags": ["test", "meeting", "summary"],
            },
            "retry_count": 0,
            "model_info": {"model_name": "mock-qwen-v1", "checkpoint_hash": "xyz789"},
        }
    )

    processor.needs_enhancement = MagicMock(return_value=False)
    processor.unload = MagicMock()
    processor.get_version_info = MagicMock(
        return_value={
            "enhancement_llm": {
                "name": "mock-qwen-v1",
                "checkpoint_hash": "xyz789",
                "quantization": "awq-4bit",
                "task": "text_enhancement",
            },
            "summarization_llm": {
                "name": "mock-qwen-v1",
                "checkpoint_hash": "xyz789",
                "quantization": "awq-4bit",
                "task": "summary",
            },
        }
    )

    return processor


@pytest.fixture
def mock_llm_factory(mock_llm_processor, monkeypatch):
    """Mock LLMProcessor factory."""
    from src.processors import llm

    # Mock the LLMProcessor class to return our mock
    def mock_llm_class(*args, **kwargs):
        return mock_llm_processor

    monkeypatch.setattr(llm, "LLMProcessor", mock_llm_class)

    yield mock_llm_processor


# =============================================================================
# Audio Preprocessing Mocks
# =============================================================================


@pytest.fixture
def mock_audio_metadata():
    """Mock audio metadata from preprocessor."""
    return {
        "format": "wav",
        "duration": 5.0,
        "sample_rate": 16000,
        "channels": 1,
        "normalized_path": None,  # No normalization needed
    }


@pytest.fixture
def mock_audio_preprocessor(mock_audio_metadata):
    """Mock audio preprocessor."""
    preprocessor = MagicMock()
    preprocessor.preprocess = MagicMock(return_value=mock_audio_metadata)

    return preprocessor


@pytest.fixture
def mock_audio_preprocessor_factory(mock_audio_preprocessor, monkeypatch):
    """Mock AudioPreprocessor factory."""
    from src.processors.audio import preprocessor

    def mock_init(*args, **kwargs):
        return mock_audio_preprocessor

    monkeypatch.setattr(preprocessor, "AudioPreprocessor", mock_init)

    return preprocessor.AudioPreprocessor


# =============================================================================
# Task Data Structures
# =============================================================================


@pytest.fixture
def sample_task_id():
    """Generate a sample task ID."""
    return str(uuid4())


@pytest.fixture
def sample_task_params(sample_task_id, tmp_path):
    """Sample task parameters for processing."""
    # Create a dummy audio file
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"dummy audio data")

    return {
        "task_id": sample_task_id,
        "audio_path": str(audio_file),
        "features": ["clean_transcript", "summary"],
        "template_id": "meeting_notes_v1",
        "asr_backend": "whisper",
    }


@pytest.fixture
def sample_task_result():
    """Sample task result structure."""
    return {
        "versions": {
            "pipeline_version": "1.0.0",
            "asr_backend": {
                "backend": "whisper",
                "model_variant": "mock-v1",
                "model_path": "/mock/path/to/model",
                "checkpoint_hash": "abc123",
                "compute_type": "float16",
                "decoding_params": {"beam_size": 5},
            },
            "llm": {
                "name": "mock-qwen-v1",
                "checkpoint_hash": "xyz789",
                "quantization": "awq-4bit",
            },
        },
        "metrics": {
            "input_duration_seconds": 5.0,
            "processing_time_seconds": 25.5,
            "rtf": 5.1,
            "vad_coverage": 0.85,
            "asr_confidence_avg": 0.92,
            "edit_rate_cleaning": 0.0,
        },
        "results": {
            "clean_transcript": "This is a test transcription.",
            "summary": {
                "title": "Test Summary",
                "abstract": "Brief summary",
                "main_points": ["Point 1", "Point 2"],
                "tags": ["test", "summary"],
            },
        },
    }


# =============================================================================
# File System Mocks
# =============================================================================


@pytest.fixture
def mock_model_paths(tmp_path, monkeypatch):
    """Mock model directory paths."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create model directories
    (models_dir / "whisper" / "erax-wow-turbo").mkdir(parents=True)
    (models_dir / "chunkformer-rnnt-large-vie").mkdir(parents=True)
    (models_dir / "llm" / "qwen3-4b-awq").mkdir(parents=True)

    # Create marker files
    (models_dir / "whisper" / "erax-wow-turbo" / "config.json").touch()
    (models_dir / "chunkformer-rnnt-large-vie" / "config.yaml").touch()
    (models_dir / "llm" / "qwen3-4b-awq" / "config.json").touch()

    from src import config

    monkeypatch.setattr(config.settings, "models_dir", models_dir)

    return models_dir


# =============================================================================
# RQ Job Mocks
# =============================================================================


@pytest.fixture
def mock_rq_job(sample_task_id):
    """Mock RQ job object."""
    job = MagicMock()
    job.id = sample_task_id
    job.get_status = MagicMock(return_value="queued")
    job.get_result = MagicMock(return_value=None)

    return job


@pytest.fixture
def mock_get_current_job(mock_rq_job, monkeypatch):
    """Mock rq.get_current_job()."""
    import rq

    monkeypatch.setattr(rq, "get_current_job", lambda: mock_rq_job)

    return mock_rq_job

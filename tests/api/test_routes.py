"""Tests for API route handlers."""
import json
import uuid
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from litestar import Litestar
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_202_ACCEPTED,
    HTTP_404_NOT_FOUND,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_429_TOO_MANY_REQUESTS,
)
from litestar.testing import TestClient

from src.api.routes import (
    ModelsController,
    ProcessController,
    StatusController,
    TemplatesController,
)
from src.api.schemas import Feature, TaskStatus


class TestProcessController:
    """Tests for the ProcessController (POST /v1/process)."""

    @pytest.fixture
    def app(self):
        """Create a test Litestar app with ProcessController."""
        return Litestar(route_handlers=[ProcessController])

    @pytest.fixture
    def valid_audio_file(self):
        """Create a valid audio file for testing."""
        # Create a minimal valid WAV file (44 bytes header + some data)
        wav_header = (
            b"RIFF"
            + (100).to_bytes(4, "little")
            + b"WAVE"
            + b"fmt "
            + (16).to_bytes(4, "little")
            + (1).to_bytes(2, "little")  # Audio format (PCM)
            + (1).to_bytes(2, "little")  # Num channels
            + (16000).to_bytes(4, "little")  # Sample rate
            + (32000).to_bytes(4, "little")  # Byte rate
            + (2).to_bytes(2, "little")  # Block align
            + (16).to_bytes(2, "little")  # Bits per sample
            + b"data"
            + (64).to_bytes(4, "little")
            + b"\x00" * 64  # Actual audio data
        )
        return BytesIO(wav_header)

    def test_process_audio_success(self, app, valid_audio_file):
        """Test successful audio processing request."""
        with TestClient(app=app) as client:
            with patch("src.api.routes.ProcessController.process_audio") as mock_process:
                task_id = uuid.uuid4()
                mock_process.return_value = {"task_id": str(task_id)}

                response = client.post(
                    "/v1/process",
                    files={"file": ("test.wav", valid_audio_file, "audio/wav")},
                    data={
                        "features": json.dumps(["clean_transcript", "summary"]),
                        "template_id": "meeting_notes_v1",
                    },
                )

                assert response.status_code == HTTP_202_ACCEPTED
                assert "task_id" in response.json()
                # Validate UUID format
                uuid.UUID(response.json()["task_id"])

    def test_process_audio_missing_file(self, app):
        """Test request without file upload."""
        with TestClient(app=app) as client:
            response = client.post(
                "/v1/process",
                data={
                    "features": json.dumps(["clean_transcript"]),
                },
            )

            assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_process_audio_invalid_format(self, app):
        """Test request with unsupported file format."""
        with TestClient(app=app) as client:
            invalid_file = BytesIO(b"not a valid audio file")

            response = client.post(
                "/v1/process",
                files={"file": ("test.txt", invalid_file, "text/plain")},
                data={"features": json.dumps(["clean_transcript"])},
            )

            assert response.status_code == HTTP_415_UNSUPPORTED_MEDIA_TYPE

    def test_process_audio_file_too_large(self, app, valid_audio_file):
        """Test request with file exceeding size limit."""
        with TestClient(app=app) as client:
            with patch("src.config.settings.max_file_size_mb", 0.0001):  # Tiny limit
                response = client.post(
                    "/v1/process",
                    files={"file": ("test.wav", valid_audio_file, "audio/wav")},
                    data={"features": json.dumps(["clean_transcript"])},
                )

                assert response.status_code == HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_process_audio_missing_template_id(self, app, valid_audio_file):
        """Test request with summary feature but missing template_id."""
        with TestClient(app=app) as client:
            response = client.post(
                "/v1/process",
                files={"file": ("test.wav", valid_audio_file, "audio/wav")},
                data={"features": json.dumps(["summary"])},
                # template_id intentionally missing
            )

            assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY
            assert "template_id" in response.text.lower()

    def test_process_audio_queue_full(self, app, valid_audio_file):
        """Test backpressure when queue is full."""
        with TestClient(app=app) as client:
            with patch("src.api.routes.check_queue_depth") as mock_check:
                mock_check.return_value = False  # Queue is full

                response = client.post(
                    "/v1/process",
                    files={"file": ("test.wav", valid_audio_file, "audio/wav")},
                    data={
                        "features": json.dumps(["clean_transcript"]),
                    },
                )

                assert response.status_code == HTTP_429_TOO_MANY_REQUESTS

    def test_process_audio_default_features(self, app, valid_audio_file):
        """Test that default features are applied when not specified."""
        with TestClient(app=app) as client:
            with patch("src.api.routes.ProcessController.process_audio") as mock_process:
                task_id = uuid.uuid4()
                mock_process.return_value = {"task_id": str(task_id)}

                response = client.post(
                    "/v1/process",
                    files={"file": ("test.wav", valid_audio_file, "audio/wav")},
                    data={"template_id": "meeting_notes_v1"},
                    # features not specified - should use defaults
                )

                assert response.status_code == HTTP_202_ACCEPTED


class TestStatusController:
    """Tests for the StatusController (GET /v1/status/{task_id})."""

    @pytest.fixture
    def app(self):
        """Create a test Litestar app with StatusController."""
        return Litestar(route_handlers=[StatusController])

    def test_get_status_not_found(self, app):
        """Test status check for non-existent task."""
        with TestClient(app=app) as client:
            task_id = uuid.uuid4()
            with patch("src.api.routes.get_task_from_redis") as mock_get:
                mock_get.return_value = None

                response = client.get(f"/v1/status/{task_id}")

                assert response.status_code == HTTP_404_NOT_FOUND

    def test_get_status_pending(self, app):
        """Test status check for pending task."""
        with TestClient(app=app) as client:
            task_id = uuid.uuid4()
            with patch("src.api.routes.get_task_from_redis") as mock_get:
                mock_get.return_value = {
                    "task_id": str(task_id),
                    "status": TaskStatus.PENDING,
                    "submitted_at": "2025-10-08T10:00:00",
                }

                response = client.get(f"/v1/status/{task_id}")

                assert response.status_code == HTTP_200_OK
                data = response.json()
                assert data["status"] == TaskStatus.PENDING
                assert data["task_id"] == str(task_id)

    def test_get_status_processing(self, app):
        """Test status check for task in processing."""
        with TestClient(app=app) as client:
            task_id = uuid.uuid4()
            with patch("src.api.routes.get_task_from_redis") as mock_get:
                mock_get.return_value = {
                    "task_id": str(task_id),
                    "status": TaskStatus.PROCESSING_ASR,
                    "submitted_at": "2025-10-08T10:00:00",
                }

                response = client.get(f"/v1/status/{task_id}")

                assert response.status_code == HTTP_200_OK
                data = response.json()
                assert data["status"] == TaskStatus.PROCESSING_ASR

    def test_get_status_complete(self, app):
        """Test status check for completed task with all fields."""
        with TestClient(app=app) as client:
            task_id = uuid.uuid4()
            with patch("src.api.routes.get_task_from_redis") as mock_get:
                mock_get.return_value = {
                    "task_id": str(task_id),
                    "status": TaskStatus.COMPLETE,
                    "submitted_at": "2025-10-08T10:00:00",
                    "completed_at": "2025-10-08T10:05:00",
                    "versions": {
                        "pipeline_version": "1.0.0",
                        "asr_backend": {
                            "name": "whisper",
                            "model_variant": "erax-wow-turbo",
                            "model_path": "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
                            "checkpoint_hash": "a1b2c3d4e5f6",
                            "compute_type": "int8_float16",
                            "decoding_params": {"beam_size": 5, "vad_filter": True},
                        },
                        "llm": {
                            "name": "cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
                            "checkpoint_hash": "f6g7h8i9j0k1",
                            "quantization": "awq-4bit",
                            "chat_template": "qwen3_nonthinking",
                            "thinking": False,
                            "reasoning_parser": None,
                            "structured_output": {
                                "backend": "json_schema",
                                "schema_id": "meeting_notes_v1",
                                "schema_hash": "sha256:abc123",
                            },
                            "decoding_params": {
                                "temperature": 0.3,
                                "top_p": 0.9,
                                "top_k": 20,
                                "repetition_penalty": 1.05,
                            },
                        },
                    },
                    "metrics": {
                        "input_duration_seconds": 100.0,
                        "processing_time_seconds": 10.0,
                        "rtf": 0.1,
                        "vad_coverage": 0.9,
                        "asr_confidence_avg": 0.95,
                    },
                    "results": {
                        "clean_transcript": "This is a test transcript.",
                        "summary": {
                            "title": "Test Summary",
                            "tags": ["test", "demo"],
                        },
                    },
                }

                response = client.get(f"/v1/status/{task_id}")

                assert response.status_code == HTTP_200_OK
                data = response.json()
                assert data["status"] == TaskStatus.COMPLETE
                assert "versions" in data
                assert "metrics" in data
                assert "results" in data
                assert data["results"]["summary"]["tags"] == ["test", "demo"]

    def test_get_status_failed(self, app):
        """Test status check for failed task."""
        with TestClient(app=app) as client:
            task_id = uuid.uuid4()
            with patch("src.api.routes.get_task_from_redis") as mock_get:
                mock_get.return_value = {
                    "task_id": str(task_id),
                    "status": TaskStatus.FAILED,
                    "submitted_at": "2025-10-08T10:00:00",
                    "error": "Model loading failed",
                }

                response = client.get(f"/v1/status/{task_id}")

                assert response.status_code == HTTP_200_OK
                data = response.json()
                assert data["status"] == TaskStatus.FAILED
                assert "error" in data


class TestModelsController:
    """Tests for the ModelsController (GET /v1/models)."""

    @pytest.fixture
    def app(self):
        """Create a test Litestar app with ModelsController."""
        return Litestar(route_handlers=[ModelsController])

    def test_get_models_success(self, app):
        """Test successful retrieval of available models."""
        with TestClient(app=app) as client:
            with patch("src.api.routes.get_available_models") as mock_get:
                mock_get.return_value = {
                    "models": [
                        {
                            "backend_id": "whisper",
                            "name": "Whisper (CTranslate2)",
                            "default_variant": "erax-wow-turbo",
                            "description": "Default ASR backend",
                            "capabilities": ["transcription", "punctuation"],
                        }
                    ]
                }

                response = client.get("/v1/models")

                assert response.status_code == HTTP_200_OK
                data = response.json()
                assert "models" in data
                assert len(data["models"]) > 0
                assert data["models"][0]["backend_id"] == "whisper"

    def test_get_models_empty(self, app):
        """Test models endpoint when no models are available."""
        with TestClient(app=app) as client:
            with patch("src.api.routes.get_available_models") as mock_get:
                mock_get.return_value = {"models": []}

                response = client.get("/v1/models")

                assert response.status_code == HTTP_200_OK
                data = response.json()
                assert data["models"] == []


class TestTemplatesController:
    """Tests for the TemplatesController (GET /v1/templates)."""

    @pytest.fixture
    def app(self):
        """Create a test Litestar app with TemplatesController."""
        return Litestar(route_handlers=[TemplatesController])

    def test_get_templates_success(self, app):
        """Test successful retrieval of available templates."""
        with TestClient(app=app) as client:
            with patch("src.api.routes.scan_templates_directory") as mock_scan:
                mock_scan.return_value = {
                    "templates": [
                        {
                            "template_id": "meeting_notes_v1",
                            "name": "Meeting Notes V1",
                            "description": "Structured meeting notes",
                            "schema_url": "/v1/templates/meeting_notes_v1/schema",
                        }
                    ]
                }

                response = client.get("/v1/templates")

                assert response.status_code == HTTP_200_OK
                data = response.json()
                assert "templates" in data
                assert len(data["templates"]) > 0
                assert data["templates"][0]["template_id"] == "meeting_notes_v1"

    def test_get_templates_empty(self, app):
        """Test templates endpoint when no templates are available."""
        with TestClient(app=app) as client:
            with patch("src.api.routes.scan_templates_directory") as mock_scan:
                mock_scan.return_value = {"templates": []}

                response = client.get("/v1/templates")

                assert response.status_code == HTTP_200_OK
                data = response.json()
                assert data["templates"] == []

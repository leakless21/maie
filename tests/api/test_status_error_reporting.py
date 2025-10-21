"""
Test that the /v1/status endpoint properly reports error codes and stages.

This test verifies the fix for the issue where ASR errors (and other errors)
were not being properly exposed through the status endpoint.
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
from fakeredis import aioredis as fakeredis_async


@pytest.fixture
def redis_client():
    """Provide a fake async Redis client using fakeredis."""
    client = fakeredis_async.FakeRedis(decode_responses=True)
    return client


@pytest.mark.asyncio
async def test_status_endpoint_exposes_asr_error_code(redis_client):
    """Test that ASR processing errors are exposed with error_code and stage."""
    task_id = uuid4()
    task_key = f"task:{task_id}"

    # Simulate an ASR processing error stored in Redis
    error_data = {
        "task_id": str(task_id),
        "status": "FAILED",
        "error": "ASR transcription failed: Audio file not found",
        "error_code": "ASR_PROCESSING_ERROR",
        "stage": "asr",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "features": json.dumps(["clean_transcript"]),
        "asr_backend": "chunkformer",
    }

    await redis_client.hset(task_key, mapping=error_data)  # type: ignore
    await redis_client.expire(task_key, 3600)

    # Import here to avoid circular dependencies
    from src.api.routes import get_task_from_redis

    # Mock the get_results_redis to return our fake client
    with patch(
        "src.api.dependencies.get_results_redis", return_value=redis_client
    ):
        # Retrieve task data
        task_data = await get_task_from_redis(task_id)

    # Verify the error data is present
    assert task_data is not None
    assert task_data["status"] == "FAILED"
    assert task_data["error"] == "ASR transcription failed: Audio file not found"
    assert task_data["error_code"] == "ASR_PROCESSING_ERROR"
    assert task_data["stage"] == "asr"

    # Clean up
    await redis_client.delete(task_key)


@pytest.mark.asyncio
async def test_status_endpoint_exposes_llm_error_code(redis_client):
    """Test that LLM processing errors are exposed with error_code and stage."""
    task_id = uuid4()
    task_key = f"task:{task_id}"

    # Simulate an LLM processing error stored in Redis
    error_data = {
        "task_id": str(task_id),
        "status": "FAILED",
        "error": "Failed to load LLM model: Task exceeded maximum timeout value (300 seconds)",
        "error_code": "MODEL_LOAD_ERROR",
        "stage": "llm",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "features": json.dumps(["summary"]),
        "template_id": "meeting_notes_v1",
    }

    await redis_client.hset(task_key, mapping=error_data)  # type: ignore
    await redis_client.expire(task_key, 3600)

    from src.api.routes import get_task_from_redis

    # Mock the get_results_redis to return our fake client
    with patch(
        "src.api.dependencies.get_results_redis", return_value=redis_client
    ):
        # Retrieve task data
        task_data = await get_task_from_redis(task_id)

    # Verify the error data is present
    assert task_data is not None
    assert task_data["status"] == "FAILED"
    assert "Task exceeded maximum timeout" in task_data["error"]
    assert task_data["error_code"] == "MODEL_LOAD_ERROR"
    assert task_data["stage"] == "llm"

    # Clean up
    await redis_client.delete(task_key)


@pytest.mark.asyncio
async def test_status_endpoint_exposes_preprocessing_error_code(redis_client):
    """Test that preprocessing errors are exposed with error_code and stage."""
    task_id = uuid4()
    task_key = f"task:{task_id}"

    # Simulate a preprocessing error stored in Redis
    error_data = {
        "task_id": str(task_id),
        "status": "FAILED",
        "error": "Audio preprocessing failed: Invalid audio format",
        "error_code": "AUDIO_PREPROCESSING_ERROR",
        "stage": "preprocessing",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    await redis_client.hset(task_key, mapping=error_data)  # type: ignore
    await redis_client.expire(task_key, 3600)

    from src.api.routes import get_task_from_redis

    # Mock the get_results_redis to return our fake client
    with patch(
        "src.api.dependencies.get_results_redis", return_value=redis_client
    ):
        # Retrieve task data
        task_data = await get_task_from_redis(task_id)

    # Verify the error data is present
    assert task_data is not None
    assert task_data["status"] == "FAILED"
    assert task_data["error"] == "Audio preprocessing failed: Invalid audio format"
    assert task_data["error_code"] == "AUDIO_PREPROCESSING_ERROR"
    assert task_data["stage"] == "preprocessing"

    # Clean up
    await redis_client.delete(task_key)


@pytest.mark.asyncio
async def test_status_schema_validation_with_error_fields(redis_client):
    """Test that StatusResponseSchema accepts error_code and stage fields."""
    from src.api.schemas import StatusResponseSchema

    task_id = uuid4()

    # Test that schema validation passes with error fields
    response_data = {
        "task_id": task_id,
        "status": "FAILED",
        "error": "Processing failed",
        "error_code": "ASR_PROCESSING_ERROR",
        "stage": "asr",
        "submitted_at": datetime.now(timezone.utc),
        "completed_at": datetime.now(timezone.utc),
    }

    # This should not raise a validation error
    response = StatusResponseSchema(**response_data)

    assert response.task_id == task_id
    assert response.status == "FAILED"
    assert response.error == "Processing failed"
    assert response.error_code == "ASR_PROCESSING_ERROR"
    assert response.stage == "asr"


@pytest.mark.asyncio
async def test_status_schema_accepts_none_for_optional_error_fields(redis_client):
    """Test that StatusResponseSchema accepts None for optional error fields."""
    from src.api.schemas import StatusResponseSchema

    task_id = uuid4()

    # Test successful completion without error fields
    response_data = {
        "task_id": task_id,
        "status": "COMPLETE",
        "submitted_at": datetime.now(timezone.utc),
        "completed_at": datetime.now(timezone.utc),
    }

    # This should not raise a validation error
    response = StatusResponseSchema(**response_data)

    assert response.task_id == task_id
    assert response.status == "COMPLETE"
    assert response.error is None
    assert response.error_code is None
    assert response.stage is None

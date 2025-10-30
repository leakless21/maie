"""
Mock factories for creating realistic mock objects in integration tests.

This module provides factory functions for creating properly-structured mock objects
that simulate real components without loading actual models or services.

Usage:
    from tests.fixtures.mock_factories import create_mock_asr_output, create_mock_asr_processor

    # Create a realistic ASR output
    asr_output = create_mock_asr_output(text="Hello world")

    # Create a mock ASR processor
    mock_asr = create_mock_asr_processor(output=asr_output)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

from src.processors.base import ASRResult


# ============================================================================
# ASR Result Factory
# ============================================================================


def create_mock_asr_output(
    text: str = "This is a test transcription from the audio file.",
    segments: Optional[List[Dict[str, Any]]] = None,
    language: str = "en",
    confidence: float = 0.95,
    duration: Optional[float] = None,
) -> ASRResult:
    """
    Create a realistic mock ASR output using real ASRResult dataclass.

    Args:
        text: Transcribed text
        segments: Optional list of time-aligned segments
        language: Language code (default: en)
        confidence: Confidence score 0-1 (default: 0.95)
        duration: Optional duration in seconds

    Returns:
        ASRResult instance with proper structure
    """
    if segments is None:
        segments = [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "This is a test",
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "transcription from the audio file.",
            },
        ]

    return ASRResult(
        text=text,
        segments=segments,
        language=language,
        confidence=confidence,
        duration=duration,
    )


# ============================================================================
# ASR Processor Factory
# ============================================================================


def create_mock_asr_processor(
    backend: str = "whisper",
    output: Optional[ASRResult] = None,
    version_info: Optional[Dict[str, Any]] = None,
) -> Mock:
    """
    Create a mock ASR processor that returns realistic results.

    Args:
        backend: ASR backend name (default: whisper)
        output: Optional MockASRResult to return (uses default if None)
        version_info: Optional version info dict

    Returns:
        Mock object configured as ASR processor
    """
    if output is None:
        output = create_mock_asr_output()

    if version_info is None:
        version_info = {
            "backend": backend,
            "model_variant": "large-v3",
            "model_path": f"/data/models/{backend}/large-v3",
            "checkpoint_hash": "mock_hash_abc123",
            "compute_type": "float16",
            "decoding_params": {
                "beam_size": 5,
                "vad_filter": True,
            },
        }

    mock_processor = Mock()
    mock_processor.execute.return_value = output
    mock_processor.unload.return_value = None
    mock_processor.get_version_info.return_value = version_info

    # Make processor support len() if needed for compatibility
    mock_processor.__len__ = Mock(return_value=1)

    return mock_processor


# ============================================================================
# LLM Result Factory
# ============================================================================


@dataclass
class MockLLMEnhancementResult:
    """Mock LLM enhancement result."""

    enhanced_text: str
    enhancement_applied: bool = False
    edit_rate: float = 0.0
    edit_distance: int = 0


@dataclass
class MockLLMSummaryResult:
    """Mock LLM summary result."""

    summary: Dict[str, Any] = field(
        default_factory=lambda: {
            "title": "Test Audio Transcription",
            "main_points": [
                "Audio file contains test transcription",
                "Quality is good",
            ],
            "tags": ["test", "audio", "transcription"],
        }
    )
    retry_count: int = 0
    model_info: Dict[str, Any] = field(
        default_factory=lambda: {
            "model_name": "qwen3-4b-instruct",
            "checkpoint_hash": "mock_llm_hash_xyz",
        }
    )


def create_mock_llm_processor(
    backend: str = "vllm",
    enhancement_result: Optional[MockLLMEnhancementResult] = None,
    summary_result: Optional[MockLLMSummaryResult] = None,
    needs_enhancement: bool = False,
    version_info: Optional[Dict[str, Any]] = None,
) -> Mock:
    """
    Create a mock LLM processor that returns realistic results.

    Args:
        backend: LLM backend name (default: vllm)
        enhancement_result: Optional enhancement result (uses default if None)
        summary_result: Optional summary result (uses default if None)
        needs_enhancement: Whether model needs enhancement (default: False for Whisper)
        version_info: Optional version info dict

    Returns:
        Mock object configured as LLM processor
    """
    if enhancement_result is None:
        enhancement_result = MockLLMEnhancementResult(
            enhanced_text="This is a test transcription from the audio file.",
            enhancement_applied=False,
            edit_rate=0.0,
            edit_distance=0,
        )

    if summary_result is None:
        summary_result = MockLLMSummaryResult()

    if version_info is None:
        version_info = {
            "model_name": "qwen3-4b-instruct",
            "checkpoint_hash": "mock_llm_hash_xyz",
            "backend": backend,
            "quantization": "awq-4bit",
        }

    mock_processor = Mock()

    # Mock enhancement method
    mock_processor.enhance_text.return_value = {
        "enhanced_text": enhancement_result.enhanced_text,
        "enhancement_applied": enhancement_result.enhancement_applied,
        "edit_rate": enhancement_result.edit_rate,
        "edit_distance": enhancement_result.edit_distance,
    }

    # Mock summary method
    mock_processor.generate_summary.return_value = {
        "summary": summary_result.summary,
        "retry_count": summary_result.retry_count,
        "model_info": summary_result.model_info,
    }

    # Mock needs_enhancement
    mock_processor.needs_enhancement.return_value = needs_enhancement

    # Mock lifecycle methods
    mock_processor.unload.return_value = None
    mock_processor.get_version_info.return_value = version_info

    # Make processor support len() if needed for compatibility
    mock_processor.__len__ = Mock(return_value=1)

    return mock_processor


# ============================================================================
# Redis Factory
# ============================================================================


def create_mock_redis_client() -> Mock:
    """
    Create a mock Redis client for testing.

    Returns:
        Mock Redis client with basic methods
    """
    mock_redis = Mock()

    # Store data in mock dictionary
    redis_data = {}

    def mock_hset(name: str, mapping: Dict[str, Any]) -> int:
        """Mock hset method."""
        if name not in redis_data:
            redis_data[name] = {}
        redis_data[name].update(mapping)
        return len(mapping)

    def mock_hgetall(name: str) -> Dict[str, Any]:
        """Mock hgetall method."""
        return redis_data.get(name, {})

    def mock_hget(name: str, key: str) -> Optional[str]:
        """Mock hget method."""
        return redis_data.get(name, {}).get(key)

    def mock_exists(name: str) -> bool:
        """Mock exists method."""
        return name in redis_data

    def mock_flushall() -> bool:
        """Mock flushall method."""
        redis_data.clear()
        return True

    # Bind mock methods
    mock_redis.hset.side_effect = mock_hset
    mock_redis.hgetall.side_effect = mock_hgetall
    mock_redis.hget.side_effect = mock_hget
    mock_redis.exists.side_effect = mock_exists
    mock_redis.flushall.side_effect = mock_flushall

    return mock_redis


# ============================================================================
# Complete Pipeline Factory
# ============================================================================


def create_mock_pipeline_components(
    asr_backend: str = "whisper",
    llm_backend: str = "vllm",
    asr_output: Optional[ASRResult] = None,
    llm_summary: Optional[MockLLMSummaryResult] = None,
    needs_enhancement: bool = False,
) -> Dict[str, Mock]:
    """
    Create a complete set of mock pipeline components.

    Args:
        asr_backend: ASR backend name
        llm_backend: LLM backend name
        asr_output: Optional custom ASR output
        llm_summary: Optional custom LLM summary
        needs_enhancement: Whether enhancement is needed

    Returns:
        Dictionary with keys: 'asr', 'llm', 'redis'
    """
    return {
        "asr": create_mock_asr_processor(backend=asr_backend, output=asr_output),
        "llm": create_mock_llm_processor(
            backend=llm_backend,
            summary_result=llm_summary,
            needs_enhancement=needs_enhancement,
        ),
        "redis": create_mock_redis_client(),
    }

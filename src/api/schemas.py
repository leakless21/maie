"""Pydantic request/response models for the MAIE API."""

import json
import mimetypes
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional
from uuid import UUID

from litestar.datastructures import UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# Enums
# =============================================================================


class TaskStatus(str, Enum):
    """Task processing status."""

    PENDING = "PENDING"
    PREPROCESSING = "PREPROCESSING"
    PROCESSING_ASR = "PROCESSING_ASR"
    PROCESSING_LLM = "PROCESSING_LLM"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class Feature(str, Enum):
    """Available output features."""

    RAW_TRANSCRIPT = "raw_transcript"
    CLEAN_TRANSCRIPT = "clean_transcript"
    SUMMARY = "summary"
    ENHANCEMENT_METRICS = "enhancement_metrics"


# =============================================================================
# Request Models
# =============================================================================


class ProcessRequestSchema(BaseModel):
    """
    Request schema for the /v1/process endpoint.

    This schema defines the multipart form data structure for audio processing requests.
    All parameters except 'file' are optional and have sensible defaults.
    """

    file: Any = Field(
        ...,
        description="The audio file to process (multipart file upload). Supported formats: WAV, MP3, M4A, FLAC. Maximum size: 100MB.",
        json_schema_extra={"format": "binary", "type": "string"},
    )
    features: List[Feature] = Field(
        default=[Feature.CLEAN_TRANSCRIPT, Feature.SUMMARY],
        description="Desired outputs. Available options: 'raw_transcript', 'clean_transcript', 'summary', 'enhancement_metrics'. Default: ['clean_transcript', 'summary']. Tags are embedded in summary output via the template schema.",
        json_schema_extra={
            "examples": [
                ["clean_transcript", "summary"],
                ["raw_transcript"],
                [
                    "raw_transcript",
                    "clean_transcript",
                    "summary",
                    "enhancement_metrics",
                ],
            ]
        },
    )
    template_id: Optional[str] = Field(
        None,
        description="The summary format template ID. Required if 'summary' is in features. Use /v1/templates to get available templates.",
        json_schema_extra={"examples": ["meeting_notes_v1", "interview_summary_v1"]},
    )
    asr_backend: Optional[str] = Field(
        default="chunkformer",
        description="ASR backend selection. Available options: 'whisper', 'chunkformer'(default). Use /v1/models to get available backends.",
        json_schema_extra={
            "examples": ["whisper", "chunkformer"],
            "enum": ["whisper", "chunkformer"],
        },
    )

    @field_validator("features", mode="before")
    @classmethod
    def _coerce_features(cls, value: Any) -> List[Feature]:
        """Convert features from various input formats to List[Feature]."""
        if isinstance(value, str):
            # Handle JSON string format
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [Feature(f) if isinstance(f, str) else f for f in parsed]
                except (json.JSONDecodeError, ValueError):
                    pass
            # Handle single feature as string
            return [Feature(value)]
        elif isinstance(value, list):
            return [Feature(f) if isinstance(f, str) else f for f in value]
        else:
            return [Feature.CLEAN_TRANSCRIPT, Feature.SUMMARY]

    @field_validator("file", mode="before")
    @classmethod
    def _coerce_file(cls, value: Any) -> UploadFile:
        if isinstance(value, UploadFile):
            return value
        if isinstance(value, (str, Path)):
            filename = Path(value).name if isinstance(value, Path) else value
            content_type = (
                mimetypes.guess_type(str(filename))[0] or "application/octet-stream"
            )
            return UploadFile(content_type=content_type, filename=str(filename))
        if isinstance(value, Mapping):
            payload = dict(value)
            filename = payload.get("filename") or "upload"
            content_type = (
                payload.get("content_type")
                or mimetypes.guess_type(filename)[0]
                or "application/octet-stream"
            )
            file_data = payload.get("file_data") or payload.get("data")
            if isinstance(file_data, str):
                file_data = file_data.encode()
            headers = dict(payload.get("headers") or {})
            if "size" in payload and "content-length" not in headers:
                headers["content-length"] = str(payload["size"])
            return UploadFile(
                content_type=content_type,
                filename=filename,
                file_data=file_data,
                headers=headers,
            )
        raise TypeError("file must be an UploadFile or mapping")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "description": "Basic processing with default settings",
                    "value": {
                        "features": ["clean_transcript", "summary"],
                        "template_id": "meeting_notes_v1",
                        "asr_backend": "whisper",
                    },
                },
                {
                    "description": "Raw transcript only",
                    "value": {"features": ["raw_transcript"], "asr_backend": "whisper"},
                },
                {
                    "description": "All features with ChunkFormer backend",
                    "value": {
                        "features": [
                            "raw_transcript",
                            "clean_transcript",
                            "summary",
                            "enhancement_metrics",
                        ],
                        "template_id": "meeting_notes_v1",
                        "asr_backend": "chunkformer",
                    },
                },
            ]
        },
    )

    @model_validator(mode="after")
    def check_template_required(self):
        """Ensure template_id is present when summary is requested.

        This runs after standard validation so `features` is normalized to list of Feature
        or strings. Receives the validated model instance.
        """
        features = getattr(self, "features", None)
        template_id = getattr(self, "template_id", None)
        if not features:
            return self

        normalized = [getattr(f, "value", f) for f in features]
        if Feature.SUMMARY.value in normalized and not template_id:
            raise ValueError("template_id is required when summary is in features")
        return self


# =============================================================================
# Response Models
# =============================================================================


class ProcessResponse(BaseModel):
    """Response for POST /v1/process endpoint."""

    task_id: UUID = Field(description="Unique task identifier")
    status: Literal["PENDING"] = Field(description="Initial task status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
                "status": "PENDING",
            }
        }
    )


class ASRBackendSchema(BaseModel):
    """Schema for ASR backend version information."""

    name: str = Field(..., description="Name of the ASR backend")
    model_variant: str = Field(..., description="Variant of the model")
    model_path: str = Field(..., description="Path to the model")
    checkpoint_hash: str = Field(..., description="Hash of the model checkpoint")
    compute_type: str = Field(..., description="Compute type used")
    decoding_params: Dict[str, Any] = Field(..., description="Decoding parameters")


class LLMSchema(BaseModel):
    """Schema for LLM version information."""

    name: str = Field(..., description="Name of the LLM")
    checkpoint_hash: str = Field(..., description="Hash of the model checkpoint")
    quantization: str = Field(..., description="Quantization type")
    thinking: bool = Field(..., description="Whether thinking is enabled")
    reasoning_parser: Optional[str] = Field(
        None, description="Reasoning parser used, if any"
    )
    structured_output: Dict[str, Any] = Field(
        ..., description="Structured output configuration"
    )
    decoding_params: Dict[str, Any] = Field(..., description="Decoding parameters")


class VersionsSchema(BaseModel):
    """Schema for version information."""

    pipeline_version: str = Field(..., description="Version of the pipeline")
    asr_backend: ASRBackendSchema = Field(..., description="ASR backend information")
    llm: LLMSchema = Field(..., description="Summarization LLM information")


class MetricsSchema(BaseModel):
    """Schema for processing metrics."""

    input_duration_seconds: float = Field(
        ..., description="Duration of the input audio in seconds"
    )
    processing_time_seconds: float = Field(
        ..., description="Time taken for processing in seconds"
    )
    rtf: float = Field(..., description="Real-Time Factor")
    vad_coverage: float = Field(..., description="VAD coverage ratio")
    asr_confidence_avg: float = Field(..., description="Average ASR confidence")
    edit_rate_cleaning: float | None = Field(
        default=None, description="Edit distance rate for enhancement"
    )


class ResultsSchema(BaseModel):
    """Schema for processing results."""

    raw_transcript: Optional[str] = Field(None, description="Raw transcript from ASR")
    clean_transcript: Optional[str] = Field(None, description="Cleaned transcript")
    summary: Optional[Dict[str, Any]] = Field(
        None, description="Structured summary with embedded tags"
    )


class StatusResponseSchema(BaseModel):
    """Response schema for the /v1/status/{task_id} endpoint."""

    task_id: UUID = Field(..., description="Unique identifier for the processing task")
    status: str = Field(
        ...,
        description="Current status of the task (pending, processing, completed, failed)",
    )
    error: Optional[str] = Field(None, description="Error message if the task failed")
    error_code: Optional[str] = Field(
        None,
        description="Error code for categorizing the failure (e.g., ASR_PROCESSING_ERROR, MODEL_LOAD_ERROR)",
    )
    stage: Optional[str] = Field(
        None,
        description="Processing stage where the error occurred (e.g., preprocessing, asr, llm)",
    )
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    versions: Optional[VersionsSchema] = Field(
        None, description="Version information for reproducibility"
    )
    metrics: Optional[MetricsSchema] = Field(None, description="Processing metrics")
    results: Optional[ResultsSchema] = Field(None, description="Processing results")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "description": "Successful completion",
                    "value": {
                        "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
                        "status": "COMPLETE",
                        "versions": {
                            "pipeline_version": "1.0.0",
                            "asr_backend": {
                                "name": "whisper",
                                "model_variant": "erax-wow-turbo",
                                "model_path": "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
                                "checkpoint_hash": "a1b2c3d4...",
                                "compute_type": "int8_float16",
                                "decoding_params": {"beam_size": 5, "vad_filter": True},
                            },
                            "llm": {
                                "name": "qwen3",
                                "checkpoint_hash": "z9y8x7w6...",
                                "quantization": "awq-4bit",
                                "thinking": False,
                                "reasoning_parser": None,
                                "structured_output": {
                                    "title": "string",
                                    "main_points": ["string"],
                                    "tags": ["string"],
                                },
                                "decoding_params": {"temperature": 0.3, "top_p": 0.9},
                            },
                        },
                        "metrics": {
                            "input_duration_seconds": 2701.3,
                            "processing_time_seconds": 162.8,
                            "rtf": 0.06,
                            "vad_coverage": 0.88,
                            "asr_confidence_avg": 0.91,
                        },
                        "results": {
                            "clean_transcript": "The meeting on October 4th...",
                            "summary": {
                                "title": "Q4 Budget Planning",
                                "main_points": ["Budget approved"],
                                "tags": ["Finance", "Budget"],
                            },
                        },
                    },
                },
                {
                    "description": "Failed with ASR error",
                    "value": {
                        "task_id": "a1b2c3d4-5e6f-7g8h-9i0j-1k2l3m4n5o6p",
                        "status": "FAILED",
                        "error": "ASR transcription failed: Audio file not found",
                        "error_code": "ASR_PROCESSING_ERROR",
                        "stage": "asr",
                        "submitted_at": "2025-10-20T07:43:53Z",
                        "completed_at": "2025-10-20T07:45:20Z",
                    },
                },
            ]
        }
    )


class ModelInfoSchema(BaseModel):
    """Schema for individual model information."""

    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Display name of the model")
    description: str = Field(..., description="Description of the model capabilities")
    type: str = Field(..., description="Type of model (ASR, TTS, etc.)")
    version: str = Field(..., description="Version of the model")
    supported_languages: List[str] = Field(
        default_factory=list, description="List of supported languages"
    )


class ModelsResponseSchema(BaseModel):
    """Response schema for the /v1/models endpoint."""

    models: List[ModelInfoSchema] = Field(..., description="List of available models")


class TemplateInfoSchema(BaseModel):
    """Schema for individual template information."""

    id: str = Field(..., description="Unique identifier for the template")
    name: str = Field(..., description="Display name of the template")
    description: str = Field(..., description="Description of the template")
    schema_url: str = Field(..., description="URL to the JSON schema for this template")
    parameters: dict = Field(
        default_factory=dict, description="Default parameters for the template"
    )
    example: Optional[Dict[str, Any]] = Field(
        None, description="Example of the template output"
    )
    example: Optional[Dict[str, Any]] = Field(
        None, description="Example of the template output"
    )


class TemplatesResponseSchema(BaseModel):
    """Response schema for the /v1/templates endpoint."""

    templates: List[TemplateInfoSchema] = Field(
        ..., description="List of available templates"
    )


class HealthResponse(BaseModel):
    """Response for GET /health endpoint."""

    status: Literal["healthy", "unhealthy"]
    version: str
    redis_connected: bool
    queue_depth: int
    worker_active: bool

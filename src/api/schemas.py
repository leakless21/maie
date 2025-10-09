"""Pydantic request/response models for the MAIE API."""

from datetime import datetime 
from typing import List, Optional, Literal, Dict, Any
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field, field_validator

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
    """Request schema for the /v1/process endpoint."""
    
    file: Any = Field(
        ...,
        description="The audio file to process (multipart file upload)"
    )
    features: List[Feature] = Field(
        default=[Feature.CLEAN_TRANSCRIPT, Feature.SUMMARY],
        description="Desired outputs. Default: ['clean_transcript', 'summary']. Tags are embedded in summary output via the template schema."
    )
    template_id: Optional[str] = Field(
        None,
        description="The summary format. Required if 'summary' is in features"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": ["clean_transcript", "summary"],
                "template_id": "meeting_notes_v1"
            }
    }

    @field_validator('template_id')
    @classmethod
    def validate_template_id(cls, v, info):
        """Validate that template_id is provided when summary is requested.

        Uses Pydantic v2 FieldValidator style: `info.data` contains other field values.
        `features` will be coerced to a List[Feature], so compare with Feature.SUMMARY.
        """
        features = info.data.get('features')
        if features and Feature.SUMMARY in features and not v:
            raise ValueError('template_id is required when summary is in features')
        return v
    
# =============================================================================
# Response Models
# =============================================================================

class ProcessResponse(BaseModel):
    """Response for POST /v1/process endpoint."""
    task_id: UUID = Field(description="Unique task identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b"
            }
        }

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
    chat_template: str = Field(..., description="Chat template used")
    thinking: bool = Field(..., description="Whether thinking is enabled")
    reasoning_parser: Optional[str] = Field(None, description="Reasoning parser used, if any")
    structured_output: Dict[str, Any] = Field(..., description="Structured output configuration")
    decoding_params: Dict[str, Any] = Field(..., description="Decoding parameters")


class VersionsSchema(BaseModel):
    """Schema for version information."""
    
    pipeline_version: str = Field(..., description="Version of the pipeline")
    asr_backend: ASRBackendSchema = Field(..., description="ASR backend information")
    llm: LLMSchema = Field(..., description="Summarization LLM information")


class MetricsSchema(BaseModel):
    """Schema for processing metrics."""
    
    input_duration_seconds: float = Field(..., description="Duration of the input audio in seconds")
    processing_time_seconds: float = Field(..., description="Time taken for processing in seconds")
    rtf: float = Field(..., description="Real-Time Factor")
    vad_coverage: float = Field(..., description="VAD coverage ratio")
    asr_confidence_avg: float = Field(..., description="Average ASR confidence")
    edit_rate_cleaning: float | None = Field(
        default=None,
        description="Edit distance rate for enhancement"
    )

class ResultsSchema(BaseModel):
    """Schema for processing results."""
    
    raw_transcript: Optional[str] = Field(None, description="Raw transcript from ASR")
    clean_transcript: Optional[str] = Field(None, description="Cleaned transcript")
    summary: Optional[Dict[str, Any]] = Field(None, description="Structured summary with embedded tags")


class StatusResponseSchema(BaseModel):
    """Response schema for the /v1/status/{task_id} endpoint."""
    
    task_id: UUID = Field(
        ...,
        description="Unique identifier for the processing task"
    )
    status: str = Field(
        ...,
        description="Current status of the task (pending, processing, completed, failed)"
    )
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    versions: Optional[VersionsSchema] = Field(
        None,
        description="Version information for reproducibility"
    )
    metrics: Optional[MetricsSchema] = Field(
        None,
        description="Processing metrics"
    )
    results: Optional[ResultsSchema] = Field(
        None,
        description="Processing results"
    )           
    class Config:
        json_schema_extra = {
            "example": {
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
                        "decoding_params": {"beam_size": 5, "vad_filter": True}
                    },
                    "llm": {
                        "name": "qwen3",
                        "checkpoint_hash": "z9y8x7w6...",
                        "quantization": "awq-4bit",
                        "chat_template": "qwen3_nonthinking",
                        "thinking": False,
                        "reasoning_parser": None,
                        "structured_output": {"title": "string", "main_points": ["string"], "tags": ["string"]},
                        "decoding_params": {"temperature": 0.3, "top_p": 0.9}
                    }
                },
                "metrics": {
                    "input_duration_seconds": 2701.3,
                    "processing_time_seconds": 162.8,
                    "rtf": 0.06,
                    "vad_coverage": 0.88,
                    "asr_confidence_avg": 0.91
                },
                "results": {
                    "clean_transcript": "The meeting on October 4th...",
                    "summary": {
                        "title": "Q4 Budget Planning",
                        "main_points": ["Budget approved"],
                        "tags": ["Finance", "Budget"]
                    }
                }
            }
        }


class ModelInfoSchema(BaseModel):
    """Schema for individual model information."""
    
    id: str = Field(
        ...,
        description="Unique identifier for the model"
    )
    name: str = Field(
        ...,
        description="Display name of the model"
    )
    description: str = Field(
        ...,
        description="Description of the model capabilities"
    )
    type: str = Field(
        ...,
        description="Type of model (ASR, TTS, etc.)"
    )
    version: str = Field(
        ...,
        description="Version of the model"
    )
    supported_languages: List[str] = Field(
        default_factory=list,
        description="List of supported languages"
    )


class ModelsResponseSchema(BaseModel):
    """Response schema for the /v1/models endpoint."""
    
    models: List[ModelInfoSchema] = Field(
        ...,
        description="List of available models"
    )


class TemplateInfoSchema(BaseModel):
    """Schema for individual template information."""
    
    id: str = Field(
        ...,
        description="Unique identifier for the template"
    )
    name: str = Field(
        ...,
        description="Display name of the template"
    )
    description: str = Field(
        ...,
        description="Description of the template"
    )
    schema_url: str = Field(
        ...,
        description="URL to the JSON schema for this template"
    )
    parameters: dict = Field(
        default_factory=dict,
        description="Default parameters for the template"
    )


class TemplatesResponseSchema(BaseModel):
    """Response schema for the /v1/templates endpoint."""
    
    templates: List[TemplateInfoSchema] = Field(
        ...,
        description="List of available templates"
    )
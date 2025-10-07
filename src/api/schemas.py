"""Pydantic request/response models for the MAIE API."""

from typing import List, Optional, Literal, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ProcessRequestSchema(BaseModel):
    """Request schema for the /v1/process endpoint."""
    
    file: Any = Field(
        ...,
        description="The audio file to process (multipart file upload)"
    )
    features: List[Literal["raw_transcript", "clean_transcript", "summary", "enhancement_metrics"]] = Field(
        default=["clean_transcript", "summary"],
        description="Desired outputs. Default: ['clean_transcript', 'summary']. Note: tags are embedded in summary output via the template schema."
    )
    template_id: Optional[str] = Field(
        None,
        description="The summary format. Required if 'summary' is in features"
    )
    
    @field_validator('template_id')
    def validate_template_id(cls, v, values):
        """Validate that template_id is provided when summary is requested."""
        if 'features' in values.data and 'summary' in values.data['features'] and not v:
            raise ValueError('template_id is required when summary is in features')
        return v


class ASRBackendSchema(BaseModel):
    """Schema for ASR backend version information."""
    
    name: str = Field(..., description="Name of the ASR backend")
    model_variant: str = Field(..., description="Variant of the model")
    model_path: str = Field(..., description="Path to the model")
    checkpoint_hash: str = Field(..., description="Hash of the model checkpoint")
    compute_type: str = Field(..., description="Compute type used")
    decoding_params: Dict[str, Any] = Field(..., description="Decoding parameters")


class SummarizationLLMSchema(BaseModel):
    """Schema for summarization LLM version information."""
    
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
    summarization_llm: SummarizationLLMSchema = Field(..., description="Summarization LLM information")


class MetricsSchema(BaseModel):
    """Schema for processing metrics."""
    
    input_duration_seconds: float = Field(..., description="Duration of the input audio in seconds")
    processing_time_seconds: float = Field(..., description="Time taken for processing in seconds")
    rtf: float = Field(..., description="Real-Time Factor")
    vad_coverage: float = Field(..., description="VAD coverage ratio")
    asr_confidence_avg: float = Field(..., description="Average ASR confidence")
    edit_rate_cleaning: float = Field(..., description="Edit rate during cleaning")


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
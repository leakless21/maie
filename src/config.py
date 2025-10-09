"""
Central configuration management using Pydantic Settings.
All environment variables are loaded and validated here.
"""
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    # ============================================================
    # Core Settings
    # ============================================================
    pipeline_version: str = Field(default="1.0.0", description="Pipeline version for NFR-1")
    environment: Literal["development", "production"] = Field(default="development")
    debug: bool = Field(default=False, description="Enable debug logging")
    
    # ============================================================
    # API Server Settings
    # ============================================================
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    secret_api_key: str = Field(default="your_secret_api_key_here", description="API authentication key")
    max_file_size_mb: int = Field(default=500, description="Maximum upload size in MB")
    
    # ============================================================
    # Redis Settings
    # ============================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for queue (DB 0)"
    )
    redis_results_db: int = Field(default=1, description="Redis DB for task results")
    max_queue_depth: int = Field(default=50, description="Maximum queue size for backpressure")
    
    # ============================================================
    # ASR Settings
    # ============================================================
    whisper_model_variant: str = Field(
        default="erax-wow-turbo",
        description="Whisper model variant (erax-wow-turbo, large-v3, etc.)"
    )
    whisper_beam_size: int = Field(default=5, description="Beam size for decoding")
    whisper_vad_filter: bool = Field(default=True, description="Enable VAD filtering")
    whisper_compute_type: str = Field(default="int8_float16", description="CTranslate2 compute type")

    # ============================================================
    # LLM Settings - Enhancement Task
    # ============================================================
    llm_enhance_model: str = Field(
        default="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
        description="LLM model for text enhancement"
    )
    llm_enhance_gpu_memory_utilization: float = Field(
        default=0.95,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization for vLLM"
    )

    llm_enhance_max_model_len: int = Field(default=32768, description="Maximum context length")
    llm_enhance_temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    llm_enhance_top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    llm_enhance_top_k: int = Field(default=20, ge=1, description="Top-k sampling")
    llm_enhance_max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    # ============================================================
    # LLM Settings
    # ============================================================
    llm_sum_model: str = Field(
        default="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
        description="LLM model for summarization"
    )
    llm_sum_gpu_memory_utilization: float = Field(
        default=0.95,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization for vLLM"
    )
    llm_sum_max_model_len: int = Field(default=32768, description="Maximum context length")
    llm_sum_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    llm_sum_top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    llm_sum_top_k: int = Field(default=20, ge=1, description="Top-k sampling")
    llm_sum_max_tokens: int = Field(default=4096, description="Maximum tokens to generate")

    # ============================================================
    # File Paths
    # ============================================================
    audio_dir: Path = Field(default=Path("data/audio"), description="Audio upload directory")
    models_dir: Path = Field(default=Path("data/models"), description="Model weights directory")
    templates_dir: Path = Field(default=Path("templates"), description="JSON schema templates")
    chat_templates_dir: Path = Field(
        default=Path("assets/chat-templates"),
        description="Jinja chat templates"
    )
    
    # ============================================================
    # Worker Settings
    # ============================================================
    worker_name: str = Field(default="maie-worker", description="Worker identifier")
    job_timeout: int = Field(default=600, description="Job timeout in seconds")
    result_ttl: int = Field(default=86400, description="Result retention in seconds (24h)")
    
    # ============================================================
    # Pydantic Settings Config
    # ============================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore unknown env vars
    )
    
    @field_validator("audio_dir", "models_dir", "templates_dir", "chat_templates_dir")
    @classmethod
    def create_directories(cls, path: Path) -> Path:
        """Ensure directories exist on initialization."""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_model_path(self, model_type: str) -> Path:
        """Get full path to model directory."""
        return self.models_dir / model_type
    
    def get_template_path(self, template_id: str) -> Path:
        """Get full path to template JSON file."""
        return self.templates_dir / f"{template_id}.json"


# Global settings instance
settings = Settings()
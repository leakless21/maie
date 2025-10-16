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
    pipeline_version: str = Field(
        default="1.0.0", description="Pipeline version for NFR-1"
    )
    environment: Literal["development", "production", "test"] = Field(
        default="development"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    # ============================================================
    # Logging Settings
    # ============================================================
    # Environment-driven logging configuration centralization
    log_level: str = Field(
        default="DEBUG",
        description="Logging level (e.g., DEBUG, INFO, WARNING)",
    )
    log_dir: Path = Field(default=Path("logs"), description="Directory for log files")
    log_rotation: str = Field(
        default="500 MB",
        description="Rotation policy (e.g., '500 MB', '00:00' for daily)",
    )
    log_retention: str = Field(
        default="30 days", description="Retention policy for rotated logs"
    )
    log_compression: str = Field(
        default="zip", description="Compression for rotated logs (zip, gz, etc.)"
    )
    log_console_serialize: bool = Field(
        default=False,
        description="Serialize console logs as JSON (recommended true in containers)",
    )
    log_file_serialize: bool = Field(
        default=False, description="Serialize file logs as JSON"
    )
    loguru_diagnose: bool = Field(
        default=False, description="Enable diagnose traces (disable in production)"
    )
    loguru_backtrace: bool = Field(
        default=True, description="Enable backtrace on exceptions"
    )
    loguru_format: str | None = Field(
        default=None,
        description="Optional Loguru format string for development console",
    )
    enable_loguru: bool = Field(
        default=True,
        description="Global kill switch to enable/disable Loguru configuration",
    )

    # ============================================================
    # API Server Settings
    # ============================================================
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    secret_api_key: str = Field(
        default="your_secret_api_key_here", description="API authentication key"
    )
    max_file_size_mb: int = Field(default=500, description="Maximum upload size in MB")

    # ============================================================
    # Redis Settings
    # ============================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for queue (DB 0)",
    )
    redis_results_db: int = Field(default=1, description="Redis DB for task results")
    max_queue_depth: int = Field(
        default=50, description="Maximum queue size for backpressure"
    )

    # ============================================================
    # ASR Settings
    # ============================================================
    whisper_model_path: str = Field(
        default="data/models/era-x-wow-turbo-v1.1-ct2",
        description="Path to Whisper model directory",
    )
    whisper_model_variant: str = Field(
        default="erax-wow-turbo",
        description="Whisper model variant (erax-wow-turbo, large-v3, etc.)",
    )
    whisper_beam_size: int = Field(default=5, description="Beam size for decoding")
    whisper_vad_filter: bool = Field(default=True, description="Enable VAD filtering")
    whisper_vad_min_silence_ms: int = Field(
        default=500, description="Minimum silence duration for VAD (ms)"
    )
    whisper_vad_speech_pad_ms: int = Field(
        default=400, description="Speech padding for VAD (ms)"
    )
    whisper_device: str = Field(
        default="cuda", description="Device for whisper computation (cpu, cuda, auto)"
    )
    whisper_compute_type: str = Field(
        default="float16", description="CTranslate2 compute type"
    )
    whisper_cpu_fallback: bool = Field(
        default=False,
        description="Fallback to CPU if CUDA fails (disabled for offline deployment)",
    )
    whisper_condition_on_previous_text: bool = Field(
        default=True,
        description="Use previous text as context (set False for Distil-Whisper models)",
    )
    whisper_language: str | None = Field(
        default=None,
        description="Force specific language code (e.g., 'en', 'vi') or None for auto-detection",
    )
    whisper_cpu_threads: int | None = Field(
        default=None,
        description="Number of CPU threads for Whisper inference (None = auto, affects OMP_NUM_THREADS)",
    )

    # ============================================================
    # ChunkFormer Settings (Long-form / chunked ASR backend)
    # ============================================================
    chunkformer_model_path: str = Field(
        default="data/models/chunkformer-ctc-large-vie",
        description="Path to ChunkFormer model directory",
    )
    chunkformer_model_variant: str = Field(
        default="khanhld/chunkformer-large-vie",
        description="ChunkFormer model variant or HF repo id",
    )
    chunkformer_chunk_size: int = Field(
        default=64, description="Chunk size in frames for chunked processing"
    )
    chunkformer_left_context_size: int = Field(
        default=128,
        description="Left context window size (frames) applied to each chunk",
    )
    chunkformer_right_context_size: int = Field(
        default=128,
        description="Right context window size (frames) applied to each chunk",
    )
    chunkformer_total_batch_duration: int = Field(
        default=14400,
        description="Maximum total batch duration for ChunkFormer processing (seconds)",
    )
    chunkformer_return_timestamps: bool = Field(
        default=True,
        description="Whether ChunkFormer should return word/segment timestamps",
    )
    chunkformer_device: str = Field(
        default="cuda",
        description="Device for ChunkFormer computation (cpu, cuda, auto)",
    )
    chunkformer_batch_size: int | None = Field(
        default=None,
        description="Batch size for ChunkFormer (None = sequential/unbatched)",
    )
    chunkformer_cpu_fallback: bool = Field(
        default=False,
        description="Fallback to CPU if CUDA fails for ChunkFormer (disabled for offline deployment)",
    )

    # ============================================================
    # LLM Settings - Enhancement Task
    # ============================================================
    llm_enhance_model: str = Field(
        default="data/models/qwen3-4b-instruct-2507-awq",
        description="LLM model for text enhancement",
    )
    llm_enhance_gpu_memory_utilization: float = Field(
        default=0.95,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization for vLLM (0.75 accounts for ~3.7GB used by desktop environment on RTX 3060)",
    )

    llm_enhance_max_model_len: int = Field(
        default=32768,
        description="Maximum context length (32K tokens, balanced for RTX 3060 12GB with desktop)",
    )
    llm_enhance_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    llm_enhance_top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling threshold"
    )
    llm_enhance_top_k: int | None = Field(
        default=None, ge=1, description="Top-k sampling"
    )
    llm_enhance_max_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate"
    )
    llm_enhance_use_beam_search: bool = Field(
        default=False, description="Use beam search decoding strategy"
    )
    llm_enhance_quantization: str | None = Field(
        default=None, description="Quantization method (awq, compressed-tensors, etc.)"
    )

    # ============================================================
    # LLM Settings - Summarization Task
    # ============================================================
    llm_sum_model: str = Field(
        default="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
        description="LLM model for summarization",
    )
    llm_sum_gpu_memory_utilization: float = Field(
        default=0.95, ge=0.1, le=1.0, description="GPU memory utilization for vLLM"
    )
    llm_sum_max_model_len: int = Field(
        default=32768, description="Maximum context length"
    )
    llm_sum_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    llm_sum_top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling threshold"
    )
    llm_sum_top_k: int | None = Field(default=None, ge=1, description="Top-k sampling")
    llm_sum_max_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate"
    )
    llm_sum_quantization: str | None = Field(
        default=None, description="Quantization method (awq, compressed-tensors, etc.)"
    )

    # ============================================================
    # File Paths
    # ============================================================
    audio_dir: Path = Field(
        default=Path("data/audio"), description="Audio upload directory"
    )
    models_dir: Path = Field(
        default=Path("data/models"), description="Model weights directory"
    )
    templates_dir: Path = Field(
        default=Path("templates"), description="JSON schema templates"
    )
    chat_templates_dir: Path = Field(
        default=Path("assets/chat-templates"), description="Jinja chat templates"
    )

    # ============================================================
    # Worker Settings
    # ============================================================
    worker_name: str = Field(default="maie-worker", description="Worker identifier")
    job_timeout: int = Field(default=600, description="Job timeout in seconds")
    result_ttl: int = Field(
        default=86400, description="Result retention in seconds (24h)"
    )
    worker_concurrency: int = Field(default=2, description="Worker concurrency level")
    worker_prefetch_multiplier: int = Field(
        default=4, description="Prefetch multiplier for worker queue"
    )
    worker_prefetch_timeout: int = Field(
        default=30, description="Timeout for worker fetch operations (seconds)"
    )

    # ============================================================
    # Feature Flags
    # ============================================================
    enable_enhancement: bool = Field(
        default=True, description="Enable LLM-based enhancement post-processing"
    )

    # ============================================================
    # Validation
    # ============================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed_levels = {
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }
        upper_value = value.upper()
        if upper_value not in allowed_levels:
            raise ValueError(
                f"Invalid log level: {value}. Must be one of {allowed_levels}."
            )
        return upper_value

    @field_validator("log_retention")
    @classmethod
    def validate_log_retention(cls, value: str) -> str:
        """
        Validate retention format to avoid silent mistakes like non-parsable values.
        Accept simple strings like '30 days' or '1 week'. The actual parsing is delegated to Loguru.
        """
        if not value or not value.strip():
            raise ValueError("log_retention cannot be empty.")
        return value

    @field_validator("log_dir")
    @classmethod
    def validate_log_dir(cls, value: Path) -> Path:
        if value.is_file():
            raise ValueError("log_dir must be a directory path.")
        return value

    @field_validator("audio_dir", "models_dir", "templates_dir", "chat_templates_dir")
    @classmethod
    def create_directories(cls, path: Path) -> Path:
        """Ensure directories exist on initialization when safe to create."""
        try:
            if path.exists():
                return path

            for ancestor in path.parents:
                if ancestor == Path("/"):
                    break
                if ancestor.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    return path
            return path
        except PermissionError:
            raise

    @field_validator("api_port")
    @classmethod
    def validate_api_port(cls, value: int) -> int:
        if not (1 <= value <= 65535):
            raise ValueError("api_port must be between 1 and 65535")
        return value

    @field_validator(
        "whisper_cpu_threads",
        "chunkformer_batch_size",
        "llm_enhance_top_k",
        "llm_enhance_max_tokens",
        "llm_sum_top_k",
        "llm_sum_max_tokens",
        mode="before",
    )
    @classmethod
    def coerce_optional_ints(cls, value):
        """Parse optional integer fields, converting empty strings to None."""
        if value == "" or value is None:
            return None
        if isinstance(value, str):
            return int(value)
        return value

    def get_model_path(self, model_type: str) -> Path:
        """Get full path to model directory."""
        return self.models_dir / model_type

    def get_template_path(self, template_id: str) -> Path:
        """Get full path to template JSON file."""
        return self.templates_dir / f"{template_id}.json"


settings = Settings()


__all__ = ["Settings", "settings"]

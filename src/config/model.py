from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Mapping, Tuple, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.validation import coerce_optional_int, blank_to_none


ModelT = TypeVar("ModelT", bound=BaseModel)


class LoggingSettings(BaseModel):
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"))
    log_rotation: str = Field(default="500 MB")
    log_retention: str = Field(default="30 days")
    log_console_serialize: bool = Field(default=False)
    log_file_serialize: bool = Field(default=False)
    log_compression: str = Field(default="zip")
    enable_loguru: bool = Field(default=True)
    loguru_diagnose: bool = Field(default=False)
    loguru_backtrace: bool = Field(default=True)
    loguru_format: str | None = Field(default=None)

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        allowed = {
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }
        upper_value = value.upper()
        if upper_value not in allowed:
            raise ValueError(
                f"Invalid log level: {value}. Must be one of {', '.join(sorted(allowed))}."
            )
        return upper_value

    @field_validator("log_dir")
    @classmethod
    def ensure_directory_not_file(cls, value: Path) -> Path:
        if value.is_file():
            raise ValueError("log_dir must reference a directory.")
        return value


class ApiSettings(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    secret_key: SecretStr = Field(default=SecretStr("your_secret_api_key_here"))
    fallback_keys: Tuple[str, ...] = Field(
        default=("test-key-123456789012345678901234567890",)
    )
    max_file_size_mb: float = Field(default=500.0)

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if not (1 <= value <= 65535):
            raise ValueError("api_port must be between 1 and 65535")
        return value


class RedisSettings(BaseModel):
    url: str = Field(default="redis://localhost:6379/0")
    results_db: int = Field(default=1)
    max_queue_depth: int = Field(default=50)

    model_config = ConfigDict(validate_assignment=True)


class AsrSettings(BaseModel):
    whisper_model_path: str = Field(
        default="data/models/openai-whisper-large",
    )
    whisper_model_variant: str = Field(default="openai-large")
    whisper_beam_size: int = Field(default=5)
    whisper_vad_filter: bool = Field(default=False)
    whisper_vad_min_silence_ms: int = Field(default=500)
    whisper_vad_speech_pad_ms: int = Field(default=400)
    whisper_device: str = Field(default="cuda")
    whisper_compute_type: str = Field(default="float16")
    whisper_cpu_fallback: bool = Field(default=False)
    whisper_condition_on_previous_text: bool = Field(default=False)
    whisper_language: str | None = Field(default=None)
    whisper_cpu_threads: int | None = Field(default=None)
    whisper_word_timestamps: bool = Field(
        default=True,
        description="Enable word-level timestamps. Required for accurate segment timestamps in faster-whisper.",
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("whisper_language", mode="before")
    @classmethod
    def empty_languages_to_none(cls, value: Any) -> str | None:
        return blank_to_none(value)

    @field_validator("whisper_cpu_threads", mode="before")
    @classmethod
    def convert_optional_threads(cls, value: Any) -> int | None:
        return coerce_optional_int(value)


class ChunkformerSettings(BaseModel):
    chunkformer_model_path: str = Field(
        default="data/models/chunkformer-rnnt-large-vie"
    )
    chunkformer_model_variant: str = Field(
        default="khanhld/chunkformer-rnnt-large-vie",
    )
    chunkformer_chunk_size: int = Field(default=64, description="Chunk size in frames")
    chunkformer_left_context_size: int = Field(
        default=128, description="Left context size in frames"
    )
    chunkformer_right_context_size: int = Field(
        default=128, description="Right context size in frames"
    )
    chunkformer_total_batch_duration: int = Field(
        default=14400, description="Total batch duration in seconds"
    )
    chunkformer_return_timestamps: bool = Field(default=True)
    chunkformer_device: str = Field(default="cuda")
    chunkformer_batch_size: int | None = Field(default=None)
    chunkformer_cpu_fallback: bool = Field(default=False)

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("chunkformer_batch_size", mode="before")
    @classmethod
    def convert_optional_batch(cls, value: Any) -> int | None:
        return coerce_optional_int(value)


class LlmEnhanceSettings(BaseModel):
    model: str = Field(default="data/models/qwen3-4b-instruct-2507-awq")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    max_model_len: int = Field(default=32768)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None)
    use_beam_search: bool = Field(default=False)
    quantization: str | None = Field(default=None)
    max_num_seqs: int | None = Field(
        default=None,
        ge=1,
        description="Maximum in-flight sequences vLLM should schedule concurrently",
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Upper bound on total tokens (prompt + decode) processed per scheduler step",
    )
    max_num_partial_prefills: int | None = Field(
        default=None,
        ge=1,
        description="Enables chunked prefill when >1 to overlap scheduling for long prompts",
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator(
        "top_k",
        "max_tokens",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_num_partial_prefills",
        mode="before",
    )
    @classmethod
    def optional_ints(cls, value: Any) -> int | None:
        return coerce_optional_int(value)


class LlmSumSettings(BaseModel):
    model: str = Field(default="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    max_model_len: int = Field(default=32768)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None)
    quantization: str | None = Field(default=None)
    max_num_seqs: int | None = Field(
        default=None,
        ge=1,
        description="Maximum concurrent sequences for summary vLLM runs",
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Token budget (prompt + decode) per scheduler step when summarizing",
    )
    max_num_partial_prefills: int | None = Field(
        default=None,
        ge=1,
        description="Chunked prefill configuration for summary workloads",
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator(
        "top_k",
        "max_tokens",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_num_partial_prefills",
        mode="before",
    )
    @classmethod
    def optional_ints(cls, value: Any) -> int | None:
        return coerce_optional_int(value)


class PathsSettings(BaseModel):
    audio_dir: Path = Field(default=Path("data/audio"))
    models_dir: Path = Field(default=Path("data/models"))
    templates_dir: Path = Field(default=Path("templates"))

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("audio_dir", "models_dir", "templates_dir")
    @classmethod
    def ensure_paths(cls, value: Path) -> Path:
        if value.is_file():
            raise ValueError("Directory paths must refer to directories, not files.")
        return value


class WorkerSettings(BaseModel):
    worker_name: str = Field(default="maie-worker")
    job_timeout: int = Field(default=600)
    result_ttl: int = Field(default=86400)
    worker_concurrency: int = Field(default=2)
    worker_prefetch_multiplier: int = Field(default=4)
    worker_prefetch_timeout: int = Field(default=30)

    model_config = ConfigDict(validate_assignment=True)


class DiarizationSettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable speaker diarization")
    model_path: str = Field(
        default="data/models/pyannote-speaker-diarization-community-1",
        description="LOCAL PATH to pyannote speaker diarization model for FULLY OFFLINE operation (no HuggingFace/network calls)",
    )
    overlap_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum IoU threshold for speaker overlap detection",
    )
    require_cuda: bool = Field(
        default=False,
        description="Require CUDA for diarization; skip gracefully if False and no CUDA",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for speaker embedding model (pyannote 3.x uses config.yaml defaults)",
    )
    segmentation_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for segmentation model (pyannote 3.x uses config.yaml defaults)",
    )

    model_config = ConfigDict(validate_assignment=True)


class CleanupSettings(BaseModel):
    audio_cleanup_interval: int = Field(
        default=3600, description="Interval in seconds between audio cleanup runs"
    )
    log_cleanup_interval: int = Field(
        default=86400, description="Interval in seconds between log cleanup runs"
    )
    cache_cleanup_interval: int = Field(
        default=1800, description="Interval in seconds between cache cleanup runs"
    )
    disk_monitor_interval: int = Field(
        default=300, description="Interval in seconds between disk monitoring runs"
    )

    audio_retention_days: int = Field(
        default=7, description="Number of days to keep audio files"
    )
    logs_retention_days: int = Field(
        default=7, description="Number of days to keep log files"
    )

    disk_threshold_pct: int = Field(
        default=80, description="Disk usage percentage threshold for alerts"
    )
    emergency_cleanup: bool = Field(
        default=False, description="Enable automatic emergency cleanup on disk alerts"
    )
    check_dir: str = Field(
        default=".", description="Directory to monitor for disk usage"
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator(
        "audio_cleanup_interval",
        "log_cleanup_interval",
        "cache_cleanup_interval",
        "disk_monitor_interval",
    )
    @classmethod
    def validate_intervals(cls, value: int) -> int:
        """Ensure cleanup intervals are positive and reasonable (10s to 86400s = 1 day)."""
        if value < 10:
            raise ValueError(
                f"Cleanup interval must be at least 10 seconds, got {value}"
            )
        if value > 86400:
            raise ValueError(
                f"Cleanup interval should not exceed 86400 seconds (1 day), got {value}"
            )
        return value

    @field_validator("audio_retention_days", "logs_retention_days")
    @classmethod
    def validate_retention_days(cls, value: int) -> int:
        """Ensure retention periods are non-negative."""
        if value < 0:
            raise ValueError(f"Retention days cannot be negative, got {value}")
        if value > 365:
            raise ValueError(f"Retention days should not exceed 365, got {value}")
        return value

    @field_validator("disk_threshold_pct")
    @classmethod
    def validate_disk_threshold(cls, value: int) -> int:
        """Ensure disk threshold is between 0 and 100."""
        if value < 0 or value > 100:
            raise ValueError(
                f"Disk threshold percentage must be between 0 and 100, got {value}"
            )
        return value

    @field_validator("check_dir")
    @classmethod
    def validate_check_dir(cls, value: str) -> str:
        """Ensure check_dir is a valid path string."""
        if not value or not isinstance(value, str):
            raise ValueError(f"check_dir must be a non-empty string, got {value}")
        return value


class VADSettings(BaseModel):
    """Voice Activity Detection (VAD) settings."""

    enabled: bool = Field(
        default=True,
        description="Enable Voice Activity Detection preprocessing",
    )
    backend: str = Field(
        default="silero",
        description="VAD backend to use (currently supports 'silero')",
    )
    silero_model_path: str | None = Field(
        default=None,
        description="Optional explicit ONNX model path; if not provided, uses silero-vad's built-in loader",
    )
    silero_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Speech confidence threshold for Silero VAD",
    )
    silero_sampling_rate: int = Field(
        default=16000,
        ge=8000,
        description="Audio sampling rate for Silero VAD",
    )
    min_speech_duration_ms: int = Field(
        default=200,
        ge=0,
        description="Minimum speech segment duration in milliseconds",
    )
    max_speech_duration_ms: int = Field(
        default=60000,
        ge=0,
        description="Maximum continuous speech duration in milliseconds",
    )
    min_silence_duration_ms: int = Field(
        default=500,
        ge=0,
        description="Minimum silence duration between speech segments in milliseconds",
    )
    window_size_samples: int = Field(
        default=512,
        ge=1,
        description="Window size for VAD in samples",
    )
    device: str = Field(
        default="cuda",
        description="Device to use for VAD processing ('cuda' or 'cpu')",
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        """Ensure backend is supported."""
        supported = {"silero"}
        if value not in supported:
            raise ValueError(
                f"Unsupported VAD backend: {value}. Supported: {', '.join(supported)}"
            )
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Ensure device is valid."""
        supported = {"cuda", "cpu"}
        if value not in supported:
            raise ValueError(
                f"Invalid device: {value}. Must be one of: {', '.join(supported)}"
            )
        return value


class FeatureFlags(BaseModel):
    enable_enhancement: bool = Field(default=True)

    model_config = ConfigDict(validate_assignment=True)


class AppSettings(BaseSettings):
    pipeline_version: str = Field(default="1.0.0")
    environment: Literal["development", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    verbose_components: bool = Field(default=False)

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    asr: AsrSettings = Field(default_factory=AsrSettings)
    chunkformer: ChunkformerSettings = Field(default_factory=ChunkformerSettings)
    diarization: DiarizationSettings = Field(default_factory=DiarizationSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    llm_enhance: LlmEnhanceSettings = Field(default_factory=LlmEnhanceSettings)
    llm_sum: LlmSumSettings = Field(default_factory=LlmSumSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    cleanup: CleanupSettings = Field(default_factory=CleanupSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="APP_",
        case_sensitive=False,
        validate_default=True,
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    _SECRET_FIELDS: ClassVar[set[str]] = {"secret_api_key"}

    def ensure_directories(self) -> Dict[str, Path]:
        """
        Create configured directories if they do not already exist.

        Returns:
            Mapping of directory attribute name to the resolved Path.
        """
        directories = {
            "audio_dir": self.paths.audio_dir,
            "models_dir": self.paths.models_dir,
            "templates_dir": self.paths.templates_dir,
        }
        created: Dict[str, Path] = {}
        for name, path in directories.items():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created[name] = path
        return created

    def get_model_path(self, model_type: str) -> Path:
        return self.paths.models_dir / model_type

    def apply_profile(self, profile: Mapping[str, Any]) -> "AppSettings":
        """
        Apply a profile to the settings instance, respecting fields set via environment variables.
        """

        def _apply(model: ModelT, overrides: Mapping[str, Any]) -> ModelT:
            updates: Dict[str, Any] = {}
            fields_set = getattr(model, "model_fields_set", set())

            for field_name, override_value in overrides.items():
                if hasattr(model, field_name):
                    current = getattr(model, field_name)
                    if isinstance(current, BaseModel) and isinstance(
                        override_value, Mapping
                    ):
                        updated_submodel = _apply(current, override_value)
                        if updated_submodel is not current:
                            updates[field_name] = updated_submodel
                    elif field_name not in fields_set:
                        updates[field_name] = override_value
                else:
                    updates[field_name] = override_value

            if updates:
                merged = model.model_dump(mode="python")
                merged.update(updates)
                return type(model).model_validate(merged)
            return model

        updated = _apply(self, profile)
        return cast(AppSettings, updated)

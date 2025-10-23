from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Mapping, Tuple, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return int(stripped)
    return int(value)


def _blank_to_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


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
        default="data/models/era-x-wow-turbo-v1.1-ct2",
    )
    whisper_model_variant: str = Field(default="erax-wow-turbo")
    whisper_beam_size: int = Field(default=5)
    whisper_vad_filter: bool = Field(default=True)
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
        return _blank_to_none(value)

    @field_validator("whisper_cpu_threads", mode="before")
    @classmethod
    def convert_optional_threads(cls, value: Any) -> int | None:
        return _coerce_optional_int(value)


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
        return _coerce_optional_int(value)


class LlmEnhanceSettings(BaseModel):
    model: str = Field(default="data/models/qwen3-4b-instruct-2507-awq")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    max_model_len: int = Field(default=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
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
        return _coerce_optional_int(value)


class LlmSumSettings(BaseModel):
    model: str = Field(default="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    max_model_len: int = Field(default=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
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
        return _coerce_optional_int(value)


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
    llm_enhance: LlmEnhanceSettings = Field(default_factory=LlmEnhanceSettings)
    llm_sum: LlmSumSettings = Field(default_factory=LlmSumSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
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

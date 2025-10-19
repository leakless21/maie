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
    whisper_condition_on_previous_text: bool = Field(default=True)
    whisper_language: str | None = Field(default=None)
    whisper_cpu_threads: int | None = Field(default=None)

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
    chunkformer_model_path: str = Field(default="data/models/chunkformer-ctc-large-vie")
    chunkformer_model_variant: str = Field(
        default="khanhld/chunkformer-large-vie",
    )
    chunkformer_chunk_size: int = Field(default=64)
    chunkformer_left_context_size: int = Field(default=128)
    chunkformer_right_context_size: int = Field(default=128)
    chunkformer_total_batch_duration: int = Field(default=14400)
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
        description="Maximum concurrent sequences for summarization vLLM runs",
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Token budget (prompt + decode) per scheduler step when summarizing",
    )
    max_num_partial_prefills: int | None = Field(
        default=None,
        ge=1,
        description="Chunked prefill configuration for summarization workloads",
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
    environment: Literal["development", "production"] = Field(
        default="development"
    )
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

    _COMPAT_FIELDS: ClassVar[Mapping[str, Tuple[str, ...]]] = {
        "log_level": ("logging", "log_level"),
        "log_dir": ("logging", "log_dir"),
        "log_rotation": ("logging", "log_rotation"),
        "log_retention": ("logging", "log_retention"),
        "log_console_serialize": ("logging", "log_console_serialize"),
        "log_file_serialize": ("logging", "log_file_serialize"),
        "log_compression": ("logging", "log_compression"),
        "enable_loguru": ("logging", "enable_loguru"),
        "loguru_diagnose": ("logging", "loguru_diagnose"),
        "loguru_backtrace": ("logging", "loguru_backtrace"),
        "loguru_format": ("logging", "loguru_format"),
        "api_host": ("api", "host"),
        "api_port": ("api", "port"),
        "secret_api_key": ("api", "secret_key"),
        "fallback_api_keys": ("api", "fallback_keys"),
        "max_file_size_mb": ("api", "max_file_size_mb"),
        "redis_url": ("redis", "url"),
        "redis_results_db": ("redis", "results_db"),
        "max_queue_depth": ("redis", "max_queue_depth"),
        "whisper_model_path": ("asr", "whisper_model_path"),
        "whisper_model_variant": ("asr", "whisper_model_variant"),
        "whisper_beam_size": ("asr", "whisper_beam_size"),
        "whisper_vad_filter": ("asr", "whisper_vad_filter"),
        "whisper_vad_min_silence_ms": ("asr", "whisper_vad_min_silence_ms"),
        "whisper_vad_speech_pad_ms": ("asr", "whisper_vad_speech_pad_ms"),
        "whisper_device": ("asr", "whisper_device"),
        "whisper_compute_type": ("asr", "whisper_compute_type"),
        "whisper_cpu_fallback": ("asr", "whisper_cpu_fallback"),
        "whisper_condition_on_previous_text": (
            "asr",
            "whisper_condition_on_previous_text",
        ),
        "whisper_language": ("asr", "whisper_language"),
        "whisper_cpu_threads": ("asr", "whisper_cpu_threads"),
        "chunkformer_model_path": ("chunkformer", "chunkformer_model_path"),
        "chunkformer_model_variant": ("chunkformer", "chunkformer_model_variant"),
        "chunkformer_chunk_size": ("chunkformer", "chunkformer_chunk_size"),
        "chunkformer_left_context_size": ("chunkformer", "chunkformer_left_context_size"),
        "chunkformer_right_context_size": (
            "chunkformer",
            "chunkformer_right_context_size",
        ),
        "chunkformer_total_batch_duration": (
            "chunkformer",
            "chunkformer_total_batch_duration",
        ),
        "chunkformer_return_timestamps": ("chunkformer", "chunkformer_return_timestamps"),
        "chunkformer_device": ("chunkformer", "chunkformer_device"),
        "chunkformer_batch_size": ("chunkformer", "chunkformer_batch_size"),
        "chunkformer_cpu_fallback": ("chunkformer", "chunkformer_cpu_fallback"),
        "llm_enhance_model": ("llm_enhance", "model"),
        "llm_enhance_gpu_memory_utilization": (
            "llm_enhance",
            "gpu_memory_utilization",
        ),
        "llm_enhance_max_model_len": ("llm_enhance", "max_model_len"),
        "llm_enhance_temperature": ("llm_enhance", "temperature"),
        "llm_enhance_top_p": ("llm_enhance", "top_p"),
        "llm_enhance_top_k": ("llm_enhance", "top_k"),
        "llm_enhance_max_tokens": ("llm_enhance", "max_tokens"),
        "llm_enhance_use_beam_search": ("llm_enhance", "use_beam_search"),
        "llm_enhance_quantization": ("llm_enhance", "quantization"),
        "llm_enhance_max_num_seqs": ("llm_enhance", "max_num_seqs"),
        "llm_enhance_max_num_batched_tokens": (
            "llm_enhance",
            "max_num_batched_tokens",
        ),
        "llm_enhance_max_num_partial_prefills": (
            "llm_enhance",
            "max_num_partial_prefills",
        ),
        "llm_sum_model": ("llm_sum", "model"),
        "llm_sum_gpu_memory_utilization": ("llm_sum", "gpu_memory_utilization"),
        "llm_sum_max_model_len": ("llm_sum", "max_model_len"),
        "llm_sum_temperature": ("llm_sum", "temperature"),
        "llm_sum_top_p": ("llm_sum", "top_p"),
        "llm_sum_top_k": ("llm_sum", "top_k"),
        "llm_sum_max_tokens": ("llm_sum", "max_tokens"),
        "llm_sum_quantization": ("llm_sum", "quantization"),
        "llm_sum_max_num_seqs": ("llm_sum", "max_num_seqs"),
        "llm_sum_max_num_batched_tokens": (
            "llm_sum",
            "max_num_batched_tokens",
        ),
        "llm_sum_max_num_partial_prefills": (
            "llm_sum",
            "max_num_partial_prefills",
        ),
        "audio_dir": ("paths", "audio_dir"),
        "models_dir": ("paths", "models_dir"),
        "templates_dir": ("paths", "templates_dir"),
        "worker_name": ("worker", "worker_name"),
        "job_timeout": ("worker", "job_timeout"),
        "result_ttl": ("worker", "result_ttl"),
        "worker_concurrency": ("worker", "worker_concurrency"),
        "worker_prefetch_multiplier": ("worker", "worker_prefetch_multiplier"),
        "worker_prefetch_timeout": ("worker", "worker_prefetch_timeout"),
        "enable_enhancement": ("features", "enable_enhancement"),
    }

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

    def get_template_path(self, template_id: str) -> Path:
        return self.paths.templates_dir / f"{template_id}.json"

    def _resolve_compat(self, name: str) -> Any:
        path = self._COMPAT_FIELDS[name]
        value: Any = self
        for segment in path:
            value = getattr(value, segment)
        if name in self._SECRET_FIELDS and isinstance(value, SecretStr):
            return value.get_secret_value()
        return value

    def __getattr__(self, name: str) -> Any:
        if name in self._COMPAT_FIELDS:
            return self._resolve_compat(name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._COMPAT_FIELDS:
            path = self._COMPAT_FIELDS[name]
            target: Any = self
            for segment in path[:-1]:
                target = BaseModel.__getattribute__(target, segment)
            field_name = path[-1]
            if name in self._SECRET_FIELDS:
                value = value if isinstance(value, SecretStr) else SecretStr(str(value))
            current_value = BaseModel.__getattribute__(target, field_name)
            if isinstance(current_value, Path):
                value = Path(value)
            elif isinstance(current_value, tuple):
                if isinstance(value, str):
                    value = (value,)
                elif isinstance(value, list):
                    value = tuple(value)
            setattr(target, field_name, value)
            stored_value = BaseModel.__getattribute__(target, field_name)
            if name in self._SECRET_FIELDS and isinstance(stored_value, SecretStr):
                stored_value = stored_value.get_secret_value()
            object.__setattr__(self, name, stored_value)
            return
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._COMPAT_FIELDS:
            object.__delattr__(self, name)
            return
        super().__delattr__(name)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        for alias in self._COMPAT_FIELDS:
            object.__setattr__(self, alias, self._resolve_compat(alias))

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

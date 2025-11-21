from __future__ import annotations

from pathlib import Path
from typing import Dict

ProfileData = Dict[str, object]

DEVELOPMENT_PROFILE: ProfileData = {
    "environment": "development",
    "debug": True,
    "verbose_components": True,
    "logging": {
        "log_level": "DEBUG",
        "log_console_serialize": False,
        "log_file_serialize": True,
        "log_rotation": "50 MB",
        "log_retention": "3 days",
    },
    "api": {
        "secret_key": "dev_api_key_change_in_production",
        "max_file_size_mb": 100,
    },
    "redis": {
        "max_queue_depth": 10,
    },
    "worker": {
        "worker_name": "maie-worker-dev",
        "job_timeout": 360,  # 6 minutes - reasonable for ASR + diarization + LLM
        "result_ttl": 3600,
        "worker_concurrency": 1,
        "worker_prefetch_multiplier": 2,
        "worker_prefetch_timeout": 10,
    },
    "cleanup": {
        "audio_cleanup_interval": 60,
        "log_cleanup_interval": 3600,
        "cache_cleanup_interval": 300,
        "disk_monitor_interval": 30,
        "audio_retention_days": 1,
        "logs_retention_days": 1,
        "disk_threshold_pct": 90,
        "emergency_cleanup": False,
    },
    "llm_enhance": {
        "gpu_memory_utilization": 0.6,
        "max_model_len": 16384,
    },
    "llm_sum": {
        "gpu_memory_utilization": 0.6,
        "max_model_len": 16384,
    },
    "features": {
        "enable_enhancement": False,
    },
    "diarization": {
        "enabled": True,
    },
}

PRODUCTION_PROFILE: ProfileData = {
    "environment": "production",
    "debug": False,
    "verbose_components": False,
    "logging": {
        "log_level": "INFO",
        "log_console_serialize": True,
        "log_file_serialize": True,
        "log_rotation": "1 GB",
        "log_retention": "30 days",
    },
    "api": {
        "secret_key": "CHANGE_ME_IN_PRODUCTION",
        "max_file_size_mb": 500,
    },
    "redis": {
        "max_queue_depth": 100,
    },
    "paths": {
        "audio_dir": Path("/app/data/audio"),
        "models_dir": Path("/app/data/models"),
        "templates_dir": Path("/app/templates"),
    },
    "worker": {
        "worker_name": "maie-worker-prod",
        "job_timeout": 600,
        "result_ttl": 86400,
        "worker_concurrency": 2,
        "worker_prefetch_multiplier": 4,
        "worker_prefetch_timeout": 30,
    },
    "cleanup": {
        "audio_cleanup_interval": 3600,
        "log_cleanup_interval": 86400,
        "cache_cleanup_interval": 1800,
        "disk_monitor_interval": 300,
        "audio_retention_days": 7,
        "logs_retention_days": 7,
        "disk_threshold_pct": 85,
        "emergency_cleanup": True,
    },
    "llm_enhance": {
        "gpu_memory_utilization": 0.9,
        "max_model_len": 32768,
    },
    "llm_sum": {
        "gpu_memory_utilization": 0.9,
        "max_model_len": 32768,
    },
    "features": {
        "enable_enhancement": True,
    },
}


PROFILES: Dict[str, ProfileData] = {
    "development": DEVELOPMENT_PROFILE,
    "production": PRODUCTION_PROFILE,
}

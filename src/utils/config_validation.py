"""Configuration-specific validation utilities for the MAIE project.

This module provides validation functions specifically designed for 
configuration settings and parameters.
"""

from typing import Any, Dict, List, Union
from .types import ErrorContext
from .validation import (
    validate_range, validate_positive, validate_port, validate_percentage,
    validate_choice, coerce_optional_int, coerce_optional_str
)


def validate_audio_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate audio-related configuration settings.
    
    Args:
        settings: Dictionary containing audio settings to validate
        
    Returns:
        Validated settings dictionary
        
    Raises:
        ValueError: If any setting is invalid
        
    Examples:
        >>> validate_audio_settings({"sample_rate": 16000, "chunk_duration": 30})
        {'sample_rate': 16000, 'chunk_duration': 30}
    """
    validated = {}
    
    # Validate sample rate (common rates: 800, 16000, 22050, 4100, 48000)
    if "sample_rate" in settings:
        sample_rate = coerce_optional_int(settings["sample_rate"])
        if sample_rate is not None:
            validate_range(sample_rate, 8000, 192000, "sample_rate")
            validated["sample_rate"] = sample_rate
    
    # Validate chunk duration in seconds
    if "chunk_duration" in settings:
        chunk_duration = coerce_optional_int(settings["chunk_duration"])
        if chunk_duration is not None:
            validate_range(chunk_duration, 1, 300, "chunk_duration")  # 1 second to 5 minutes
            validated["chunk_duration"] = chunk_duration
    
    # Validate audio format
    if "audio_format" in settings:
        audio_format = coerce_optional_str(settings["audio_format"])
        if audio_format is not None:
            allowed_formats = ["mp3", "wav", "flac", "m4a", "opus"]
            validate_choice(audio_format.lower(), allowed_formats, "audio_format")
            validated["audio_format"] = audio_format.lower()
    
    # Validate volume normalization
    if "normalize_volume" in settings:
        validated["normalize_volume"] = bool(settings["normalize_volume"])
    
    return validated


def validate_llm_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate LLM-related configuration settings.
    
    Args:
        settings: Dictionary containing LLM settings to validate
        
    Returns:
        Validated settings dictionary
        
    Raises:
        ValueError: If any setting is invalid
    """
    validated = {}
    
    # Validate model name
    if "model_name" in settings:
        model_name = coerce_optional_str(settings["model_name"])
        if model_name is not None and model_name.strip():
            validated["model_name"] = model_name.strip()
    
    # Validate temperature
    if "temperature" in settings:
        temperature = settings["temperature"]
        if temperature is not None:
            temperature = float(temperature)
            validate_range(temperature, 0.0, 2.0, "temperature")
            validated["temperature"] = temperature
    
    # Validate max tokens
    if "max_tokens" in settings:
        max_tokens = coerce_optional_int(settings["max_tokens"])
        if max_tokens is not None:
            validate_positive(max_tokens, "max_tokens")
            validate_range(max_tokens, 1, 32768, "max_tokens")
            validated["max_tokens"] = max_tokens
    
    # Validate top_p
    if "top_p" in settings:
        top_p = settings["top_p"]
        if top_p is not None:
            top_p = float(top_p)
            validate_range(top_p, 0.0, 1.0, "top_p")
            validated["top_p"] = top_p
    
    # Validate frequency_penalty
    if "frequency_penalty" in settings:
        freq_penalty = settings["frequency_penalty"]
        if freq_penalty is not None:
            freq_penalty = float(freq_penalty)
            validate_range(freq_penalty, -2.0, 2.0, "frequency_penalty")
            validated["frequency_penalty"] = freq_penalty
    
    # Validate presence_penalty
    if "presence_penalty" in settings:
        pres_penalty = settings["presence_penalty"]
        if pres_penalty is not None:
            pres_penalty = float(pres_penalty)
            validate_range(pres_penalty, -2.0, 2.0, "presence_penalty")
            validated["presence_penalty"] = pres_penalty
    
    return validated


def validate_api_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate API-related configuration settings.
    
    Args:
        settings: Dictionary containing API settings to validate
        
    Returns:
        Validated settings dictionary
        
    Raises:
        ValueError: If any setting is invalid
    """
    validated = {}
    
    # Validate port
    if "port" in settings:
        port = settings["port"]
        if port is not None:
            validated["port"] = validate_port(port)
    
    # Validate host
    if "host" in settings:
        host = coerce_optional_str(settings["host"])
        if host is not None:
            # Basic host validation - should be a valid hostname or IP
            if not (isinstance(host, str) and len(host) <= 255):
                raise ValueError("host must be a valid string with max length 255")
            validated["host"] = host
    
    # Validate rate limiting
    if "rate_limit" in settings:
        rate_limit = coerce_optional_int(settings["rate_limit"])
        if rate_limit is not None:
            validate_positive(rate_limit, "rate_limit")
            validated["rate_limit"] = rate_limit
    
    # Validate request timeout
    if "request_timeout" in settings:
        timeout = coerce_optional_int(settings["request_timeout"])
        if timeout is not None:
            validate_positive(timeout, "request_timeout")
            validated["request_timeout"] = timeout
    
    # Validate max request size
    if "max_request_size" in settings:
        max_size = coerce_optional_int(settings["max_request_size"])
        if max_size is not None:
            validate_positive(max_size, "max_request_size")
            validated["max_request_size"] = max_size
    
    return validated


def validate_cleanup_intervals(intervals: Dict[str, int]) -> Dict[str, int]:
    """Validate cleanup interval configurations.
    
    Args:
        intervals: Dictionary containing cleanup intervals to validate
        
    Returns:
        Validated intervals dictionary
        
    Raises:
        ValueError: If any interval is invalid
    """
    validated = {}
    
    # Validate audio cleanup interval (in hours)
    if "audio_cleanup_hours" in intervals:
        hours = intervals["audio_cleanup_hours"]
        if hours is not None:
            hours = int(hours)
            validate_range(hours, 1, 8760, "audio_cleanup_hours")  # Max 1 year
            validated["audio_cleanup_hours"] = hours
    
    # Validate log cleanup interval (in hours)
    if "log_cleanup_hours" in intervals:
        hours = intervals["log_cleanup_hours"]
        if hours is not None:
            hours = int(hours)
            validate_range(hours, 1, 8760, "log_cleanup_hours")  # Max 1 year
            validated["log_cleanup_hours"] = hours
    
    # Validate cache cleanup interval (in hours)
    if "cache_cleanup_hours" in intervals:
        hours = intervals["cache_cleanup_hours"]
        if hours is not None:
            hours = int(hours)
            validate_range(hours, 1, 8760, "cache_cleanup_hours")  # Max 1 year
            validated["cache_cleanup_hours"] = hours
    
    return validated


def validate_retention_periods(periods: Dict[str, int]) -> Dict[str, int]:
    """Validate retention period configurations.
    
    Args:
        periods: Dictionary containing retention periods to validate
        
    Returns:
        Validated periods dictionary
        
    Raises:
        ValueError: If any period is invalid
    """
    validated = {}
    
    # Validate audio retention period (in days)
    if "audio_retention_days" in periods:
        days = periods["audio_retention_days"]
        if days is not None:
            days = int(days)
            validate_range(days, 1, 3650, "audio_retention_days")  # Max 10 years
            validated["audio_retention_days"] = days
    
    # Validate log retention period (in days)
    if "log_retention_days" in periods:
        days = periods["log_retention_days"]
        if days is not None:
            days = int(days)
            validate_range(days, 1, 3650, "log_retention_days")  # Max 10 years
            validated["log_retention_days"] = days
    
    # Validate result retention period (in days)
    if "result_retention_days" in periods:
        days = periods["result_retention_days"]
        if days is not None:
            days = int(days)
            validate_range(days, 1, 3650, "result_retention_days")  # Max 10 years
            validated["result_retention_days"] = days
    
    return validated


def validate_disk_thresholds(thresholds: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
    """Validate disk usage threshold configurations.
    
    Args:
        thresholds: Dictionary containing disk threshold percentages to validate
        
    Returns:
        Validated thresholds dictionary
        
    Raises:
        ValueError: If any threshold is invalid
    """
    validated = {}
    
    # Validate disk usage warning threshold
    if "disk_warning_threshold" in thresholds:
        threshold = thresholds["disk_warning_threshold"]
        if threshold is not None:
            threshold = float(threshold)
            validate_percentage(threshold, "disk_warning_threshold")
            validated["disk_warning_threshold"] = threshold
    
    # Validate disk usage critical threshold
    if "disk_critical_threshold" in thresholds:
        threshold = thresholds["disk_critical_threshold"]
        if threshold is not None:
            threshold = float(threshold)
            validate_percentage(threshold, "disk_critical_threshold")
            validated["disk_critical_threshold"] = threshold
    
    # Validate minimum free space threshold (in GB)
    if "min_free_space_gb" in thresholds:
        space = thresholds["min_free_space_gb"]
        if space is not None:
            space = float(space)
            validate_positive(space, "min_free_space_gb")
            validated["min_free_space_gb"] = space
    
    return validated


def validate_worker_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate worker-related configuration settings.
    
    Args:
        settings: Dictionary containing worker settings to validate
        
    Returns:
        Validated settings dictionary
        
    Raises:
        ValueError: If any setting is invalid
    """
    validated = {}
    
    # Validate number of workers
    if "num_workers" in settings:
        num_workers = coerce_optional_int(settings["num_workers"])
        if num_workers is not None:
            validate_range(num_workers, 1, 64, "num_workers")  # Reasonable upper limit
            validated["num_workers"] = num_workers
    
    # Validate worker timeout
    if "worker_timeout" in settings:
        timeout = coerce_optional_int(settings["worker_timeout"])
        if timeout is not None:
            validate_positive(timeout, "worker_timeout")
            validated["worker_timeout"] = timeout
    
    # Validate batch size
    if "batch_size" in settings:
        batch_size = coerce_optional_int(settings["batch_size"])
        if batch_size is not None:
            validate_range(batch_size, 1, 1024, "batch_size")
            validated["batch_size"] = batch_size
    
    return validated


def validate_logging_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate logging configuration settings.
    
    Args:
        settings: Dictionary containing logging settings to validate
        
    Returns:
        Validated settings dictionary
        
    Raises:
        ValueError: If any setting is invalid
    """
    validated = {}
    
    # Validate log level
    if "log_level" in settings:
        log_level = coerce_optional_str(settings["log_level"])
        if log_level is not None:
            allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            validate_choice(log_level.upper(), allowed_levels, "log_level")
            validated["log_level"] = log_level.upper()
    
    # Validate log retention
    if "log_retention_days" in settings:
        days = coerce_optional_int(settings["log_retention_days"])
        if days is not None:
            validate_range(days, 1, 365, "log_retention_days")
            validated["log_retention_days"] = days
    
    # Validate log rotation size
    if "log_rotation_size_mb" in settings:
        size = coerce_optional_int(settings["log_rotation_size_mb"])
        if size is not None:
            validate_positive(size, "log_rotation_size_mb")
            validated["log_rotation_size_mb"] = size
    
    return validated


def validate_all_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate all configuration sections.
    
    Args:
        config: Complete configuration dictionary to validate
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If any configuration setting is invalid
    """
    validated_config = {}
    
    # Validate each section if present
    if "audio" in config:
        validated_config["audio"] = validate_audio_settings(config["audio"])
    
    if "llm" in config:
        validated_config["llm"] = validate_llm_settings(config["llm"])
    
    if "api" in config:
        validated_config["api"] = validate_api_settings(config["api"])
    
    if "cleanup" in config:
        validated_config["cleanup"] = validate_cleanup_intervals(config["cleanup"])
    
    if "retention" in config:
        validated_config["retention"] = validate_retention_periods(config["retention"])
    
    if "disk" in config:
        validated_config["disk"] = validate_disk_thresholds(config["disk"])
    
    if "worker" in config:
        validated_config["worker"] = validate_worker_settings(config["worker"])
    
    if "logging" in config:
        validated_config["logging"] = validate_logging_settings(config["logging"])
    
    return validated_config
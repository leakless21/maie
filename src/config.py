"""
Configuration module for Modular Audio Intelligence Engine (MAIE)

This module defines the Pydantic Settings configuration class for the MAIE system
as specified in TDD Section 3.6 and 5.2. It provides environment variable definitions,
configuration validation, and default values for all required settings.

The configuration supports the 5-6 environment variables mentioned in the TDD document
and follows the minimal configuration approach for on-premises deployment.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    MAIE Application Settings
    
    Configuration class that defines all required settings for the MAIE system.
    Values are loaded from environment variables with appropriate defaults.
    """
    
    # ============================================================================
    # CORE API & AUTHENTICATION
    # ============================================================================
    
    # API authentication key for MAIE services (TDD Section 3.1)
    secret_api_key: str = "your_secret_api_key_here"
    
    # ============================================================================
    # REDIS CONFIGURATION
    # ============================================================================
    
    # Redis connection URL for task queues and data storage
    redis_url: str = "redis://localhost:6379/0"
    
    # Maximum queue depth for backpressure management (NFR-5)
    max_queue_depth: int = 1000
    
    # ============================================================================
    # AUDIO PROCESSING
    # ============================================================================
    
    # Whisper model variant for audio transcription (TDD Section 3.3.1)
    whisper_model_variant: str = "erax-wow-turbo"
    
    # Directory for uploaded audio files
    audio_dir: str = "/data/audio"
    
    # ============================================================================
    # LLM CONFIGURATION
    # ============================================================================
    
    # Maximum context length for LLM processing (TDD Section 3.2)
    llm_context_length: int = 4096
    
    # Maximum model length for vLLM
    max_model_len: int = 32768
    
    # LLM sampling temperature (0.0 to 1.0)
    temperature: float = 0.7
    
    # Maximum tokens for LLM generation
    max_tokens: int = 2048
    
    # Top-p sampling parameter for LLM
    top_p: float = 0.9
    
    # GPU memory utilization for vLLM (0.0 to 1.0)
    gpu_memory_utilization: float = 0.9
    
    # ============================================================================
    # MODEL PATHS & DIRECTORIES
    # ============================================================================
    
    # Directory for AI model weights
    model_dir: str = "/data/models"
    
    # Specific model paths (TDD Section 3.7)
    whisper_model_path: str = "/data/models/whisper/erax-wow-turbo"
    chunkformer_model_path: str = "/data/models/chunkformer/large-vie"
    llm_model_path: str = "/data/models/llm/qwen3-4b-awq"
    
    # Redis data directory
    redis_data_dir: str = "/data/redis"
    
    # Application templates directory
    templates_dir: str = "/app/templates"
    
    # Chat templates directory for vLLM integration
    chat_templates_dir: str = "/app/assets/chat-templates"
    
    # Default chat template name
    default_chat_template: str = "qwen3_nonthinking"
    
    # ============================================================================
    # DEVELOPMENT & DEBUGGING
    # ============================================================================
    
    # Enable debug mode (true/false)
    debug: bool = False
    
    # Log level (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"
    
    class Config:
        """
        Configuration for the Settings class
        """
        # Load environment variables from .env file if present
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        # Use underscore as separator in environment variables
        # e.g., SECRET_API_KEY maps to secret_api_key
        env_nested_delimiter = "_"
        
        # Case insensitive environment variables
        case_sensitive = False


# Create a singleton instance of the settings
settings = Settings()


def get_settings() -> Settings:
    """
    Get the application settings instance
    
    Returns:
        Settings: The application settings instance
    """
    return settings


# Validate configuration at startup
def validate_config():
    """
    Validate the configuration settings
    
    Performs basic validation checks on the configuration to ensure
    required values are present and valid.
    """
    # Check that required directories exist if they should
    required_dirs = [
        settings.audio_dir,
        settings.model_dir,
        settings.templates_dir
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Warning: Directory does not exist: {directory}")
    
    # Validate API key is not default value in production
    if settings.secret_api_key == "your_secret_api_key_here" and not settings.debug:
        print("Warning: Using default API key. This should be changed in production.")
    
    # Validate model paths exist if models should be pre-downloaded
    model_paths = [
        settings.whisper_model_path,
        settings.chunkformer_model_path,
        settings.llm_model_path
    ]
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Model path does not exist: {model_path}. "
                  f"Run scripts/download_models.sh before starting workers.")


if __name__ == "__main__":
    # Validate configuration when module is run directly
    validate_config()
    print("Configuration validation complete")
    print(f"API Key: {'*' * len(settings.secret_api_key) if settings.secret_api_key else 'Not set'}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"Audio Directory: {settings.audio_dir}")
    print(f"Model Directory: {settings.model_dir}")
    print(f"Debug Mode: {settings.debug}")
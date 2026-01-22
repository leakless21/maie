#!/usr/bin/env python3
"""
Ollama Configuration Helper for MAIE.

Reads MAIE configuration and provides Ollama-compatible settings.
Validates Ollama installation, server status, and model availability.
"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Default model for Jetson/edge deployments
DEFAULT_MODEL = "ministral-3:3b"


def get_ollama_settings() -> dict:
    """
    Get Ollama configuration from MAIE settings or environment.
    
    Returns:
        dict with host, port, enhance_model, summary_model, base_url
    """
    host = os.getenv("OLLAMA_HOST", "localhost")
    port = os.getenv("OLLAMA_PORT", "11434")
    
    # Try to load from MAIE settings if available
    enhance_model = DEFAULT_MODEL
    summary_model = DEFAULT_MODEL
    
    try:
        from src.config import settings
        enhance_model = settings.llm_server.enhance_model_name or DEFAULT_MODEL
        summary_model = settings.llm_server.summary_model_name or DEFAULT_MODEL
    except ImportError:
        # MAIE not installed, use defaults/env vars
        enhance_model = os.getenv("OLLAMA_ENHANCE_MODEL", DEFAULT_MODEL)
        summary_model = os.getenv("OLLAMA_SUMMARY_MODEL", DEFAULT_MODEL)
    
    return {
        "host": host,
        "port": port,
        "enhance_model": enhance_model,
        "summary_model": summary_model,
        "base_url": f"http://{host}:{port}/v1",
    }


def check_ollama_installed() -> bool:
    """Check if Ollama is installed on the system."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_ollama_running(host: str = "localhost", port: str = "11434") -> bool:
    """
    Check if Ollama server is running and responding.
    
    Args:
        host: Ollama server host
        port: Ollama server port
        
    Returns:
        True if server is running and healthy
    """
    try:
        url = f"http://{host}:{port}/api/version"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def list_local_models(host: str = "localhost", port: str = "11434") -> list:
    """
    Get list of locally available Ollama models.
    
    Args:
        host: Ollama server host
        port: Ollama server port
        
    Returns:
        List of model names available locally
    """
    try:
        url = f"http://{host}:{port}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return [model["name"] for model in data.get("models", [])]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError):
        return []


def pull_model(model_name: str) -> bool:
    """
    Pull a model using Ollama CLI.
    
    Args:
        model_name: Name of the model to pull (e.g., 'ministral-3b:3b')
        
    Returns:
        True if pull succeeded
    """
    try:
        print(f"Pulling model: {model_name}...")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model: {e}", file=sys.stderr)
        return False


def model_exists(model_name: str, host: str = "localhost", port: str = "11434") -> bool:
    """
    Check if a specific model is available locally.
    
    Args:
        model_name: Model name to check
        host: Ollama server host
        port: Ollama server port
        
    Returns:
        True if model is available
    """
    models = list_local_models(host, port)
    # Check exact match or base name match (e.g., 'ministral-3b:3b' matches 'ministral-3b:3b')
    for m in models:
        if m == model_name or m.startswith(model_name.split(":")[0]):
            return True
    return False


def get_maie_env_config() -> str:
    """
    Generate MAIE environment configuration for Ollama.
    
    Returns:
        String with environment variable exports
    """
    config = get_ollama_settings()
    return f"""# MAIE Ollama Configuration
# Add these to your .env file or export them:

APP_LLM_BACKEND=vllm_server
APP_LLM_SERVER__ENHANCE_BASE_URL={config['base_url']}
APP_LLM_SERVER__SUMMARY_BASE_URL={config['base_url']}
APP_LLM_SERVER__ENHANCE_MODEL_NAME={config['enhance_model']}
APP_LLM_SERVER__SUMMARY_MODEL_NAME={config['summary_model']}
APP_LLM_SUM__STRUCTURED_OUTPUTS_ENABLED=false
"""


def print_status(verbose: bool = False):
    """Print current Ollama status."""
    config = get_ollama_settings()
    installed = check_ollama_installed()
    running = check_ollama_running(config["host"], config["port"])
    
    print("=" * 60)
    print("Ollama Status for MAIE")
    print("=" * 60)
    print(f"Installed:      {'✓ Yes' if installed else '✗ No'}")
    print(f"Server Running: {'✓ Yes' if running else '✗ No'}")
    print(f"Host:           {config['host']}")
    print(f"Port:           {config['port']}")
    print(f"API Base URL:   {config['base_url']}")
    print("-" * 60)
    print(f"Enhance Model:  {config['enhance_model']}")
    print(f"Summary Model:  {config['summary_model']}")
    
    if running:
        models = list_local_models(config["host"], config["port"])
        enhance_ready = model_exists(config["enhance_model"], config["host"], config["port"])
        summary_ready = model_exists(config["summary_model"], config["host"], config["port"])
        
        print("-" * 60)
        print(f"Enhance Model Ready: {'✓ Yes' if enhance_ready else '✗ No (needs pull)'}")
        print(f"Summary Model Ready: {'✓ Yes' if summary_ready else '✗ No (needs pull)'}")
        
        if verbose and models:
            print("-" * 60)
            print("Available Models:")
            for m in models:
                print(f"  - {m}")
    
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ollama configuration helper for MAIE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --status              Show Ollama status
  %(prog)s --check               Check if Ollama is running (exit 0/1)
  %(prog)s --list-models         List available models
  %(prog)s --show-config         Show MAIE configuration for Ollama
  %(prog)s --pull                Pull required models
  %(prog)s --ensure              Ensure Ollama is ready (start if needed, pull models)
""",
    )
    parser.add_argument("--status", action="store_true", help="Show detailed Ollama status")
    parser.add_argument("--check", action="store_true", help="Check if Ollama is running")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--show-config", action="store_true", help="Show MAIE configuration")
    parser.add_argument("--pull", action="store_true", help="Pull required models")
    parser.add_argument("--ensure", action="store_true", help="Ensure Ollama is ready")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    config = get_ollama_settings()

    if args.status:
        print_status(verbose=args.verbose)
        return 0

    if args.check:
        running = check_ollama_running(config["host"], config["port"])
        if args.verbose:
            print(f"Ollama running: {running}")
        return 0 if running else 1

    if args.list_models:
        if not check_ollama_running(config["host"], config["port"]):
            print("Error: Ollama is not running", file=sys.stderr)
            return 1
        models = list_local_models(config["host"], config["port"])
        if models:
            print("Available models:")
            for m in models:
                print(f"  - {m}")
        else:
            print("No models found")
        return 0

    if args.show_config:
        print(get_maie_env_config())
        return 0

    if args.pull:
        if not check_ollama_installed():
            print("Error: Ollama is not installed", file=sys.stderr)
            return 1
        if not check_ollama_running(config["host"], config["port"]):
            print("Error: Ollama server is not running. Start with: ollama serve", file=sys.stderr)
            return 1
        
        models_to_pull = set([config["enhance_model"], config["summary_model"]])
        for model in models_to_pull:
            if not model_exists(model, config["host"], config["port"]):
                if not pull_model(model):
                    return 1
            else:
                print(f"Model '{model}' is already available")
        return 0

    if args.ensure:
        # Check installation
        if not check_ollama_installed():
            print("Error: Ollama is not installed", file=sys.stderr)
            print("Install from: https://ollama.ai/download", file=sys.stderr)
            return 1
        
        # Check/start server
        if not check_ollama_running(config["host"], config["port"]):
            print("Ollama server not running. Please start it with: ollama serve")
            return 1
        
        # Pull models
        models_to_pull = set([config["enhance_model"], config["summary_model"]])
        for model in models_to_pull:
            if not model_exists(model, config["host"], config["port"]):
                if not pull_model(model):
                    return 1
        
        print("\n✓ Ollama is ready for MAIE!")
        print("\nConfiguration:")
        print(get_maie_env_config())
        return 0

    # Default: show status
    print_status(verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())

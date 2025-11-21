#!/usr/bin/env python3
"""
vLLM Server Configuration Helper

Reads MAIE configuration from src/config/model.py and outputs
vLLM server CLI arguments for launching a local vLLM server.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import MAIE configuration (no logging to avoid stdout pollution)
from src.config import settings


def get_vllm_args(model_type: str = "enhance") -> list[str]:
    """
    Generate vLLM server CLI arguments from MAIE configuration.
    
    Args:
        model_type: "enhance" or "summary" to select which model config to use
    
    Returns:
        List of CLI arguments for vLLM server
    """
    # Select config based on model type
    if model_type == "summary":
        config = settings.llm_sum
    else:
        config = settings.llm_enhance
    
    # Get server settings
    port = os.getenv("VLLM_SERVER_PORT", "8001")
    host = os.getenv("VLLM_SERVER_HOST", "0.0.0.0")
    
    # Build arguments
    args = [
        "--host", host,
        "--port", port,
        "--model", config.model,
        "--served-model-name", f"maie-{model_type}",  # Expose model with consistent name
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-model-len", str(config.max_model_len),
    ]
    
    # Add optional arguments if configured
    if config.quantization:
        args.extend(["--quantization", config.quantization])
    
    if config.max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(config.max_num_seqs)])
    
    if config.max_num_batched_tokens is not None:
        args.extend(["--max-num-batched-tokens", str(config.max_num_batched_tokens)])
    
    # Add tensor parallel if multiple GPUs
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices and "," in cuda_visible_devices:
        gpu_count = len(cuda_visible_devices.split(","))
        args.extend(["--tensor-parallel-size", str(gpu_count)])
    
    return args


def print_config_info(model_type: str = "enhance"):
    """Print configuration information for debugging."""
    if model_type == "summary":
        config = settings.llm_sum
    else:
        config = settings.llm_enhance
    
    port = os.getenv("VLLM_SERVER_PORT", "8001")
    host = os.getenv("VLLM_SERVER_HOST", "0.0.0.0")
    
    # Print to stderr to avoid mixing with vLLM args on stdout
    print("=" * 70, file=sys.stderr)
    print("vLLM Server Configuration (from MAIE settings)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Model Type: {model_type}", file=sys.stderr)
    print(f"Model Path: {config.model}", file=sys.stderr)
    print(f"Server Host: {host}", file=sys.stderr)
    print(f"Server Port: {port}", file=sys.stderr)
    print(f"GPU Memory: {config.gpu_memory_utilization}", file=sys.stderr)
    print(f"Max Model Len: {config.max_model_len}", file=sys.stderr)
    if config.quantization:
        print(f"Quantization: {config.quantization}", file=sys.stderr)
    if config.max_num_seqs:
        print(f"Max Num Seqs: {config.max_num_seqs}", file=sys.stderr)
    if config.max_num_batched_tokens:
        print(f"Max Batched Tokens: {config.max_num_batched_tokens}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate vLLM server CLI arguments from MAIE config"
    )
    parser.add_argument(
        "--model-type",
        choices=["enhance", "summary"],
        default="enhance",
        help="Which model config to use (default: enhance)"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration information before printing args"
    )
    
    args = parser.parse_args()
    
    if args.show_config:
        print_config_info(args.model_type)
    
    # Print vLLM arguments (one per line for easy parsing)
    vllm_args = get_vllm_args(args.model_type)
    print(" ".join(vllm_args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Safe wrapper for infer_vllm.py that handles long Vietnamese text better.

This script:
1. Reads transcript from stdin or file
2. Automatically sets reasonable max_tokens
3. Handles proper argument escaping

Usage:
  echo "Your text" | python examples/infer_vllm_safe.py --template-id interview_transcript_v1
  python examples/infer_vllm_safe.py --file transcript.txt --template-id interview_transcript_v1
"""

import sys
import os
import argparse
import json

# Ensure repository root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.config import configure_logging, get_logger
from src.processors.llm import LLMProcessor


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters for English, ~2 for Vietnamese"""
    return len(text) // 2  # Conservative estimate for Vietnamese


def main() -> int:
    logger = configure_logging() or get_logger()

    parser = argparse.ArgumentParser(
        description="Safe vLLM inference with automatic token management"
    )
    parser.add_argument(
        "--task",
        choices=["enhancement", "summary"],
        default="summary",
        help="Task to run"
    )
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--file", help="Read text from file")
    input_group.add_argument("--text", help="Direct text input (will be read from stdin if not provided)")
    
    parser.add_argument("--template-id", help="Template id for summary task")
    parser.add_argument("--model-path", help="Optional model path override")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-tokens", type=int, help="Override max tokens (auto-calculated if not set)")
    parser.add_argument(
        "--auto-max-tokens-ratio",
        type=float,
        default=0.4,
        help="Ratio of estimated input tokens for output (default: 0.4 for summaries)"
    )
    args = parser.parse_args()

    # Read input text
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            input_text = f.read()
        logger.info(f"Read {len(input_text)} characters from {args.file}")
    elif args.text:
        input_text = args.text
    else:
        # Read from stdin
        logger.info("Reading from stdin (press Ctrl+D when done)...")
        input_text = sys.stdin.read()
    
    if not input_text.strip():
        logger.error("No input text provided")
        return 1

    # Automatically calculate max_tokens if not provided
    if args.max_tokens is None:
        estimated_input_tokens = estimate_tokens(input_text)
        
        if args.task == "summary":
            # For summaries: output ~30-40% of input length
            auto_max_tokens = max(512, int(estimated_input_tokens * args.auto_max_tokens_ratio))
            # Cap at reasonable limit
            auto_max_tokens = min(auto_max_tokens, 8192)
        else:
            # For enhancement: output ~1:1 with input
            auto_max_tokens = max(1024, int(estimated_input_tokens * 1.2))
            auto_max_tokens = min(auto_max_tokens, 16384)
        
        logger.info(
            f"Auto-calculated max_tokens: {auto_max_tokens} "
            f"(estimated input: {estimated_input_tokens} tokens)"
        )
        args.max_tokens = auto_max_tokens

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. GPU is required for vLLM.")
            print(json.dumps({
                "error": "gpu_required",
                "message": "CUDA not available; vLLM requires GPU"
            }))
            return 2
    except ImportError:
        logger.error("PyTorch is not installed.")
        return 2

    # Build kwargs
    overrides = {}
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.top_p is not None:
        overrides["top_p"] = args.top_p
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens

    # Initialize processor
    processor = (
        LLMProcessor(model_path=args.model_path)
        if args.model_path
        else LLMProcessor()
    )

    try:
        if args.task == "enhancement":
            result = processor.execute(input_text, task="enhancement", **overrides)
            output_text = getattr(result, "text", None) or str(result)
            print(output_text)
        else:
            # Summary task
            result = processor.execute(
                input_text,
                task="summary",
                template_id=args.template_id,
                **overrides,
            )
            
            # Log which API was used
            method = (
                getattr(result, "metadata", {}).get("method")
                if hasattr(result, "metadata")
                else None
            )
            if method == "chat_api":
                logger.info("✓ Used vLLM chat() API")
            
            # Output structured summary
            structured = (
                getattr(result, "metadata", {}).get("structured_summary")
                if hasattr(result, "metadata")
                else None
            )
            
            if structured is not None:
                print(json.dumps(structured, ensure_ascii=False, indent=2))
            else:
                output_text = getattr(result, "text", None) or str(result)
                print(output_text)
        
        return 0
        
    except Exception as e:
        logger.exception(f"LLM inference failed: {e}")
        print(json.dumps({
            "error": "llm_inference_failed",
            "message": str(e)
        }))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Simple vLLM server test - no JSON parsing, just raw output.

Usage:
  python examples/test_vllm_simple.py --text "hello world"
  python examples/test_vllm_simple.py --text "your text here" --task summary
"""

import os
import sys
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.config import configure_logging, get_logger
from src.processors.llm import LLMProcessor
from src.utils.device import ensure_cuda_available


def main() -> int:
    logger = configure_logging() or get_logger()

    parser = argparse.ArgumentParser(description="Simple vLLM server test - no JSON parsing")
    parser.add_argument("--text", required=True, help="Text to process")
    parser.add_argument(
        "--task",
        choices=["enhancement", "summary"],
        default="enhancement",
        help="Task to perform",
    )
    parser.add_argument(
        "--template-id",
        default="generic_summary_v2",
        help="Template ID for summary task",
    )

    args = parser.parse_args()

    try:
        ensure_cuda_available()
    except Exception as e:
        logger.warning(f"CUDA check failed: {e}")

    # Initialize processor
    processor = LLMProcessor()

    print(f"\n{'='*80}")
    print(f"Task: {args.task}")
    print(f"Text length: {len(args.text)} chars")
    print(f"{'='*80}\n")

    # Disable structured outputs to get raw response
    result = processor.execute(
        text=args.text,
        task=args.task,
        template_id=args.template_id if args.task == "summary" else None,
        # Override to disable structured outputs
        **{"_disable_structured_outputs": True}
    )

    print(f"\n{'='*80}")
    print("RAW OUTPUT:")
    print(f"{'='*80}")
    print(result.text)
    print(f"\n{'='*80}")
    print("METADATA:")
    print(f"{'='*80}")
    print(f"Task: {result.metadata.get('task')}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Model: {result.model_info}")
    print(f"Validation: {result.metadata.get('validation', 'N/A')}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

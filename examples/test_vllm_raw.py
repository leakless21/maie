#!/usr/bin/env python3
"""
Simple vLLM server test - raw output without JSON parsing.

Usage:
  python examples/test_vllm_raw.py --text "hello world"
  python examples/test_vllm_raw.py --text "your text here" --task summary
"""

import os
import sys
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Disable structured outputs before importing config
os.environ["MAIE_LLM_SUM_STRUCTURED_OUTPUTS_ENABLED"] = "false"

from src.config import configure_logging, get_logger
from src.processors.llm import LLMProcessor
from src.utils.device import ensure_cuda_available


def main() -> int:
    logger = configure_logging() or get_logger()

    parser = argparse.ArgumentParser(
        description="Simple vLLM server test - raw output without JSON parsing"
    )
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
    logger.info("Initializing LLM processor...")
    processor = LLMProcessor()

    print(f"\n{'='*80}")
    print(f"Task: {args.task}")
    print(f"Text length: {len(args.text)} chars")
    print(f"Structured outputs: DISABLED")
    print(f"{'='*80}\n")

    # Execute with summary template
    result = processor.execute(
        text=args.text,
        task=args.task,
        template_id=args.template_id if args.task == "summary" else None,
    )

    print(f"\n{'='*80}")
    print("RAW LLM OUTPUT:")
    print(f"{'='*80}")
    print(result.text)
    print(f"\n{'='*80}")
    print("METADATA:")
    print(f"{'='*80}")
    print(f"Task: {result.metadata.get('task')}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Model: {result.model_info.get('model_name', 'unknown') if result.model_info else 'unknown'}")
    print(f"Validation: {result.metadata.get('validation', 'N/A')}")
    if "error" in result.metadata:
        print(f"Error: {result.metadata['error']}")
    if "parse_error" in result.metadata:
        print(f"Parse error: {result.metadata['parse_error']}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

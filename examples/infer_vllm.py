#!/usr/bin/env python3
"""
CLI: vLLM-based LLM inference using MAIE's LLMProcessor.

Usage examples:
  python examples/infer_vllm.py --task enhancement --text "hello world"
  python examples/infer_vllm.py --task summary --text "...transcript..." --template-id meeting

Prints the resulting text (enhanced transcript or summary JSON) to stdout.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import os
import sys

# Ensure repository root is on sys.path when running directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import configure_logging, get_logger
from src.processors.llm import LLMProcessor


def main() -> int:
    logger = configure_logging() or get_logger()

    parser = argparse.ArgumentParser(description="LLM inference via vLLM")
    parser.add_argument(
        "--task",
        choices=["enhancement", "summary"],
        default="enhancement",
        help="Task to run",
    )
    parser.add_argument(
        "--text", required=True, help="Input text (e.g., transcript) to process"
    )
    parser.add_argument("--template-id", help="Template id when task=summary")
    parser.add_argument("--model-path", help="Optional model path override")
    # Common sampling overrides
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.top_p is not None:
        overrides["top_p"] = args.top_p
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens

    processor = (
        LLMProcessor(model_path=args.model_path) if args.model_path else LLMProcessor()
    )

    try:
        if args.task == "enhancement":
            result = processor.execute(args.text, task="enhancement", **overrides)
            # LLMResult with .text
            output_text = getattr(result, "text", None) or str(result)
            print(output_text)
        else:
            # summary
            result = processor.execute(
                args.text,
                task="summary",
                template_id=args.template_id,
                **overrides,
            )
            # Expect LLMResult.text to carry rendered/serialized summary; if metadata has structured JSON, prefer that
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
    except Exception as e:  # noqa: BLE001 - top-level CLI error reporting
        logger.exception("LLM inference failed: {}", e)
        print(json.dumps({"error": "llm_inference_failed", "message": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

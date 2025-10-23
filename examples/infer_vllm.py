#!/usr/bin/env python3
"""
CLI: vLLM-based LLM inference using MAIE's LLMProcessor with Chat API.

Usage examples:
  # Text enhancement (uses generate API)
  python examples/infer_vllm.py --task enhancement --text "hello world"

  # Summary with chat API (new approach)
  python examples/infer_vllm.py --task summary --text "...transcript..." --template-id generic_summary_v2

  # Legacy summary (old templates with fallback)
  python examples/infer_vllm.py --task summary --text "...transcript..." --template-id generic_summary_v1

Prints the resulting text (enhanced transcript or summary JSON) to stdout.

Note: Summary task now uses vLLM's chat() API with OpenAI-format messages.
The chat API automatically handles chat template formatting and stop tokens.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Any, Dict

# Ensure repository root is on sys.path when running directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Mitigate CUDA memory fragmentation when running standalone.
# Respect any value the user already provided.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.config import configure_logging, get_logger  # noqa: E402
from src.processors.llm import LLMProcessor  # noqa: E402


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

    # Enforce GPU-only for vLLM in all environments.
    # Fail fast if CUDA is not available to avoid silently falling back to CPU.
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            logger.error("CUDA is not available. GPU is required for vLLM.")
            print(
                json.dumps(
                    {
                        "error": "gpu_required",
                        "message": "CUDA not available; vLLM examples require GPU",
                    }
                )
            )
            return 2
    except ImportError:
        logger.error("PyTorch is not installed. GPU is required for vLLM.")
        print(
            json.dumps(
                {
                    "error": "gpu_required",
                    "message": "PyTorch not installed; vLLM examples require GPU",
                }
            )
        )
        return 2

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
            # summary - now uses chat API automatically
            result = processor.execute(
                args.text,
                task="summary",
                template_id=args.template_id,
                **overrides,
            )

            # Check if chat API was used
            method = (
                getattr(result, "metadata", {}).get("method")
                if hasattr(result, "metadata")
                else None
            )
            if method == "chat_api":
                logger.info("✓ Used vLLM chat() API with OpenAI-format messages")
            else:
                logger.info("↻ Used fallback generate() API with old template")

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

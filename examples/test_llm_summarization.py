#!/usr/bin/env python3
"""
Test LLM summary with the actual ASR transcript to debug truncation issues.

Usage:
  python examples/test_llm_summarization.py --transcript-file <path> --template interview_transcript_v1
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import configure_logging, get_logger
from src.config.logging import get_module_logger
from src.processors.llm import LLMProcessor


def main() -> int:
    logger = configure_logging() or get_logger()
    logger = get_module_logger(__name__)

    parser = argparse.ArgumentParser(description="Test LLM summary")
    parser.add_argument("--transcript-file", help="Path to transcript file")
    parser.add_argument("--transcript-text", help="Transcript text directly")
    parser.add_argument(
        "--template",
        default="interview_transcript_v1",
        help="Template ID to use",
    )
    parser.add_argument("--max-tokens", type=int, help="Override max_tokens")
    args = parser.parse_args()

    # Get transcript
    if args.transcript_file:
        transcript_path = Path(args.transcript_file)
        if not transcript_path.exists():
            logger.error(f"Transcript file not found: {transcript_path}")
            return 2
        transcript = transcript_path.read_text(encoding="utf-8")
    elif args.transcript_text:
        transcript = args.transcript_text
    else:
        logger.error("Either --transcript-file or --transcript-text must be provided")
        return 2

    # Log input metrics
    char_count = len(transcript)
    word_count = len(transcript.split())
    logger.info(
        f"=== INPUT TRANSCRIPT === | {char_count:,} chars | {word_count:,} words"
    )
    logger.info(f"Preview (first 200 chars): {transcript[:200]}...")

    # Load LLM
    try:
        logger.info("Loading LLM model...")
        llm = LLMProcessor()
        llm._load_model()
        logger.info("LLM model loaded successfully")
    except Exception as e:
        logger.exception(f"Failed to load LLM: {e}")
        return 3

    # Generate summary
    try:
        logger.info(f"Generating summary with template: {args.template}")

        # Override max_tokens if provided
        runtime_overrides = {}
        if args.max_tokens:
            runtime_overrides["max_tokens"] = args.max_tokens
            logger.info(f"Overriding max_tokens to: {args.max_tokens}")

        result = llm.generate_summary(
            transcript=transcript,
            template_id=args.template,
            runtime_overrides=runtime_overrides if runtime_overrides else None,
        )

        if result.get("summary"):
            summary_str = json.dumps(result["summary"], indent=2, ensure_ascii=False)
            summary_char_count = len(summary_str)
            summary_word_count = len(summary_str.split())

            logger.info(
                f"=== OUTPUT SUMMARY === | {summary_char_count:,} chars | {summary_word_count:,} words"
            )
            logger.info(f"Retry count: {result.get('retry_count', 0)}")

            print("\n" + "=" * 80)
            print("SUMMARY OUTPUT:")
            print("=" * 80)
            print(summary_str)
            print("=" * 80 + "\n")

            return 0
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Summary generation failed: {error_msg}")
            return 1

    except Exception as e:
        logger.exception(f"Summary generation failed: {e}")
        return 1
    finally:
        try:
            llm.unload()
            logger.info("LLM model unloaded")
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
CLI: Audio preprocessing (validate + normalize) using MAIE's AudioPreprocessor.

Usage examples:
  python examples/infer_preprocess.py --input-audio data/audio/sample.wav
  python examples/infer_preprocess.py --input-audio input.mp3 --output-audio out.wav

Outputs concise JSON to stdout with keys:
  format, duration, sample_rate, channels, normalized (bool), normalized_path
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import os
import sys

# Ensure repository root is on sys.path when running directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import configure_logging, get_logger  # noqa: E402
from src.config.logging import get_module_logger  # noqa: E402
from src.processors.audio import AudioPreprocessor  # noqa: E402


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def main() -> int:
    logger = configure_logging() or get_logger()
    logger = get_module_logger(__name__)

    parser = argparse.ArgumentParser(
        description="Validate and normalize audio for ASR input"
    )
    parser.add_argument(
        "--input-audio",
        required=True,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--output-audio",
        required=False,
        help=(
            "Optional explicit path to write the normalized file. If provided and"
            " normalization occurs, the normalized file will be moved to this path."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input_audio)
    if not input_path.exists():
        logger.error("Input audio does not exist: {}", input_path)
        print(
            json.dumps(
                {"error": "input_not_found", "path": str(input_path)},
                default=_json_default,
            )
        )
        return 2

    pre = AudioPreprocessor()
    try:
        metadata: Dict[str, Any] = pre.preprocess(input_path)
        normalized_path = metadata.get("normalized_path")

        # If user provided explicit output path, move normalized file there
        if args.output_audio:
            target_path = Path(args.output_audio)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if normalized_path:
                shutil.move(str(normalized_path), str(target_path))
                metadata["normalized_path"] = target_path
            else:
                # No normalization occurred; copy original to requested output
                shutil.copy2(str(input_path), str(target_path))
                metadata["normalized_path"] = target_path

        # Add convenience boolean
        metadata["normalized"] = bool(metadata.get("normalized_path"))

        print(json.dumps(metadata, default=_json_default))
        return 0
    except Exception as e:  # noqa: BLE001 - top-level CLI error reporting
        logger.exception("Preprocessing failed: {}", e)
        print(
            json.dumps(
                {"error": "preprocess_failed", "message": str(e)},
                default=_json_default,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

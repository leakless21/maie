#!/usr/bin/env python3
"""
CLI: ASR inference with ChunkFormer backend using MAIE pipeline helpers.

Usage examples:
  python examples/infer_asr_chunkformer.py --audio data/audio/sample.wav
  python examples/infer_asr_chunkformer.py --audio input.wav --model-path data/models/chunkformer

Prints the transcription to stdout. Use --json to also include RTF/confidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import os
import sys

# Ensure repository root is on sys.path when running directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import configure_logging, get_logger
from src.config.logging import get_module_logger
from src.processors.audio import AudioPreprocessor
from src.worker.pipeline import (
    execute_asr_transcription,
    load_asr_model,
    unload_asr_model,
)


def main() -> int:
    logger = configure_logging() or get_logger()
    logger = get_module_logger(__name__)

    parser = argparse.ArgumentParser(description="ASR with ChunkFormer backend")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--model-path", help="Optional model path override")
    parser.add_argument("--left-context", type=int, default=None)
    parser.add_argument("--right-context", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("Audio not found: {}", audio_path)
        if args.json:
            print(json.dumps({"error": "input_not_found", "path": str(audio_path)}))
        return 2

    config: Dict[str, Any] = {}
    if args.model_path:
        config["model_path"] = args.model_path
    if args.left_context is not None:
        config["left_context"] = args.left_context
    if args.right_context is not None:
        config["right_context"] = args.right_context

    preprocessor = AudioPreprocessor()
    try:
        metadata = preprocessor.preprocess(audio_path)
    except Exception as e:
        logger.exception("Audio preprocessing failed: {}", e)
        if args.json:
            print(
                json.dumps(
                    {"error": "preprocessing_failed", "message": str(e), "path": str(audio_path)}
                )
            )
        return 3

    if metadata.get("normalized_path"):
        logger.info(
            "Audio normalized",
            original_format=metadata.get("format"),
            duration=metadata.get("duration"),
        )

    logger.info(
        "Audio preprocessing complete",
        duration=metadata.get("duration"),
        sample_rate=metadata.get("sample_rate"),
        channels=metadata.get("channels"),
        normalized=metadata.get("normalized_path") is not None,
    )

    processing_audio_path = metadata.get("normalized_path") or audio_path
    audio_duration = float(metadata.get("duration", 0.0))
    processing_audio_path = str(processing_audio_path)

    asr = None
    try:
        asr = load_asr_model("chunkformer", **config)
        # Phase 1: execute_asr_transcription now returns (ASRResult, rtf, metadata)
        asr_result, rtf, _meta = execute_asr_transcription(
            asr, processing_audio_path, audio_duration
        )
        
        if args.json:
            result_dict = {
                "transcript": asr_result.text,
                "rtf": rtf,
                "confidence": asr_result.confidence,
            }
            # Include segments if available (Phase 1 enhancement)
            if asr_result.segments:
                result_dict["segments"] = asr_result.segments
            print(json.dumps(result_dict))
        else:
            # Use segments by default - print each segment with timestamp
            if asr_result.segments:
                for segment in asr_result.segments:
                    start = segment.get("start", "")
                    end = segment.get("end", "")
                    text = segment.get("text", "").strip()
                    if text:
                        # Handle both numeric (Whisper) and string (ChunkFormer) timestamps
                        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                            print(f"[{start:.2f}s - {end:.2f}s] {text}")
                        else:
                            print(f"[{start} - {end}] {text}")
            else:
                # Fallback to full transcript if no segments
                print(asr_result.text)
        return 0
    except Exception as e:  # noqa: BLE001 - top-level CLI error reporting
        logger.exception("ChunkFormer inference failed: {}", e)
        if args.json:
            print(json.dumps({"error": "inference_failed", "message": str(e)}))
        return 1
    finally:
        if asr is not None:
            try:
                unload_asr_model(asr)
            except Exception:
                logger.warning("Failed to unload ASR model cleanly")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
CLI: ASR inference with Whisper backend using MAIE pipeline helpers.

Usage examples:
  python examples/infer_asr_whisper.py --audio data/audio/sample.wav
  python examples/infer_asr_whisper.py --audio input.wav --model-path data/models/whisper

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

from src.config import configure_logging, get_logger  # noqa: E402
from src.config.logging import get_module_logger  # noqa: E402
from src.processors.audio import AudioPreprocessor  # noqa: E402
from src.processors.asr.factory import ASRFactory  # noqa: E402


def main() -> int:
    logger = configure_logging() or get_logger()
    logger = get_module_logger(__name__)

    parser = argparse.ArgumentParser(description="ASR with Whisper backend")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    # Accept both --model-path and --model for convenience
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        help="Optional model path override",
    )
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument(
        "--language",
        help="Force language code for transcription (e.g., vi). Defaults to auto-detect.",
        default="vi",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        help="Override Whisper task. Defaults to backend configuration.",
        default="transcribe",
    )
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
    if args.beam_size is not None:
        config["beam_size"] = args.beam_size
    if args.vad_filter:
        config["vad_filter"] = True

    exec_kwargs: Dict[str, Any] = {}
    if args.language:
        exec_kwargs["language"] = args.language
    if args.task:
        exec_kwargs["task"] = args.task

    preprocessor = AudioPreprocessor()
    try:
        metadata = preprocessor.preprocess(audio_path)
    except Exception as e:
        logger.exception("Audio preprocessing failed: {}", e)
        if args.json:
            print(
                json.dumps(
                    {
                        "error": "preprocessing_failed",
                        "message": str(e),
                        "path": str(audio_path),
                    }
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
        asr = ASRFactory.create("whisper", **config)
        logger.info("ASR model loaded", backend="whisper")

        # Execute ASR transcription
        import time

        start_time = time.time()
        asr_result = asr.execute(
            audio_data=open(processing_audio_path, "rb").read(),
            **exec_kwargs,
        )
        processing_time = time.time() - start_time

        # Calculate RTF (Real-Time Factor)
        rtf = processing_time / audio_duration if audio_duration > 0 else 0.0

        if args.json:
            result_dict = {
                "transcript": asr_result.text,
                "rtf": rtf,
                "confidence": asr_result.confidence,
            }
            # Include segments if available
            if asr_result.segments:
                result_dict["segments"] = asr_result.segments
            print(json.dumps(result_dict))
        else:
            # Use segments by default - print each segment with timestamp
            if asr_result.segments:
                for segment in asr_result.segments:
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    text = segment.get("text", "").strip()
                    if text:
                        # Whisper uses numeric timestamps (float seconds)
                        print(f"[{start:.2f}s - {end:.2f}s] {text}")
            else:
                # Fallback to full transcript if no segments
                print(asr_result.text)
        return 0
    except Exception as e:  # noqa: BLE001 - top-level CLI error reporting
        logger.exception("Whisper inference failed: {}", e)
        if args.json:
            print(json.dumps({"error": "inference_failed", "message": str(e)}))
        return 1
    finally:
        if asr is not None:
            try:
                asr.unload()
                logger.info("ASR model unloaded")
            except Exception:
                logger.warning("Failed to unload ASR model cleanly")


if __name__ == "__main__":
    raise SystemExit(main())

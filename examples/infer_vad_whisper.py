#!/usr/bin/env python3
"""
CLI: VAD + ASR inference with Whisper backend using MAIE processors.

This example demonstrates how to use Voice Activity Detection (VAD) before ASR.
VAD preprocesses audio to detect speech segments, reducing noise and improving ASR accuracy.

Usage examples:
  python examples/infer_vad_whisper.py --audio data/audio/sample.wav
  python examples/infer_vad_whisper.py --audio input.wav --model-path data/models/whisper --vad-threshold 0.5
  python examples/infer_vad_whisper.py --audio input.wav --no-vad  # Skip VAD, run ASR only

Outputs transcription with VAD segment statistics via logging.
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
from src.config.model import VADSettings  # noqa: E402
from src.processors.audio import AudioPreprocessor  # noqa: E402
from src.processors.asr.factory import ASRFactory  # noqa: E402
from src.processors.vad.factory import VADFactory  # noqa: E402


def main() -> int:
    logger = configure_logging() or get_logger()
    logger = get_module_logger(__name__)

    parser = argparse.ArgumentParser(description="VAD + ASR with Whisper backend")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    # Accept both --model-path and --model for convenience
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        help="Optional ASR model path override",
    )
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD confidence threshold (0.0-1.0). Default: 0.5",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Skip VAD processing, run ASR only",
    )
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
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("Audio not found: {}", audio_path)
        if args.json:
            print(json.dumps({"error": "input_not_found", "path": str(audio_path)}))
        return 2

    # Prepare ASR config
    asr_config: Dict[str, Any] = {}
    if args.model_path:
        asr_config["model_path"] = args.model_path
    if args.beam_size is not None:
        asr_config["beam_size"] = args.beam_size

    # Prepare ASR exec kwargs
    asr_exec_kwargs: Dict[str, Any] = {}
    if args.language:
        asr_exec_kwargs["language"] = args.language
    if args.task:
        asr_exec_kwargs["task"] = args.task

    # Step 1: Audio preprocessing
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

    # Step 2: VAD processing (optional)
    vad_result = None
    if not args.no_vad:
        try:
            vad_config = VADSettings(silero_threshold=args.vad_threshold, device="cuda")
            vad = VADFactory.create("silero", **vad_config.model_dump())
            logger.info(
                "VAD model loaded",
                backend="silero",
                threshold=args.vad_threshold,
                device=vad_config.device,
            )

            # Run VAD detection on the preprocessed audio file
            import time

            start_time = time.time()
            vad_result = vad.detect_speech(processing_audio_path)
            vad_time = time.time() - start_time

            logger.info(
                "VAD detection complete",
                num_segments=len(vad_result.segments),
                speech_ratio=f"{vad_result.speech_ratio:.1%}",
                total_duration=f"{vad_result.total_duration:.2f}s",
                speech_duration=f"{vad_result.speech_duration:.2f}s",
                processing_time=f"{vad_time:.2f}s",
            )

            # Log speech segments for debugging
            speech_segments = vad_result.get_speech_segments()
            logger.debug(
                "Speech segments detected",
                count=len(speech_segments),
                segments=[
                    {
                        "start": f"{seg.start:.2f}s",
                        "end": f"{seg.end:.2f}s",
                        "duration": f"{seg.duration():.2f}s",
                    }
                    for seg in speech_segments[:5]  # Log first 5 segments
                ],
            )

            vad.unload()
            logger.info("VAD model unloaded")
        except Exception as e:
            logger.warning("VAD processing failed (continuing with ASR): {}", e)
            vad_result = None

    # Step 3: ASR transcription
    asr = None
    try:
        asr = ASRFactory.create("whisper", **asr_config)
        logger.info("ASR model loaded", backend="whisper")

        # Execute ASR transcription
        import time

        start_time = time.time()
        asr_result = asr.execute(
            audio_data=open(processing_audio_path, "rb").read(),
            **asr_exec_kwargs,
        )
        processing_time = time.time() - start_time

        # Calculate RTF (Real-Time Factor)
        rtf = processing_time / audio_duration if audio_duration > 0 else 0.0

        confidence_str = (
            f"{asr_result.confidence:.2%}"
            if asr_result.confidence is not None
            else "unknown"
        )
        logger.info(
            "ASR transcription complete",
            rtf=f"{rtf:.2f}x",
            processing_time=f"{processing_time:.2f}s",
            confidence=confidence_str,
        )

        if args.json:
            result_dict = {
                "transcript": asr_result.text,
                "rtf": rtf,
                "confidence": asr_result.confidence if asr_result.confidence is not None else 0.0,
            }
            # Include VAD result if available
            if vad_result:
                result_dict["vad"] = {
                    "num_segments": len(vad_result.segments),
                    "speech_ratio": vad_result.speech_ratio,
                    "total_duration": vad_result.total_duration,
                    "speech_duration": vad_result.speech_duration,
                }
            # Include ASR segments if available
            if asr_result.segments:
                result_dict["segments"] = asr_result.segments
            print(json.dumps(result_dict))
        else:
            # Log VAD statistics if available
            if vad_result:
                logger.info(
                    "VAD Results",
                    speech_segments=len(vad_result.segments),
                    speech_ratio=f"{vad_result.speech_ratio:.1%}",
                    speech_duration=f"{vad_result.speech_duration:.2f}s / {vad_result.total_duration:.2f}s",
                )

            # Log transcription
            if asr_result.segments:
                logger.info("Transcription segments:")
                for segment in asr_result.segments:
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    text = segment.get("text", "").strip()
                    if text:
                        logger.info(
                            "Segment",
                            start=f"{start:.2f}s",
                            end=f"{end:.2f}s",
                            text=text,
                        )
            else:
                # Log full transcript if no segments
                logger.info("Full Transcript", text=asr_result.text)

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

#!/usr/bin/env python3
"""
Extract the ASR transcript from an audio file and save it to a text file.

Usage:
  python examples/test_extract_asr_transcript.py --audio data/audio/sample.mp3 --output transcript.txt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import configure_logging, get_logger
from src.config.logging import get_module_logger
from src.processors.audio import AudioPreprocessor
from src.worker.pipeline import execute_asr_transcription, load_asr_model, unload_asr_model


def main() -> int:
    logger = configure_logging() or get_logger()
    logger = get_module_logger(__name__)

    parser = argparse.ArgumentParser(description="Extract ASR transcript from audio")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument(
        "--output", default="transcript.txt", help="Output transcript file"
    )
    parser.add_argument(
        "--backend", default="chunkformer", help="ASR backend (whisper or chunkformer)"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 2

    # Preprocess audio
    try:
        logger.info("Preprocessing audio...")
        preprocessor = AudioPreprocessor()
        metadata = preprocessor.preprocess(audio_path)
        processing_audio_path = metadata.get("normalized_path") or audio_path
        audio_duration = float(metadata.get("duration", 0.0))
        logger.info(
            f"Audio preprocessed: duration={audio_duration:.2f}s, path={processing_audio_path}"
        )
    except Exception as e:
        logger.exception(f"Audio preprocessing failed: {e}")
        return 3

    # Load ASR model
    asr = None
    try:
        logger.info(f"Loading ASR model: {args.backend}")
        asr = load_asr_model(args.backend)
        logger.info("ASR model loaded")

        # Transcribe
        logger.info("Transcribing audio...")
        asr_result, rtf, metadata = execute_asr_transcription(
            asr, str(processing_audio_path), audio_duration
        )

        # Log metrics
        char_count = len(asr_result.text)
        word_count = len(asr_result.text.split())
        segment_count = len(asr_result.segments) if asr_result.segments else 0

        logger.info(
            f"=== ASR OUTPUT === | {char_count:,} chars | {word_count:,} words | {segment_count} segment(s) | RTF: {rtf:.3f}"
        )

        # Save transcript
        output_path = Path(args.output)
        output_path.write_text(asr_result.text, encoding="utf-8")
        logger.info(f"Transcript saved to: {output_path}")

        print(f"\nTranscript saved to: {output_path}")
        print(f"Characters: {char_count:,}")
        print(f"Words: {word_count:,}")
        print(f"Segments: {segment_count}")
        print(f"RTF: {rtf:.3f}")

        return 0

    except Exception as e:
        logger.exception(f"ASR transcription failed: {e}")
        return 1
    finally:
        if asr is not None:
            try:
                unload_asr_model(asr)
                logger.info("ASR model unloaded")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

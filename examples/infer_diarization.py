#!/usr/bin/env python3
"""
Example: Using diarization with MAIE.

This script demonstrates how to use the speaker diarization feature
with both Whisper and ChunkFormer ASR backends.

Usage:
    python examples/infer_diarization.py --audio tests/assets/test_audio.wav --backend whisper
    python examples/infer_diarization.py --audio tests/assets/test_audio.wav --backend chunkformer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path to allow imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    """Run diarization example."""
    parser = argparse.ArgumentParser(
        description="Demonstrate diarization with MAIE ASR backends"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="whisper",
        choices=["whisper", "chunkformer"],
        help="ASR backend to use",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=True,
        help="Enable diarization",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/models/speaker-diarization-community-1",
        help="Path to diarization model",
    )

    args = parser.parse_args()

    # Validate audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    print(f"Audio file: {audio_path}")
    print(f"ASR backend: {args.backend}")
    print(f"Diarization: {'enabled' if args.diarize else 'disabled'}")
    print()

    # =========================================================================
    # Step 1: Load ASR backend
    # =========================================================================
    try:
        from src.processors.asr.factory import ASRFactory

        print(f"[1/3] Loading {args.backend} ASR backend...")
        asr_model = ASRFactory.create(backend_type=args.backend)
        print(f"      ✓ {args.backend} loaded successfully")

    except Exception as e:
        print(f"Error loading ASR model: {e}")
        return 1

    # =========================================================================
    # Step 2: Run ASR transcription
    # =========================================================================
    try:
        print(f"[2/3] Transcribing audio...")
        
        # Use the same audio preprocessing as the ASR pipeline
        from src.processors.audio import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        metadata = preprocessor.preprocess(audio_path)
        
        # Use preprocessed audio if available, otherwise original
        processing_audio_path = metadata.get("normalized_path") or audio_path
        print(f"      - Using audio: {processing_audio_path}")
        
        with open(processing_audio_path, "rb") as f:
            audio_bytes = f.read()

        asr_result = asr_model.execute(audio_bytes)
        print(f"      ✓ Transcription complete")
        print(f"      - Duration: {asr_result.segments[-1]['end']:.1f}s" if asr_result.segments else "      - No segments")
        print(f"      - Text: {asr_result.text[:100]}...")
        print(f"      - Segments: {len(asr_result.segments) if asr_result.segments else 0}")

    except Exception as e:
        print(f"Error during ASR: {e}")
        return 1

    # =========================================================================
    # Step 3: Apply diarization (optional)
    # =========================================================================
    merged_segs = []
    if args.diarize:
        try:
            from src.processors.audio.diarizer import get_diarizer

            print(f"[3/3] Applying speaker diarization...")

            # Create diarizer
            diarizer = get_diarizer(
                model_path=args.model_path,
                require_cuda=False,
            )

            if diarizer:
                # Run diarization on the same preprocessed audio used for ASR
                diar_spans = diarizer.diarize(str(processing_audio_path), num_speakers=None)

                if diar_spans:
                    # Convert ASR segments to proper format for alignment
                    asr_segs_for_diar = []
                    if asr_result.segments:
                        for seg in asr_result.segments:
                            class ASRSeg:
                                def __init__(self, start, end, text, words=None):
                                    self.start = start
                                    self.end = end
                                    self.text = text
                                    self.words = words

                            asr_segs_for_diar.append(
                                ASRSeg(seg["start"], seg["end"], seg["text"], seg.get("words"))
                            )

                    # Check if word timestamps are available
                    has_word_timestamps = diarizer.has_word_timestamps(asr_segs_for_diar)
                    
                    if has_word_timestamps:
                        # Use WhisperX-style word-level assignment
                        diarized_segs = diarizer.assign_word_speakers_whisperx_style(
                            diar_spans, asr_segs_for_diar
                        )
                    else:
                        # Skip diarization, keep segments as-is with speaker=None
                        from src.processors.audio.diarizer import DiarizedSegment
                        diarized_segs = []
                        for seg in asr_segs_for_diar:
                            diarized_segs.append(DiarizedSegment(
                                start=seg.start,
                                end=seg.end,
                                text=seg.text,
                                speaker=None,
                            ))

                    # Merge adjacent same-speaker segments
                    merged_segs = diarizer.merge_adjacent_same_speaker(diarized_segs)

                    print(f"      ✓ Diarization complete")
                    print(f"      - Speaker segments: {len(merged_segs)}")
                    print(
                        f"      - Unique speakers: {len(set(s.speaker for s in merged_segs if s.speaker))}"
                    )

                    # Display diarized transcript
                    print()
                    print("Speaker-attributed transcript:")
                    print("-" * 80)
                    for i, seg in enumerate(merged_segs, 1):
                        speaker = seg.speaker or "Unknown"
                        start_min, start_sec = divmod(seg.start, 60)
                        end_min, end_sec = divmod(seg.end, 60)
                        print(
                            f"[{i:2d}] {start_min:02.0f}:{start_sec:05.2f}-{end_min:02.0f}:{end_sec:05.2f} "
                            f"{speaker:>8}: {seg.text}"
                        )
                else:
                    print("      ✗ Diarization returned no speakers")
            else:
                print("      ✗ Diarizer not available (GPU/model issue)")

        except Exception as e:
            print(f"Error during diarization: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        print("[3/3] Diarization skipped (use --diarize to enable)")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 80)
    print("Summary:")
    print(f"  ASR Backend: {args.backend}")
    print(f"  Total text: {len(asr_result.text)} characters")
    print(f"  Word count: {len(asr_result.text.split()) if asr_result.text else 0}")
    print(f"  Segments: {len(asr_result.segments) if asr_result.segments else 0}")
    if args.diarize:
        print(f"  Diarized: Yes ({len(merged_segs) if 'merged_segs' in locals() else 0} speaker segments)")
    else:
        print("  Diarized: No")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
CLI: Speaker diarization inference (standalone, no ASR required).

Standalone testing tool for speaker diarization with configurable batch sizes 
to test memory management and performance tradeoffs. Does NOT require ASR.

Usage examples:
  # Basic diarization with default batch sizes
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav

  # Test with custom batch sizes
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav \\
    --embedding-batch-size 16 --segmentation-batch-size 16

  # Monitor memory usage during processing
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav \\
    --monitor-memory --json

  # Specify number of speakers if known
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav \\
    --num-speakers 2 --monitor-memory

  # Compare different batch size configurations
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav \\
    --test-batch-sizes "8,16,32,64" --monitor-memory

Output: Speaker segments with timing and speaker labels, optional memory stats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import psutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Ensure repository root is on sys.path when running directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import configure_logging, get_logger  # noqa: E402
from src.config.logging import get_module_logger  # noqa: E402
from src.processors.audio import AudioPreprocessor  # noqa: E402
from src.processors.audio.diarizer import get_diarizer  # noqa: E402


class MemoryMonitor:
    """Simple memory usage monitor for testing."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.peak_memory = None
        self.start_time = None

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

    def update(self):
        """Update peak memory usage."""
        current = self.process.memory_info().rss / 1024 / 1024  # MB
        if current > self.peak_memory:
            self.peak_memory = current

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return stats."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        current = self.process.memory_info().rss / 1024 / 1024  # MB
        return {
            "elapsed_seconds": round(elapsed, 2),
            "start_memory_mb": round(self.start_memory, 2),
            "peak_memory_mb": round(self.peak_memory, 2),
            "end_memory_mb": round(current, 2),
            "delta_memory_mb": round(current - (self.start_memory or 0), 2),
        }


def format_diarization_output(segments: List[Dict[str, Any]]) -> str:
    """Format diarization results for human-readable display."""
    lines = []
    current_speaker = None

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start

        # Speaker change indicator
        if speaker != current_speaker:
            lines.append(f"\n[Speaker {speaker}] ({start:.2f}s → {end:.2f}s, {duration:.2f}s)")
            current_speaker = speaker
        else:
            lines.append(f"({start:.2f}s → {end:.2f}s, {duration:.2f}s)")

    return "\n".join(lines)


def run_diarization_test(
    audio_path: Path,
    embedding_batch_size: int = 32,
    segmentation_batch_size: int = 32,
    num_speakers: Optional[int] = None,
    monitor_memory: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Run diarization with specified configuration and return results.

    Args:
        audio_path: Path to audio file
        embedding_batch_size: Batch size for embedding model
        segmentation_batch_size: Batch size for segmentation model
        num_speakers: Optional number of speakers (if known)
        monitor_memory: Whether to monitor memory usage

    Returns:
        Tuple of (diarization_result, memory_stats)
    """
    logger = get_module_logger(__name__)
    memory_monitor = MemoryMonitor() if monitor_memory else None

    if memory_monitor:
        memory_monitor.start()

    try:
        # Step 1: Preprocess audio
        logger.info(
            "Preprocessing audio",
            path=str(audio_path),
            embedding_batch_size=embedding_batch_size,
            segmentation_batch_size=segmentation_batch_size,
        )
        preprocessor = AudioPreprocessor()
        metadata = preprocessor.preprocess(audio_path)

        if not metadata:
            raise ValueError("Audio preprocessing returned empty metadata")

        if memory_monitor:
            memory_monitor.update()

        processing_audio_path = metadata.get("normalized_path") or audio_path
        audio_duration = float(metadata.get("duration", 0.0))
        sample_rate = int(metadata.get("sample_rate", 16000))

        logger.info(
            "Audio preprocessed successfully",
            duration=audio_duration,
            sample_rate=sample_rate,
            normalized=metadata.get("normalized_path") is not None,
        )

        # Step 2: Load diarization model with configured batch sizes
        logger.info(
            "Loading speaker diarization model",
            embedding_batch_size=embedding_batch_size,
            segmentation_batch_size=segmentation_batch_size,
            num_speakers=num_speakers,
        )

        diarizer = get_diarizer(
            embedding_batch_size=embedding_batch_size,
            segmentation_batch_size=segmentation_batch_size,
        )

        if not diarizer:
            raise ValueError("Failed to load diarization model")

        if memory_monitor:
            memory_monitor.update()

        logger.info("Diarization model loaded successfully")

        # Step 3: Run diarization
        logger.info("Running speaker diarization")
        diarized_segments = diarizer.diarize(
            str(processing_audio_path), num_speakers=num_speakers
        )

        if memory_monitor:
            memory_monitor.update()

        if not diarized_segments:
            raise ValueError("Diarization returned no segments")

        logger.info(
            "Diarization complete", num_segments=len(diarized_segments)
        )

        result = {
            "success": True,
            "audio_file": str(audio_path),
            "duration_seconds": audio_duration,
            "sample_rate": sample_rate,
            "batch_sizes": {
                "embedding_batch_size": embedding_batch_size,
                "segmentation_batch_size": segmentation_batch_size,
            },
            "num_speakers": num_speakers,
            "segments": diarized_segments,
            "num_segments": len(diarized_segments),
        }

        memory_stats = memory_monitor.stop() if memory_monitor else None

        return result, memory_stats

    except Exception as e:
        logger.exception("Diarization test failed: {}", e)
        result = {
            "success": False,
            "error": str(e),
            "audio_file": str(audio_path),
        }
        memory_stats = memory_monitor.stop() if memory_monitor else None
        return result, memory_stats


def main() -> int:
    """Main CLI entry point."""
    configure_logging()
    logger = get_module_logger(__name__)

    parser = argparse.ArgumentParser(
        description="Speaker diarization inference (standalone, no ASR required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic diarization with default batch sizes (32)
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav

  # Test with custom batch sizes
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav \\
    --embedding-batch-size 16 --segmentation-batch-size 16

  # Specify number of speakers if known
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav \\
    --num-speakers 2

  # Monitor memory usage
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav \\
    --monitor-memory --json

  # Compare batch size configurations (memory fix validation)
  python examples/infer_diarization_standalone.py --audio data/audio/sample.wav \\
    --test-batch-sizes "8,16,32,64" --monitor-memory
        """,
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to input audio file (WAV, MP3, OGG, FLAC, etc.)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for speaker embedding extraction (default: 32, official pyannote.audio default)",
    )
    parser.add_argument(
        "--segmentation-batch-size",
        type=int,
        default=32,
        help="Batch size for speech segmentation (default: 32, official pyannote.audio default)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Optional: number of speakers (if known, improves accuracy)",
    )
    parser.add_argument(
        "--monitor-memory",
        action="store_true",
        help="Monitor and report memory usage during processing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for parsing/automation)",
    )
    parser.add_argument(
        "--test-batch-sizes",
        help='Test multiple batch sizes (comma-separated, e.g. "8,16,32,64")',
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("Audio file not found: {}", audio_path)
        if args.json:
            print(json.dumps({"error": "audio_not_found", "path": str(audio_path)}))
        return 2

    if args.verbose:
        logger.info("Verbose logging enabled")

    # Test multiple batch size configurations if requested
    if args.test_batch_sizes:
        try:
            batch_sizes = [int(x.strip()) for x in args.test_batch_sizes.split(",")]
        except ValueError:
            logger.error(
                "Invalid batch sizes format. Expected comma-separated integers."
            )
            return 1

        logger.info(
            "Testing multiple batch size configurations",
            batch_sizes=batch_sizes,
        )

        all_results = []
        for batch_size in batch_sizes:
            logger.info(
                "Running test with batch_size={}",
                batch_size,
            )
            result, memory_stats = run_diarization_test(
                audio_path,
                embedding_batch_size=batch_size,
                segmentation_batch_size=batch_size,
                num_speakers=args.num_speakers,
                monitor_memory=args.monitor_memory,
            )

            if memory_stats:
                result["memory"] = memory_stats

            all_results.append(result)

            if not result.get("success"):
                logger.warning(
                    "Test failed for batch_size={}: {}",
                    batch_size,
                    result.get("error"),
                )

        if args.json:
            print(json.dumps(all_results, indent=2))
        else:
            print("\n" + "=" * 80)
            print("BATCH SIZE COMPARISON RESULTS")
            print("=" * 80)
            for result in all_results:
                batch_size = result.get("batch_sizes", {}).get(
                    "embedding_batch_size"
                )
                success = result.get("success", False)
                status = "✓ PASS" if success else "✗ FAIL"
                print(f"\nBatch Size: {batch_size:3d} - {status}")

                if success:
                    print(f"  Segments: {result.get('num_segments', 0)}")
                    if "memory" in result:
                        mem = result["memory"]
                        print(
                            f"  Memory: {mem.get('start_memory_mb'):.1f}MB "
                            f"→ {mem.get('peak_memory_mb'):.1f}MB "
                            f"(+{mem.get('delta_memory_mb'):.1f}MB, "
                            f"{mem.get('elapsed_seconds'):.1f}s)"
                        )
                else:
                    print(f"  Error: {result.get('error')}")

        return 0

    # Single test with specified batch sizes
    result, memory_stats = run_diarization_test(
        audio_path,
        embedding_batch_size=args.embedding_batch_size,
        segmentation_batch_size=args.segmentation_batch_size,
        num_speakers=args.num_speakers,
        monitor_memory=args.monitor_memory,
    )

    if memory_stats:
        result["memory"] = memory_stats

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result.get("success"):
            print("\n" + "=" * 80)
            print("SPEAKER DIARIZATION RESULTS")
            print("=" * 80)
            print(f"\nAudio:       {result['audio_file']}")
            print(f"Duration:    {result['duration_seconds']:.1f}s")
            print(f"Sample Rate: {result['sample_rate']} Hz")
            print(
                f"Batch Size:  {result['batch_sizes']['embedding_batch_size']} "
                f"(embedding) / {result['batch_sizes']['segmentation_batch_size']} (segmentation)"
            )
            print(f"Speakers:    {result['num_speakers'] or 'auto-detect'}")
            print(f"Segments:    {result['num_segments']}\n")

            if result["segments"]:
                print(format_diarization_output(result["segments"]))
            else:
                print("(No segments produced)")

            if "memory" in result:
                mem = result["memory"]
                print("\n" + "-" * 80)
                print("MEMORY STATISTICS")
                print("-" * 80)
                print(f"Start:       {mem['start_memory_mb']:.1f} MB")
                print(f"Peak:        {mem['peak_memory_mb']:.1f} MB")
                print(f"End:         {mem['end_memory_mb']:.1f} MB")
                print(f"Delta:       {mem['delta_memory_mb']:+.1f} MB")
                print(f"Elapsed:     {mem['elapsed_seconds']:.1f}s")
        else:
            print("✗ DIARIZATION FAILED")
            print(f"Error: {result['error']}")
            logger.error("Diarization test failed: {}", result.get("error"))
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())

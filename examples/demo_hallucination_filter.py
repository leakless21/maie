#!/usr/bin/env python3
"""
Demo script showing ASR hallucination filtering in action.

This script demonstrates how to use the HallucinationFilter to clean up
common ASR hallucinations from transcription segments.

Usage:
    python examples/demo_hallucination_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.asr_filters import FilterConfig, HallucinationFilter


def demo_basic_filtering():
    """Demonstrate basic hallucination filtering."""
    print("=" * 70)
    print("Demo 1: Basic Repetition Filtering")
    print("=" * 70)

    # Create filter with repetition detection
    config = FilterConfig(
        enabled=True,
        max_repeated_words=3,
        max_repeated_phrases=2,
        min_segment_length=2,
    )
    filter = HallucinationFilter(config)

    # Sample segments with hallucinations
    segments = [
        {"text": "Welcome to our meeting", "start": 0.0, "end": 2.0},
        {
            "text": "thank you thank you thank you thank you",
            "start": 2.0,
            "end": 4.0,
        },  # Hallucination
        {"text": "Let's discuss the quarterly results", "start": 4.0, "end": 7.0},
        {"text": "music", "start": 7.0, "end": 7.5},  # Hallucination
        {"text": "The revenue increased by fifteen percent", "start": 7.5, "end": 10.0},
        {
            "text": "subscribe and like subscribe and like",
            "start": 10.0,
            "end": 12.0,
        },  # Hallucination
        {"text": "Any questions from the team", "start": 12.0, "end": 14.0},
    ]

    print("\nOriginal segments:")
    for i, seg in enumerate(segments, 1):
        print(f"  [{i}] {seg['text']}")

    # Apply filtering
    filtered = filter.filter_segments(segments)

    print(f"\nFiltered segments (removed {len(segments) - len(filtered)} hallucinations):")
    for i, seg in enumerate(filtered, 1):
        print(f"  [{i}] {seg['text']}")


def demo_confidence_filtering():
    """Demonstrate confidence-based filtering."""
    print("\n" + "=" * 70)
    print("Demo 2: Confidence-Based Filtering")
    print("=" * 70)

    # Create filter with confidence threshold
    config = FilterConfig(
        enabled=True,
        min_segment_confidence=0.8,
    )
    filter = HallucinationFilter(config)

    # Segments with varying confidence levels
    segments = [
        {"text": "Clear and confident speech", "confidence": 0.95, "start": 0.0, "end": 2.0},
        {
            "text": "Uncertain mumbled words",
            "confidence": 0.45,
            "start": 2.0,
            "end": 4.0,
        },  # Low confidence
        {"text": "Back to clear audio", "confidence": 0.92, "start": 4.0, "end": 6.0},
        {
            "text": "Background noise interpreted",
            "confidence": 0.55,
            "start": 6.0,
            "end": 8.0,
        },  # Low confidence
        {"text": "Final clear statement", "confidence": 0.88, "start": 8.0, "end": 10.0},
    ]

    print("\nOriginal segments with confidence:")
    for i, seg in enumerate(segments, 1):
        conf = seg.get("confidence", "N/A")
        print(f"  [{i}] ({conf:.2f}) {seg['text']}")

    # Apply filtering
    filtered = filter.filter_segments(segments)

    print(f"\nFiltered segments (kept {len(filtered)}/{len(segments)} high-confidence):")
    for i, seg in enumerate(filtered, 1):
        conf = seg.get("confidence", "N/A")
        print(f"  [{i}] ({conf:.2f}) {seg['text']}")


def demo_word_probability_filtering():
    """Demonstrate word probability-based filtering."""
    print("\n" + "=" * 70)
    print("Demo 3: Word Probability Filtering")
    print("=" * 70)

    # Create filter with word probability threshold
    config = FilterConfig(
        enabled=True,
        min_word_probability=0.7,
    )
    filter = HallucinationFilter(config)

    # Segments with word-level probabilities
    segments = [
        {
            "text": "High quality transcription",
            "words": [
                {"word": "High", "probability": 0.92, "start": 0.0, "end": 0.3},
                {"word": "quality", "probability": 0.88, "start": 0.3, "end": 0.7},
                {"word": "transcription", "probability": 0.85, "start": 0.7, "end": 1.5},
            ],
            "start": 0.0,
            "end": 1.5,
        },
        {
            "text": "Uncertain garbage words",
            "words": [
                {"word": "Uncertain", "probability": 0.45, "start": 1.5, "end": 2.0},
                {"word": "garbage", "probability": 0.52, "start": 2.0, "end": 2.5},
                {"word": "words", "probability": 0.38, "start": 2.5, "end": 3.0},
            ],
            "start": 1.5,
            "end": 3.0,
        },  # Low avg probability
        {
            "text": "Another confident segment",
            "words": [
                {"word": "Another", "probability": 0.89, "start": 3.0, "end": 3.4},
                {"word": "confident", "probability": 0.91, "start": 3.4, "end": 4.0},
                {"word": "segment", "probability": 0.87, "start": 4.0, "end": 4.5},
            ],
            "start": 3.0,
            "end": 4.5,
        },
    ]

    print("\nOriginal segments with average word probability:")
    for i, seg in enumerate(segments, 1):
        if "words" in seg and seg["words"]:
            avg_prob = sum(w["probability"] for w in seg["words"]) / len(seg["words"])
            print(f"  [{i}] (avg={avg_prob:.2f}) {seg['text']}")
        else:
            print(f"  [{i}] (no word data) {seg['text']}")

    # Apply filtering
    filtered = filter.filter_segments(segments)

    print(f"\nFiltered segments (kept {len(filtered)}/{len(segments)} high-probability):")
    for i, seg in enumerate(filtered, 1):
        if "words" in seg and seg["words"]:
            avg_prob = sum(w["probability"] for w in seg["words"]) / len(seg["words"])
            print(f"  [{i}] (avg={avg_prob:.2f}) {seg['text']}")
        else:
            print(f"  [{i}] (no word data) {seg['text']}")


def demo_multilingual_filtering():
    """Demonstrate language-specific pattern filtering."""
    print("\n" + "=" * 70)
    print("Demo 4: Multilingual Pattern Filtering")
    print("=" * 70)

    # Create filters for different languages
    configs = {
        "English": FilterConfig(enabled=True, language="en"),
        "Vietnamese": FilterConfig(enabled=True, language="vi"),
        "Japanese": FilterConfig(enabled=True, language="ja"),
    }

    # Sample segments in different languages
    test_cases = [
        ("English", "thank you for watching"),
        ("English", "This is important content"),
        ("Vietnamese", "cảm ơn đã xem"),
        ("Vietnamese", "Nội dung quan trọng"),
        ("Japanese", "ご視聴ありがとうございました"),
        ("Japanese", "重要な内容です"),
    ]

    print("\nTesting language-specific patterns:")
    for lang, text in test_cases:
        filter = HallucinationFilter(configs[lang])
        segment = {"text": text, "start": 0.0, "end": 1.0}
        should_filter = filter.should_filter_segment(segment)

        status = "❌ FILTERED" if should_filter else "✅ KEPT"
        print(f"  {lang:12} | {status} | {text}")


def main():
    """Run all demos."""
    print("\n🎯 ASR Hallucination Filtering Demonstrations\n")

    try:
        demo_basic_filtering()
        demo_confidence_filtering()
        demo_word_probability_filtering()
        demo_multilingual_filtering()

        print("\n" + "=" * 70)
        print("✅ All demos completed successfully!")
        print("=" * 70)
        print("\nTo use hallucination filtering in your application:")
        print("  1. Enable in config: ASR__HALLUCINATION_FILTER_ENABLED=true")
        print("  2. Adjust thresholds as needed for your use case")
        print("  3. Monitor filtered segment counts in logs")
        print("\n")

        return 0

    except Exception as e:
        print(f"\n❌ Demo failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

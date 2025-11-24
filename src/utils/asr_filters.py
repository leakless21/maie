"""
ASR hallucination detection and filtering utilities.

Provides tools to detect and filter common hallucinations that ASR models
(particularly Whisper) often generate on silence, noise, or low-quality audio.

Common hallucination types:
- Repeated phrases or words
- Background music transcriptions ("Thanks for watching", "Subscribe", etc.)
- Noise interpreted as speech
- Low-confidence segments
- Common filler patterns

Usage:
    from src.utils.asr_filters import HallucinationFilter
    
    filter = HallucinationFilter()
    
    # Filter single text
    clean_text = filter.filter_text("Thank you. Thank you. Thank you.")
    
    # Filter segments from ASR result
    filtered_segments = filter.filter_segments(segments)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config.logging import get_module_logger

logger = get_module_logger(__name__)


@dataclass
class FilterConfig:
    """Configuration for hallucination filtering."""

    # Enable/disable filtering
    enabled: bool = False

    # Repetition detection
    max_repeated_words: int = 3  # Maximum consecutive identical words
    max_repeated_phrases: int = 2  # Maximum repeated phrases in a segment

    # Confidence thresholds
    min_segment_confidence: Optional[float] = None  # Minimum confidence for segments
    min_word_probability: Optional[float] = None  # Minimum word probability

    # Pattern matching
    pattern_file: Optional[str] = None  # Path to hallucination patterns JSON
    custom_patterns: Optional[List[str]] = None  # Additional regex patterns

    # Segment filtering
    min_segment_length: int = 1  # Minimum words per segment
    max_segment_length: Optional[int] = None  # Maximum words per segment

    # Language-specific settings
    language: Optional[str] = None  # Target language code (e.g., 'en', 'vi')


class HallucinationFilter:
    """
    Filter for detecting and removing ASR hallucinations.

    Supports multiple detection strategies:
    - Pattern matching against known hallucinations
    - Repetition detection (word and phrase level)
    - Confidence/probability thresholding
    - Language-specific filtering
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize hallucination filter.

        Args:
            config: Filter configuration. If None, uses defaults with filtering disabled.
        """
        self.config = config or FilterConfig()
        self._hallucination_patterns: Set[str] = set()
        self._regex_patterns: List[re.Pattern] = []

        if self.config.enabled:
            self._load_patterns()
            logger.info(
                "HallucinationFilter initialized",
                patterns_loaded=len(self._hallucination_patterns),
                regex_patterns=len(self._regex_patterns),
                language=self.config.language or "all",
            )

    def _load_patterns(self) -> None:
        """Load hallucination patterns from file and custom patterns."""
        # Load from file if specified
        if self.config.pattern_file:
            pattern_path = Path(self.config.pattern_file)
            if pattern_path.exists():
                try:
                    with open(pattern_path, encoding="utf-8") as f:
                        data = json.load(f)

                    # Load language-specific patterns
                    if self.config.language and self.config.language in data:
                        patterns = data[self.config.language]
                        self._hallucination_patterns.update(
                            p.lower() for p in patterns.get("exact", [])
                        )
                        for regex in patterns.get("regex", []):
                            self._regex_patterns.append(re.compile(regex, re.IGNORECASE))

                    # Load common patterns (apply to all languages)
                    if "common" in data:
                        patterns = data["common"]
                        self._hallucination_patterns.update(
                            p.lower() for p in patterns.get("exact", [])
                        )
                        for regex in patterns.get("regex", []):
                            self._regex_patterns.append(re.compile(regex, re.IGNORECASE))

                    logger.info(
                        "Loaded hallucination patterns from file",
                        file=str(pattern_path),
                        exact_patterns=len(self._hallucination_patterns),
                        regex_patterns=len(self._regex_patterns),
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load hallucination patterns from file: {}",
                        e,
                        file=str(pattern_path),
                    )

        # Add custom patterns
        if self.config.custom_patterns:
            for pattern in self.config.custom_patterns:
                try:
                    self._regex_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(
                        "Invalid custom regex pattern: {} - {}",
                        pattern,
                        e,
                    )

    def _is_repetitive(self, text: str) -> bool:
        """
        Check if text contains excessive repetition.

        Args:
            text: Text to check

        Returns:
            True if text is repetitive
        """
        if not text.strip():
            return False

        words = text.split()
        if len(words) < 2:
            return False

        # Check for repeated words
        consecutive_count = 1
        prev_word = words[0].lower()

        for word in words[1:]:
            current_word = word.lower()
            if current_word == prev_word:
                consecutive_count += 1
                if consecutive_count >= self.config.max_repeated_words:
                    logger.debug(
                        "Detected word repetition",
                        word=current_word,
                        count=consecutive_count,
                        text=text[:100],
                    )
                    return True
            else:
                consecutive_count = 1
                prev_word = current_word

        # Check for repeated phrases (3-5 words)
        for phrase_len in range(3, 6):
            if len(words) < phrase_len * 2:
                continue

            phrases: Dict[str, int] = {}
            for i in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[i : i + phrase_len]).lower()
                phrases[phrase] = phrases.get(phrase, 0) + 1
                if phrases[phrase] >= self.config.max_repeated_phrases:
                    logger.debug(
                        "Detected phrase repetition",
                        phrase=phrase,
                        count=phrases[phrase],
                        text=text[:100],
                    )
                    return True

        return False

    def _matches_pattern(self, text: str) -> bool:
        """
        Check if text matches known hallucination patterns.

        Args:
            text: Text to check

        Returns:
            True if text matches a hallucination pattern
        """
        text_lower = text.lower().strip()

        # Check exact patterns
        if text_lower in self._hallucination_patterns:
            logger.debug("Matched exact hallucination pattern", text=text)
            return True

        # Check regex patterns
        for pattern in self._regex_patterns:
            if pattern.search(text):
                logger.debug(
                    "Matched regex hallucination pattern",
                    text=text,
                    pattern=pattern.pattern,
                )
                return True

        return False

    def _is_low_quality(
        self,
        segment: Dict[str, Any],
        min_confidence: Optional[float] = None,
        min_probability: Optional[float] = None,
    ) -> bool:
        """
        Check if segment has low quality indicators.

        Args:
            segment: Segment dict with optional confidence/probability fields
            min_confidence: Minimum confidence threshold
            min_probability: Minimum word probability threshold

        Returns:
            True if segment is low quality
        """
        # Check segment-level confidence
        if min_confidence is not None and "confidence" in segment:
            confidence = segment.get("confidence")
            if confidence is not None and confidence < min_confidence:
                logger.debug(
                    "Low segment confidence",
                    confidence=confidence,
                    threshold=min_confidence,
                    text=segment.get("text", "")[:50],
                )
                return True

        # Check word-level probabilities
        if min_probability is not None and "words" in segment:
            words = segment.get("words", [])
            if words:
                avg_prob = sum(w.get("probability", 1.0) for w in words) / len(words)
                if avg_prob < min_probability:
                    logger.debug(
                        "Low word probabilities",
                        avg_probability=avg_prob,
                        threshold=min_probability,
                        text=segment.get("text", "")[:50],
                    )
                    return True

        return False

    def should_filter_segment(self, segment: Dict[str, Any]) -> bool:
        """
        Determine if a segment should be filtered out.

        Args:
            segment: Segment dictionary with 'text' key and optional metadata

        Returns:
            True if segment should be filtered
        """
        if not self.config.enabled:
            return False

        text = segment.get("text", "").strip()

        # Empty or very short segments
        if not text:
            return True

        words = text.split()
        word_count = len(words)

        # Check length constraints
        if word_count < self.config.min_segment_length:
            logger.debug("Segment too short", text=text, word_count=word_count)
            return True

        if self.config.max_segment_length and word_count > self.config.max_segment_length:
            logger.debug("Segment too long", text=text, word_count=word_count)
            return True

        # Check for repetition
        if self._is_repetitive(text):
            return True

        # Check against known patterns
        if self._matches_pattern(text):
            return True

        # Check quality metrics
        if self._is_low_quality(
            segment,
            self.config.min_segment_confidence,
            self.config.min_word_probability,
        ):
            return True

        return False

    def filter_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of ASR segments.

        Args:
            segments: List of segment dictionaries

        Returns:
            Filtered list of segments
        """
        if not self.config.enabled or not segments:
            return segments

        filtered = []
        filtered_count = 0

        for segment in segments:
            if not self.should_filter_segment(segment):
                filtered.append(segment)
            else:
                filtered_count += 1

        if filtered_count > 0:
            logger.info(
                "Filtered hallucinated segments",
                original_count=len(segments),
                filtered_count=filtered_count,
                remaining_count=len(filtered),
            )

        return filtered

    def filter_text(self, text: str) -> str:
        """
        Filter hallucinations from raw text.

        Note: This is a simple approach that checks the entire text.
        For segment-based filtering, use filter_segments() instead.

        Args:
            text: Raw transcription text

        Returns:
            Filtered text
        """
        if not self.config.enabled or not text:
            return text

        # Check if entire text is a hallucination
        if self._matches_pattern(text):
            logger.info("Filtered entire text as hallucination", text=text[:100])
            return ""

        # Check for repetition
        if self._is_repetitive(text):
            logger.info("Filtered repetitive text", text=text[:100])
            return ""

        return text


def create_filter_from_config() -> HallucinationFilter:
    """
    Create HallucinationFilter from application configuration.

    Reads configuration from src.config settings.

    Returns:
        Configured HallucinationFilter instance
    """
    from src import config as cfg

    # Build filter config from settings
    filter_config = FilterConfig(
        enabled=getattr(cfg.settings.asr, "hallucination_filter_enabled", False),
        max_repeated_words=getattr(cfg.settings.asr, "hallucination_max_repeated_words", 3),
        max_repeated_phrases=getattr(cfg.settings.asr, "hallucination_max_repeated_phrases", 2),
        min_segment_confidence=getattr(
            cfg.settings.asr, "hallucination_min_segment_confidence", None
        ),
        min_word_probability=getattr(
            cfg.settings.asr, "hallucination_min_word_probability", None
        ),
        pattern_file=getattr(cfg.settings.asr, "hallucination_pattern_file", None),
        min_segment_length=getattr(cfg.settings.asr, "hallucination_min_segment_length", 1),
        max_segment_length=getattr(cfg.settings.asr, "hallucination_max_segment_length", None),
        language=getattr(cfg.settings.asr, "whisper_language", None),
    )

    return HallucinationFilter(filter_config)

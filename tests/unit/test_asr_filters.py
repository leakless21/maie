"""
Tests for ASR hallucination filtering.
"""

from __future__ import annotations

from src.utils.asr_filters import (
    FilterConfig,
    HallucinationFilter,
    create_filter_from_config,
)


class TestFilterConfig:
    """Test FilterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FilterConfig()
        assert config.enabled is False
        assert config.max_repeated_words == 3
        assert config.max_repeated_phrases == 2
        assert config.min_segment_confidence is None
        assert config.min_word_probability is None
        assert config.pattern_file is None
        assert config.custom_patterns is None
        assert config.min_segment_length == 1
        assert config.max_segment_length is None
        assert config.language is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FilterConfig(
            enabled=True,
            max_repeated_words=5,
            max_repeated_phrases=3,
            min_segment_confidence=0.8,
            min_word_probability=0.7,
            pattern_file="custom_patterns.json",
            custom_patterns=["test.*pattern"],
            min_segment_length=2,
            max_segment_length=50,
            language="en",
        )
        assert config.enabled is True
        assert config.max_repeated_words == 5
        assert config.max_repeated_phrases == 3
        assert config.min_segment_confidence == 0.8
        assert config.min_word_probability == 0.7
        assert config.pattern_file == "custom_patterns.json"
        assert config.custom_patterns == ["test.*pattern"]
        assert config.min_segment_length == 2
        assert config.max_segment_length == 50
        assert config.language == "en"


class TestHallucinationFilter:
    """Test HallucinationFilter class."""

    def test_disabled_filter_returns_unchanged(self):
        """When filtering is disabled, segments should pass through unchanged."""
        config = FilterConfig(enabled=False)
        filter = HallucinationFilter(config)

        segments = [
            {"text": "thank you thank you thank you", "start": 0.0, "end": 1.0},
            {"text": "normal text here", "start": 1.0, "end": 2.0},
        ]

        result = filter.filter_segments(segments)
        assert result == segments
        assert len(result) == 2

    def test_repetitive_words_detection(self):
        """Test detection of repetitive words."""
        config = FilterConfig(enabled=True, max_repeated_words=3)
        filter = HallucinationFilter(config)

        # Should be filtered (4 repetitions)
        segment_repeated = {
            "text": "thank you thank you thank you thank you",
            "start": 0.0,
            "end": 1.0,
        }
        assert filter.should_filter_segment(segment_repeated) is True

        # Should not be filtered (2 repetitions)
        segment_ok = {"text": "thank you thank you", "start": 0.0, "end": 1.0}
        assert filter.should_filter_segment(segment_ok) is False

    def test_repeated_phrases_detection(self):
        """Test detection of repeated phrases."""
        config = FilterConfig(enabled=True, max_repeated_phrases=2)
        filter = HallucinationFilter(config)

        # Should be filtered (phrase repeated 3 times)
        segment_repeated = {
            "text": "how are you how are you how are you",
            "start": 0.0,
            "end": 2.0,
        }
        assert filter.should_filter_segment(segment_repeated) is True

        # Should not be filtered (phrase appears once)
        segment_ok = {"text": "how are you today", "start": 0.0, "end": 1.0}
        assert filter.should_filter_segment(segment_ok) is False

    def test_empty_segment_filtering(self):
        """Test filtering of empty or whitespace-only segments."""
        config = FilterConfig(enabled=True)
        filter = HallucinationFilter(config)

        assert filter.should_filter_segment({"text": "", "start": 0.0, "end": 1.0}) is True
        assert filter.should_filter_segment({"text": "   ", "start": 0.0, "end": 1.0}) is True
        assert filter.should_filter_segment({"text": "\n\t", "start": 0.0, "end": 1.0}) is True

    def test_segment_length_filtering(self):
        """Test filtering based on segment length."""
        config = FilterConfig(
            enabled=True,
            min_segment_length=2,
            max_segment_length=5,
        )
        filter = HallucinationFilter(config)

        # Too short (1 word)
        segment_short = {"text": "hello", "start": 0.0, "end": 0.5}
        assert filter.should_filter_segment(segment_short) is True

        # Just right (2 words)
        segment_ok_min = {"text": "hello world", "start": 0.0, "end": 1.0}
        assert filter.should_filter_segment(segment_ok_min) is False

        # Just right (5 words)
        segment_ok_max = {
            "text": "this is a good sentence",
            "start": 0.0,
            "end": 2.0,
        }
        assert filter.should_filter_segment(segment_ok_max) is False

        # Too long (6 words)
        segment_long = {
            "text": "this is a very long sentence here",
            "start": 0.0,
            "end": 3.0,
        }
        assert filter.should_filter_segment(segment_long) is True

    def test_confidence_threshold_filtering(self):
        """Test filtering based on confidence thresholds."""
        config = FilterConfig(
            enabled=True,
            min_segment_confidence=0.8,
        )
        filter = HallucinationFilter(config)

        # Low confidence
        segment_low = {
            "text": "uncertain text",
            "confidence": 0.5,
            "start": 0.0,
            "end": 1.0,
        }
        assert filter.should_filter_segment(segment_low) is True

        # High confidence
        segment_high = {
            "text": "certain text",
            "confidence": 0.9,
            "start": 0.0,
            "end": 1.0,
        }
        assert filter.should_filter_segment(segment_high) is False

        # No confidence data (should not filter)
        segment_no_conf = {"text": "text without confidence", "start": 0.0, "end": 1.0}
        assert filter.should_filter_segment(segment_no_conf) is False

    def test_word_probability_filtering(self):
        """Test filtering based on word probability thresholds."""
        config = FilterConfig(
            enabled=True,
            min_word_probability=0.7,
        )
        filter = HallucinationFilter(config)

        # Low word probabilities
        segment_low = {
            "text": "uncertain words here",
            "words": [
                {"word": "uncertain", "probability": 0.5, "start": 0.0, "end": 0.3},
                {"word": "words", "probability": 0.6, "start": 0.3, "end": 0.6},
                {"word": "here", "probability": 0.5, "start": 0.6, "end": 0.9},
            ],
            "start": 0.0,
            "end": 1.0,
        }
        # Average: (0.5 + 0.6 + 0.5) / 3 = 0.533 < 0.7
        assert filter.should_filter_segment(segment_low) is True

        # High word probabilities
        segment_high = {
            "text": "certain words here",
            "words": [
                {"word": "certain", "probability": 0.9, "start": 0.0, "end": 0.3},
                {"word": "words", "probability": 0.85, "start": 0.3, "end": 0.6},
                {"word": "here", "probability": 0.8, "start": 0.6, "end": 0.9},
            ],
            "start": 0.0,
            "end": 1.0,
        }
        # Average: (0.9 + 0.85 + 0.8) / 3 = 0.85 > 0.7
        assert filter.should_filter_segment(segment_high) is False

    def test_filter_segments_list(self):
        """Test filtering a list of segments."""
        config = FilterConfig(enabled=True, max_repeated_words=3, min_segment_length=2)
        filter = HallucinationFilter(config)

        segments = [
            {"text": "good segment here", "start": 0.0, "end": 1.0},  # Keep
            {"text": "short", "start": 1.0, "end": 1.5},  # Filter (too short)
            {
                "text": "thank thank thank thank",
                "start": 1.5,
                "end": 2.5,
            },  # Filter (repetitive)
            {"text": "another good segment", "start": 2.5, "end": 3.5},  # Keep
            {"text": "", "start": 3.5, "end": 4.0},  # Filter (empty)
            {"text": "final good segment", "start": 4.0, "end": 5.0},  # Keep
        ]

        result = filter.filter_segments(segments)
        assert len(result) == 3
        assert result[0]["text"] == "good segment here"
        assert result[1]["text"] == "another good segment"
        assert result[2]["text"] == "final good segment"

    def test_filter_text_simple(self):
        """Test simple text filtering."""
        config = FilterConfig(enabled=True, max_repeated_words=3)
        filter = HallucinationFilter(config)

        # Repetitive text should be filtered
        result = filter.filter_text("thank thank thank thank")
        assert result == ""

        # Normal text should pass through
        result = filter.filter_text("this is normal text")
        assert result == "this is normal text"

    def test_case_insensitive_matching(self):
        """Test that repetition detection is case-insensitive."""
        config = FilterConfig(enabled=True, max_repeated_words=3)
        filter = HallucinationFilter(config)

        segment = {
            "text": "Thank THANK thank Thank",
            "start": 0.0,
            "end": 1.0,
        }
        assert filter.should_filter_segment(segment) is True

    def test_custom_regex_patterns(self):
        """Test custom regex patterns."""
        config = FilterConfig(
            enabled=True,
            custom_patterns=[
                r"(?i)^music$",  # Match "music" (case-insensitive)
                r"^\d+$",  # Match numbers only
            ],
        )
        filter = HallucinationFilter(config)

        # Should match custom patterns
        assert filter.should_filter_segment({"text": "music", "start": 0.0, "end": 1.0}) is True
        assert filter.should_filter_segment({"text": "MUSIC", "start": 0.0, "end": 1.0}) is True
        assert filter.should_filter_segment({"text": "12345", "start": 0.0, "end": 1.0}) is True

        # Should not match
        assert (
            filter.should_filter_segment({"text": "background music", "start": 0.0, "end": 1.0})
            is False
        )
        assert filter.should_filter_segment({"text": "test", "start": 0.0, "end": 1.0}) is False

    def test_multiple_filters_combined(self):
        """Test that multiple filter conditions work together."""
        config = FilterConfig(
            enabled=True,
            max_repeated_words=3,
            min_segment_length=2,
            min_segment_confidence=0.8,
        )
        filter = HallucinationFilter(config)

        segments = [
            # Good segment
            {
                "text": "good quality segment",
                "confidence": 0.9,
                "start": 0.0,
                "end": 1.0,
            },
            # Filter: too short
            {"text": "short", "confidence": 0.9, "start": 1.0, "end": 1.5},
            # Filter: low confidence
            {
                "text": "uncertain long segment",
                "confidence": 0.5,
                "start": 1.5,
                "end": 2.5,
            },
            # Filter: repetitive
            {
                "text": "word word word word",
                "confidence": 0.9,
                "start": 2.5,
                "end": 3.5,
            },
            # Good segment
            {
                "text": "another good segment",
                "confidence": 0.95,
                "start": 3.5,
                "end": 4.5,
            },
        ]

        result = filter.filter_segments(segments)
        assert len(result) == 2
        assert result[0]["text"] == "good quality segment"
        assert result[1]["text"] == "another good segment"

    def test_disabled_filter_no_processing(self):
        """Test that disabled filter does no processing."""
        config = FilterConfig(enabled=False)
        filter = HallucinationFilter(config)

        # Even with obvious hallucinations, nothing should be filtered
        segments = [
            {"text": "thank thank thank thank thank", "start": 0.0, "end": 1.0},
            {"text": "", "start": 1.0, "end": 1.5},
            {"text": "x", "start": 1.5, "end": 2.0},
        ]

        result = filter.filter_segments(segments)
        assert result == segments
        assert len(result) == 3


class TestCreateFilterFromConfig:
    """Tests for filter factory that reads application settings."""

    def test_create_filter_from_config_uses_app_settings(self):
        """Test that filter is created using application settings."""
        from src import config as cfg
        from src.utils.asr_filters import create_filter_from_config

        # Ensure settings are what we expect
        assert cfg.settings.asr.hallucination.enabled is True
        assert cfg.settings.asr.hallucination.max_repeated_words == 3

        filter = create_filter_from_config()

        assert isinstance(filter, HallucinationFilter)
        assert filter.config.enabled is cfg.settings.asr.hallucination.enabled
        assert (
            filter.config.max_repeated_words
            == cfg.settings.asr.hallucination.max_repeated_words
        )
        assert (
            filter.config.max_repeated_phrases
            == cfg.settings.asr.hallucination.max_repeated_phrases
        )


class TestFilterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_segments_list(self):
        """Test filtering empty segments list."""
        config = FilterConfig(enabled=True)
        filter = HallucinationFilter(config)

        result = filter.filter_segments([])
        assert result == []

    def test_none_segments_list(self):
        """Test filtering None segments list (when disabled, returns early)."""
        config = FilterConfig(enabled=False)
        filter = HallucinationFilter(config)

        # When disabled, empty list is returned unchanged
        result = filter.filter_segments([])
        assert result == []

    def test_segment_without_text_key(self):
        """Test segment without 'text' key defaults to empty string."""
        config = FilterConfig(enabled=True)
        filter = HallucinationFilter(config)

        segment = {"start": 0.0, "end": 1.0}  # Missing 'text' key
        # Should filter empty text
        assert filter.should_filter_segment(segment) is True

    def test_malformed_segment(self):
        """Test handling of malformed segment data."""
        config = FilterConfig(enabled=True, min_word_probability=0.7)
        filter = HallucinationFilter(config)

        # Segment with malformed words list
        segment = {
            "text": "test text",
            "words": [
                {"word": "test"},  # Missing probability
                {"word": "text", "probability": 0.8},
            ],
            "start": 0.0,
            "end": 1.0,
        }

        # Should not crash, should handle gracefully
        # Word without probability should default to 1.0
        result = filter.should_filter_segment(segment)
        # Average: (1.0 + 0.8) / 2 = 0.9 > 0.7
        assert result is False

    def test_invalid_regex_pattern_in_custom(self):
        """Test that invalid regex patterns are handled gracefully."""
        config = FilterConfig(
            enabled=True,
            custom_patterns=[
                r"(?i)^valid$",  # Valid pattern
                r"[invalid(",  # Invalid regex
            ],
        )

        # Should not crash during initialization
        filter = HallucinationFilter(config)

        # Should work with valid pattern, ignore invalid one
        assert filter.should_filter_segment({"text": "valid", "start": 0.0, "end": 1.0}) is True
        assert filter.should_filter_segment({"text": "other", "start": 0.0, "end": 1.0}) is False

    def test_unicode_text_handling(self):
        """Test proper handling of Unicode text."""
        config = FilterConfig(enabled=True, max_repeated_words=3)
        filter = HallucinationFilter(config)

        # Vietnamese text with repetition
        segment_vi = {
            "text": "cảm ơn cảm ơn cảm ơn cảm ơn",
            "start": 0.0,
            "end": 1.0,
        }
        assert filter.should_filter_segment(segment_vi) is True

        # Japanese text
        segment_ja = {
            "text": "ありがとう ありがとう ありがとう ありがとう",
            "start": 0.0,
            "end": 1.0,
        }
        assert filter.should_filter_segment(segment_ja) is True

        # Chinese text
        segment_zh = {
            "text": "谢谢 谢谢 谢谢 谢谢",
            "start": 0.0,
            "end": 1.0,
        }
        assert filter.should_filter_segment(segment_zh) is True

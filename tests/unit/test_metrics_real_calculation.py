# tests/unit/test_metrics_real_calculation.py
"""
TDD-driven tests for metrics calculation with real data.

Verifies:
1. Audio duration flows correctly from AudioPreprocessor
2. RTF (Real-Time Factor) calculation uses actual timing data
3. Edit rate calculation uses actual Levenshtein distance
4. Metrics include all required fields per TDD FR-5
"""

import time

from src.worker.pipeline import _calculate_edit_rate, calculate_metrics


class TestEditRateCalculation:
    """Test the Levenshtein distance-based edit rate calculation."""

    def test_identical_strings_return_zero(self):
        """Identical strings should have 0.0 edit rate."""
        result = _calculate_edit_rate("hello world", "hello world")
        assert result == 0.0

    def test_completely_different_strings(self):
        """Completely different strings should have 1.0 edit rate."""
        result = _calculate_edit_rate("abc", "xyz")
        assert result == 1.0

    def test_single_character_change(self):
        """Single character change should have proportional edit rate."""
        result = _calculate_edit_rate("hello", "helo")  # 1 deletion in 5 chars
        assert 0.15 < result < 0.25  # ~20% change

    def test_empty_string_handling(self):
        """Empty strings should be handled correctly."""
        assert _calculate_edit_rate("", "") == 0.0
        assert _calculate_edit_rate("hello", "") == 1.0
        assert _calculate_edit_rate("", "hello") == 1.0

    def test_case_sensitive(self):
        """Edit rate should be case-sensitive."""
        result = _calculate_edit_rate("Hello", "hello")
        assert result > 0.0  # Case change counts as edit

    def test_real_transcription_cleaning(self):
        """Test with realistic transcription cleaning scenario."""
        original = "um... like, you know, the meeting was, uh, productive"
        cleaned = "The meeting was productive"
        result = _calculate_edit_rate(original, cleaned)
        # Should show significant change but not 1.0
        assert 0.3 < result < 0.8

    def test_minor_punctuation_changes(self):
        """Test with minor punctuation corrections."""
        original = "hello world"
        cleaned = "Hello world."
        result = _calculate_edit_rate(original, cleaned)
        # Should show small change
        assert 0.0 < result < 0.3


class TestCalculateMetrics:
    """Test the metrics calculation function with real timing data."""

    def test_basic_metrics_structure(self):
        """Test that all required metrics fields are present."""
        start_time = time.time() - 5.0  # Simulate 5 seconds of processing

        metrics = calculate_metrics(
            transcription="Hello world",
            clean_transcript=None,
            start_time=start_time,
            audio_duration=10.0,
            asr_rtf=0.3,
        )

        # Verify required fields per TDD FR-5
        assert "processing_time_seconds" in metrics
        assert "rtf" in metrics
        assert "asr_rtf" in metrics
        assert "transcription_length" in metrics
        assert "audio_duration" in metrics

    def test_total_processing_time_is_accurate(self):
        """Test that total_processing_time reflects actual elapsed time."""
        start_time = time.time()
        time.sleep(0.1)  # Simulate some processing

        metrics = calculate_metrics(
            transcription="Test",
            clean_transcript=None,
            start_time=start_time,
            audio_duration=1.0,
            asr_rtf=0.5,
        )

        # Should be approximately 0.1 seconds (with small tolerance)
        assert 0.08 < metrics["processing_time_seconds"] < 0.15

    def test_rtf_calculation_with_10_second_audio(self):
        """Test RTF calculation: processing_time / audio_duration."""
        start_time = time.time() - 2.0  # 2 seconds processing time

        metrics = calculate_metrics(
            transcription="Sample transcription",
            clean_transcript=None,
            start_time=start_time,
            audio_duration=10.0,  # 10 second audio
            asr_rtf=0.15,
        )

        # RTF = 2.0 / 10.0 = 0.2
        expected_rtf = 2.0 / 10.0
        assert abs(metrics["rtf"] - expected_rtf) < 0.05

    def test_rtf_calculation_with_60_second_audio(self):
        """Test RTF with longer audio file."""
        start_time = time.time() - 5.0  # 5 seconds processing

        metrics = calculate_metrics(
            transcription="Longer transcription",
            clean_transcript=None,
            start_time=start_time,
            audio_duration=60.0,  # 1 minute audio
            asr_rtf=0.08,
        )

        # RTF = 5.0 / 60.0 = 0.083
        expected_rtf = 5.0 / 60.0
        assert abs(metrics["rtf"] - expected_rtf) < 0.01

    def test_rtf_handles_zero_audio_duration(self):
        """Test that zero audio duration doesn't cause division by zero."""
        start_time = time.time() - 1.0

        metrics = calculate_metrics(
            transcription="Test",
            clean_transcript=None,
            start_time=start_time,
            audio_duration=0.0,  # Edge case
            asr_rtf=0.0,
        )

        assert metrics["rtf"] == 0

    def test_transcription_length_calculated(self):
        """Test that transcription length is correctly calculated."""
        metrics = calculate_metrics(
            transcription="Hello world, this is a test transcription.",
            clean_transcript=None,
            start_time=time.time(),
            audio_duration=10.0,
            asr_rtf=0.2,
        )

        assert metrics["transcription_length"] == len(
            "Hello world, this is a test transcription."
        )

    def test_audio_duration_preserved(self):
        """Test that audio_duration is preserved in metrics."""
        audio_duration = 42.5

        metrics = calculate_metrics(
            transcription="Test",
            clean_transcript=None,
            start_time=time.time(),
            audio_duration=audio_duration,
            asr_rtf=0.3,
        )

        assert metrics["audio_duration"] == audio_duration

    def test_asr_rtf_preserved(self):
        """Test that ASR RTF is preserved in metrics."""
        asr_rtf = 0.234

        metrics = calculate_metrics(
            transcription="Test",
            clean_transcript=None,
            start_time=time.time(),
            audio_duration=10.0,
            asr_rtf=asr_rtf,
        )

        assert metrics["asr_rtf"] == asr_rtf


class TestMetricsWithEnhancement:
    """Test metrics when text enhancement is performed."""

    def test_edit_rate_included_when_enhanced(self):
        """Test that edit_rate_cleaning is included when transcription is enhanced."""
        metrics = calculate_metrics(
            transcription="um hello world like you know",
            clean_transcript="Hello world.",
            start_time=time.time(),
            audio_duration=10.0,
            asr_rtf=0.2,
        )

        assert "edit_rate_cleaning" in metrics
        assert 0.0 < metrics["edit_rate_cleaning"] < 1.0

    def test_edit_rate_not_included_when_no_enhancement(self):
        """Test that edit_rate_cleaning is NOT included when no enhancement."""
        metrics = calculate_metrics(
            transcription="Hello world",
            clean_transcript=None,  # No enhancement
            start_time=time.time(),
            audio_duration=10.0,
            asr_rtf=0.2,
        )

        assert "edit_rate_cleaning" not in metrics

    def test_edit_rate_not_included_when_unchanged(self):
        """Test that edit_rate_cleaning is NOT included when transcript unchanged."""
        transcription = "Hello world"

        metrics = calculate_metrics(
            transcription=transcription,
            clean_transcript=transcription,  # Same as original
            start_time=time.time(),
            audio_duration=10.0,
            asr_rtf=0.2,
        )

        assert "edit_rate_cleaning" not in metrics

    def test_edit_rate_uses_real_calculation(self):
        """Test that edit_rate uses actual Levenshtein distance, not length diff."""
        # These have same length but different content
        original = "hello world"
        enhanced = "world hello"  # Reordered

        metrics = calculate_metrics(
            transcription=original,
            clean_transcript=enhanced,
            start_time=time.time(),
            audio_duration=10.0,
            asr_rtf=0.2,
        )

        # Calculate expected edit rate using the actual function
        expected_edit_rate = _calculate_edit_rate(original, enhanced)

        assert "edit_rate_cleaning" in metrics
        assert abs(metrics["edit_rate_cleaning"] - expected_edit_rate) < 0.01

    def test_realistic_cleaning_scenario(self):
        """Test with realistic transcription cleaning scenario."""
        original = "um... so like, the meeting, you know, it was really, uh, productive and stuff"
        cleaned = "The meeting was really productive."

        metrics = calculate_metrics(
            transcription=original,
            clean_transcript=cleaned,
            start_time=time.time() - 3.5,
            audio_duration=12.0,
            asr_rtf=0.25,
        )

        # Verify all metrics are present
        assert "edit_rate_cleaning" in metrics
        assert "processing_time_seconds" in metrics
        assert "rtf" in metrics

        # Edit rate should show significant change
        assert 0.3 < metrics["edit_rate_cleaning"] < 0.8

        # RTF should be calculated correctly
        assert abs(metrics["rtf"] - (3.5 / 12.0)) < 0.05


class TestAudioDurationFlow:
    """Test that audio_duration flows correctly from AudioPreprocessor to metrics."""

    def test_audio_duration_from_preprocessing_metadata(self):
        """
        Test that audio_duration from AudioPreprocessor metadata
        flows correctly into calculate_metrics.

        This simulates the flow:
        1. AudioPreprocessor.preprocess() returns {"duration": X}
        2. extract_audio_duration() gets X
        3. calculate_metrics() receives X
        4. metrics["audio_duration"] == X
        """
        # Simulate AudioPreprocessor metadata
        preprocessing_metadata = {
            "duration": 42.5,
            "sample_rate": 16000,
            "channels": 1,
        }

        # Extract duration (as done in pipeline)
        audio_duration = preprocessing_metadata.get("duration", 10.0)

        # Calculate metrics with that duration
        metrics = calculate_metrics(
            transcription="Test transcription",
            clean_transcript=None,
            start_time=time.time() - 5.0,
            audio_duration=audio_duration,
            asr_rtf=0.15,
        )

        # Verify audio_duration flowed through correctly
        assert metrics["audio_duration"] == 42.5

        # Verify RTF calculation used the correct duration
        expected_rtf = 5.0 / 42.5
        assert abs(metrics["rtf"] - expected_rtf) < 0.01

    def test_audio_duration_fallback_when_missing(self):
        """Test fallback when duration is missing from preprocessing metadata."""
        preprocessing_metadata = {
            "sample_rate": 16000,
            # duration key missing
        }

        # Fallback to default (as done in pipeline)
        audio_duration = preprocessing_metadata.get("duration", 10.0)

        metrics = calculate_metrics(
            transcription="Test",
            clean_transcript=None,
            start_time=time.time(),
            audio_duration=audio_duration,
            asr_rtf=0.2,
        )

        # Should use fallback value
        assert metrics["audio_duration"] == 10.0

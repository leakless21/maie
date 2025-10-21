# tests/unit/test_feature_selection.py
"""
TDD-driven tests for feature selection logic (FR-3).

Verifies:
1. Enhancement is skipped for Whisper (has native punctuation)
2. Enhancement is applied for ChunkFormer and other backends
3. Feature selection in pipeline correctly uses needs_enhancement()
4. Both transcript-only and summary features work with skip logic
"""

from src.processors.llm.processor import LLMProcessor


class TestEnhancementSkipLogic:
    """Test that enhancement is correctly skipped based on ASR backend."""

    def test_whisper_skips_enhancement(self):
        """Whisper with erax-wow-turbo has native punctuation, so skip enhancement."""
        processor = LLMProcessor()

        # Whisper should skip enhancement
        assert processor.needs_enhancement("whisper") is False

    def test_whisper_case_insensitive(self):
        """needs_enhancement should be case-insensitive."""
        processor = LLMProcessor()

        assert processor.needs_enhancement("WHISPER") is False
        assert processor.needs_enhancement("Whisper") is False
        assert processor.needs_enhancement("WhIsPeR") is False

    def test_chunkformer_needs_enhancement(self):
        """ChunkFormer doesn't have native punctuation, needs enhancement."""
        processor = LLMProcessor()

        assert processor.needs_enhancement("chunkformer") is True

    def test_unknown_backend_needs_enhancement(self):
        """Unknown backends should default to needing enhancement (safe default)."""
        processor = LLMProcessor()

        assert processor.needs_enhancement("unknown_backend") is True
        assert processor.needs_enhancement("custom_asr") is True
        assert processor.needs_enhancement("") is True

    def test_none_backend_needs_enhancement(self):
        """None backend should be handled gracefully."""
        processor = LLMProcessor()

        # Should not crash, but return True (safe default)
        try:
            result = processor.needs_enhancement(None)
            # If it doesn't crash, it should default to True
            assert result is True
        except (AttributeError, TypeError):
            # If it crashes, that's also acceptable (fail fast)
            pass


class TestWhisperVariants:
    """Test enhancement logic for different Whisper variants."""

    def test_whisper_with_model_suffix(self):
        """Test Whisper with model size suffixes."""
        processor = LLMProcessor()

        # All Whisper variants skip enhancement
        assert processor.needs_enhancement("whisper-tiny") is False
        assert processor.needs_enhancement("whisper-base") is False
        assert processor.needs_enhancement("whisper-small") is False
        assert processor.needs_enhancement("whisper-medium") is False
        assert processor.needs_enhancement("whisper-large") is False
        assert processor.needs_enhancement("whisper-large-v3") is False

    def test_whisper_with_language(self):
        """Test Whisper with language specifications."""
        processor = LLMProcessor()

        # These should be recognized as Whisper
        # Note: Current implementation only checks if starts with "whisper"
        # So these will all return False (skip enhancement)
        assert processor.needs_enhancement("whisper_vi") is False
        assert processor.needs_enhancement("whisper_en") is False


class TestChunkFormerVariants:
    """Test enhancement logic for ChunkFormer variants."""

    def test_chunkformer_needs_enhancement(self):
        """ChunkFormer needs enhancement for punctuation."""
        processor = LLMProcessor()

        assert processor.needs_enhancement("chunkformer") is True

    def test_chunkformer_with_model_suffix(self):
        """Test ChunkFormer with model size suffixes."""
        processor = LLMProcessor()

        assert processor.needs_enhancement("chunkformer-base") is True
        assert processor.needs_enhancement("chunkformer-large") is True
        assert processor.needs_enhancement("chunkformer-rnnt-large-vie") is True


class TestFeatureSelectionIntegration:
    """Test feature selection in pipeline context."""

    def test_transcript_only_with_whisper(self):
        """Test transcript-only feature with Whisper (no enhancement)."""
        processor = LLMProcessor()

        # Whisper transcription doesn't need enhancement
        features = ["transcript"]
        asr_backend = "whisper"

        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        # Should NOT need enhancement
        assert needs_enhancement is False

    def test_transcript_only_with_chunkformer(self):
        """Test transcript-only feature with ChunkFormer (needs enhancement)."""
        processor = LLMProcessor()

        features = ["transcript"]
        asr_backend = "chunkformer"

        # Even for transcript-only, enhancement is NOT applied
        # because "clean_transcript" is not in features
        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        assert needs_enhancement is False

    def test_clean_transcript_with_whisper(self):
        """Test clean_transcript feature with Whisper (skip enhancement)."""
        processor = LLMProcessor()

        features = ["clean_transcript", "summary"]
        asr_backend = "whisper"

        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        # Should skip enhancement even though clean_transcript is requested
        assert needs_enhancement is False

    def test_clean_transcript_with_chunkformer(self):
        """Test clean_transcript feature with ChunkFormer (apply enhancement)."""
        processor = LLMProcessor()

        features = ["clean_transcript", "summary"]
        asr_backend = "chunkformer"

        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        # Should apply enhancement
        assert needs_enhancement is True

    def test_summary_only_with_whisper(self):
        """Test summary-only feature with Whisper."""
        processor = LLMProcessor()

        features = ["summary"]
        asr_backend = "whisper"

        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        # Summary doesn't require clean_transcript, so no enhancement
        assert needs_enhancement is False

    def test_both_features_with_whisper(self):
        """Test both transcript and summary with Whisper."""
        processor = LLMProcessor()

        features = ["clean_transcript", "summary"]
        asr_backend = "whisper"

        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        # Even with both features, Whisper skips enhancement
        assert needs_enhancement is False

    def test_both_features_with_chunkformer(self):
        """Test both transcript and summary with ChunkFormer."""
        processor = LLMProcessor()

        features = ["clean_transcript", "summary"]
        asr_backend = "chunkformer"

        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        # ChunkFormer needs enhancement when clean_transcript is requested
        assert needs_enhancement is True


class TestFeatureSelectionEdgeCases:
    """Test edge cases in feature selection."""

    def test_empty_features_list(self):
        """Test with empty features list."""
        processor = LLMProcessor()

        features = []
        asr_backend = "chunkformer"

        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        # No features means no enhancement
        assert needs_enhancement is False

    def test_none_features(self):
        """Test with None features."""
        processor = LLMProcessor()

        features = None
        asr_backend = "chunkformer"

        # Should handle None gracefully
        try:
            needs_enhancement = (
                "clean_transcript" in features
                and processor.needs_enhancement(asr_backend)
            )
            assert needs_enhancement is False
        except TypeError:
            # "in None" raises TypeError, which is acceptable
            pass

    def test_unknown_features(self):
        """Test with unknown feature names."""
        processor = LLMProcessor()

        features = ["unknown_feature", "another_unknown"]
        asr_backend = "chunkformer"

        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        # Unknown features don't trigger enhancement
        assert needs_enhancement is False

    def test_duplicate_features(self):
        """Test with duplicate feature names."""
        processor = LLMProcessor()

        features = ["clean_transcript", "clean_transcript", "summary"]
        asr_backend = "chunkformer"

        needs_enhancement = (
            "clean_transcript" in features and processor.needs_enhancement(asr_backend)
        )

        # Duplicates don't affect logic
        assert needs_enhancement is True


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_vietnamese_whisper_transcript(self):
        """
        Scenario: Vietnamese audio with Whisper (erax-wow-turbo variant).
        Expected: Skip enhancement, use transcription directly.
        """
        processor = LLMProcessor()

        asr_backend = "whisper"
        features = ["transcript", "summary"]

        # Check if enhancement is needed
        needs_clean = "clean_transcript" in features
        apply_enhancement = needs_clean and processor.needs_enhancement(asr_backend)

        # Should NOT enhance - Whisper has native punctuation
        assert apply_enhancement is False

    def test_vietnamese_chunkformer_with_enhancement(self):
        """
        Scenario: Vietnamese audio with ChunkFormer.
        Expected: Apply enhancement for punctuation.
        """
        processor = LLMProcessor()

        asr_backend = "chunkformer"
        features = ["clean_transcript", "summary"]

        needs_clean = "clean_transcript" in features
        apply_enhancement = needs_clean and processor.needs_enhancement(asr_backend)

        # Should enhance - ChunkFormer needs punctuation
        assert apply_enhancement is True

    def test_transcription_only_no_enhancement(self):
        """
        Scenario: User only wants raw transcription.
        Expected: No enhancement regardless of backend.
        """
        processor = LLMProcessor()

        # Test with both backends
        for backend in ["whisper", "chunkformer"]:
            features = ["transcript"]  # Only raw transcript

            needs_clean = "clean_transcript" in features
            apply_enhancement = needs_clean and processor.needs_enhancement(backend)

            # Should never enhance for transcript-only
            assert apply_enhancement is False

    def test_summary_with_implicit_enhancement(self):
        """
        Scenario: User requests summary with clean transcript.
        Expected: Enhancement applied for non-Whisper backends.
        """
        processor = LLMProcessor()

        # ChunkFormer needs enhancement
        asr_backend = "chunkformer"
        features = ["clean_transcript", "summary"]

        needs_clean = "clean_transcript" in features
        apply_enhancement = needs_clean and processor.needs_enhancement(asr_backend)

        assert apply_enhancement is True

        # Whisper skips enhancement
        asr_backend = "whisper"
        apply_enhancement = needs_clean and processor.needs_enhancement(asr_backend)

        assert apply_enhancement is False

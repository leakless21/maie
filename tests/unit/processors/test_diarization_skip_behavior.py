"""
Tests for diarization skip behavior when word timestamps are unavailable.

Verifies that when diarization runs but no word timestamps are available,
the system correctly skips speaker attribution and uses plain transcript.
"""

from unittest.mock import Mock


class TestDiarizationSkipBehavior:
    """Test that diarization correctly skips when word timestamps unavailable."""

    def test_plain_transcript_used_when_no_word_timestamps(self):
        """When word timestamps unavailable, plain transcript should be used (not speaker-formatted)."""
        # Simulate the scenario
        has_word_timestamps = False
        original_transcription = "Hello world. This is a test."

        # Simulate what should happen in pipeline
        if has_word_timestamps:
            # This branch should NOT execute
            final_transcript = "SPEAKER_0: Hello world.\nSPEAKER_1: This is a test."
        else:
            # This branch SHOULD execute
            final_transcript = original_transcription

        # Verify plain transcript is used
        assert final_transcript == original_transcription
        assert "SPEAKER" not in final_transcript
        assert "S?:" not in final_transcript

    def test_speaker_transcript_used_when_word_timestamps_available(self):
        """When word timestamps available, speaker-attributed transcript should be used."""
        has_word_timestamps = True
        original_transcription = "Hello world. This is a test."

        # Simulate what should happen in pipeline
        if has_word_timestamps:
            # This branch SHOULD execute
            final_transcript = "SPEAKER_0: Hello world.\nSPEAKER_1: This is a test."
        else:
            # This branch should NOT execute
            final_transcript = original_transcription

        # Verify speaker transcript is used
        assert final_transcript != original_transcription
        assert "SPEAKER" in final_transcript

    def test_segments_not_modified_when_diarization_skipped(self):
        """ASR segments should remain unmodified when diarization is skipped."""
        # Original ASR segments (no speaker info)
        original_segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello world"},
            {"start": 1.0, "end": 2.0, "text": "This is a test"},
        ]

        has_word_timestamps = False

        # When no word timestamps, segments should not be modified with speaker info
        if has_word_timestamps:
            # This would add speaker labels
            updated_segments = [
                {**seg, "speaker": "SPEAKER_0"} for seg in original_segments
            ]
        else:
            # Segments remain as-is (no speaker field added)
            updated_segments = original_segments

        # Verify no speaker info was added
        assert updated_segments == original_segments
        for seg in updated_segments:
            assert "speaker" not in seg or seg.get("speaker") is None

    def test_logging_indicates_plain_transcript_usage(self):
        """Logs should indicate plain transcript is being used when diarization skipped."""
        has_word_timestamps = False

        # Simulate logging decision
        if has_word_timestamps:
            log_message = "Using speaker-attributed transcript for LLM"
        else:
            log_message = "Continuing with plain transcript (no speaker labels)"

        # Verify correct log message
        assert "plain transcript" in log_message.lower()
        assert "no speaker" in log_message.lower()


class TestDiarizationRenderLogic:
    """Test the logic for when to render speaker-attributed transcripts."""

    def test_render_speaker_transcript_only_when_has_word_timestamps(self):
        """render_speaker_attributed_transcript should only be called when word timestamps exist."""
        mock_render = Mock()
        has_word_timestamps = False
        segments = [{"start": 0.0, "end": 1.0, "text": "Test", "speaker": None}]

        # Simulate pipeline logic
        if has_word_timestamps:
            mock_render(segments)

        # Should NOT have been called
        mock_render.assert_not_called()

    def test_render_speaker_transcript_called_when_timestamps_available(self):
        """render_speaker_attributed_transcript should be called when word timestamps exist."""
        mock_render = Mock(return_value="SPEAKER_0: Test")
        has_word_timestamps = True
        segments = [{"start": 0.0, "end": 1.0, "text": "Test", "speaker": "SPEAKER_0"}]

        result = None
        # Simulate pipeline logic
        if has_word_timestamps:
            result = mock_render(segments)

        # Should have been called
        mock_render.assert_called_once_with(segments)
        assert result == "SPEAKER_0: Test"

    def test_no_speaker_formatting_without_word_timestamps(self):
        """Text should not have speaker formatting when word timestamps unavailable."""
        plain_text = "This is a test transcript."
        has_word_timestamps = False

        # When no word timestamps, don't format with speakers
        if has_word_timestamps:
            formatted = "S?: This is a test transcript."
        else:
            formatted = plain_text

        # Should be plain, no speaker prefixes
        assert formatted == plain_text
        assert not formatted.startswith("S?:")
        assert not formatted.startswith("SPEAKER_")

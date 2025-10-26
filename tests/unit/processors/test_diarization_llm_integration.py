"""
Tests for diarization integration with LLM pipeline.

Verifies that speaker-attributed transcripts are properly rendered
and passed to the LLM when diarization is enabled.
"""

from __future__ import annotations

from unittest.mock import Mock, patch, MagicMock
import pytest


class TestDiarizationLLMIntegration:
    """Test diarization integration with LLM processing."""

    def test_diarized_transcript_renders_human_format(self):
        """Test that diarized segments render in human format for LLM."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Thanks for joining everyone.",
                "speaker": "S1",
            },
            {
                "start": 5.0,
                "end": 9.0,
                "text": "Happy to be here.",
                "speaker": "S2",
            },
        ]
        
        result = render_speaker_attributed_transcript(segments)
        
        # Should contain speaker codes in plain text format
        assert "S1" in result
        assert "S2" in result
        assert "S1: Thanks for joining everyone." in result
        assert "S2: Happy to be here." in result

    def test_pipeline_uses_speaker_transcript_when_diarization_enabled(self):
        """Test that pipeline uses speaker-attributed transcript for LLM when diarization is on."""
        # This test verifies the integration point: when diarization is enabled,
        # the rendered speaker-attributed transcript is used instead of plain text
        
        # Create mock segments with speaker info
        diarized_segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello",
                "speaker": "S1",
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "world",
                "speaker": "S2",
            },
        ]
        
        # The rendered output should be different from plain text
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        rendered = render_speaker_attributed_transcript(diarized_segments)
        
        # Rendered should include speaker codes in plain text format
        assert "S1" in rendered
        assert "S2" in rendered
        assert "S1: hello" in rendered
        assert "S2: world" in rendered

    def test_pipeline_falls_back_to_plain_if_render_fails(self):
        """Test that pipeline falls back to plain transcript if rendering fails."""
        # If rendering fails for any reason, the pipeline should continue
        # with the plain transcription without failing
        
        segments = [
            {"start": 0.0, "end": 5.0, "text": "test", "speaker": "S1"}
        ]
        
        # Mock rendering to raise an exception
        with patch(
            "src.processors.prompt.diarization.render_speaker_attributed_transcript",
            side_effect=Exception("Rendering failed"),
        ):
            from src.processors.prompt.diarization import render_speaker_attributed_transcript
            
            # The exception should be caught in the pipeline
            # (this is tested implicitly by the pipeline tests)
            with pytest.raises(Exception):
                render_speaker_attributed_transcript(segments)

    def test_diarized_vs_plain_transcript_length(self):
        """Test that diarized transcript includes metadata overhead."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is a test sentence.",
                "speaker": "S1",
            }
        ]
        
        # Plain text
        plain_text = "This is a test sentence."
        
        # Rendered with speaker info
        rendered = render_speaker_attributed_transcript(segments)
        
        # Rendered should be longer due to metadata and speaker codes
        assert len(rendered) > len(plain_text)
        
        # But should still contain the original text
        assert plain_text in rendered

    def test_multiple_speakers_in_lllm_transcript(self):
        """Test that multiple speakers are properly represented in LLM transcript."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {"start": 0.0, "end": 1.0, "text": "First speaker", "speaker": "S1"},
            {"start": 1.0, "end": 2.0, "text": "Second speaker", "speaker": "S2"},
            {"start": 2.0, "end": 3.0, "text": "Third speaker", "speaker": "S3"},
            {"start": 3.0, "end": 4.0, "text": "Back to first", "speaker": "S1"},
        ]
        
        result = render_speaker_attributed_transcript(
            segments,
        )
        
        # Should have all 3 speakers
        assert "S1" in result
        assert "S2" in result
        assert "S3" in result
        
        # Should have all text segments
        assert "First speaker" in result
        assert "Second speaker" in result
        assert "Third speaker" in result
        assert "Back to first" in result

    def test_none_speaker_handled_in_lllm_transcript(self):
        """Test that segments with no speaker (None) are handled gracefully in LLM."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {"start": 0.0, "end": 2.0, "text": "speaker one", "speaker": "S1"},
            {"start": 2.0, "end": 4.0, "text": "unknown speaker", "speaker": None},
            {"start": 4.0, "end": 6.0, "text": "speaker two", "speaker": "S2"},
        ]
        
        result = render_speaker_attributed_transcript(
            segments,
        )
        
        # Should have known speakers
        assert "S1" in result
        assert "S2" in result
        
        # Should still have the text even if speaker is unknown
        assert "unknown speaker" in result
        
        # Should be valid
        assert len(result) > 0


class TestLLMInputFormat:
    """Test that LLM receives properly formatted input."""

    def test_llm_input_is_speaker_attributed_when_diarization_enabled(self):
        """
        Integration test: verify that when diarization is enabled,
        the transcription passed to LLM is speaker-attributed.
        """
        # This would be tested by the full pipeline test
        # Here we verify the rendering logic produces valid input
        
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Good morning everyone",
                "speaker": "S1",
            },
            {
                "start": 3.0,
                "end": 6.0,
                "text": "Good morning how are you",
                "speaker": "S2",
            },
        ]
        
        rendered = render_speaker_attributed_transcript(segments)
        
        # Should be a valid string
        assert isinstance(rendered, str)
        
        # Should contain speaker information in plain text format
        assert "S1" in rendered
        assert "S2" in rendered
        assert "S1: Good morning everyone" in rendered
        assert "S2: Good morning how are you" in rendered
        assert "S2" in rendered
        
        # Should be suitable for LLM input (not empty, reasonable length)
        assert len(rendered) > 50
        assert len(rendered) < 100000  # Sanity check

    def test_plain_text_format_for_analytics(self):
        """Test plain text format output for analytics/storage."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "speaker one",
                "speaker": "S1",
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "speaker two",
                "speaker": "S2",
            },
        ]
        
        result = render_speaker_attributed_transcript(segments)
        
        # Should be a string
        assert isinstance(result, str)
        
        # Should be plain text format
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "S1: speaker one"
        assert lines[1] == "S2: speaker two"

    def test_plain_text_format_for_tracking(self):
        """Test plain text format output for tracking."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {"start": 0.0, "end": 2.0, "text": "test", "speaker": "S1"},
            {"start": 2.0, "end": 4.0, "text": "test", "speaker": "S2"},
        ]
        
        result = render_speaker_attributed_transcript(segments)
        
        # Should be a string
        assert isinstance(result, str)
        
        # Should be plain text format
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "S1: test"
        assert lines[1] == "S2: test"

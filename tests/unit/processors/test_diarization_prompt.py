"""Unit tests for diarization prompt packaging."""

from __future__ import annotations

import pytest

from src.processors.prompt.diarization import (
    DiarizedSegment,
    render_speaker_attributed_transcript,
)


class TestRenderSpeakerAttributedTranscript:
    """Tests for the simplified rendering function."""

    def test_render_with_dict_segments(self):
        """Test rendering with dict segments."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "S1"},
            {"start": 5.0, "end": 10.0, "text": "World", "speaker": "S2"},
        ]

        output = render_speaker_attributed_transcript(segments)

        assert isinstance(output, str)
        assert "S1: Hello" in output
        assert "S2: World" in output

    def test_render_with_segment_objects(self):
        """Test rendering with DiarizedSegment objects."""
        segments = [
            DiarizedSegment(start=0.0, end=5.0, text="Hello", speaker="S1"),
            DiarizedSegment(start=5.0, end=10.0, text="World", speaker="S2"),
        ]

        output = render_speaker_attributed_transcript(segments)

        assert isinstance(output, str)
        assert "S1: Hello" in output
        assert "S2: World" in output

    def test_render_with_unknown_speaker(self):
        """Test rendering with None speaker."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Some text", "speaker": None},
        ]

        output = render_speaker_attributed_transcript(segments)
        assert "S?: Some text" in output

    def test_empty_segments(self):
        """Test rendering empty segment list."""
        output = render_speaker_attributed_transcript([])
        assert output == ""

    def test_missing_segment_fields(self):
        """Test rendering with incomplete segment dicts."""
        segments = [
            {"text": "Hello"},  # Missing start, end, speaker
        ]

        output = render_speaker_attributed_transcript(segments)

        # Should gracefully handle missing fields with defaults
        assert isinstance(output, str)
        assert "S?: Hello" in output

    def test_empty_text_filtered(self):
        """Test that empty text segments are filtered out."""
        segments = [
            {"text": "Hello", "speaker": "S1"},
            {"text": "", "speaker": "S2"},  # Empty text
            {"text": "   ", "speaker": "S3"},  # Whitespace only
            {"text": "World", "speaker": "S4"},
        ]

        output = render_speaker_attributed_transcript(segments)
        lines = output.split("\n")
        
        assert len(lines) == 2  # Only S1 and S4
        assert "S1: Hello" in output
        assert "S4: World" in output
        assert "S2:" not in output
        assert "S3:" not in output

    def test_multiple_speakers(self):
        """Test rendering with multiple speakers."""
        segments = [
            {"text": "First speaker", "speaker": "S1"},
            {"text": "Second speaker", "speaker": "S2"},
            {"text": "Back to first", "speaker": "S1"},
            {"text": "Third speaker", "speaker": "S3"},
        ]

        output = render_speaker_attributed_transcript(segments)
        lines = output.split("\n")
        
        assert len(lines) == 4
        assert lines[0] == "S1: First speaker"
        assert lines[1] == "S2: Second speaker"
        assert lines[2] == "S1: Back to first"
        assert lines[3] == "S3: Third speaker"

    def test_consolidate_unknown_speaker_blocks(self):
        """Consolidate consecutive unknown speaker segments into one line."""
        segments = [
            {"text": "First unknown", "speaker": None},
            {"text": "Second unknown", "speaker": None},
            {"text": "Known speaker", "speaker": "S1"},
            {"text": "Third unknown", "speaker": None},
            {"text": "Fourth unknown", "speaker": None},
        ]

        output = render_speaker_attributed_transcript(segments)
        lines = output.split("\n")

        assert len(lines) == 3
        assert lines[0] == "S?: First unknown Second unknown"
        assert lines[1] == "S1: Known speaker"
        assert lines[2] == "S?: Third unknown Fourth unknown"

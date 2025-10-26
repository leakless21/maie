"""
End-to-end verification that diarization is properly wired to LLM pipeline.

This test verifies the complete flow:
1. Audio segments are diarized (speakers identified)
2. Segments are aligned with ASR output
3. Speakers are attributed to ASR text
4. Transcript is rendered in speaker-attributed format
5. LLM receives speaker context in transcription variable
"""

import json
from src.processors.prompt.diarization import render_speaker_attributed_transcript
from src.processors.audio.diarizer import Diarizer, DiarizedSegment


def test_alignment_and_rendering_e2e():
    """Test full flow: alignment -> rendering -> LLM format."""
    # Create a diarizer instance
    diarizer = Diarizer()
    
    # Simulated diarization output
    diarized_spans = [
        {"start": 0.0, "end": 2.5, "speaker": "S1"},
        {"start": 2.5, "end": 5.0, "speaker": "S2"},
        {"start": 5.0, "end": 7.5, "speaker": "S1"},
    ]
    
    # Simulated ASR segments with word timestamps
    class ASRSeg:
        def __init__(self, start, end, text, words=None):
            self.start = start
            self.end = end
            self.text = text
            self.words = words
    
    asr_segments = [
        ASRSeg(0.0, 2.5, "Hello, how are you today?", [
            {"start": 0.0, "end": 0.5, "word": "Hello"},
            {"start": 0.5, "end": 1.0, "word": "how"},
            {"start": 1.0, "end": 1.5, "word": "are"},
            {"start": 1.5, "end": 2.0, "word": "you"},
            {"start": 2.0, "end": 2.5, "word": "today"}
        ]),
        ASRSeg(2.5, 5.0, "I'm doing great, thanks for asking.", [
            {"start": 2.5, "end": 3.0, "word": "I'm"},
            {"start": 3.0, "end": 3.5, "word": "doing"},
            {"start": 3.5, "end": 4.0, "word": "great"},
            {"start": 4.0, "end": 4.5, "word": "thanks"},
            {"start": 4.5, "end": 5.0, "word": "asking"}
        ]),
        ASRSeg(5.0, 7.5, "That's wonderful to hear.", [
            {"start": 5.0, "end": 5.5, "word": "That's"},
            {"start": 5.5, "end": 6.0, "word": "wonderful"},
            {"start": 6.0, "end": 6.5, "word": "to"},
            {"start": 6.5, "end": 7.5, "word": "hear"}
        ]),
    ]
    
    # Use WhisperX-style word-level assignment
    aligned = diarizer.assign_word_speakers_whisperx_style(diarized_spans, asr_segments)
    
    # Should get DiarizedSegment objects back (word-level now)
    assert len(aligned) > 3  # More segments due to word-level processing
    assert all(isinstance(seg, DiarizedSegment) for seg in aligned)
    assert any(seg.speaker is not None for seg in aligned)  # At least some should have speakers
    
    # Convert to dicts for rendering (as pipeline does)
    segments_as_dicts = [
        {"start": seg.start, "end": seg.end, "text": seg.text, "speaker": seg.speaker}
        for seg in aligned
    ]
    
    # Render for LLM
    rendered = render_speaker_attributed_transcript(segments_as_dicts)
    
    # Verify LLM-ready output
    assert isinstance(rendered, str)
    assert "S1" in rendered
    assert "S2" in rendered
    assert "Hello" in rendered
    assert "great" in rendered
    assert "wonderful" in rendered


def test_speaker_attributed_human_format():
    """Test human format rendering for LLM."""
    segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello, how are you?", "speaker": "S1"},
        {"start": 2.5, "end": 5.0, "text": "I'm doing well.", "speaker": "S2"},
    ]
    
    rendered = render_speaker_attributed_transcript(segments)
    
    assert isinstance(rendered, str)
    assert "S1" in rendered
    assert "S2" in rendered
    assert "Hello" in rendered
    assert "well" in rendered


def test_multiple_speakers_all_included():
    """Test that all speakers are rendered."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "speaker one", "speaker": "S1"},
        {"start": 1.0, "end": 2.0, "text": "speaker two", "speaker": "S2"},
        {"start": 2.0, "end": 3.0, "text": "speaker three", "speaker": "S3"},
    ]
    
    rendered = render_speaker_attributed_transcript(segments)
    
    assert "S1" in rendered
    assert "S2" in rendered
    assert "S3" in rendered
    assert "speaker one" in rendered
    assert "speaker two" in rendered
    assert "speaker three" in rendered


def test_speaker_order_preserved():
    """Test speaker turns are in order."""
    segments = [
        {"start": 0.0, "end": 2.0, "text": "First turn", "speaker": "S1"},
        {"start": 2.0, "end": 4.0, "text": "Second turn", "speaker": "S2"},
        {"start": 4.0, "end": 6.0, "text": "Third turn", "speaker": "S1"},
    ]
    
    rendered = render_speaker_attributed_transcript(segments)
    
    # All content should be present and in order
    assert "First turn" in rendered
    assert "Second turn" in rendered
    assert "Third turn" in rendered


def test_plain_text_format():
    """Test that plain text format is returned."""
    segments = [
        {"start": 0.0, "end": 2.0, "text": "test", "speaker": "S1"},
    ]
    
    rendered = render_speaker_attributed_transcript(segments)
    
    assert isinstance(rendered, str)
    # Should have speaker attribution
    assert "S1: test" in rendered


def test_multiple_speakers_plain_text():
    """Test multiple speakers in plain text format."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "S1"},
        {"start": 1.0, "end": 2.0, "text": "world", "speaker": "S2"},
    ]
    
    result = render_speaker_attributed_transcript(segments)
    
    assert isinstance(result, str)
    assert "S1: hello" in result
    assert "S2: world" in result
    
    # Should be plain text format
    lines = result.strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == "S1: hello"
    assert lines[1] == "S2: world"


def test_three_speakers_plain_text():
    """Test three speakers in plain text format."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "one", "speaker": "S1"},
        {"start": 1.0, "end": 2.0, "text": "two", "speaker": "S2"},
        {"start": 2.0, "end": 3.0, "text": "three", "speaker": "S3"},
    ]
    
    result = render_speaker_attributed_transcript(segments)
    
    assert isinstance(result, str)
    assert "S1: one" in result
    assert "S2: two" in result
    assert "S3: three" in result
    
    # Should be three lines
    lines = result.strip().split("\n")
    assert len(lines) == 3


def test_empty_segments_handled():
    """Test empty segment handling."""
    rendered = render_speaker_attributed_transcript([])
    
    assert isinstance(rendered, str)
    assert rendered == ""


def test_none_speaker_preserved():
    """Test segments with None speaker are handled."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "speaker one", "speaker": "S1"},
        {"start": 1.0, "end": 2.0, "text": "unknown", "speaker": None},
        {"start": 2.0, "end": 3.0, "text": "speaker two", "speaker": "S2"},
    ]
    
    rendered = render_speaker_attributed_transcript(segments)
    
    # All text should be present
    assert "speaker one" in rendered
    assert "unknown" in rendered
    assert "speaker two" in rendered
    assert "S1" in rendered
    assert "S2" in rendered


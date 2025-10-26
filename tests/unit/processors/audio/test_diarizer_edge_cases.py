"""
Hardcore edge case tests for diarization.

Tests extreme scenarios:
- Zero-duration segments
- Overlapping segments with identical times
- Very short segments (< 0.1s)
- Very long text that needs splitting
- Unicode and special characters
- Empty text
- Many speakers (10+)
- Rapid speaker changes
- Extremely unbalanced speaker overlap
- Floating point precision edge cases
"""

from __future__ import annotations

import pytest
from src.processors.audio.diarizer import Diarizer, DiarizedSegment


class TestHardcoreEdgeCases:
    """Extreme edge cases that could break the implementation."""

    @pytest.fixture
    def diarizer(self):
        """Create diarizer instance."""
        return Diarizer(overlap_threshold=0.3)

    def test_zero_duration_asr_segment(self, diarizer):
        """Test handling of zero-duration ASR segment."""
        diarized = [{"start": 0.0, "end": 2.0, "speaker": "S1"}]
        asr = [DiarizedSegment(start=1.0, end=1.0, text="instant")]  # Zero duration
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        # Should handle gracefully
        assert len(result) >= 0
        
    def test_zero_duration_diarization_span(self, diarizer):
        """Test handling of zero-duration diarization span."""
        diarized = [{"start": 1.0, "end": 1.0, "speaker": "S1"}]  # Zero duration
        asr = [DiarizedSegment(start=0.0, end= 2.0, text="hello world")]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        # Should still produce output
        assert len(result) > 0

    def test_extremely_short_segment(self, diarizer):
        """Test segments shorter than 0.01 seconds."""
        diarized = [{"start": 0.0, "end": 0.001, "speaker": "S1"}]
        asr = [DiarizedSegment(start=0.0, end=0.001, text="a")]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        assert len(result) == 1
        assert result[0].speaker == "S1"

    def test_very_long_text_proportional_split(self, diarizer):
        """Test proportional split with very long text (1000+ words)."""
        # Generate long text
        words = [f"word{i}" for i in range(1000)]
        long_text = " ".join(words)
        
        diarized = [
            {"start": 0.0, "end": 50.0, "speaker": "S1"},
            {"start": 50.0, "end": 100.0, "speaker": "S2"},
        ]
        asr = [DiarizedSegment(start=0.0, end=100.0, text=long_text)]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        # Should split text proportionally
        if len(result) > 1:
            total_words = sum(len(seg.text.split()) for seg in result)
            # All words should be preserved
            assert total_words == 1000

    def test_empty_text_segment(self, diarizer):
        """Test handling of empty text."""
        diarized = [{"start": 0.0, "end": 2.0, "speaker": "S1"}]
        asr = [DiarizedSegment(start=0.0, end=2.0, text="")]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        assert len(result) == 1
        assert result[0].text == ""
        assert result[0].speaker == "S1"

    def test_whitespace_only_text(self, diarizer):
        """Test handling of whitespace-only text."""
        diarized = [{"start": 0.0, "end": 2.0, "speaker": "S1"}]
        asr = [DiarizedSegment(start=0.0, end=2.0, text="   \t\n   ")]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        assert len(result) == 1
        assert result[0].speaker == "S1"

    def test_unicode_and_emoji_text(self, diarizer):
        """Test handling of unicode, emoji, and special characters."""
        diarized = [{"start": 0.0, "end": 2.0, "speaker": "S1"}]
        text = "Hello 世界 🌍 café naïve"
        asr = [DiarizedSegment(start=0.0, end=2.0, text=text)]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        assert len(result) == 1
        assert result[0].text == text
        assert result[0].speaker == "S1"

    def test_many_speakers(self, diarizer):
        """Test handling of many speakers (15+)."""
        # 15 speakers in sequence
        diarized = [
            {"start": float(i), "end": float(i + 1), "speaker": f"S{i+1}"}
            for i in range(15)
        ]
        asr = [
            DiarizedSegment(start=float(i), end=float(i + 1), text=f"speaker {i+1}")
            for i in range(15)
        ]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        assert len(result) == 15
        # Verify all speakers assigned correctly
        speakers = [seg.speaker for seg in result]
        assert len(set(speakers)) == 15

    def test_rapid_speaker_alternation(self, diarizer):
        """Test rapid speaker changes every 0.1 seconds."""
        diarized = []
        asr = []
        
        for i in range(20):
            start = i * 0.1
            end = (i + 1) * 0.1
            speaker = "S1" if i % 2 == 0 else "S2"
            diarized.append({"start": start, "end": end, "speaker": speaker})
            asr.append(DiarizedSegment(start=start, end=end, text=f"word{i}"))
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        # Should handle rapid alternation
        assert len(result) == 20

    def test_extremely_unbalanced_overlap(self, diarizer):
        """Test when one speaker has 99% overlap, another 1%."""
        diarized = [
            {"start": 0.0, "end": 99.0, "speaker": "S1"},
            {"start": 99.0, "end": 100.0, "speaker": "S2"},
        ]
        asr = [DiarizedSegment(start=0.0, end=100.0, text="long speech")]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        # Should assign to dominant speaker (S1 has 99%)
        assert len(result) >= 1
        # First/main segment should be S1
        assert result[0].speaker == "S1"

    def test_floating_point_precision_edge(self, diarizer):
        """Test floating point precision issues."""
        # Times that could cause floating point issues
        diarized = [{"start": 0.1 + 0.2, "end": 0.4 + 0.5, "speaker": "S1"}]  # 0.3, 0.9
        asr = [DiarizedSegment(start=0.3, end=0.9, text="test")]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        assert len(result) == 1
        assert result[0].speaker == "S1"

    def test_negative_times(self, diarizer):
        """Test handling of negative times (should not happen but test robustness)."""
        diarized = [{"start": -1.0, "end": 1.0, "speaker": "S1"}]
        asr = [DiarizedSegment(start=0.0, end=1.0, text="hello")]
        
        # Should handle without crashing
        result = diarizer.align_diarization_with_asr(diarized, asr)
        assert len(result) >= 0

    def test_times_out_of_order(self, diarizer):
        """Test segments with start > end (invalid but test robustness)."""
        diarized = [{"start": 2.0, "end": 1.0, "speaker": "S1"}]  # Reversed
        asr = [DiarizedSegment(start=1.0, end=2.0, text="hello")]
        
        # Should handle without crashing
        result = diarizer.align_diarization_with_asr(diarized, asr)
        assert len(result) >= 0

    def test_merge_with_single_segment(self, diarizer):
        """Test merging when there's only one segment."""
        segments = [DiarizedSegment(start=0.0, end=2.0, text="hello", speaker="S1")]
        
        result = diarizer.merge_adjacent_same_speaker(segments)
        
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_merge_empty_list(self, diarizer):
        """Test merging with empty list."""
        result = diarizer.merge_adjacent_same_speaker([])
        
        assert len(result) == 0

    def test_merge_all_none_speakers(self, diarizer):
        """Test merging when all speakers are None."""
        segments = [
            DiarizedSegment(start=0.0, end=1.0, text="one", speaker=None),
            DiarizedSegment(start=1.0, end=2.0, text="two", speaker=None),
            DiarizedSegment(start=2.0, end=3.0, text="three", speaker=None),
        ]
        
        result = diarizer.merge_adjacent_same_speaker(segments)
        
        # None speakers should not be merged
        assert len(result) == 3

    def test_proportional_split_single_word(self, diarizer):
        """Test proportional split with single word."""
        diarized = [
            {"start": 0.0, "end": 1.0, "speaker": "S1"},
            {"start": 1.0, "end": 2.0, "speaker": "S2"},
        ]
        asr = [DiarizedSegment(start=0.0, end=2.0, text="word")]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        # Should handle single word gracefully
        assert len(result) >= 1
        # Word should not be lost
        all_text = " ".join(seg.text for seg in result)
        assert "word" in all_text

    def test_three_way_speaker_overlap(self, diarizer):
        """Test three speakers overlapping same segment."""
        diarized = [
            {"start": 0.0, "end": 3.0, "speaker": "S1"},  # 33% each
            {"start": 0.0, "end": 3.0, "speaker": "S2"},
            {"start": 0.0, "end": 3.0, "speaker": "S3"},
        ]
        asr = [DiarizedSegment(start=0.0, end=3.0, text="hello world test")]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        # Should handle 3-way tie gracefully
        assert len(result) >= 1
        # All text should be preserved
        all_text = " ".join(seg.text for seg in result)
        assert "hello" in all_text

    def test_alignment_with_dict_and_dataclass_mixed(self, diarizer):
        """Test alignment works with mixed input types."""
        # Mix dict and dataclass
        diarized = [
            {"start": 0.0, "end": 1.0, "speaker": "S1"},  # dict
        ]
        asr = [
            DiarizedSegment(start=0.0, end=1.0, text="hello"),  # dataclass
        ]
        
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        assert len(result) == 1
        assert isinstance(result[0], DiarizedSegment)

    def test_very_large_segment_count(self, diarizer):
        """Test with 1000+ segments (performance/scale test)."""
        diarized = [
            {"start": float(i), "end": float(i + 1), "speaker": f"S{i % 10}"}
            for i in range(1000)
        ]
        asr = [
            DiarizedSegment(start=float(i), end=float(i + 1), text=f"segment {i}")
            for i in range(1000)
        ]
        
        # This tests performance and memory
        result = diarizer.align_diarization_with_asr(diarized, asr)
        
        assert len(result) == 1000

    def test_iou_calculation_precision(self, diarizer):
        """Test IoU calculation with precise edge cases."""
        # Exact adjacency (no overlap)
        iou1 = diarizer._calculate_iou((0.0, 1.0), (1.0, 2.0))
        assert iou1 == 0.0
        
        # Perfect overlap
        iou2 = diarizer._calculate_iou((0.0, 2.0), (0.0, 2.0))
        assert iou2 == 1.0
        
        # One completely contains other
        iou3 = diarizer._calculate_iou((0.0, 10.0), (2.0, 3.0))
        assert 0.0 < iou3 < 1.0
        
        # Half overlap
        iou4 = diarizer._calculate_iou((0.0, 2.0), (1.0, 3.0))
        assert 0.3 < iou4 < 0.4  # Should be 1/3

    def test_merge_preserves_order(self, diarizer):
        """Test that merging preserves chronological order."""
        segments = [
            DiarizedSegment(start=0.0, end=1.0, text="first", speaker="S1"),
            DiarizedSegment(start=1.0, end=2.0, text="second", speaker="S1"),
            DiarizedSegment(start=2.0, end=3.0, text="third", speaker="S2"),
            DiarizedSegment(start=3.0, end=4.0, text="fourth", speaker="S2"),
        ]
        
        result = diarizer.merge_adjacent_same_speaker(segments)
        
        # Should be 2 merged segments
        assert len(result) == 2
        # Order should be preserved
        assert result[0].text == "first second"
        assert result[1].text == "third fourth"
        # Times should be correct
        assert result[0].start == 0.0
        assert result[0].end == 2.0
        assert result[1].start == 2.0
        assert result[1].end == 4.0


class TestRenderingEdgeCases:
    """Edge cases for transcript rendering."""

    def test_render_with_very_long_speaker_text(self):
        """Test rendering with extremely long text."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        long_text = " ".join([f"word{i}" for i in range(1000)])
        segments = [
            {"start": 0.0, "end": 100.0, "text": long_text, "speaker": "S1"}
        ]
        
        result = render_speaker_attributed_transcript(segments)
        
        # Should handle long text
        assert isinstance(result, str)
        assert "word0" in result
        assert "word999" in result

    def test_render_with_special_characters_in_text(self):
        """Test rendering with special characters that could break formatting."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Text with: colons | pipes & symbols", "speaker": "S1"},
            {"start": 1.0, "end": 2.0, "text": "More [brackets] {braces} <angles>", "speaker": "S2"},
        ]
        
        result = render_speaker_attributed_transcript(segments)
        
        # Should preserve special characters
        assert "colons" in result
        assert "brackets" in result

    def test_render_with_newlines_in_text(self):
        """Test rendering when text contains newlines."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {"start": 0.0, "end": 1.0, "text": "First line\nSecond line\nThird line", "speaker": "S1"},
        ]
        
        result = render_speaker_attributed_transcript(segments)
        
        # Should handle newlines
        assert isinstance(result, str)

    def test_plain_text_format_with_special_characters(self):
        """Test plain text rendering with special characters."""
        from src.processors.prompt.diarization import render_speaker_attributed_transcript
        
        segments = [
            {"start": 0.0, "end": 1.0, "text": 'Quote: "hello" and slash: \\ backslash', "speaker": "S1"},
        ]
        
        result = render_speaker_attributed_transcript(segments)
        
        # Should be plain text with speaker attribution
        assert isinstance(result, str)
        assert "S1: Quote: \"hello\" and slash: \\ backslash" in result

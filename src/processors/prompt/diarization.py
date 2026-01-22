"""
Diarization prompt packaging for LLM input.

This module transforms speaker-attributed ASR segments into plain text format
suitable for LLM input: S1: text\nS2: more text\n...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DiarizedSegment:
    """A transcript segment with speaker attribution."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"DiarizedSegment(start={self.start:.1f}, end={self.end:.1f}, "
            f"speaker={self.speaker}, text={self.text[:20]!r}...)"
        )


def render_speaker_attributed_transcript(
    segments: List[Dict[str, Any]]
) -> str:
    """
    Render speaker-attributed segments as plain text.
    
    Format: S1: text\nS2: more text\n...
    
    Args:
        segments: List of diarized segment dicts or DiarizedSegment objects with 'speaker' and 'text' fields
        
    Returns:
        Plain text with speaker attribution
    """
    lines = []
    for seg in segments:
        # Handle both dict and DiarizedSegment objects
        if isinstance(seg, dict):
            speaker = seg.get("speaker") or "S?"
            text = seg.get("text", "").strip()
        else:
            # Assume it's a DiarizedSegment or similar object
            speaker = getattr(seg, "speaker", None) or "S?"
            text = getattr(seg, "text", "").strip()
        
        if not text:
            continue

        if speaker == "S?" and lines and lines[-1][0] == "S?":
            lines[-1] = ("S?", f"{lines[-1][1]} {text}".strip())
        else:
            lines.append((speaker, text))
    return "\n".join(f"{speaker}: {text}" for speaker, text in lines)

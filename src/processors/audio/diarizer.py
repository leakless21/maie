"""
Diarization processor for speaker attribution in MAIE.

This module provides speaker diarization functionality using pyannote.audio,
with graceful fallback when GPU is unavailable.

Features:
- Speaker segmentation via pyannote.audio
- WhisperX-style word-level speaker assignment using temporal overlap
- Segment merging for readability
- Graceful CUDA handling
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from src.config.logging import get_module_logger
from src.utils.device import has_cuda

logger = get_module_logger(__name__)


@dataclass
class DiarizedSegment:
    """ASR segment with speaker attribution."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    def __repr__(self) -> str:
        return f"DiarizedSegment(start={self.start}, end={self.end}, speaker={self.speaker}, text={self.text[:20]!r}...)"


class Diarizer:
    """
    Speaker diarization processor using pyannote.audio.

    Attributes:
        model_path: Path to the pyannote speaker diarization model
        require_cuda: Whether to fail if CUDA is unavailable
        embedding_batch_size: Batch size for embedding model (controls memory usage)
        segmentation_batch_size: Batch size for segmentation model (controls memory usage)
        model: Lazy-loaded pyannote model instance
    """

    def __init__(
        self,
        model_path: str = "data/models/speaker-diarization-community-1",
        require_cuda: bool = False,
        embedding_batch_size: int = 32,
        segmentation_batch_size: int = 32,
    ):
        """
        Initialize the Diarizer.

        Args:
            model_path: Path to the speaker diarization model
            require_cuda: If True, raise when CUDA unavailable. If False, skip diarization gracefully.
            embedding_batch_size: Batch size for speaker embedding model (default 32 per official docs)
            segmentation_batch_size: Batch size for segmentation model (default 32 per official docs)
        """
        self.model_path = model_path
        self.require_cuda = require_cuda
        self.embedding_batch_size = embedding_batch_size
        self.segmentation_batch_size = segmentation_batch_size
        self.model = None

    def _load_pyannote_model(self) -> object:
        """
        Load pyannote speaker diarization model lazily.
        
        NOTE: pyannote.audio 3.x requires using HuggingFace model identifiers
        like "pyannote/speaker-diarization-3.1" instead of local paths.
        Batch sizes are configured in the model's config.yaml file.
        
        Reference: https://github.com/pyannote/pyannote-audio/blob/main/tutorials

        Returns:
            Loaded model callable or None if loading fails.

        Raises:
            RuntimeError: If CUDA is required but unavailable.
        """
        logger.info(
            f"_load_pyannote_model called with batch sizes: "
            f"embedding={self.embedding_batch_size}, "
            f"segmentation={self.segmentation_batch_size}"
        )
        
        if self.model is not None:
            logger.info("Model already loaded, returning cached instance")
            return self.model

        # Check CUDA availability
        if self.require_cuda and not has_cuda():
            logger.error("CUDA is required but not available; cannot load diarization model")
            raise RuntimeError("CUDA is required but not available")

        if not has_cuda():
            logger.warning("CUDA not available; diarization will be skipped")
            return None

        logger.info("Starting pyannote model load process...")
        
        try:
            # Lazy import of pyannote
            logger.info("Importing pyannote.audio.Pipeline...")
            from pyannote.audio import Pipeline

            logger.info("Imports successful, loading pipeline...")
            
            # pyannote.audio 3.x uses HuggingFace model hub format
            # Convert local path references to HF model IDs
            model_id = self.model_path
            
            # Map common local paths to HuggingFace model IDs
            if "speaker-diarization-3.1" in model_id:
                model_id = "pyannote/speaker-diarization-3.1"
            elif "speaker-diarization-community-1" in model_id or "community" in model_id:
                # community-1 not available in 3.x, use 3.1 instead
                model_id = "pyannote/speaker-diarization-3.1"
                logger.info(
                    f"Mapped {self.model_path} to {model_id} "
                    "(pyannote 3.x uses HuggingFace model hub)"
                )
            
            logger.info(f"Loading pyannote pipeline: {model_id}")
            
            # Load model from HuggingFace (uses local cache if available)
            model = Pipeline.from_pretrained(model_id)
            
            # Move model to GPU if available
            import torch
            device = torch.device("cuda" if has_cuda() else "cpu")
            model.to(device)  # type: ignore[union-attr]
            
            # Log batch size configuration info
            # NOTE: In pyannote 3.x, batch sizes are set in the model's config.yaml
            # Default is 32 for both embedding and segmentation
            logger.info(
                f"Pyannote model loaded successfully on {device}. "
                f"Requested batch sizes: embedding={self.embedding_batch_size}, "
                f"segmentation={self.segmentation_batch_size}. "
                f"Note: pyannote 3.x batch sizes are configured in model's config.yaml"
            )
            
            self.model = model
            return self.model

        except ImportError as e:
            logger.error(f"pyannote.audio not installed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load diarization model: {e}", exc_info=True)
            return None


    def diarize(
        self, audio_path: str, num_speakers: Optional[int] = None
    ) -> Optional[list]:
        """
        Run speaker diarization on audio file.

        Args:
            audio_path: Path to audio file
            num_speakers: Optional number of speakers (if known)

        Returns:
            List of diarization spans with speaker info, or None if diarization unavailable
        """
        # Try to load model
        model = self._load_pyannote_model()
        if model is None:
            logger.warning("Diarization model unavailable; skipping diarization")
            return None

        try:
            logger.info(
                f"Running diarization on {audio_path} with batch_sizes: "
                f"embedding={self.embedding_batch_size}, segmentation={self.segmentation_batch_size}"
            )


            # Run diarization
            # NOTE: Batch sizes should have been configured during model load in _load_pyannote_model()
            if num_speakers is not None:
                diarization = model(audio_path, num_speakers=num_speakers)
            else:
                diarization = model(audio_path)

            # Convert pyannote output to simple list of spans
            # pyannote returns an iterator of (segment, _, speaker) tuples
            spans = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                spans.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "speaker": speaker,
                    }
                )

            logger.info(f"Diarization complete: {len(spans)} speaker segments")
            
            return spans

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return None

    def has_word_timestamps(self, asr_segments: list) -> bool:
        """
        Check if ASR segments contain word-level timestamps.
        
        Args:
            asr_segments: List of ASR segments to check
            
        Returns:
            True if word-level timestamps are available, False otherwise
        """
        if not asr_segments:
            return False
        
        first_seg = asr_segments[0]
        words = getattr(first_seg, 'words', None)
        if not words and isinstance(first_seg, dict):
            words = first_seg.get('words')
        
        return words is not None and len(words) > 0

    def assign_word_speakers_whisperx_style(
        self, diar_spans: list, asr_segments: list
    ) -> list[DiarizedSegment]:
        """
        Assign speakers to words using WhisperX-style temporal overlap.
        
        This method replicates WhisperX's approach:
        1. For each word, find the diarization segment with maximum temporal overlap
        2. Assign the speaker from that segment to the word
        3. If no overlap exists, leave speaker as None
        
        Args:
            diar_spans: List of diarization spans from pyannote
            asr_segments: List of ASR segments with word-level timestamps
            
        Returns:
            List of DiarizedSegment with speaker attribution at word level
        """
        result = []
        
        for asr_idx, asr_seg in enumerate(asr_segments):
            # Get words from the segment
            words = getattr(asr_seg, 'words', None)
            if not words and isinstance(asr_seg, dict):
                words = asr_seg.get('words')
            
            if not words:
                logger.warning(f"ASR segment {asr_idx} has no word timestamps, skipping")
                continue
            
            # Process each word individually
            for word in words:
                word_start = word.get('start', 0) if isinstance(word, dict) else getattr(word, 'start', 0)
                word_end = word.get('end', 0) if isinstance(word, dict) else getattr(word, 'end', 0)
                word_text = word.get('word', '') if isinstance(word, dict) else getattr(word, 'word', '')
                
                # Find the diarization segment with maximum temporal overlap
                best_speaker = None
                max_overlap_duration = 0
                
                for diar in diar_spans:
                    # Handle both dict and object formats
                    if isinstance(diar, dict):
                        diar_start = diar.get("start", 0)
                        diar_end = diar.get("end", 0)
                        diar_speaker = diar.get("speaker")
                    else:
                        diar_start = getattr(diar, "start", 0)
                        diar_end = getattr(diar, "end", 0)
                        diar_speaker = getattr(diar, "speaker", None)
                    
                    if not diar_speaker:
                        continue
                    
                    # Check for temporal overlap
                    if word_start < diar_end and word_end > diar_start:
                        # Calculate overlap duration
                        overlap_start = max(word_start, diar_start)
                        overlap_end = min(word_end, diar_end)
                        overlap_duration = overlap_end - overlap_start
                        
                        if overlap_duration > max_overlap_duration:
                            max_overlap_duration = overlap_duration
                            best_speaker = diar_speaker
                
                # Create word segment with assigned speaker
                result.append(DiarizedSegment(
                    start=word_start,
                    end=word_end,
                    text=word_text,
                    speaker=best_speaker,
                ))
        
        return result


    def merge_adjacent_same_speaker(
        self, segments: list[DiarizedSegment],
    ) -> list[DiarizedSegment]:
        """
        Merge adjacent segments with the same speaker, including intelligent Unknown handling.

        Args:
            segments: List of diarized segments

        Returns:
            List of merged segments
        """
        if not segments:
            return []

        # First pass: merge same speakers
        merged = []
        current = segments[0]

        for next_seg in segments[1:]:
            # Merge if same speaker (not None) and adjacent (or very close)
            if (
                current.speaker is not None
                and current.speaker == next_seg.speaker
                and abs(current.end - next_seg.start) < 0.01
            ):
                # Merge by extending current segment
                current = DiarizedSegment(
                    start=current.start,
                    end=next_seg.end,
                    text=self._merge_text(current.text, next_seg.text),
                    speaker=current.speaker,
                )
            else:
                # Different speaker or gap: save current and start new
                merged.append(current)
                current = next_seg

        # Add last segment
        merged.append(current)

        # Second pass: intelligently assign Unknown segments to adjacent speakers
        return self._merge_unknown_segments(merged)

    def _merge_unknown_segments(self, segments: list[DiarizedSegment]) -> list[DiarizedSegment]:
        """
        Intelligently merge Unknown (None) segments with adjacent identified speakers.
        
        This helps reduce fragmentation by assigning Unknown segments to the most likely speaker
        based on temporal proximity and context.
        
        Args:
            segments: List of segments after initial same-speaker merging
            
        Returns:
            List of segments with Unknown segments intelligently assigned
        """
        if not segments:
            return []
            
        result = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # If current segment has a speaker, keep it as-is
            if current.speaker is not None:
                result.append(current)
                i += 1
                continue
                
            # Current segment is Unknown - try to assign it intelligently
            assigned_speaker = None
            
            # Look at previous segment
            if result and result[-1].speaker is not None:
                prev_speaker = result[-1].speaker
                prev_end = result[-1].end
                
                # If gap is small (< 0.5s), likely same speaker
                if abs(prev_end - current.start) < 0.5:
                    assigned_speaker = prev_speaker
            
            # Look at next segment
            if assigned_speaker is None and i + 1 < len(segments):
                next_seg = segments[i + 1]
                if next_seg.speaker is not None:
                    next_speaker = next_seg.speaker
                    next_start = next_seg.start
                    
                    # If gap is small (< 0.5s), likely same speaker
                    if abs(current.end - next_start) < 0.5:
                        assigned_speaker = next_speaker
            
            # If we found a likely speaker, assign it
            if assigned_speaker is not None:
                # Merge with previous segment if it's the same speaker
                if result and result[-1].speaker == assigned_speaker:
                    result[-1] = DiarizedSegment(
                        start=result[-1].start,
                        end=current.end,
                        text=self._merge_text(result[-1].text, current.text),
                        speaker=assigned_speaker,
                    )
                else:
                    # Create new segment with assigned speaker
                    result.append(DiarizedSegment(
                        start=current.start,
                        end=current.end,
                        text=current.text,
                        speaker=assigned_speaker,
                    ))
            else:
                # Keep as Unknown if no clear assignment
                result.append(current)
                
            i += 1
            
        return result

    def _merge_text(self, text1: str, text2: str) -> str:
        """
        Intelligently merge two text segments, avoiding double spaces.
        
        Args:
            text1: First text segment
            text2: Second text segment
            
        Returns:
            Merged text with proper spacing
        """
        if not text1:
            return text2
        if not text2:
            return text1
            
        # Strip whitespace from both texts
        text1 = text1.strip()
        text2 = text2.strip()
        
        # If either text is empty after stripping, return the non-empty one
        if not text1:
            return text2
        if not text2:
            return text1
            
        # Join with a single space
        return f"{text1} {text2}"


def get_diarizer(
    model_path: str = "data/models/speaker-diarization-community-1",
    require_cuda: bool = False,
    embedding_batch_size: int = 32,
    segmentation_batch_size: int = 32,
) -> Optional[Diarizer]:
    """
    Factory function to create and return a Diarizer instance.

    Gracefully returns None if diarization is unavailable (e.g., no CUDA when required).

    Args:
        model_path: Path to the diarization model
        require_cuda: If True, returns None when CUDA unavailable
        embedding_batch_size: Batch size for speaker embedding (default 32 per official pyannote.audio docs)
        segmentation_batch_size: Batch size for segmentation (default 32 per official pyannote.audio docs)

    Returns:
        Diarizer instance or None if unavailable
    """
    try:
        if require_cuda and not has_cuda():
            logger.warning("Diarization requires CUDA but not available; skipping")
            return None

        diarizer = Diarizer(
            model_path=model_path,
            require_cuda=require_cuda,
            embedding_batch_size=embedding_batch_size,
            segmentation_batch_size=segmentation_batch_size,
        )
        return diarizer
    except Exception as e:
        logger.error(f"Failed to create diarizer: {e}")
        return None
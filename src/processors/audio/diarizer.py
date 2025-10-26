"""
Diarization processor for speaker attribution in MAIE.

This module provides speaker diarization functionality using pyannote.audio,
with graceful fallback when GPU is unavailable.

Features:
- Speaker segmentation via pyannote.audio
- IoU-based alignment with ASR segments
- Proportional splitting without word-level timestamps
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
        overlap_threshold: Minimum IoU threshold for segment overlap (0.0-1.0)
        embedding_batch_size: Batch size for embedding model (controls memory usage)
        segmentation_batch_size: Batch size for segmentation model (controls memory usage)
        model: Lazy-loaded pyannote model instance
    """

    def __init__(
        self,
        model_path: str = "data/models/speaker-diarization-community-1",
        require_cuda: bool = False,
        overlap_threshold: float = 0.3,
        embedding_batch_size: int = 32,
        segmentation_batch_size: int = 32,
    ):
        """
        Initialize the Diarizer.

        Args:
            model_path: Path to the speaker diarization model
            require_cuda: If True, raise when CUDA unavailable. If False, skip diarization gracefully.
            overlap_threshold: Minimum IoU for considering overlap (default 0.3 = 30%)
            embedding_batch_size: Batch size for speaker embedding model (default 32 per official docs)
            segmentation_batch_size: Batch size for segmentation model (default 32 per official docs)
        """
        self.model_path = model_path
        self.require_cuda = require_cuda
        self.overlap_threshold = overlap_threshold
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

    @staticmethod
    def _calculate_iou(interval_a: tuple[float, float], interval_b: tuple[float, float]) -> float:
        """
        Calculate Intersection over Union (IoU) for two time intervals.

        Args:
            interval_a: (start, end) tuple for first interval
            interval_b: (start, end) tuple for second interval

        Returns:
            IoU value between 0.0 and 1.0
        """
        start_a, end_a = interval_a
        start_b, end_b = interval_b

        # Calculate intersection
        intersection_start = max(start_a, start_b)
        intersection_end = min(end_a, end_b)
        intersection = max(0.0, intersection_end - intersection_start)

        # Calculate union
        union_start = min(start_a, start_b)
        union_end = max(end_a, end_b)
        union = union_end - union_start

        # Calculate IoU
        iou = intersection / union if union > 0 else 0.0
        return iou

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

            # DEBUG: Log diarization input details
            import os
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                logger.debug(
                    "Diarization input details",
                    audio_path=audio_path,
                    file_size_bytes=file_size,
                    num_speakers=num_speakers,
                    embedding_batch_size=self.embedding_batch_size,
                    segmentation_batch_size=self.segmentation_batch_size,
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
            
            # DEBUG: Log diarization output details
            speaker_count = len(set(span["speaker"] for span in spans))
            logger.debug(
                "Diarization output details",
                span_count=len(spans),
                speaker_count=speaker_count,
                speakers=list(set(span["speaker"] for span in spans)),
                total_duration=sum(span["end"] - span["start"] for span in spans) if spans else 0,
            )
            
            # DEBUG: Log first few diarization spans for analysis
            if spans:
                logger.debug("First 5 diarization spans:")
                for i, span in enumerate(spans[:5]):
                    logger.debug(f"  Span {i}: {span['start']:.3f}s-{span['end']:.3f}s speaker={span['speaker']}")
                logger.debug("Last 5 diarization spans:")
                for i, span in enumerate(spans[-5:], len(spans)-5):
                    logger.debug(f"  Span {i}: {span['start']:.3f}s-{span['end']:.3f}s speaker={span['speaker']}")
            
            return spans

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return None

    def align_diarization_with_asr(
        self, diar_spans: list, asr_segments: list
    ) -> list[DiarizedSegment]:
        """
        Align diarization spans with ASR segments using IoU-based matching.

        Algorithm:
        1. For each ASR segment, find overlapping diarization spans (IoU >= threshold)
        2. If no speakers: keep segment with speaker=None
        3. If one speaker: assign entire segment to that speaker
        4. If multiple speakers:
           a. If one speaker covers >=0.7 of segment: assign entire segment to that speaker
           b. Otherwise: split proportionally at the dominant speaker boundary

        Args:
            diar_spans: List of diarization spans from pyannote
            asr_segments: List of ASR segments to align

        Returns:
            List of DiarizedSegment with speaker attribution
        """
        # DEBUG: Log alignment input details
        logger.debug(
            "Diarization alignment input",
            diar_spans_count=len(diar_spans),
            asr_segments_count=len(asr_segments),
            overlap_threshold=self.overlap_threshold,
        )
        
        # DEBUG: Log first few ASR segments for analysis
        if asr_segments:
            logger.debug("First 3 ASR segments:")
            for i, seg in enumerate(asr_segments[:3]):
                logger.debug(f"  ASR {i}: {seg.start:.3f}s-{seg.end:.3f}s text='{seg.text[:50]}...'")
            logger.debug("Last 3 ASR segments:")
            for i, seg in enumerate(asr_segments[-3:], len(asr_segments)-3):
                logger.debug(f"  ASR {i}: {seg.start:.3f}s-{seg.end:.3f}s text='{seg.text[:50]}...'")
        
        result = []

        for asr_idx, asr_seg in enumerate(asr_segments):
            asr_interval = (asr_seg.start, asr_seg.end)
            asr_duration = asr_seg.end - asr_seg.start

            # DEBUG: Log processing of each ASR segment
            logger.debug(f"Processing ASR segment {asr_idx}: {asr_seg.start:.3f}s-{asr_seg.end:.3f}s (duration: {asr_duration:.3f}s)")

            # Find overlapping diarization spans
            overlaps = []
            overlap_candidates = []
            
            for diar_idx, diar in enumerate(diar_spans):
                # Handle both dict and dataclass diarization spans
                diar_start = (
                    diar["start"] if isinstance(diar, dict) else diar.start
                )
                diar_end = diar["end"] if isinstance(diar, dict) else diar.end
                diar_speaker = (
                    diar["speaker"] if isinstance(diar, dict) else diar.speaker
                )

                diar_interval = (diar_start, diar_end)
                iou = self._calculate_iou(asr_interval, diar_interval)
                
                # Calculate overlap duration and percentage
                overlap_duration = min(asr_seg.end, diar_end) - max(asr_seg.start, diar_start)
                overlap_pct = overlap_duration / asr_duration if asr_duration > 0 else 0
                
                # Track all candidates for debugging
                overlap_candidates.append({
                    "diar_idx": diar_idx,
                    "speaker": diar_speaker,
                    "start": diar_start,
                    "end": diar_end,
                    "overlap_duration": overlap_duration,
                    "overlap_pct": overlap_pct,
                    "iou": iou,
                })

                if iou >= self.overlap_threshold:
                    overlaps.append(
                        {
                            "speaker": diar_speaker,
                            "start": diar_start,
                            "end": diar_end,
                            "overlap_pct": overlap_pct,
                            "iou": iou,
                        }
                    )
            
            # DEBUG: Log overlap analysis for this ASR segment
            logger.debug(f"ASR {asr_idx} overlap analysis:")
            logger.debug(f"  Found {len(overlaps)} overlaps above threshold {self.overlap_threshold}")
            logger.debug(f"  Total candidates checked: {len(overlap_candidates)}")
            
            # Log top 5 candidates by IoU for debugging
            top_candidates = sorted(overlap_candidates, key=lambda x: x["iou"], reverse=True)[:5]
            logger.debug(f"  Top 5 candidates by IoU:")
            for i, cand in enumerate(top_candidates):
                logger.debug(f"    {i+1}. IoU={cand['iou']:.4f} overlap={cand['overlap_pct']:.3f} speaker={cand['speaker']} {cand['start']:.3f}s-{cand['end']:.3f}s")

            # Decide how to handle overlaps
            if not overlaps:
                # No speakers: keep segment with speaker=None
                logger.debug(f"ASR {asr_idx}: No overlaps found, assigning speaker=None")
                result.append(
                    DiarizedSegment(
                        start=asr_seg.start,
                        end=asr_seg.end,
                        text=asr_seg.text,
                        speaker=None,
                    )
                )
            elif len(overlaps) == 1:
                # Single speaker: assign entire segment
                logger.debug(f"ASR {asr_idx}: Single overlap found, assigning speaker={overlaps[0]['speaker']}")
                result.append(
                    DiarizedSegment(
                        start=asr_seg.start,
                        end=asr_seg.end,
                        text=asr_seg.text,
                        speaker=overlaps[0]["speaker"],
                    )
                )
            else:
                # Multiple speakers: check for dominant speaker
                max_overlap = max(ovlp["overlap_pct"] for ovlp in overlaps)
                if max_overlap >= 0.7:
                    # Dominant speaker (>=70% coverage)
                    dominant = next(
                        ovlp for ovlp in overlaps if ovlp["overlap_pct"] == max_overlap
                    )
                    result.append(
                        DiarizedSegment(
                            start=asr_seg.start,
                            end=asr_seg.end,
                            text=asr_seg.text,
                            speaker=dominant["speaker"],
                        )
                    )
                else:
                    # No dominant speaker: proportional split
                    # Sort overlaps by start time to determine split order
                    overlaps_sorted = sorted(
                        overlaps, key=lambda x: x["start"]
                    )

                    # Find the proportional split point based on max overlap
                    max_overlap_speaker = overlaps_sorted[0]
                    split_point = min(asr_seg.end, max_overlap_speaker["end"])

                    # Split text proportionally
                    split_ratio = (split_point - asr_seg.start) / asr_duration
                    words = asr_seg.text.split()

                    if words:
                        split_word_idx = max(1, int(len(words) * split_ratio))
                        first_text = " ".join(words[:split_word_idx])
                        second_text = " ".join(words[split_word_idx:])

                        # Add first segment
                        result.append(
                            DiarizedSegment(
                                start=asr_seg.start,
                                end=split_point,
                                text=first_text,
                                speaker=max_overlap_speaker["speaker"],
                            )
                        )

                        # Add second segment with next speaker
                        if second_text.strip():
                            next_speaker = (
                                overlaps_sorted[1]["speaker"]
                                if len(overlaps_sorted) > 1
                                else max_overlap_speaker["speaker"]
                            )
                            result.append(
                                DiarizedSegment(
                                    start=split_point,
                                    end=asr_seg.end,
                                    text=second_text,
                                    speaker=next_speaker,
                                )
                            )
                    else:
                        # Empty text, keep as-is
                        result.append(
                            DiarizedSegment(
                                start=asr_seg.start,
                                end=asr_seg.end,
                                text=asr_seg.text,
                                speaker=max_overlap_speaker["speaker"],
                            )
                        )

        return result

    def merge_adjacent_same_speaker(
        self, segments: list[DiarizedSegment],
    ) -> list[DiarizedSegment]:
        """
        Merge adjacent segments with the same speaker.

        Args:
            segments: List of diarized segments

        Returns:
            List of merged segments
        """
        # DEBUG: Log merge input details
        logger.debug(
            "Diarization merge input",
            input_segments_count=len(segments),
            speakers=list(set(seg.speaker for seg in segments if seg.speaker)),
        )
        
        if not segments:
            return []

        merged = []
        current = segments[0]

        for next_seg in segments[1:]:
            # Merge if same speaker (not None) and adjacent (or very close)
            # Don't merge None speakers as they represent uncertainty
            if (
                current.speaker is not None
                and current.speaker == next_seg.speaker
                and abs(current.end - next_seg.start) < 0.01
            ):
                # Merge by extending current segment
                current = DiarizedSegment(
                    start=current.start,
                    end=next_seg.end,
                    text=f"{current.text} {next_seg.text}",
                    speaker=current.speaker,
                )
            else:
                # Different speaker or gap: save current and start new
                merged.append(current)
                current = next_seg

        # Add last segment
        merged.append(current)

        # DEBUG: Log merge output details
        logger.debug(
            "Diarization merge output",
            output_segments_count=len(merged),
            speakers=list(set(seg.speaker for seg in merged if seg.speaker)),
            reduction_ratio=len(segments) / len(merged) if merged else 0,
        )

        return merged


def get_diarizer(
    model_path: str = "data/models/speaker-diarization-community-1",
    require_cuda: bool = False,
    overlap_threshold: float = 0.1,
    embedding_batch_size: int = 32,
    segmentation_batch_size: int = 32,
) -> Optional[Diarizer]:
    """
    Factory function to create and return a Diarizer instance.

    Gracefully returns None if diarization is unavailable (e.g., no CUDA when required).

    Args:
        model_path: Path to the diarization model
        require_cuda: If True, returns None when CUDA unavailable
        overlap_threshold: IoU threshold for overlap detection
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
            overlap_threshold=overlap_threshold,
            embedding_batch_size=embedding_batch_size,
            segmentation_batch_size=segmentation_batch_size,
        )
        return diarizer
    except Exception as e:
        logger.error(f"Failed to create diarizer: {e}")
        return None

# Speaker Diarization Integration Plan for MAIE

## Overview

This document outlines a simple, performant plan to integrate speaker diarization into MAIE (Modular Audio Intelligence Engine) using pyannote.audio and a local, offline model. The approach favors minimal code, clear behavior, and easy maintenance.

## Background

### Current MAIE Capabilities

- Audio transcription using Whisper/ChunkFormer ASR backends
- Structured summarization with LLM enhancement
- Support for meetings, interviews, and other multi-speaker content
- Segment-level timestamps but no speaker identification

### Benefits of Diarization

- Speaker attribution for each segment
- More readable multi-speaker transcripts
- Better summaries with speaker context
- Automatic participant roster generation

### Technical Foundation

- Library: pyannote.audio (already in dependencies)
- Model: local `data/models/speaker-diarization-community-1` (present)
- Pattern: `Pipeline.from_pretrained(local_path, use_auth_token=False)` → `.to(device)`
- Integration point: after ASR, before LLM

## Minimal Device Handling

Keep device logic minimal and per-feature:

- Use a tiny helper for device selection and CUDA requirements.
- Gate diarization at the feature level; avoid global API hard-stops.
- Reuse a single diarization pipeline instance per process.

### Minimal Device Helper

```python
# src/util/device.py
import torch

def get_torch_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def require_cuda(feature_name: str = "feature") -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"{feature_name} requires a CUDA GPU")
```

Use `require_cuda("diarization")` when diarization is configured to require GPU (see config). vLLM already enforces CUDA for LLM where applicable.

## Diarization Processor (Lean)

```python
# src/processors/audio/diarizer.py
from pathlib import Path
from typing import Optional, List, Dict
from pyannote.audio import Pipeline
import torch

from src.util.device import get_torch_device, require_cuda

class DiarizationConfig:
    enabled: bool = True
    model_path: str = "data/models/speaker-diarization-community-1"
    require_cuda: bool = True

class SpeakerDiarizer:
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.pipeline = None
        self.device = get_torch_device()

    def _ensure_loaded(self) -> None:
        if self.pipeline is not None:
            return
        if self.config.require_cuda:
            require_cuda("diarization")
        self.pipeline = Pipeline.from_pretrained(
            self.config.model_path,
            use_auth_token=False,
        )
        self.pipeline.to(self.device)

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None):
        if not self.config.enabled:
            return None
        self._ensure_loaded()
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        diar = self.pipeline(audio_path, **kwargs)
        # Convert diarization to MAIE segments: [{start, end, speaker}]
        return diar

    def unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
```

## Alignment Strategy (Pragmatic IoU/Overlap)

Align diarization speaker segments with ASR segments using simple overlap/IoU heuristics. If Whisper word timestamps are available, prefer word-level alignment for better attribution; otherwise align at the segment level by maximum IoU.

### ASR Backends and Timestamps (MAIE)

- faster-whisper (WhisperBackend)
  - Start/end timestamps are float seconds per segment.
  - MAIE enables `word_timestamps=True` by default (see `settings.asr.whisper_word_timestamps`), which improves segment timing accuracy and exposes word timings inside faster-whisper. Our current ASRResult stores segment timings; if we extend it to include words, we will assign speakers per word using word midpoint contained in a diarization segment.
  - Default alignment: segment-level IoU with diarization segments.

- ChunkFormer (ChunkFormerBackend)
  - Timestamps are strings in the format `[HH:MM:SS.mmm]` per segment (e.g., `[00:01:05.500]`).
  - MAIE sets `chunkformer_return_timestamps=True` by default to include start/end in outputs. The normalization utility below converts bracketed strings to float seconds.
  - Default alignment: segment-level IoU with diarization segments.

Recommendation: run ASR and diarization sequentially on the same GPU to avoid contention and reduce OOM risk. Load diarization pipeline once per process and reuse.

Backend-specific caveats:
- faster-whisper segments are high quality when `word_timestamps=True`; without it, segment times can be skewed (we keep it enabled by default).
- ChunkFormer segments can sometimes lack timestamps in fallback paths; skip speaker assignment for segments with invalid or missing times.

## Token‑Efficient, Readable Timestamp Format

Goal: keep timestamps compact for LLM token budgets while remaining easy to read and parse. We adopt a canonical internal format and provide guidance for API responses.

### Canonical Internal Format

- Use float seconds with 2–3 decimals (centiseconds or milliseconds).
- Represent segment boundaries as a pair under a single key `t` to avoid repeating key names:

```json
{
  "t": [12.34, 15.67],
  "speaker": "S1",
  "text": "Hello there."
}
```

Rationale:
- Arrays are compact; avoiding repeated keys like `start`/`end` saves tokens.
- Float seconds are shorter than HH:MM:SS.mmm and trivial to parse.
- Keep `speaker` and `text` as full words for readability and likely better tokenization than uncommon abbreviations.

Precision guidelines:
- Default to 2 decimals (0.01s ≈ 10ms), matching typical diarization resolution.
- Use 3 decimals only when you need ms‑level precision.
- Trim trailing zeros when serializing (e.g., 12.30 → 12.3) if desired.

### Words (Optional)

When word timestamps are available (faster‑whisper), store words compactly under `w` with the same `t` format:

```json
{
  "t": [12.34, 15.67],
  "speaker": "S1",
  "text": "Hello there.",
  "w": [
    {"t": [12.34, 12.70], "text": "Hello"},
    {"t": [12.90, 13.20], "text": "there."}
  ]
}
```

### Speaker Labels

- Use short labels like `S1`, `S2`, … for compactness and readability.
- Keep a mapping in metadata if needed (e.g., `"speakers": {"S1": {...}}`).

### Backend Mappings

- faster‑whisper (float seconds): map `start`/`end` directly to `t`.
- ChunkFormer (bracket strings): normalize `[HH:MM:SS.mmm]` → float seconds, then map to `t`.

### API Response Modes (Optional)

- Storage/LLM mode (compact): use `t` arrays as above.
- UI mode (human‑friendly): render `mm:ss.s` strings for display only (do not store as primary).

### Timestamp Normalization Utilities

```python
# src/processors/audio/timestamp_utils.py
import re
from typing import Any, Dict

def normalize_timestamp(timestamp: Any) -> float:
    if isinstance(timestamp, (float, int)):
        return float(timestamp)
    if isinstance(timestamp, str):
        ts = timestamp.strip("[]")
        m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})", ts)
        if m:
            h, m_, s, ms = m.groups()
            return int(h) * 3600 + int(m_) * 60 + int(s) + int(ms) / 1000.0
        return float(ts)
    raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")

def normalize_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(segment)
    if "start" in out:
        out["start"] = normalize_timestamp(out["start"])
    if "end" in out:
        out["end"] = normalize_timestamp(out["end"])
    return out
```

### Overlap + Alignment

```python
from typing import List, Dict, Any

def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = overlap(a0, a1, b0, b1)
    if inter == 0.0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

def align_diarization_with_asr(
    diar_segments: List[Dict[str, Any]],
    asr_segments: List[Dict[str, Any]],
    overlap_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    diar = [normalize_segment(s) for s in diar_segments]
    asr = [normalize_segment(s) for s in asr_segments]
    aligned: List[Dict[str, Any]] = []
    for seg in asr:
        a0, a1 = seg["start"], seg["end"]
        best_speaker, best = None, 0.0
        for d in diar:
            score = iou(a0, a1, d["start"], d["end"])
            if score > best:
                best = score
                best_speaker = d.get("speaker")
        seg_out = dict(seg)
        seg_out["speaker"] = best_speaker if best >= overlap_threshold else None
        seg_out["speaker_confidence"] = best
        aligned.append(seg_out)
    return aligned
```

Integrated post-processing (required):
- Merge adjacent ASR segments with the same assigned `speaker` to reduce fragmentation. Preserve boundaries when punctuation or long pauses are meaningful, but by default, contiguous segments with identical speakers are merged.
- When overlaps are ambiguous (multiple diarization segments partially overlap), you can raise `overlap_threshold` to reduce false attributions.
- If ASR segments contain `None` or invalid timestamps (possible in ChunkFormer fallbacks), drop those segments or skip speaker assignment after logging.

Reference merging snippet (apply immediately after alignment):

```python
def merge_adjacent_same_speaker(aligned: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not aligned:
        return aligned
    merged: List[Dict[str, Any]] = []
    cur = dict(aligned[0])
    for seg in aligned[1:]:
        same_speaker = cur.get("speaker") is not None and cur.get("speaker") == seg.get("speaker")
        # Merge only if same speaker and adjacent (no gap) or tiny gap (<100ms)
        contiguous = abs(float(cur["end"]) - float(seg["start"])) <= 0.1
        if same_speaker and contiguous:
            cur["end"] = float(seg["end"])  # extend
            # Concatenate text with a space to preserve readability
            if cur.get("text") and seg.get("text"):
                cur["text"] = f"{cur['text'].rstrip()} {seg['text'].lstrip()}".strip()
            elif seg.get("text"):
                cur["text"] = seg["text"]
            # Update speaker_confidence to the min or weighted avg; here we keep max as a simple heuristic
            if "speaker_confidence" in cur and "speaker_confidence" in seg:
                cur["speaker_confidence"] = max(cur["speaker_confidence"], seg["speaker_confidence"]) 
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)
    return merged

# Usage:
aligned = align_diarization_with_asr(diar_segments, asr_segments, overlap_threshold=0.3)
aligned = merge_adjacent_same_speaker(aligned)
```

## Implementation Plan (Lean)

### API + Config

- Extend request schema to accept optional feature `speaker_diarization` and optional `num_speakers` hint.
- Extend transcript segment schema with optional `speaker` field. Responses remain backward compatible when the field is absent.
- Minimal settings (following existing config style):

```python
class DiarizationSettings(BaseModel):
    enabled: bool = True
    model_path: str = "data/models/speaker-diarization-community-1"
    overlap_threshold: float = 0.3
    require_cuda: bool = True
```

### Worker Integration

After ASR, when both the global setting and the per-request feature are enabled, run diarization and align speakers to ASR segments. If CUDA is unavailable and `require_cuda=True`, skip diarization (log clearly) and continue the pipeline.

### Testing

- Unit: toggle behavior (enabled/disabled), CUDA missing → diarization skipped when required, alignment on mixed timestamp formats.
- Integration: per-request feature flag runs/omits diarization; responses remain backward compatible.

## References and Research

- DER (Diarization Error Rate) is the primary diarization evaluation metric.
- RTTM is the common ground-truth format for evaluation.
- IoU/overlap is a practical alignment heuristic for merging diarization with ASR segments; it is not a diarization evaluation metric.

## Technical Considerations

- Local model usage: `Pipeline.from_pretrained(local_path, use_auth_token=False)` then `.to(device)` works offline with no HF token.
- Load once per process and reuse to avoid startup overhead.
- Prefer word timestamps when available for better alignment.
- MAIE defaults (confirmed):
  - faster-whisper word timestamps enabled by default for accurate timing: `src/config/model.py:116`
  - ChunkFormer returns timestamps enabled by default: `src/config/model.py:151`

### Performance Notes

- Alignment is O(N×M); with typical segment counts this overhead is negligible.
- Defer threshold tuning until DER baselines exist on your data.

## Success Metrics

- Establish a DER baseline on representative data; set targets after measurement.
- Track diarization overhead relative to ASR-only processing; optimize if regressions occur.
- Maintain API backward compatibility when diarization is skipped.

## Timeline (Lean)

- Week 1: Implement diarizer + alignment utils; unit tests; extend schemas (optional `speaker`).
- Week 2: Integrate with worker and per-request flag; add simple timing/DER measurement; document configuration and behavior.

## Conclusion

This integration enhances MAIE's multi-speaker handling with a simple, GPU-aware design: a small device helper, an opt-in diarizer reused per process, and pragmatic overlap-based alignment. We avoid heavy abstractions and global GPU gating, keeping the codebase clean, minimal, and performant.

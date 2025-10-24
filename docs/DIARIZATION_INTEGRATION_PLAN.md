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
# src/utils/device.py
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

The `SpeakerDiarizer` will be initialized with the application's Pydantic settings model for a single source of truth on configuration.

```python
# src/processors/audio/diarizer.py
from pathlib import Path
from typing import Optional, List, Dict, Any
from pyannote.audio import Pipeline
import torch

# Settings are sourced from a central, Pydantic-based config loader
from src.config.loader import DiarizationSettings
from src.utils.device import get_torch_device, require_cuda

class SpeakerDiarizer:
    def __init__(self, config: DiarizationSettings):
        self.config = config
        self.pipeline = None
        # Device selection is based on config to allow CPU-only operation
        self.device = get_torch_device(prefer_cuda=self.config.require_cuda)

    def _ensure_loaded(self) -> None:
        if self.pipeline is not None:
            return
        if self.config.require_cuda:
            require_cuda("diarization")
        
        self.pipeline = Pipeline.from_pretrained(
            self.config.model_path,
            use_auth_token=False, # Required for local-only models
        )
        self.pipeline.to(self.device)

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        if not self.config.enabled:
            return None
        
        self._ensure_loaded()
        
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        
        # Get Annotation object from pyannote
        annotation = self.pipeline(audio_path, **kwargs)
        
        # Convert to a simple list of dicts
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": str(speaker),
            })
        return segments

    def unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

## Alignment Strategy (Pragmatic IoU/Overlap)

Align diarization speaker segments with ASR segments using simple overlap/IoU heuristics. The primary goal is to assign a speaker to each ASR segment. When word timestamps are available, alignment is significantly more accurate.

### Text Splitting Strategies

When a single ASR segment contains multiple speakers, the transcript text must be split.

1.  **Word-level Splitting (Preferred):** If the ASR backend provides word-level timestamps (like `faster-whisper`), assign each word to a speaker based on which speaker turn its timestamp falls into. This is the most accurate method.
2.  **Proportional Splitting (Fallback):** If word timestamps are not available, split the ASR segment's text proportionally. The text (as a sequence of words) is partitioned based on the duration of each speaker's turn within the ASR segment. This is more robust than simple character-based splitting.

### Critical Edge Case: Long ASR Segments with Multiple Speakers

When a long ASR segment contains multiple speakers, we partition the segment to ensure the entire transcript is preserved and correctly attributed.

**Scenario**: ASR segment `[10.0s - 30.0s] "hello there how are you doing today i think we should discuss the project timeline"`
- Diarization detects: Speaker A `[10.0s - 18.0s]`, Speaker B `[18.5s - 30.0s]`

**Result**: The text is split proportionally. Two sub-segments are created:
1. `[10.0s - 18.0s] "hello there how are you doing today"` → Speaker A
2. `[18.5s - 30.0s] "i think we should discuss the project timeline"` → Speaker B

### ASR Backends and Timestamps (MAIE)

- faster-whisper (WhisperBackend)
  - Start/end timestamps are float seconds per segment.
  - MAIE enables `word_timestamps=True` by default, making word-level alignment the primary strategy.
- ChunkFormer (ChunkFormerBackend)
  - Timestamps are strings `[HH:MM:SS.mmm]`. No word timestamps are available.
  - Proportional splitting will be used as a fallback.

Recommendation: run ASR and diarization sequentially on the same GPU to avoid contention and reduce OOM risk. Load diarization pipeline once per process and reuse.

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

### Timestamp Normalization Utilities

The following utilities will be in `src/processors/audio/timestamp_utils.py`.

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

### Overlap + Alignment Logic

The alignment logic will be co-located with the `SpeakerDiarizer`. The following functions provide a robust way to assign speakers, handling both single-speaker and multi-speaker ASR segments.

```python
# In src/processors/audio/diarizer.py
from typing import List, Dict, Any

def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
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
        text = seg.get("text", "")
        
        # Find all overlapping speakers
        overlapping_speakers = []
        for d in diar:
            score = _iou(a0, a1, d["start"], d["end"])
            if score >= overlap_threshold:
                overlapping_speakers.append({**d, "score": score})

        # PREFERRED: If seg has word-timestamps, use those to create new segments.
        # (Implementation omitted for brevity, but involves assigning each word to a speaker
        # and then merging consecutive words from the same speaker.)

        # FALLBACK: Proportional splitting for segments without word-timestamps.
        if not overlapping_speakers:
            aligned.append({**seg, "speaker": None, "speaker_confidence": 0.0})
        elif len(overlapping_speakers) == 1:
            speaker_info = overlapping_speakers[0]
            aligned.append({**seg, "speaker": speaker_info["speaker"], "speaker_confidence": speaker_info["score"]})
        else:
            # MULTIPLE SPEAKERS: Partition the text proportionally.
            clipped_speakers = []
            for spk in overlapping_speakers:
                overlap_start = max(a0, spk["start"])
                overlap_end = min(a1, spk["end"])
                if overlap_end > overlap_start:
                    clipped_speakers.append({
                        **spk, "start": overlap_start, "end": overlap_end,
                        "duration": overlap_end - overlap_start
                    })
            
            if not clipped_speakers: # handle edge case
                aligned.append({**seg, "speaker": None, "speaker_confidence": 0.0})
                continue

            clipped_speakers.sort(key=lambda x: x["start"])

            words = text.split()
            total_speaker_duration = sum(c["duration"] for c in clipped_speakers)
            remaining_words = list(words)

            if total_speaker_duration > 0 and remaining_words:
                # Iterate and assign word chunks to each speaker except the last.
                for spk in clipped_speakers[:-1]:
                    proportion = spk["duration"] / total_speaker_duration
                    num_words_for_speaker = round(len(words) * proportion)
                    
                    speaker_words = remaining_words[:num_words_for_speaker]
                    if not speaker_words: continue
                    
                    aligned.append({
                        "start": spk["start"], "end": spk["end"], "text": " ".join(speaker_words),
                        "speaker": spk["speaker"], "speaker_confidence": spk["score"]
                    })
                    remaining_words = remaining_words[num_words_for_speaker:]

                # Assign all leftover words to the last speaker to ensure no text is lost.
                last_spk = clipped_speakers[-1]
                if remaining_words:
                    aligned.append({
                        "start": last_spk["start"], "end": last_spk["end"], "text": " ".join(remaining_words),
                        "speaker": last_spk["speaker"], "speaker_confidence": last_spk["score"]
                    })
    return aligned
```

Integrated post-processing (required):
- Merge adjacent ASR segments with the same assigned `speaker` to reduce fragmentation.

```python
def merge_adjacent_same_speaker(aligned: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not aligned:
        return aligned
    merged: List[Dict[str, Any]] = []
    cur = dict(aligned[0])
    for seg in aligned[1:]:
        same_speaker = cur.get("speaker") is not None and cur.get("speaker") == seg.get("speaker")
        contiguous = abs(float(cur["end"]) - float(seg["start"])) <= 0.1
        if same_speaker and contiguous:
            cur["end"] = float(seg["end"])
            if cur.get("text") and seg.get("text"):
                cur["text"] = f"{cur['text'].rstrip()} {seg['text'].lstrip()}".strip()
            elif seg.get("text"):
                cur["text"] = seg["text"]
            if "speaker_confidence" in cur and "speaker_confidence" in seg:
                cur["speaker_confidence"] = max(cur["speaker_confidence"], seg["speaker_confidence"]) 
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)
    return merged
```

## Implementation Plan (Lean)

### API + Config

- Extend request schema to accept optional feature `speaker_diarization` and optional `num_speakers` hint.
- Extend transcript segment schema with optional `speaker` field.
- Minimal settings (following existing config style):

```python
class DiarizationSettings(BaseModel):
    enabled: bool = Field(default=True)
    model_path: str = Field(default="data/models/speaker-diarization-community-1")
    overlap_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    require_cuda: bool = Field(default=True)
    
    model_config = ConfigDict(validate_assignment=True)
```

Add to `AppSettings`:
```python
diarization: DiarizationSettings = Field(default_factory=DiarizationSettings)
```

### Worker Integration

After ASR, if diarization is enabled, run the alignment and merging steps. If CUDA is unavailable and `require_cuda=True`, skip diarization and log a warning.

### Testing

- Unit: toggle behavior, CUDA missing, alignment logic on mixed timestamp formats.
- Integration: per-request feature flag runs/omits diarization.
- GPU tests: Use `@pytest.mark.gpu` marker for tests requiring CUDA hardware.
- Mock strategy: Mock `pyannote.audio.Pipeline` to return a fake `Annotation` object.

## TDD Implementation Strategy

Follow Test-Driven Development approach:

1. **Unit Tests First**: Write tests for each component before implementation.
2. **Mock Strategy**: Mock `pyannote.audio.Pipeline` and `torch.cuda` for fast unit tests.
3. **Integration Tests**: Use a real pyannote model with the `@pytest.mark.gpu` marker.
4. **File Structure**: To keep the design minimal, alignment and formatting logic will be co-located with the diarizer.
   - `src/utils/device.py` (if no existing equivalent is found)
   - `src/processors/audio/diarizer.py` (to contain main diarization, alignment, and formatting logic)
   - `src/processors/audio/timestamp_utils.py`

## Conclusion

This integration enhances MAIE's multi-speaker handling with a simple, GPU-aware design: a small device helper, an opt-in diarizer reused per process, and pragmatic overlap-based alignment. By co-locating related logic, we keep the codebase clean, minimal, and performant.

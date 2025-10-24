# Diarization: Overview & Implementation Guide

This consolidated guide replaces: DIARIZATION_READY, DIARIZATION_VISUAL_SUMMARY, DIARIZATION_IMPLEMENTATION_GUIDE, DIARIZATION_CHECKLIST, and DIARIZATION_REVIEW (kept for archive). It covers design, architecture, backend differences, implementation, integration, testing, and a condensed checklist.

## Executive Overview

- Goal: Attribute speakers to ASR segments, then feed a compact speaker-aware transcript to the LLM.
- Philosophy: Diarization is an enhancement. Never fail a job because of it; skip gracefully when unavailable.
- Scope: Works with both FasterWhisper and ChunkFormer ASR backends via a unified alignment pipeline.

## Architecture

```
Audio → Preprocess → ASR → Diarization → Alignment → Merge → LLM
                                     ^                 ^
                                     |                 |
                                   pyannote        Speaker-aware
                                   spans           transcript
```

### Modules
- utils/device.py: Torch device helpers (CUDA detection, selection).
- processors/audio/diarizer.py: Pyannote pipeline + alignment + merging.
- worker/pipeline.py: Integrates diarization between ASR and LLM.

## ASR Backends and Formats

Two supported backends with different timestamp characteristics:

- FasterWhisper (Whisper adapter)
  - Segment timestamps: float seconds.
  - Adapter output today: `[{start: float, end: float, text: str}]` (no word-level timestamps).

- ChunkFormer
  - Segment timestamps: adapter normalizes `HH:MM:SS.mmm` (or `[HH:MM:SS.mmm]`) to float seconds.
  - Adapter maps `decode` → `text`.
  - Word-level timestamps: not available.

## Alignment Algorithm (IoU-based)

For each ASR segment [start,end]:
- Find diarization spans overlapping above threshold (default 0.3).
- If no speakers: keep segment with speaker=None.
- If one speaker: assign speaker directly.
- If multiple speakers: apply split policy without word timestamps:
  - If one speaker dominates (≥ 0.7 of segment duration): assign entire segment to that speaker (no split).
  - Otherwise, create at most two subsegments using a proportional split point (by time).
- Merge adjacent segments when speaker is the same and gap is small (<0.1s).

Properties:
- Ensures no words are lost in proportional splitting (leftover to last speaker).
- Reduces micro-turns and token count via dominant assignment and single-split policy.

## Error Handling (Graceful by Design)

- CUDA required but not available → log warning, return None (skip diarization).
- Model path missing or load failure → log error, return None.
- Inference/alignment exceptions → log, return best-effort results; never block LLM.

## Configuration

```env
APP_DIARIZATION__ENABLED=true
APP_DIARIZATION__MODEL_PATH=data/models/speaker-diarization-community-1
APP_DIARIZATION__OVERLAP_THRESHOLD=0.3
APP_DIARIZATION__REQUIRE_CUDA=false
```

`DiarizationSettings` is added under `AppSettings` in config.

## Integration (Worker Pipeline)

Insert after ASR and before LLM:

```python
# Get ASR segments
asr_segments = transcription_result.segments

diarized_segments = asr_segments
if settings.diarization.enabled:
    try:
        diarizer = get_diarizer()
        if diarizer:
            diar_segs = diarizer.diarize(processing_audio_path, num_speakers=None)
            if diar_segs:
                diarized_segments = diarizer.align_diarization_with_asr(diar_segs, asr_segments)
                diarized_segments = diarizer.merge_adjacent_same_speaker(diarized_segments)
                logger.info("Diarization applied", num_segments=len(diarized_segments))
    except Exception as e:
        logger.warning("Diarization failed; continuing without speaker info", error=str(e))

transcription_result.segments = diarized_segments
```

Then render for LLM using the packaging spec (see DIARIZATION_PROMPT_PACKAGING.md).

## Testing Strategy (Condensed)

- Unit tests:
  - device helpers: CUDA present/absent behavior.
  - alignment: single/no/multi speaker scenarios; proportional split preserves all words; merging behavior.
- Integration tests (GPU-marked): optional E2E with real pyannote model.

## Condensed Checklist

- Foundation
  - [ ] utils/device.py implemented + tests
- Core Diarization
  - [ ] speakers via pyannote; lazy load; graceful skips
- [ ] align: IoU + proportional splitting only (no word-level)
  - [ ] merge adjacent same-speaker spans
- Integration
  - [ ] pipeline: diarize → align → merge between ASR and LLM
  - [ ] config: DiarizationSettings wired
- Packaging
  - [ ] render `mm:ss-mm:ss S#: text` with one meta header
  - [ ] round to 0.5s
- Quality
  - [ ] unit tests pass; coverage ≥85%
  - [ ] no regressions; graceful behavior without GPU/model

## Design Rationale Highlights

- Enhancement, not dependency → job resilience
- Per-backend strategy → accuracy where possible, graceful where not
- Minimal surface to LLM → token efficient, human-readable transcript

See also: DIARIZATION_PROMPT_PACKAGING.md for the exact LLM input format.

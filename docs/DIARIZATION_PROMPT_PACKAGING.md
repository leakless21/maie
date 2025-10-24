# Diarization Prompt Packaging (Token-Efficient Format)

This spec defines how to transform ASR + diarization outputs into a compact, readable prompt for LLMs. It accounts for both FasterWhisper and ChunkFormer backends, and assumes no word-level timestamps are available from our adapters.

## Goals

- Keep dialog readable with clear speakers and time ranges.
- Minimize tokens: avoid verbose keys, floats, and repeated metadata.
- Work seamlessly regardless of ASR backend.

## ASR Backend Outputs (Observed)

The repo backends produce slightly different raw segment shapes before normalization. Below are the shapes verified from the current code.

### Whisper (faster-whisper adapter)

Source: `src/processors/asr/whisper.py` builds `segments_dict` with `start`, `end`, `text` only (no word list in the adapter at present).

```json
{
  "start": 0.0,
  "end": 1.5,
  "text": "hello world from mock"
}
```

Notes:
- Timestamps are float seconds.
- The adapter does not expose word-level timestamps currently.

### ChunkFormer

Source: `src/processors/asr/chunkformer.py` normalizes various return shapes to a list of dict segments with `start`, `end`, `text` (it also maps `decode` → `text`). Timestamps are normalized to float seconds even when the model returns strings like `HH:MM:SS.mmm` or `[HH:MM:SS.mmm]`.

Normalized example:

```json
{
  "start": 0.0,
  "end": 1.5,
  "text": "chunkformer mock transcript"
}
```

Notes:
- Language may be present at the top level (not per-segment).
- No word-level timestamps; treat segments as plain spans.

## Normalized Segment Schema (Target)

### ASR Segment (normalized)

```json
{
  "start": 12.5,
  "end": 16.2,
  "text": "Thanks for joining today."
}
```

Notes:
- Whisper adapter: float seconds; no `words` propagated today.
- ChunkFormer: adapter now normalizes `start`/`end` to float seconds.

### Diarizer Span (normalized)

```json
{
  "start": 12.4,
  "end": 16.4,
  "speaker": "S1"
}
```

### Aligned Speaker Segment (worker output)

```json
{
  "start": 12.5,
  "end": 16.2,
  "speaker": "S1",
  "text": "Thanks for joining today."
}
```

If word-level timestamps are present, words may be annotated with `speaker` as well; this is optional for LLM prompts and retained for analytics.

## Normalization Plan

1) Canonical keys
- Ensure every segment has keys: `start: float`, `end: float`, `text: str`.
- For ChunkFormer: map `decode` → `text` (already done in the backend).

2) Timestamp normalization
- If `start`/`end` are strings like "HH:MM:SS.mm" or "[HH:MM:SS.mm]" (or mmm), parse to float seconds.
- If ints, cast to float.
- If floats, keep as-is.

3) Optional fields
- Do not include `words` in LLM input (none are available in current adapters). If we extend Whisper later, keep words in analytics only.

4) Safety
- Drop segments with empty/whitespace-only `text`.
- Clip negative or NaN timestamps; enforce `0 <= start < end`.

## Confidence Availability

Confidence/score metrics are not used. Our adapters do not expose reliable confidence values; omit them from outputs and prompts.

## Split Policy Without Word Timestamps

When multiple diarizer speakers overlap an ASR segment and word-level timestamps are unavailable:

- Dominant assignment: If the max-overlap speaker covers ≥ 0.7 of the ASR segment duration, assign the entire segment to that speaker (no split). This keeps turns readable and reduces token count.
- Otherwise, single-split policy: Create at most two subsegments using a proportional split point (by time). Assign the earlier portion to the earliest-starting overlapping speaker, and the remainder to the next.
- Never create more than two subsegments from a single ASR span.

## Merging and Smoothing

- Merge adjacent segments with the same speaker.
- Min turn duration: if a turn is shorter than 1.0s, merge it with its nearest neighbor (prefer the neighbor with the larger overlap window). This reduces micro-turn noise.
- Timestamp rounding for display: round to nearest 0.5s; display as `mm:ss` or `mm:ss.m` (one decimal) only when needed.

## Rendering Rules (LLM Input)

- Merge adjacent segments with the same speaker before rendering.
- One metadata header per transcript; then line-per-turn entries.
- Time format: `mm:ss` or `mm:ss.m` only if needed (round to nearest 0.5s by default).
- Line format: `mm:ss-mm:ss S#: text` (use `S1`, `S2`, ...; do not spell out names).
- Do not include confidence markers in lines.
- Do not include word-level timestamps in the prompt (retain only in JSON for analytics).

## Prompt Template

````markdown
Meta: job_id=<id> | lang=<en> | asr=<whisper-large-v3|chunkformer-X> | diarizer=<pyannote/...>
Transcript:
<lines>
````

Example:

````markdown
Meta: job_id=9012 | lang=en | asr=whisper-large-v3 | diarizer=pyannote/speaker-diarization
Transcript:
00:00-00:05 S1: Thanks for joining everyone.
00:05-00:09 S2: Happy to be here.
00:09-00:15 S1: Agenda today is launch readiness.
````

## Backend-Specific Considerations

- FasterWhisper
  - No `words` field in current adapter; use the split policy above when segments overlap with multiple speakers.
  - Merge adjacent same-speaker turns and render.

- ChunkFormer
  - No `words` field; use the split policy above.
  - Ensure all words are preserved; assign rounding leftovers to last speaker.
  - Then merge adjacent same-speaker turns and render.

## Optional JSONL for Retrieval Tasks

When structured ingestion is needed (e.g., embedding pipelines), produce a JSONL variant alongside the human-readable transcript:

```jsonl
{"t":"00:00-00:05","s":"S1","x":"Thanks for joining everyone."}
{"t":"00:05-00:09","s":"S2","x":"Happy to be here."}
{"t":"00:09-00:15","s":"S1","x":"Agenda today is launch readiness."}
```

This is not used in prompts by default (token-heavy), but useful for storage/search.

## Quality Notes

- Roundtime: nearest 0.5s saves tokens while preserving readability.
- Only include speaker when known; if unknown, render `S?` or omit the speaker prefix (choose one policy and keep it consistent).
- Keep the transcript block separate from instructions so the system prompt can be reused unchanged.

# Audio Metrics Review for Silero VAD Integration

## Executive Summary

After analyzing the codebase and official Silero VAD documentation, **the current `AudioMetricsCollector` in `src/processors/audio/metrics.py` contains THREE functions that are NOT compatible with your Silero VAD implementation**:

1. ❌ `calculate_vad_coverage()` - Uses naive energy-based VAD (custom implementation)
2. ❌ `calculate_confidence()` - Uses generic signal power/STD heuristics
3. ❌ `validate_audio_properties()` - Valid audio metadata extraction

**Reality**: Your system uses **faster-whisper's built-in Silero VAD** (via `vad_filter=True`), which provides its own metrics internally.

---

## Current Architecture

### How VAD is Actually Used

```yaml
Pipeline Flow:
  Audio Input
    ↓
  AudioPreprocessor (ffprobe + ffmpeg normalization)
    ↓
  ASRFactory.create_with_audio_processing()
    ├─ asr_processor (WhisperBackend)
    ├─ audio_preprocessor
    └─ audio_metrics_collector (UNUSED)
    ↓
  WhisperBackend.execute()
    └─ model.transcribe(vad_filter=True)  ← Silero VAD runs HERE
    ↓
  Returns: ASRResult with basic metrics (vad_coverage=0.0, asr_confidence_avg=0.0)
```

**Key Finding**: `AudioMetricsCollector` is instantiated but **never called**. The pipeline fills placeholder values:

- `vad_coverage=0.0` (hardcoded, no actual Silero VAD metrics extracted)
- `asr_confidence_avg=0.0` (hardcoded, Whisper doesn't expose per-segment confidence)

---

## Silero VAD Technical Analysis

### What Silero VAD Provides

From official documentation ([snakers4/silero-vad](https://github.com/snakers4/silero-vad)):

**Output from `get_speech_timestamps()`:**

```python
speech_timestamps = [
    {"start": 0.5, "end": 3.2},  # milliseconds or seconds (depending on return_seconds=True)
    {"start": 5.1, "end": 8.9},
]
```

**Available Parameters:**

- `threshold` (default: 0.5) - confidence threshold for speech detection (0-1)
- `min_speech_duration_ms` (default: 250) - minimum speech segment length
- `min_silence_duration_ms` (default: 100) - minimum silence between segments
- `max_speech_duration_s` - maximum speech segment length
- `speech_pad_ms` - padding to add to speech segments
- `return_seconds` - return times in seconds (vs samples)

**Per-Frame Output** (advanced API):

```python
# Get speech probabilities for each frame (not directly exposed in main API)
# Model outputs: (batch, time_steps, classes)
# Can extract confidence per speech segment from model internals
```

### What Silero VAD Does NOT Provide (via faster-whisper)

❌ **Direct confidence scores per segment** - Model only provides binary classification (speech/no-speech)
❌ **Frame-level probabilities** - Only available if you use ONNX runtime directly and parse model outputs
❌ **Speaker information** - No diarization, just voice activity detection

---

## Analysis of Current `AudioMetricsCollector`

### Function 1: `calculate_vad_coverage()`

**Current Implementation:**

```python
def calculate_vad_coverage(self, audio_path: str) -> float:
    # Uses naive energy-based VAD on 20ms frames
    # Threshold = 0.1 * standard deviation
    # Returns ratio of active frames
```

**Problems:**

- ✗ **Does NOT use Silero VAD** - reimplements naive energy-based detection
- ✗ **Inaccurate for real audio** - hardcoded 0.1x std threshold too simplistic
- ✗ **Cannot be called during pipeline** - operates on audio file, returns static value
- ✗ **Redundant** - Silero VAD inside faster-whisper already calculated this

**Should Be Replaced With:**

```python
def extract_vad_coverage_from_whisper_segments(self, segments: List[dict]) -> float:
    """Extract VAD coverage from Whisper transcription segments.

    Args:
        segments: Whisper transcription result segments (includes timing info)

    Returns:
        Ratio of speech time to total audio duration
    """
    if not segments:
        return 0.0

    total_duration = max(seg.get('end', 0) for seg in segments)
    speech_duration = sum(seg.get('end', 0) - seg.get('start', 0)
                         for seg in segments)

    return speech_duration / total_duration if total_duration > 0 else 0.0
```

---

### Function 2: `calculate_confidence()`

**Current Implementation:**

```python
def calculate_confidence(self, audio_path: str) -> float:
    # Signal power / 100.0
    # Also considers STD / 1000.0
    # Heuristic-based, arbitrary constants
```

**Problems:**

- ✗ **Arbitrary heuristics** - Constants (100.0, 1000.0) have no basis
- ✗ **Not from Silero VAD** - Computes generic audio signal metrics
- ✗ **Misleading semantics** - Returns value called "confidence" but isn't speech confidence
- ✗ **Cannot get true ASR confidence** - Whisper (via faster-whisper v1.2.0) does NOT expose per-segment confidence scores

**What Can Be Extracted Instead:**

1. **From Silero VAD speech timestamps:**

   - Segment count and duration patterns
   - Speech/silence ratio ← actual VAD metric

2. **From Whisper segments:**
   - `no_speech_prob` field (available in newer versions, may not be in 1.2.0)
   - Average `prob` field if exposed (confidence per token)

**Should Be:**

```python
def extract_asr_confidence_from_segments(self, segments: List[dict]) -> Dict[str, float]:
    """Extract available confidence metrics from ASR segments.

    Returns:
        Dict with:
        - no_speech_prob: Average probability audio is non-speech
        - segment_count: Number of speech segments detected
        - avg_segment_duration: Average duration of speech segments
    """
    # Depends on Whisper version and available fields
```

---

### Function 3: `validate_audio_properties()`

**Current Implementation:**

```python
def validate_audio_properties(self, audio_path: str) -> Dict[str, Any]:
    # Loads audio with scipy.io.wavfile
    # Checks: sample rate (16kHz), mono, duration (0.1s - 1h)
    # Returns validation dict with issues
```

**Assessment:** ✅ **PARTIALLY VALID** but **NOT NEEDED** in pipeline

**Why It's Redundant:**

- `AudioPreprocessor._probe_audio()` already does this via ffprobe
- `AudioPreprocessor._needs_normalization()` already validates sample rate and channels
- `AudioPreprocessor.preprocess()` already validates duration (>1.0s)

**Redundancy in Pipeline:**

```
preprocess()
  ├─ _probe_audio()  ← extracts same metadata
  ├─ validates duration
  └─ _needs_normalization()  ← checks sample rate + channels

calculate_audio_properties()  ← DUPLICATE WORK
  └─ scipy.io.wavfile.read()  ← reads entire file again
```

**Impact:**

- ✗ Reads audio file twice (performance cost)
- ✗ Uses scipy instead of ffprobe (different parsing logic, potential inconsistencies)
- ✗ Never called in current pipeline anyway

---

## Recommendations

### Immediate Actions (Safe Removals)

#### 1. **Remove `AudioMetricsCollector` class entirely** ⚠️

```bash
# File: src/processors/audio/metrics.py
# Action: DELETE entire file (all 3 methods are unused/invalid)
```

**Reasoning:**

- No method is called in the current pipeline
- None of them correctly use Silero VAD
- `validate_audio_properties()` work is already done by `AudioPreprocessor`

#### 2. **Update `ASRFactory.create_with_audio_processing()`** ⚠️

```python
# Before
return {
    "asr_processor": asr_processor,
    "audio_preprocessor": AudioPreprocessor(),
    "audio_metrics_collector": AudioMetricsCollector(),  # ← Remove
}

# After
return {
    "asr_processor": asr_processor,
    "audio_preprocessor": AudioPreprocessor(),
}
```

#### 3. **Remove import from `src/processors/asr/factory.py`**

```python
# Remove this line:
from src.processors.audio.metrics import AudioMetricsCollector
```

---

### Future: Extract Real Metrics from Whisper

If you need actual VAD metrics, modify the pipeline to extract them from Whisper results:

```python
# src/processors/asr/whisper.py - in execute() method

def execute(self, audio_path: str, **kwargs) -> ASRResult:
    """Execute transcription and extract VAD metrics."""

    if self.model is None:
        raise RuntimeError("Model not loaded")

    transcribe_kwargs = self._prepare_transcribe_kwargs(kwargs)
    segments, info = self.model.transcribe(audio_path, **transcribe_kwargs)

    # Convert to list to enable multiple iterations
    segments_list = list(segments)

    # ✅ Extract real metrics from Whisper results
    vad_coverage = self._calculate_vad_coverage_from_segments(segments_list)

    # Build result with actual VAD metrics
    return ASRResult(
        transcript="\n".join([seg.get("text", "").strip() for seg in segments_list]),
        segments=segments_list,
        vad_coverage=vad_coverage,  # ← Real value from Silero VAD
        asr_confidence_avg=0.0,      # ← Whisper doesn't expose this
        language=info.language,
        duration=info.duration,
        vad_filter=transcribe_kwargs.get("vad_filter", False),
        library="faster-whisper",
    )

def _calculate_vad_coverage_from_segments(self, segments: List[dict]) -> float:
    """Calculate VAD coverage from transcription segments."""
    if not segments:
        return 0.0

    total_duration = max(seg.get('end', 0.0) for seg in segments)
    if total_duration == 0:
        return 0.0

    speech_duration = sum(seg.get('end', 0.0) - seg.get('start', 0.0)
                         for seg in segments)

    return min(1.0, speech_duration / total_duration)
```

---

## Summary Table

| Component                                   | Current                | Status            | Recommendation                                        |
| ------------------------------------------- | ---------------------- | ----------------- | ----------------------------------------------------- |
| `calculate_vad_coverage()`                  | Energy-based heuristic | ❌ Wrong approach | **Remove**                                            |
| `calculate_confidence()`                    | Signal power heuristic | ❌ Wrong approach | **Remove**                                            |
| `validate_audio_properties()`               | WAV metadata check     | ⚠️ Redundant      | **Remove** (done by `AudioPreprocessor`)              |
| `AudioMetricsCollector` (class)             | All 3 methods          | ❌ Unused         | **Remove entire class**                               |
| `ASRFactory.create_with_audio_processing()` | Returns collector      | ⚠️ Outdated       | **Remove `audio_metrics_collector` from return dict** |
| **Real VAD metrics**                        | Hardcoded 0.0          | ❌ Not extracted  | **Implement** extraction from Whisper segments        |

---

## Files to Modify

1. **`src/processors/audio/metrics.py`** → **DELETE**
2. **`src/processors/asr/factory.py`**
   - Remove `AudioMetricsCollector` import
   - Remove `audio_metrics_collector` from return dict
3. **`src/processors/asr/whisper.py`** (optional, for future enhancement)
   - Add `_calculate_vad_coverage_from_segments()` method
   - Update `execute()` to populate `vad_coverage` from segments

---

## References

- **Silero VAD Docs**: https://github.com/snakers4/silero-vad
- **faster-whisper v1.2.0**: Uses Silero VAD internally when `vad_filter=True`
- **Current Implementation**: `whisper_vad_filter` defaults to `True` in `src/config/model.py:107`

---

## Next Steps

1. ✅ Review this analysis
2. 🔧 Delete `src/processors/audio/metrics.py`
3. 🔧 Update `src/processors/asr/factory.py`
4. (Optional) Implement real VAD metric extraction in `WhisperBackend.execute()`
5. ✅ Update `unused-code-analysis.md` with final status

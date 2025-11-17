# PyAnnote-Audio 4.x Migration Guide

**Date**: November 17, 2025  
**Current Status**: pyannote-audio 4.0.1 released (September 29, 2025)  
**PyAnnote-Core Requirement**: pyannote-core 6.x

---

## Executive Summary

PyAnnote-Audio 4.0 is a **major breaking change** from 3.x. The migration requires attention to several key areas:

1. **Authentication**: `use_auth_token` → `token`
2. **Pipeline Loading**: `Pipeline.from_pretrained()` API unchanged but dependency changes affect behavior
3. **Audio I/O**: `soundfile`/`sox` backends removed (only `ffmpeg` + in-memory supported)
4. **Device Assignment**: `pipeline.to(device)` works similarly but tracks more attributes
5. **Output Format**: Diarization output now includes `exclusive_speaker_diarization`
6. **Clustering**: New VBx clustering by default (replaces agglomerative hierarchical)
7. **Python Support**: Drop Python < 3.10 support
8. **Namespace Package**: Switch to native namespace package
9. **Cache System**: `PYANNOTE_CACHE` removed, uses `huggingface_hub` cache directory

---

## 1. Breaking Changes in PyAnnote-Audio 4.0

### 1.1 Authentication Parameter Rename

**Before (3.x):**

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HUGGINGFACE_ACCESS_TOKEN"
)
```

**After (4.x):**

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",  # New model!
    token="HUGGINGFACE_ACCESS_TOKEN"  # Renamed parameter
)
```

**Impact**: Pipelines will fail to load if using `use_auth_token` parameter.

---

### 1.2 Revision Syntax Change

**Before (3.x):**

```python
# Using @ syntax for revision
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@v3.1",
    use_auth_token="token"
)
```

**After (4.x):**

```python
# Use explicit revision parameter
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    revision="main",  # Explicit parameter
    token="token"
)
```

---

### 1.3 Audio I/O Backend Removal

**Breaking Change**: Support for `sox` and `soundfile` backends removed. Only `ffmpeg` and in-memory audio supported.

**Before (3.x):**

```python
# Could use various backends
audio = Audio(backend="soundfile")  # No longer works
```

**After (4.x):**

```python
# Only ffmpeg or in-memory
# PyTorch switched from torchaudio to torchcodec
# Must have ffmpeg installed on system:
# Ubuntu: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

**Action Required**:

- Ensure `ffmpeg` is installed on all systems
- Remove any backend configuration
- Switch to in-memory audio processing or ensure ffmpeg is available

---

### 1.4 Python Version Support

**Before (3.x)**: Supported Python 3.8+

**After (4.x)**: **Requires Python 3.10+**

Update your `pyproject.toml`:

```toml
[project]
requires-python = ">=3.10"
```

---

### 1.5 Removed Tasks and Pipelines

**Removed:**

- `OverlappedSpeechDetection` task (merged into `SpeakerDiarization`)
- `OverlappedSpeechDetection` pipeline
- `Resegmentation` unmaintained pipeline
- `pyannote-audio-train` CLI command

If you use these, migrate to the main `SpeakerDiarization` pipeline/task.

---

### 1.6 Cache Directory Change

**Before (3.x):**

```python
# Custom cache using PYANNOTE_CACHE
os.environ["PYANNOTE_CACHE"] = "/custom/cache"
```

**After (4.x):**

```python
# Now uses huggingface_hub cache directory
# Set via environment variable:
os.environ["HF_HOME"] = "/custom/cache"  # For huggingface_hub

# Or use default: ~/.cache/huggingface/hub/
```

---

### 1.7 Model Instantiation Changes

**Breaking Change**: `Inference` class now only supports already instantiated models.

**Before (3.x):**

```python
from pyannote.audio import Inference

# Could pass model_name directly
inference = Inference("path/to/model")
```

**After (4.x):**

```python
from pyannote.audio import Inference
from pyannote.audio import Model

# Must instantiate model first
model = Model.from_pretrained("path/to/model")
inference = Inference(model)  # Pass instantiated model
```

---

### 1.8 Training Configuration Changes

If you train custom models, these options are **removed**:

- `multilabel` training in `SpeakerDiarization` task
- `warm_up` option
- `weigh_by_cardinality` option
- `vad_loss` option

Update any custom training scripts accordingly.

---

## 2. Updated Pipeline.from_pretrained() API

### 2.1 Full API Signature (4.x)

```python
from pyannote.audio import Pipeline
import torch

# Load pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="HUGGINGFACE_ACCESS_TOKEN",  # Changed from use_auth_token
    revision="main",  # Optional: specify model revision
    cache_dir=None,  # Uses HF_HOME by default
    force_download=False,
    proxies=None,
    resume_download=None,
    local_files_only=False,
    skip_check=False,  # New: skip dependency check
)

# Send to GPU
pipeline.to(torch.device("cuda"))

# Use pipeline
output = pipeline("/path/to/audio.wav")
```

### 2.2 New Model: pyannote/speaker-diarization-community-1

The 3.x model `speaker-diarization-3.1` is now legacy. The new 4.x model brings:

- **VBx Clustering**: Better speaker assignment and counting
- **Exclusive Diarization**: New output field for easier ASR alignment
- **Performance**: 2-3x faster processing
- **Better Accuracy**: Significant improvements across benchmarks

**Available via:**

- HuggingFace Hub: `pyannote/speaker-diarization-community-1`
- **PyAnnoteAI Premium**: `pyannote/speaker-diarization-precision-2` (requires pyannoteai-sdk)

---

## 3. Output Format Changes

### 3.1 Regular Diarization Output

**Format Unchanged** - Diarization still returns `Annotation` instances:

```python
output = pipeline("audio.wav")

# Iterate over speaker segments
for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
    print(f"{segment.start:.1f}s - {segment.end:.1f}s: {speaker}")
```

### 3.2 NEW: Exclusive Speaker Diarization

**NEW in 4.0**: Returns exclusive speaker diarization alongside regular diarization.

```python
output = pipeline("audio.wav")

# Regular diarization (overlapped speech possible)
print(output.speaker_diarization)

# NEW: Exclusive diarization (no overlaps - useful for ASR alignment)
print(output.exclusive_speaker_diarization)

# Iterate exclusive diarization
for segment, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True):
    print(f"{segment.start:.1f}s - {segment.end:.1f}s: {speaker}")
```

**Exclusive Diarization Benefits:**

- Simplifies reconciliation with ASR timestamps
- No overlapped speaker segments
- Better alignment with word-level transcription
- Backported from PyAnnoteAI Precision-2 model

---

## 4. Device Assignment and GPU Support

### 4.1 Device Assignment (Largely Unchanged)

**Before (3.x):**

```python
import torch
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cuda"))
```

**After (4.x):**

```python
import torch
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token="token")
pipeline.to(torch.device("cuda"))  # Same API
```

**Improvement**: 4.x tracks both `Model` and `nn.Module` attributes better when calling `.to()`.

### 4.2 Default Device

**Important**: Pipelines now **default to CPU** (changed in 3.0, still true in 4.x).

```python
# Explicitly move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)
```

---

## 5. PyAnnote-Core 6.x Breaking Changes

PyAnnote-Audio 4.0 requires **pyannote-core 6.x**. Key changes:

### 5.1 Annotation/Segment Handling

**Largely Compatible**: The `Annotation` and `Segment` classes remain similar, but some internal improvements:

- More type safety
- Better immutability guarantees
- Improved iteration performance

### 5.2 Powerset Utility Changes

**Segment String Representation**:

```python
# pyannote-core 6.x returns string tracks instead of int
from pyannote.core import Annotation, Segment

anno = Annotation()
for segment, track, label in anno.itertracks(yield_label=True):
    # track is now a string (not int)
    print(f"Track: {track} (type: {type(track).__name__})")
    # Output: Track: SPEAKER_00 (type: str)
```

### 5.3 No Major Breaking Changes in Core Data Structures

The main `Annotation` and `Segment` APIs remain compatible:

```python
from pyannote.core import Annotation, Segment

# Still works the same
annotation = Annotation()
segment = Segment(0.0, 1.0)
annotation[segment] = "SPEAKER_00"

# Iteration still works
for segment in annotation.itertracks():
    print(segment.start, segment.end)
```

---

## 6. Migration Checklist

### Code Changes

- [ ] Replace `use_auth_token` with `token` in all `Pipeline.from_pretrained()` calls
- [ ] Replace `@revision` syntax with `revision=` parameter
- [ ] Update model identifiers to new naming (e.g., `speaker-diarization-community-1`)
- [ ] Ensure Python version >= 3.10
- [ ] Install `ffmpeg` on all deployment systems
- [ ] Update `HF_HOME` environment variable if using custom cache
- [ ] Update training code if using custom tasks (remove `multilabel`, `warm_up`, etc.)
- [ ] Add support for new `exclusive_speaker_diarization` output

### Testing

- [ ] Test pipeline loading with new `token` parameter
- [ ] Test GPU device assignment still works
- [ ] Test diarization output parsing (especially `itertracks()`)
- [ ] Verify `exclusive_speaker_diarization` output format
- [ ] Test with both CPU and GPU
- [ ] Benchmark performance improvements

### Dependencies

- [ ] Update `pyproject.toml` to require `pyannote-audio>=4.0.0`
- [ ] Update `pyproject.toml` to require `pyannote-core>=6.0.0`
- [ ] Ensure `pyannote-pipeline>=4.0` (pinned in 4.0)
- [ ] Ensure `ffmpeg` is in deployment requirements

---

## 7. Example: Full Migration

### Current Code (3.x - Your Code)

```python
# src/processors/audio/diarizer.py (3.x)

from pyannote.audio import Pipeline
import torch

def _load_pyannote_model(self):
    model_id = "pyannote/speaker-diarization-3.1"

    # OLD: use_auth_token parameter
    model = Pipeline.from_pretrained(
        model_id,
        use_auth_token="HUGGINGFACE_ACCESS_TOKEN"  # BREAKING CHANGE
    )

    device = torch.device("cuda" if has_cuda() else "cpu")
    model.to(device)

    return model

def diarize(self, audio_path: str, num_speakers: Optional[int] = None):
    model = self._load_pyannote_model()

    diarization = model(audio_path, num_speakers=num_speakers)

    # OLD: itertracks() still works, but output labels changed
    spans = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        spans.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": speaker,
        })

    return spans
```

### Updated Code (4.x)

```python
# src/processors/audio/diarizer.py (4.x)

from pyannote.audio import Pipeline
import torch

def _load_pyannote_model(self):
    # NEW: Use 4.x model identifier
    model_id = "pyannote/speaker-diarization-community-1"

    # CHANGED: use_auth_token → token
    model = Pipeline.from_pretrained(
        model_id,
        token="HUGGINGFACE_ACCESS_TOKEN"  # Renamed parameter
    )

    device = torch.device("cuda" if has_cuda() else "cpu")
    model.to(device)  # No changes to this line

    return model

def diarize(self, audio_path: str, num_speakers: Optional[int] = None):
    model = self._load_pyannote_model()

    diarization = model(audio_path, num_speakers=num_speakers)

    # itertracks() API unchanged
    spans = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        spans.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": speaker,  # Now "SPEAKER_00" format in new model
        })

    # NEW: Can also access exclusive diarization for ASR alignment
    exclusive_spans = []
    for segment, _, speaker in diarization.exclusive_speaker_diarization.itertracks(yield_label=True):
        exclusive_spans.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": speaker,
        })

    return spans
```

---

## 8. Performance Improvements in 4.0

### Speed

- **2-3x faster** processing with VBx clustering
- New model trained with metadata caching (15x training speedup reported)
- Optimized dataloaders

Example on NVIDIA H100:

- 3.1 Community: ~31s per hour of audio
- 4.0 Community-1: ~14s per hour of audio

### Quality

Diarization error rate (DER) improvements across benchmarks:

| Dataset          | 3.1   | 4.0 Community-1 | Improvement |
| ---------------- | ----- | --------------- | ----------- |
| AISHELL-4        | 12.2% | 11.4%           | +0.8%       |
| AliMeeting (ch1) | 24.5% | 15.2%           | +9.3%       |
| AMI (IHM)        | 18.8% | 12.9%           | +5.9%       |
| DIHARD 3 (full)  | 21.4% | 14.7%           | +6.7%       |
| VoxConverse      | 11.2% | 8.5%            | +2.7%       |

---

## 9. Offline (Air-Gapped) Deployment

### 4.x Feature: Local Pipeline Storage

Pipelines can now be stored with their models in a single directory:

```bash
# Clone pipeline with embedded models (first time, needs internet)
git lfs install
git clone https://hf.co/pyannote/speaker-diarization-community-1 \
    /path/to/speaker-diarization-community-1

# Later, in air-gapped environment
from pyannote.audio import Pipeline

# Load from local directory (no internet needed)
pipeline = Pipeline.from_pretrained(
    "/path/to/speaker-diarization-community-1"
    # No token needed for local paths
)
```

---

## 10. Premium Support: PyAnnoteAI Precision-2

### Integration with PyAnnoteAI

4.0 adds built-in support for premium PyAnnoteAI cloud models:

```python
from pyannote.audio import Pipeline

# Switch one line to use premium model
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-precision-2",  # Premium model
    token="PYANNOTEAI_API_KEY"  # Use pyannoteai.ai API key
)

# Usage is identical
output = pipeline("audio.wav")
```

**Precision-2 Benefits:**

- Better accuracy than community-1
- Automatic cloud processing (or self-hosted)
- Free credits on dashboard.pyannote.ai

---

## 11. Common Migration Issues

### Issue 1: `ModuleNotFoundError: No module named 'pyannote'`

**Solution**: Install with all dependencies:

```bash
pip install "pyannote-audio[core]>=4.0.0"
# or
pixi add pyannote-audio
```

### Issue 2: `RuntimeError: ffmpeg not found`

**Solution**: Install system ffmpeg:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (with conda)
conda install -c conda-forge ffmpeg
```

### Issue 3: `TypeError: from_pretrained() got unexpected keyword argument 'use_auth_token'`

**Solution**: Rename to `token`:

```python
# OLD (fails in 4.x)
pipeline = Pipeline.from_pretrained(model_id, use_auth_token="token")

# NEW (works in 4.x)
pipeline = Pipeline.from_pretrained(model_id, token="token")
```

### Issue 4: `HuggingFaceHubHTTPError: 401 Client Error`

**Causes**:

1. Invalid or expired token
2. User hasn't agreed to model terms on HuggingFace
3. Private/gated model without permission

**Solution**:

1. Get valid token from hf.co/settings/tokens
2. Accept model conditions on HuggingFace
3. Check model access permissions

### Issue 5: `AttributeError: 'DiarizationOutput' has no attribute 'itertracks'`

**Cause**: Direct iteration on wrong object

**Solution**:

```python
# WRONG
output = pipeline("audio.wav")
for seg in output.itertracks():  # Wrong!

# RIGHT
output = pipeline("audio.wav")
for seg in output.speaker_diarization.itertracks():  # Correct
```

---

## 12. Telemetry (Optional)

4.0 includes **optional telemetry** to help improve pyannote. It's **disabled by default**.

```python
# Enable for current session
from pyannote.audio.telemetry import set_telemetry_metrics
set_telemetry_metrics(True)

# Or via environment variable
import os
os.environ["PYANNOTE_METRICS_ENABLED"] = "1"
```

**What's Tracked**: Usage metrics (audio duration, num_speakers), NOT personal data.

---

## 13. Compatibility Matrix

| Component         | 3.x                     | 4.x                             |
| ----------------- | ----------------------- | ------------------------------- |
| Python            | 3.8+                    | **3.10+**                       |
| pyannote-core     | 5.x                     | **6.x**                         |
| pyannote-database | 5.x                     | 5.x+                            |
| pyannote-metrics  | 1.x                     | 1.x+                            |
| pyannote-pipeline | 3.x                     | 4.x+                            |
| PyTorch           | 2.0+                    | 2.0+                            |
| torchaudio        | 2.0+                    | ✗ (replaced by torchcodec)      |
| Audio I/O         | sox, soundfile, ffmpeg  | **ffmpeg only**                 |
| Model             | speaker-diarization-3.1 | speaker-diarization-community-1 |

---

## 14. References

- **Official Repository**: https://github.com/pyannote/pyannote-audio
- **Release Notes**: https://github.com/pyannote/pyannote-audio/releases/tag/4.0.0
- **PyAnnoteAI**: https://pyannote.ai/
- **HuggingFace Models**: https://huggingface.co/pyannote
- **PyAnnote-Core**: https://github.com/pyannote/pyannote-core
- **Community Forum**: https://github.com/pyannote/pyannote-audio/discussions

---

## 15. Timeline and Support

- **pyannote-audio 3.4.0**: Last 3.x release (September 9, 2025) - Pinned dependencies
- **pyannote-audio 4.0.0**: Released September 29, 2025
- **pyannote-audio 4.0.1**: Released October 10, 2025 (current)
- **3.x Support**: Maintenance only; future pyannote.\* releases will break 3.x

---

## Summary Table: What Changed

| Aspect          | 3.x                     | 4.x                             | Impact          |
| --------------- | ----------------------- | ------------------------------- | --------------- |
| Auth Parameter  | `use_auth_token`        | `token`                         | **BREAKING**    |
| Revision Syntax | `@revision`             | `revision=`                     | **BREAKING**    |
| Audio Backend   | sox, soundfile, ffmpeg  | ffmpeg only                     | **BREAKING**    |
| Python          | 3.8+                    | 3.10+                           | **BREAKING**    |
| Cache Dir       | `PYANNOTE_CACHE`        | `HF_HOME`                       | **BREAKING**    |
| Model Inference | Direct model            | Instantiated model              | **BREAKING**    |
| Default Model   | speaker-diarization-3.1 | speaker-diarization-community-1 | **New**         |
| Clustering      | Agglomerative           | VBx                             | **New**         |
| Output          | speaker_diarization     | + exclusive_speaker_diarization | **Addition**    |
| Device API      | `.to(device)`           | `.to(device)`                   | **Same**        |
| itertracks()    | Works                   | Works                           | **Compatible**  |
| Performance     | Baseline                | 2-3x faster                     | **Improvement** |
| Accuracy        | ~13-21% DER             | ~8-14% DER                      | **Improvement** |

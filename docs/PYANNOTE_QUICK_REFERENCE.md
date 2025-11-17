# PyAnnote-Audio 4.x Quick Reference Card

**Print this for quick reference during migration**

---

## Top Breaking Changes

| Issue     | 3.x                     | 4.x                             | Fix                   |
| --------- | ----------------------- | ------------------------------- | --------------------- |
| Auth      | `use_auth_token="x"`    | `token="x"`                     | Rename parameter      |
| Revision  | `"model@v3.1"`          | `"model", revision="v3.1"`      | Use explicit param    |
| Python    | 3.8+                    | 3.10+                           | Update pyproject.toml |
| Audio I/O | sox, soundfile          | ffmpeg                          | Install ffmpeg        |
| Cache     | PYANNOTE_CACHE          | HF_HOME                         | Env var change        |
| Model     | speaker-diarization-3.1 | speaker-diarization-community-1 | Update identifier     |

---

## Code Changes at a Glance

### Loading Pipeline (Most Common Issue)

```python
# ❌ FAILS in 4.x
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="token"  # ← WRONG
)

# ✓ WORKS in 4.x
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",  # New model
    token="token"  # ← Changed parameter name
)
```

### Device Assignment (No Changes)

```python
# ✓ Same in both versions
import torch
pipeline.to(torch.device("cuda"))
```

### Diarization Output (Mostly Same)

```python
# ✓ Still works in 4.x
output = pipeline("audio.wav")
for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
    print(f"{segment.start} - {segment.end}: {speaker}")

# ✨ NEW in 4.x (optional)
for segment, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True):
    print(f"(exclusive) {segment.start} - {segment.end}: {speaker}")
```

---

## Installation

```bash
# Base installation (4.x)
pip install "pyannote.audio[core]>=4.0.0"

# System dependencies
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS

# Optional: Premium model support
pip install pyannoteai-sdk>=0.3.0
```

---

## Environment Variables

```bash
# Required for HuggingFace models
export HUGGINGFACE_TOKEN="hf_xxxxxxx"

# Optional: Custom cache directory
export HF_HOME="/path/to/cache"

# Optional: Telemetry (disabled by default)
export PYANNOTE_METRICS_ENABLED="0"

# Optional: Premium models
export PYANNOTEAI_API_KEY="pyannoteai_xxxxxxx"
```

---

## Common Errors & Fixes

| Error                                                     | Cause                  | Fix                                     |
| --------------------------------------------------------- | ---------------------- | --------------------------------------- |
| `TypeError: unexpected keyword argument 'use_auth_token'` | Using 3.x API with 4.x | Change to `token=`                      |
| `RuntimeError: ffmpeg not found`                          | ffmpeg not installed   | `apt-get install ffmpeg`                |
| `ModuleNotFoundError: torchcodec`                         | Missing dependency     | `pip install torchcodec`                |
| `HfHubHTTPError: 401 Client Error`                        | Invalid/missing token  | Set HUGGINGFACE_TOKEN                   |
| `AttributeError: no attribute 'itertracks'`               | Wrong object           | Use `.speaker_diarization.itertracks()` |

---

## Performance Expectations

| Metric   | 3.1        | 4.0       | Change          |
| -------- | ---------- | --------- | --------------- |
| Speed    | ~30s/hour  | ~14s/hour | **2.2x faster** |
| Accuracy | 13-21% DER | 8-14% DER | **Better**      |

---

## MAIE-Specific Changes

**File: `src/processors/audio/diarizer.py`**

```python
# Line 135 - Change THIS:
model = Pipeline.from_pretrained(model_id)

# To THIS:
model = Pipeline.from_pretrained(
    model_id,
    token=os.environ.get("HUGGINGFACE_TOKEN")
)

# Update model path (any of these):
model_id = "pyannote/speaker-diarization-community-1"  # NEW (4.x)
# OR for offline:
model_id = "/path/to/local/speaker-diarization-community-1"
```

**File: `pyproject.toml`**

```toml
# Change FROM:
requires-python = ">=3.8"
pyannote-audio = "~3.1"

# Change TO:
requires-python = ">=3.10"
pyannote-audio = ">=4.0.0,<5"
pyannote-core = ">=6.0.0,<7"
```

---

## Test Checklist

```bash
# 1. Check Python version
python --version  # Must be >= 3.10

# 2. Check ffmpeg
ffmpeg -version

# 3. Test imports
python -c "from pyannote.audio import Pipeline; print('OK')"

# 4. Test model loading
python -c "
from pyannote.audio import Pipeline
import os
os.environ['HUGGINGFACE_TOKEN'] = 'your_token_here'
p = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-community-1',
    token='your_token_here'
)
print('Model loaded successfully')
"

# 5. Test GPU
python -c "
import torch
print(f'GPU available: {torch.cuda.is_available()}')
print(f'GPU name: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')
"

# 6. Test diarization
python -c "
from pyannote.audio import Pipeline
import torch
p = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-community-1',
    token='your_token'
)
p.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# Test with actual audio file
result = p('/path/to/test.wav')
print(f'Diarization output: {len(result.speaker_diarization)} segments')
"
```

---

## Migration Priority

| Priority        | Task                                         | Time   | Impact                      |
| --------------- | -------------------------------------------- | ------ | --------------------------- |
| 🔴 **CRITICAL** | Update parameter: `use_auth_token` → `token` | 5 min  | Code breaks without this    |
| 🔴 **CRITICAL** | Install ffmpeg                               | 5 min  | Runtime errors without this |
| 🟡 **HIGH**     | Update Python requirement to 3.10+           | 5 min  | Dependency conflict         |
| 🟡 **HIGH**     | Update model identifier to community-1       | 5 min  | Old model may not work      |
| 🟢 **MEDIUM**   | Add token environment variable               | 10 min | Cleaner credentials         |
| 🟢 **MEDIUM**   | Test diarization                             | 15 min | Verify functionality        |
| 🔵 **OPTIONAL** | Add exclusive_speaker_diarization support    | 20 min | New feature                 |

---

## Before & After Comparison

### Before (3.x)

```python
from pyannote.audio import Pipeline
import torch

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HUGGINGFACE_TOKEN"
)
pipeline.to(torch.device("cuda"))
output = pipeline("audio.wav")

for segment, _, speaker in output.itertracks(yield_label=True):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {speaker}")

# Performance: ~30s per hour, 13-21% error rate
```

### After (4.x)

```python
from pyannote.audio import Pipeline
import torch

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",  # NEW
    token="HUGGINGFACE_TOKEN"  # CHANGED
)
pipeline.to(torch.device("cuda"))
output = pipeline("audio.wav")

for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True):  # More explicit
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {speaker}")

# Also available: output.exclusive_speaker_diarization (NEW)

# Performance: ~14s per hour, 8-14% error rate (2-3x faster, more accurate!)
```

---

## Documentation References

| Document                       | Purpose                      | Audience     |
| ------------------------------ | ---------------------------- | ------------ |
| PYANNOTE_RESEARCH_SUMMARY.md   | Overview & findings          | Everyone     |
| PYANNOTE_4X_MIGRATION_GUIDE.md | Comprehensive guide          | Developers   |
| PYANNOTE_MAIE_MIGRATION.md     | MAIE-specific implementation | MAIE team    |
| PYANNOTE_4X_API_REFERENCE.md   | Code examples & patterns     | Developers   |
| **This document**              | Quick reference              | Quick lookup |

---

## Key Metrics

**Speed Improvement**: 2-3x faster

- 3.1: ~30s per hour of audio
- 4.0: ~14s per hour of audio

**Accuracy Improvement**: Lower error rate

- 3.1: 13-21% Diarization Error Rate
- 4.0: 8-14% Diarization Error Rate

**Cost Savings**: Time saved per hour of audio

- At 2.2x: 16 seconds saved per hour
- Annual on 1000 hours: ~66 minutes saved

---

## Still Have Questions?

See these files in `/home/cetech/maie/docs/`:

1. **PYANNOTE_RESEARCH_SUMMARY.md** - Questions answered
2. **PYANNOTE_4X_MIGRATION_GUIDE.md** - Comprehensive guide
3. **PYANNOTE_MAIE_MIGRATION.md** - Implementation details
4. **PYANNOTE_4X_API_REFERENCE.md** - Code examples

---

**Last Updated**: November 17, 2025  
**Status**: Complete - Ready for Migration  
**Confidence**: High (Official Sources Only)

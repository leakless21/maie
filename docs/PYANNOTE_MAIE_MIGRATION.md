# PyAnnote-Audio 4.x Code Migration for MAIE

This document provides specific code changes needed to migrate MAIE to pyannote-audio 4.x.

---

## File: `src/processors/audio/diarizer.py`

### Current Issue

The current code uses pyannote-audio 3.x API:

```python
# Line 135: use_auth_token parameter (BREAKING in 4.x)
model = Pipeline.from_pretrained(model_id)

# Line 140: .to() device assignment (compatible, but improved in 4.x)
model.to(device)

# Line 229: itertracks() usage (compatible, but output format changes)
for segment, _, speaker in diarization.itertracks(yield_label=True):
```

### Changes Required

#### 1. Update Pipeline.from_pretrained() Call

**Location**: `_load_pyannote_model()` method around line 135

**Current Code**:

```python
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
```

**Updated Code (4.x)**:

```python
# Map common local paths to HuggingFace model IDs
if "speaker-diarization-3.1" in model_id:
    # Upgrade to 4.x community model
    model_id = "pyannote/speaker-diarization-community-1"
    logger.info(
        f"Mapped {self.model_path} to {model_id} "
        "(upgraded to pyannote 4.x)"
    )
elif "speaker-diarization-community-1" in model_id or "community" in model_id:
    model_id = "pyannote/speaker-diarization-community-1"
    logger.info(f"Using pyannote 4.x community model: {model_id}")
elif self.model_path.startswith('/'):
    # Local path: use as-is for offline deployment
    logger.info(f"Using local pipeline path: {model_id}")

logger.info(f"Loading pyannote pipeline: {model_id}")

# Load model from HuggingFace or local path
# CHANGED: use_auth_token → token parameter
try:
    # Try to load with token (for HuggingFace models)
    model = Pipeline.from_pretrained(
        model_id,
        token=os.environ.get("HUGGINGFACE_TOKEN")  # Will be None for local paths
    )
except (ImportError, ValueError, TypeError):
    # Fallback: local path might not need token
    model = Pipeline.from_pretrained(model_id)
```

**Simpler Version** (if you always have token):

```python
from os import environ

# Get token from environment (required for HuggingFace models)
hf_token = environ.get("HUGGINGFACE_TOKEN")
if not hf_token and not model_id.startswith('/'):
    logger.warning("HUGGINGFACE_TOKEN not set; model loading may fail")

model = Pipeline.from_pretrained(
    model_id,
    token=hf_token  # CHANGED: from use_auth_token
)
```

---

#### 2. Update Diarization Output Handling

**Location**: `diarize()` method around line 229

The `itertracks()` API is compatible, but output format may have changed with new model.

**Current Code**:

```python
# Convert pyannote output to simple list of spans
# pyannote returns an iterator of (segment, _, speaker) tuples
spans = []
for segment, _, speaker in diarization.itertracks(yield_label=True):
    spans.append(
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": self._normalize_speaker_label(speaker),
        }
    )
```

**No Changes Needed** - This still works! But you can optionally add:

```python
# Convert pyannote output to simple list of spans
spans = []
for segment, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    spans.append(
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": self._normalize_speaker_label(speaker),
        }
    )

# NEW: Also extract exclusive diarization for better ASR alignment
exclusive_spans = []
if hasattr(diarization, 'exclusive_speaker_diarization'):
    for segment, _, speaker in diarization.exclusive_speaker_diarization.itertracks(yield_label=True):
        exclusive_spans.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": self._normalize_speaker_label(speaker),
            }
        )
    logger.info(f"Also extracted {len(exclusive_spans)} exclusive diarization segments")
```

---

#### 3. Update Model Path Mapping

**Location**: `__init__()` method

**Current Code**:

```python
self.model_path = model_path  # "data/models/speaker-diarization-community-1"
```

**Updated Code**:

```python
# Map old paths to new 4.x model identifiers
if "3.1" in model_path or "3.0" in model_path:
    self.model_id = "pyannote/speaker-diarization-community-1"
    logger.info(f"Remapped {model_path} → {self.model_id} (pyannote 4.x)")
elif "community" in model_path or not model_path.startswith("/"):
    # Assume HuggingFace identifier
    self.model_id = model_path
else:
    # Local path
    self.model_id = model_path

self.model_path = model_path  # Keep for reference
```

---

### Complete Updated Method

Here's the complete `_load_pyannote_model()` method for 4.x:

```python
def _load_pyannote_model(self) -> object:
    """
    Load pyannote speaker diarization model lazily.

    NOTE: pyannote.audio 4.x requires:
    - token parameter (not use_auth_token)
    - ffmpeg installed on system
    - Python >= 3.10

    Reference: https://github.com/pyannote/pyannote-audio/releases/tag/4.0.0

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
        import os

        logger.info("Imports successful, loading pipeline...")

        # pyannote.audio 4.x uses HuggingFace model hub format
        model_id = self.model_path

        # Map common local paths to HF model IDs
        if "speaker-diarization-3" in model_id:
            # Upgrade 3.x models to 4.x
            model_id = "pyannote/speaker-diarization-community-1"
            logger.info(
                f"Mapped {self.model_path} to {model_id} "
                "(upgraded to pyannote 4.x)"
            )
        elif "community-1" in model_id and not model_id.startswith("/"):
            # Already correct format, just ensure HF identifier
            model_id = "pyannote/speaker-diarization-community-1"
        elif self.model_path.startswith('/'):
            # Local path for offline deployment
            logger.info(f"Using local pipeline path: {model_id}")

        logger.info(f"Loading pyannote pipeline: {model_id}")

        # Get HuggingFace token from environment
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")

        # Load model from HuggingFace or local path
        # CHANGED: use_auth_token → token (4.x breaking change)
        model = Pipeline.from_pretrained(
            model_id,
            token=hf_token  # CHANGED from use_auth_token
        )

        # Move model to GPU if available
        import torch
        device = torch.device("cuda" if has_cuda() else "cpu")
        model.to(device)  # type: ignore[union-attr]

        # Log batch size configuration info
        # NOTE: In pyannote 4.x, batch sizes are still configured in model's config.yaml
        # Default remains 32 for both embedding and segmentation
        logger.info(
            f"Pyannote model loaded successfully on {device}. "
            f"Requested batch sizes: embedding={self.embedding_batch_size}, "
            f"segmentation={self.segmentation_batch_size}. "
            f"Note: pyannote 4.x batch sizes are configured in model's config.yaml"
        )

        self.model = model
        return self.model

    except ImportError as e:
        logger.error(f"pyannote.audio not installed: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load diarization model: {e}", exc_info=True)
        return None
```

---

## File: `src/processors/asr/whisper.py`

### Current Issue

**Location**: Line 224

```python
"use_auth_token",  # This string is just a log message/comment
```

This doesn't need changes if it's just a reference. But if it's a parameter:

```python
# If used like this (BREAKING):
model = Pipeline.from_pretrained(model_id, use_auth_token="token")

# Change to:
model = Pipeline.from_pretrained(model_id, token="token")
```

---

## File: `pyproject.toml`

### Required Updates

**Current (3.x)**:

```toml
[project]
requires-python = ">=3.8"  # Too low

[project.dependencies]
pyannote-audio = "~3.1"  # Will break
```

**Updated (4.x)**:

```toml
[project]
requires-python = ">=3.10"  # MUST be >= 3.10

[project.dependencies]
pyannote-audio = ">=4.0.0,<5"  # Use 4.x
pyannote-core = ">=6.0.0,<7"  # Pinned at 6.x
pyannote-database = ">=5.0.0"  # Compatible
pyannote-metrics = ">=1.0.0"  # Compatible
pyannote-pipeline = ">=4.0.0"  # 4.x requirement
```

**Optional for PyAnnoteAI Premium**:

```toml
pyannoteai-sdk = ">=0.3.0"  # Optional: for precision-2 model
```

---

## File: Environment Configuration

### Required Environment Variables

**For HuggingFace Models**:

```bash
# In .env or deployment config
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx

# Optional: Custom cache directory
HF_HOME=/path/to/cache  # Default: ~/.cache/huggingface/hub/
```

**For PyAnnoteAI Premium Models**:

```bash
PYANNOTEAI_API_KEY=pyannoteai_xxxxxxxxxxxx
```

---

## Deployment Checklist

### System Requirements

- [ ] Python 3.10+ (check with `python --version`)
- [ ] ffmpeg installed (check with `ffmpeg -version`)
- [ ] CUDA 11.8+ for GPU support (check with `nvidia-smi`)
- [ ] PyTorch with CUDA support installed

### Code Changes

- [ ] Update `_load_pyannote_model()` method
- [ ] Update `Pipeline.from_pretrained()` to use `token=`
- [ ] Update model identifier to `pyannote/speaker-diarization-community-1`
- [ ] Update `pyproject.toml` Python requirement to `>=3.10`
- [ ] Update `pyproject.toml` pyannote-audio to `>=4.0.0,<5`
- [ ] Add environment variable handling for `HUGGINGFACE_TOKEN`
- [ ] Test diarization output parsing

### Testing

- [ ] Test with sample audio file
- [ ] Test GPU device assignment: `pipeline.to(torch.device("cuda"))`
- [ ] Test diarization output parsing: `diarization.itertracks()`
- [ ] Test exclusive diarization (new in 4.x): `diarization.exclusive_speaker_diarization`
- [ ] Benchmark performance improvement (should see 2-3x speedup)
- [ ] Test with both local and remote HuggingFace models
- [ ] Verify speaker label format (should still be recognizable)

### Documentation

- [ ] Update README.md with new pyannote-audio 4.x requirements
- [ ] Document new environment variables
- [ ] Update troubleshooting guide for ffmpeg requirement
- [ ] Document performance improvements

---

## Common Errors and Fixes

### Error 1: `TypeError: from_pretrained() got unexpected keyword argument 'use_auth_token'`

**Cause**: Using old 3.x API with 4.x library

**Fix**:

```python
# OLD (fails)
Pipeline.from_pretrained(model_id, use_auth_token="token")

# NEW (works)
Pipeline.from_pretrained(model_id, token="token")
```

---

### Error 2: `RuntimeError: ffmpeg not found`

**Cause**: ffmpeg not installed (4.x removed soundfile/sox support)

**Fix**:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Check installation
ffmpeg -version
```

---

### Error 3: `ModuleNotFoundError: No module named 'torchcodec'`

**Cause**: Missing torchcodec dependency (replaces torchaudio for audio I/O)

**Fix**:

```bash
pip install torchcodec
# Or reinstall pyannote-audio with all deps
pip install --upgrade "pyannote.audio[core]>=4.0.0"
```

---

### Error 4: `huggingface_hub.utils._errors.HfHubHTTPError: 401 Client Error`

**Cause**: Invalid or missing HuggingFace token

**Fix**:

```bash
# 1. Get valid token from https://hf.co/settings/tokens
# 2. Set environment variable
export HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxx"

# 3. Or authenticate with huggingface-cli
huggingface-cli login

# 4. Verify the token works
huggingface-cli whoami
```

---

### Error 5: `AttributeError: 'DiarizationOutput' object has no attribute 'itertracks'`

**Cause**: Calling itertracks on wrong object

**Fix**:

```python
# WRONG
output = pipeline("audio.wav")
for seg in output.itertracks():  # output doesn't have itertracks

# RIGHT
output = pipeline("audio.wav")
for seg in output.speaker_diarization.itertracks():  # Use .speaker_diarization
```

---

### Error 6: `Module 'torch' has no attribute 'cuda'`

**Cause**: PyTorch not built with CUDA support

**Fix**:

```bash
# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance Metrics (Before vs After)

### Speed Improvement

| Metric                 | 3.1      | 4.0      | Speedup  |
| ---------------------- | -------- | -------- | -------- |
| DIHARD 3 (~5min files) | 37s/hour | 14s/hour | **2.6x** |
| AMI (~1h files)        | 31s/hour | 14s/hour | **2.2x** |

### Accuracy Improvement (Diarization Error Rate)

| Dataset    | 3.1   | 4.0   | Improvement |
| ---------- | ----- | ----- | ----------- |
| AISHELL-4  | 12.2% | 11.4% | -0.8%       |
| AliMeeting | 24.5% | 15.2% | -9.3%       |
| DIHARD 3   | 21.4% | 14.7% | -6.7%       |

---

## Offline Deployment (Air-Gapped Environment)

For deployments without internet access:

```bash
# 1. On machine with internet:
git lfs install
git clone https://hf.co/pyannote/speaker-diarization-community-1 \
    /tmp/speaker-diarization-community-1

# 2. Copy to air-gapped machine
scp -r /tmp/speaker-diarization-community-1 user@airgapped:/path/to/models/

# 3. In air-gapped environment:
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "/path/to/models/speaker-diarization-community-1"
    # No token needed for local paths
)
```

---

## References

- **Migration Guide**: See `PYANNOTE_4X_MIGRATION_GUIDE.md` in this directory
- **Official Changelog**: https://github.com/pyannote/pyannote-audio/blob/develop/CHANGELOG.md
- **Release Notes**: https://github.com/pyannote/pyannote-audio/releases/tag/4.0.0
- **PyAnnote-Core 6.x**: https://github.com/pyannote/pyannote-core/releases/tag/6.0.0

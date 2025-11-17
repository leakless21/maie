# PyAnnote-Audio 4.x Implementation Reference

Quick reference for all API changes and new features in pyannote-audio 4.x.

---

## 1. Quick Reference Table

| Feature           | 3.x API                                               | 4.x API                                                          | Status          |
| ----------------- | ----------------------------------------------------- | ---------------------------------------------------------------- | --------------- |
| Load Pipeline     | `Pipeline.from_pretrained(model, use_auth_token="t")` | `Pipeline.from_pretrained(model, token="t")`                     | **BREAKING**    |
| Revision          | `model@v3.1`                                          | `model, revision="v3.1"`                                         | **BREAKING**    |
| Send to GPU       | `pipeline.to(torch.device("cuda"))`                   | `pipeline.to(torch.device("cuda"))`                              | ✓ Same          |
| Iterate Output    | `for seg, _, speaker in output.itertracks()`          | `for seg, _, speaker in output.speaker_diarization.itertracks()` | ✓ Same          |
| Exclusive Diar    | ✗ Not available                                       | `output.exclusive_speaker_diarization`                           | **NEW**         |
| Model             | speaker-diarization-3.1                               | speaker-diarization-community-1                                  | **New Default** |
| Clustering        | Agglomerative                                         | VBx                                                              | **New**         |
| Audio Backend     | sox, soundfile, ffmpeg                                | ffmpeg only                                                      | **BREAKING**    |
| Cache Dir         | PYANNOTE_CACHE env var                                | HF_HOME env var                                                  | **BREAKING**    |
| Model Load        | `Inference("model")`                                  | `Inference(Model.from_pretrained("model"))`                      | **BREAKING**    |
| Batch Size Config | Model kwargs                                          | model's config.yaml                                              | **Changed**     |

---

## 2. Complete API Examples

### 2.1 Loading a Pipeline

#### Basic (Simplest Case)

```python
from pyannote.audio import Pipeline
import torch

# Load community model from HuggingFace
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="YOUR_HF_TOKEN"
)

# Send to GPU
pipeline.to(torch.device("cuda"))

# Run diarization
output = pipeline("audio.wav")
```

#### With All Options

```python
from pyannote.audio import Pipeline
import torch
import os

# Load with all possible parameters
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ.get("HUGGINGFACE_TOKEN"),  # Can be None for public models
    revision="main",  # Optional: specific model version
    cache_dir=None,  # None = use HF_HOME env var (default: ~/.cache/huggingface/hub/)
    force_download=False,  # Re-download even if cached
    resume_download=False,  # Resume incomplete downloads
    local_files_only=False,  # Use only cached files
    skip_check=False,  # Skip dependency version check (advanced)
)

# Device handling
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))
else:
    print("GPU not available, using CPU (slower)")
    # No need to call .to() explicitly; CPU is default
```

#### For Offline/Air-Gapped Use

```python
from pyannote.audio import Pipeline
import torch

# Load from local directory (no internet needed)
pipeline = Pipeline.from_pretrained(
    "/path/to/speaker-diarization-community-1"
    # No token parameter needed for local paths
)

pipeline.to(torch.device("cuda"))
```

#### Premium PyAnnoteAI Model

```python
from pyannote.audio import Pipeline

# Use premium model on pyannoteai.ai cloud or self-hosted
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-precision-2",
    token="YOUR_PYANNOTEAI_API_KEY"
)

output = pipeline("audio.wav")
```

---

### 2.2 Running Diarization

#### Basic Execution

```python
# Run on audio file
output = pipeline("audio.wav")

# Run with preloaded audio
import torchaudio
waveform, sample_rate = torchaudio.load("audio.wav")
output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# With optional number of speakers
output = pipeline("audio.wav", num_speakers=2)

# With speaker bounds
output = pipeline("audio.wav", min_speakers=1, max_speakers=4)
```

#### With Progress Hook

```python
from pyannote.audio.pipelines.utils.hook import ProgressHook

with ProgressHook() as hook:
    output = pipeline("audio.wav", hook=hook)
```

#### With Timing Hook (Profile Performance)

```python
from pyannote.audio.pipelines.utils.hook import TimingHook

with TimingHook() as hook:
    output = pipeline("audio.wav", hook=hook)

print(hook)  # Shows timing breakdown
```

#### With Artifact Hook (Save Internal Steps)

```python
from pyannote.audio.pipelines.utils.hook import ArtifactHook

with ArtifactHook() as hook:
    output = pipeline("audio.wav", hook=hook)

# Access internal artifacts
embeddings = hook.embeddings  # Speaker embeddings
segmentation = hook.segmentation  # Raw segmentation scores
```

---

### 2.3 Processing Output

#### Regular Diarization

```python
output = pipeline("audio.wav")

# Iterate speaker segments
for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {speaker}")

# Get as RTTM format
with open("output.rttm", "w") as f:
    output.speaker_diarization.write_rttm(f)

# Get as dict/JSON-serializable format
diarization_dict = []
for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
    diarization_dict.append({
        "start": float(segment.start),
        "end": float(segment.end),
        "speaker": speaker,
    })
```

#### NEW: Exclusive Diarization (4.x Feature)

```python
output = pipeline("audio.wav")

# NEW: Exclusive diarization (no overlapping speakers)
# Useful for ASR alignment when overlapped speech isn't needed
for segment, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {speaker} (exclusive)")

# Compare regular vs exclusive
print(f"Regular diarization: {len(output.speaker_diarization)} segments")
print(f"Exclusive diarization: {len(output.exclusive_speaker_diarization)} segments")
# exclusive typically has fewer segments (no overlaps)
```

#### Access Speaker Count

```python
output = pipeline("audio.wav")

# Get number of speakers detected
diarization = output.speaker_diarization
num_speakers = len(set(label for _, _, label in diarization.itertracks(yield_label=True)))
print(f"Detected {num_speakers} speakers")
```

---

### 2.4 Device Management

#### Explicit GPU Assignment

```python
import torch
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token="token")

# Send to specific GPU
device = torch.device("cuda:0")  # First GPU
pipeline.to(device)

# Or auto-select first available GPU
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))
else:
    print("No GPU; using CPU")
```

#### CPU-Only Execution

```python
# No .to() call needed; CPU is default
output = pipeline("audio.wav")  # Runs on CPU
```

#### Mixed Precision (Advanced)

```python
import torch
from torch.amp import autocast_mode

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token="token")
pipeline.to(torch.device("cuda"))

# Run with automatic mixed precision
with autocast_mode(device_type="cuda", dtype=torch.float16):
    output = pipeline("audio.wav")
```

---

## 3. Error Handling

### Proper Exception Handling

```python
from pyannote.audio import Pipeline
import logging
import torch

logger = logging.getLogger(__name__)

def load_diarizer(model_id: str, token: str = None):
    """Load diarization pipeline with proper error handling."""
    try:
        logger.info(f"Loading {model_id}...")
        pipeline = Pipeline.from_pretrained(
            model_id,
            token=token,
            skip_check=False,  # Verify dependencies
        )

        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Moving pipeline to {device}")
        pipeline.to(device)

        logger.info("Pipeline loaded successfully")
        return pipeline

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install pyannote.audio[core]>=4.0.0")
        return None

    except (ValueError, RuntimeError) as e:
        if "ffmpeg" in str(e).lower():
            logger.error("ffmpeg not found. Install with: apt-get install ffmpeg")
        elif "401" in str(e):
            logger.error("Invalid or missing HuggingFace token")
            logger.error("Get token from: https://hf.co/settings/tokens")
        else:
            logger.error(f"Failed to load pipeline: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return None

def diarize_safely(pipeline, audio_path: str, num_speakers: int = None):
    """Run diarization with error handling."""
    try:
        output = pipeline(audio_path, num_speakers=num_speakers)
        return output
    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_path}")
        return None
    except RuntimeError as e:
        if "cuda" in str(e).lower():
            logger.warning("GPU error; retrying on CPU...")
            pipeline.to(torch.device("cpu"))
            output = pipeline(audio_path, num_speakers=num_speakers)
            return output
        else:
            logger.error(f"Diarization failed: {e}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error during diarization: {e}", exc_info=True)
        return None
```

---

## 4. Configuration and Batch Sizes

### Understanding Batch Sizes in 4.x

In pyannote-audio 4.x, batch sizes are **configured in the model's `config.yaml`**, not passed as parameters.

```python
# NO LONGER WORKS in 4.x
pipeline = Pipeline.from_pretrained(
    model_id,
    embedding_batch_size=16,  # ✗ Parameter removed
    segmentation_batch_size=16,  # ✗ Parameter removed
)

# Default batch sizes (in model's config.yaml)
# embedding_batch_size: 32
# segmentation_batch_size: 32
```

If you need different batch sizes, you must **modify the model's config file**:

```python
# Load pipeline
pipeline = Pipeline.from_pretrained(model_id, token=token)

# Access internal batch sizes
seg_batch_size = pipeline._segmentation.batch_size  # May not be publicly accessible
```

Or configure after loading:

```python
# Some pipeline internals may expose batch_size attributes
# Check pipeline implementation for specifics
```

---

## 5. Dependency Management

### Required Dependencies (4.x)

```toml
[project.dependencies]
# Core
pyannote-audio = ">=4.0.0,<5.0.0"
pyannote-core = ">=6.0.0,<7.0.0"
pyannote-database = ">=5.0.0"
pyannote-metrics = ">=1.0.0"
pyannote-pipeline = ">=4.0.0,<5.0.0"

# ML Framework
torch = ">=2.0.0"
lightning = ">=2.0.0"  # Switched from pytorch-lightning
torchcodec = ">=0.1.0"  # NEW: Replaces torchaudio for audio I/O

# Audio I/O
# NO LONGER: soundfile, sox
# ONLY: ffmpeg (system package)

# Optional
pyannoteai-sdk = ">=0.3.0"  # For premium Precision-2 model
```

### Removing Deprecated Dependencies

If upgrading from 3.x, remove:

```toml
# NO LONGER NEEDED
- pytorch-lightning  # Use 'lightning' instead
- soundfile  # Use ffmpeg instead
- torchaudio  # Use torchcodec instead (for audio I/O)
```

---

## 6. Performance Tips

### Memory Optimization

```python
# For limited GPU memory:
# 1. Ensure batch sizes are low (in model's config.yaml)
# 2. Use lower precision (if model supports it)
# 3. Process shorter audio chunks

import torch
pipeline = pipeline.to("cuda")

# Try lower precision
if torch.cuda.is_available():
    pipeline = pipeline.half()  # FP16 precision
```

### Speed Optimization

```python
# 4.x features for speed:
# - VBx clustering (automatic, faster than agglomerative)
# - Optimized dataloaders
# - Metadata caching in training

# For inference:
# - Use GPU (2-3x speedup)
# - Batch process multiple files
# - Use in-memory audio (avoid disk I/O)
```

### Profiling Performance

```python
from pyannote.audio.pipelines.utils.hook import TimingHook
import time

start = time.time()
with TimingHook() as hook:
    output = pipeline("audio.wav", hook=hook)
elapsed = time.time() - start

print(f"Total time: {elapsed:.2f}s")
print(f"Timing breakdown:\n{hook}")
```

---

## 7. Model Comparison Matrix

| Aspect         | 3.1 Community | 4.0 Community-1 | Precision-2 Premium |
| -------------- | ------------- | --------------- | ------------------- |
| Source         | HuggingFace   | HuggingFace     | PyAnnoteAI Cloud    |
| License        | MIT (open)    | MIT (open)      | Commercial          |
| Speed          | Baseline      | 2-3x faster     | 2-3x faster         |
| Accuracy (DER) | 13-21%        | 8-14%           | <5%                 |
| Clustering     | Agglomerative | VBx             | VBx                 |
| Exclusive Diar | ✗ No          | ✓ Yes           | ✓ Yes               |
| Supports GPU   | ✓ Yes         | ✓ Yes           | ✓ Yes (cloud)       |
| Offline        | ✓ Yes         | ✓ Yes           | ✗ Requires internet |
| Model Size     | ~150MB        | ~150MB          | Cloud-based         |

---

## 8. Common Patterns

### Pattern 1: Pipeline with Fallback

```python
def get_diarization_pipeline():
    """Get diarization pipeline with fallback."""
    try:
        # Try primary model
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=os.environ.get("HUGGINGFACE_TOKEN")
        )
    except Exception as e:
        logger.warning(f"Failed to load community model: {e}")
        try:
            # Fallback to local model
            pipeline = Pipeline.from_pretrained(
                "/path/to/local/model"
            )
            logger.info("Loaded local model as fallback")
        except Exception as e2:
            logger.error(f"Failed to load fallback model: {e2}")
            return None

    # Try GPU, fallback to CPU
    try:
        pipeline.to(torch.device("cuda"))
    except RuntimeError:
        logger.warning("GPU unavailable, using CPU")

    return pipeline
```

### Pattern 2: Batch Processing with Progress

```python
from pathlib import Path
from pyannote.audio.pipelines.utils.hook import ProgressHook

def process_audio_files(audio_dir: str, output_dir: str):
    """Process multiple audio files with progress tracking."""
    pipeline = Pipeline.from_pretrained(model_id, token=token)
    pipeline.to(torch.device("cuda"))

    audio_files = list(Path(audio_dir).glob("*.wav"))
    results = {}

    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing {audio_file.name}...")

        try:
            with ProgressHook() as hook:
                output = pipeline(str(audio_file), hook=hook)

            # Save result
            output_file = Path(output_dir) / f"{audio_file.stem}.rttm"
            with open(output_file, "w") as f:
                output.speaker_diarization.write_rttm(f)

            results[audio_file.name] = "success"
        except Exception as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")
            results[audio_file.name] = f"error: {e}"

    return results
```

### Pattern 3: Real-time/Streaming Processing

```python
def process_long_audio_chunked(audio_path: str, chunk_duration: float = 30.0):
    """Process long audio in chunks."""
    import librosa

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    chunk_samples = int(chunk_duration * sr)

    all_diarization = []

    for start in range(0, len(y), chunk_samples):
        end = min(start + chunk_samples, len(y))
        chunk = y[start:end]

        # Save chunk temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile
            soundfile.write(f.name, chunk, sr)

            # Process chunk
            output = pipeline(f.name)

            # Adjust timestamps
            time_offset = start / sr
            for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
                all_diarization.append({
                    "start": float(segment.start) + time_offset,
                    "end": float(segment.end) + time_offset,
                    "speaker": speaker,
                })

            # Clean up
            import os
            os.unlink(f.name)

    return all_diarization
```

---

## 9. Troubleshooting Checklist

```python
# Debug information
import sys
import torch
from pyannote.audio import __version__ as pya_version

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyAnnote-Audio: {pya_version}")

# Check ffmpeg
import subprocess
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    print(f"ffmpeg: {'Available' if result.returncode == 0 else 'Not found'}")
except FileNotFoundError:
    print("ffmpeg: Not found (required for 4.x)")

# Check dependencies
def check_imports():
    imports = [
        "pyannote.audio",
        "pyannote.core",
        "pyannote.database",
        "pyannote.metrics",
        "pyannote.pipeline",
        "torch",
        "lightning",
        "torchcodec",
    ]
    for module in imports:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")

check_imports()
```

---

## References

- **GitHub**: https://github.com/pyannote/pyannote-audio
- **Releases**: https://github.com/pyannote/pyannote-audio/releases/tag/4.0.0
- **HuggingFace**: https://huggingface.co/pyannote
- **PyAnnoteAI**: https://www.pyannote.ai/

# Jetson Nano Branch Plan

> **Goal**: Maintain a simplified ASR-only version for Jetson Nano on the `jetson` branch while keeping code synchronized with `main`.

| Field | Value |
|-------|-------|
| **Target Platform** | NVIDIA Jetson Nano / Orin Nano |
| **Branch** | `jetson` |
| **Parent Branch** | `main` |
| **Feature Scope** | ASR only (no LLM summarization) |
| **Last Updated** | December 2025 |

---

## Table of Contents

1. [Branch Strategy](#1-branch-strategy)
2. [Code Synchronization Workflows](#2-code-synchronization-workflows)
3. [Refactoring Plan: Shared Core Architecture](#3-refactoring-plan-shared-core-architecture)
4. [Simplified Edge Server Setup](#4-simplified-edge-server-setup)
5. [Jetson-Specific Dependencies](#5-jetson-specific-dependencies)
6. [Merge Strategy Script](#6-merge-strategy-script)
7. [Implementation Checklist](#7-implementation-checklist)

---

## 1. Branch Strategy

### Branch Structure

```
main (default)
│   ├── Full features: ASR + diarization + LLM summarization
│   ├── Platform: x86_64 (Linux/Docker)
│   ├── Dependencies: vllm, flashinfer, pyannote-audio, etc.
│   └── Server: Full async API + Redis/RQ workers
│
└── jetson (this branch)
    ├── Reduced features: ASR only
    ├── Platform: ARM64 (Jetson Nano/Orin)
    ├── Dependencies: Custom wheels, no vllm/flashinfer
    └── Server: Simplified synchronous edge API
```

### What Differs Between Branches

| Component | `main` | `jetson` |
|-----------|--------|----------|
| `pyproject.toml` | Full deps, x86_64 wheels | Minimal deps, ARM64 custom wheels |
| Entry point | `src/api/main.py` | `src/api/edge_main.py` |
| Config profile | `production` | `jetson` |
| LLM support | ✅ vLLM + templates | ❌ Disabled |
| Diarization | ✅ pyannote-audio | ❌ Disabled (no torchcodec) |
| Worker model | Async + Redis/RQ | Synchronous inline |
| Dockerfile | `Dockerfile` | `Dockerfile.jetson` |

### What Stays Identical

- `src/processors/` — ASR processors (whisper, chunkformer)
- `src/utils/` — Utility functions
- `src/config/` — Configuration system (profiles differ, loader is same)
- `tests/unit/` — Unit tests for shared components
- Core pipeline logic in `src/worker/pipeline.py` (feature-flag controlled)

---

## 2. Code Synchronization Workflows

### 2.1 Cherry-Pick (Selective Updates)

Use when you want **specific commits** from `main`:

```bash
# 1. Find the commit hash on main
git log main --oneline

# 2. Switch to jetson branch
git checkout jetson

# 3. Cherry-pick the specific commit
git cherry-pick <commit-hash>

# 4. Resolve conflicts if any (keep jetson-specific files)
git status
# Edit conflicting files
git add .
git cherry-pick --continue
```

**Best for**: Bug fixes, new ASR features, utility improvements.

### 2.2 Merge with Ours Strategy (Periodic Sync)

Use for **regular syncs** while preserving jetson-specific files:

```bash
# 1. Fetch latest main
git fetch origin main

# 2. Start merge (don't auto-commit)
git checkout jetson
git merge origin/main --no-commit --no-ff

# 3. Keep jetson-specific files
git checkout --ours pyproject.toml
git checkout --ours Dockerfile.jetson
git checkout --ours src/api/edge_main.py
# ... other jetson-specific files

# 4. Review and commit
git diff --cached  # Review what will be committed
git commit -m "chore: sync with main branch"
```

**Best for**: Monthly sync, major version updates.

### 2.3 Rebase (Clean Linear History)

Use to keep `jetson` as a **clean layer** on top of `main`:

```bash
# 1. Fetch latest
git fetch origin main

# 2. Rebase jetson onto main
git checkout jetson
git rebase origin/main

# 3. Resolve conflicts at each commit
# For each conflict:
git status
# Edit files, then:
git add .
git rebase --continue

# 4. Force push (if remote exists)
git push --force-with-lease origin jetson
```

**Best for**: When jetson changes are minimal, keeping history clean.

⚠️ **Warning**: Don't rebase if others are working on the jetson branch.

---

## 3. Refactoring Plan: Shared Core Architecture

### Goal

Structure code so **core logic is identical** on both branches, with differences isolated to:
- Configuration profiles
- Entry points
- Dependency manifests

### Target Directory Structure

```
src/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── main.py           # Full server (main branch default)
│   ├── edge_main.py      # Simplified server (jetson default)
│   ├── routes.py         # Shared route definitions
│   ├── schemas.py        # Shared request/response schemas
│   ├── dependencies.py   # Shared dependency injection
│   └── middleware.py     # Shared middleware
├── config/
│   ├── __init__.py
│   ├── model.py          # AppSettings model (shared)
│   ├── loader.py         # Config loader (shared)
│   ├── profiles.py       # Platform profiles (contains JETSON_PROFILE)
│   └── logging.py        # Logging config (shared)
├── processors/
│   ├── __init__.py
│   ├── asr/              # ASR processors (shared)
│   │   ├── whisper.py
│   │   └── chunkformer.py
│   ├── vad/              # VAD processors (shared)
│   ├── diarization/      # Diarization (disabled on jetson via config)
│   └── llm/              # LLM processors (disabled on jetson via config)
├── worker/
│   ├── __init__.py
│   ├── pipeline.py       # Core pipeline (shared, feature-flag controlled)
│   └── main.py           # RQ worker (main branch only)
└── utils/                # Shared utilities
    ├── __init__.py
    ├── audio.py
    └── metrics.py
```

### Refactoring Tasks

#### Task 3.1: Extract Feature Flags

Modify `src/config/model.py` to support feature toggles:

```python
class FeatureFlags(BaseModel):
    """Feature flags for platform-specific capabilities."""
    enable_llm: bool = True
    enable_diarization: bool = True
    enable_enhancement: bool = True
    enable_redis_queue: bool = True

class AppSettings(BaseSettings):
    # ... existing fields ...
    features: FeatureFlags = FeatureFlags()
```

#### Task 3.2: Create Jetson Profile

Add to `src/config/profiles.py`:

```python
JETSON_PROFILE = AppSettings(
    environment="jetson",
    debug=False,
    
    # Feature flags - disable heavy components
    features=FeatureFlags(
        enable_llm=False,
        enable_diarization=False,
        enable_enhancement=False,
        enable_redis_queue=False,
    ),
    
    # ASR config - use efficient models
    asr=ASRSettings(
        default_backend="chunkformer",
        whisper_device="cuda",
        whisper_compute_type="float16",
        chunkformer_device="cuda",
    ),
    
    # Paths for Jetson
    paths=PathSettings(
        models_dir=Path("/home/jetson/maie/data/models"),
        audio_dir=Path("/tmp/maie/audio"),
        output_dir=Path("/tmp/maie/output"),
    ),
    
    # Conservative resource limits
    api=APISettings(
        max_file_size_mb=50,
        request_timeout=300,
    ),
    
    # Logging - minimal retention
    logging=LoggingSettings(
        log_rotation="50 MB",
        log_retention="3 days",
    ),
)

# Register in PROFILES dict
PROFILES = {
    "development": DEVELOPMENT_PROFILE,
    "production": PRODUCTION_PROFILE,
    "edge": EDGE_PROFILE,
    "jetson": JETSON_PROFILE,  # <-- Add this
}
```

#### Task 3.3: Guard Optional Imports

Wrap optional imports in try/except:

```python
# src/processors/llm/__init__.py
try:
    from .vllm_processor import VLLMProcessor
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    VLLMProcessor = None

# src/processors/diarization/__init__.py
try:
    from .pyannote_processor import PyAnnoteProcessor
    HAS_DIARIZATION = True
except ImportError:
    HAS_DIARIZATION = False
    PyAnnoteProcessor = None
```

#### Task 3.4: Conditional Pipeline Steps

Modify `src/worker/pipeline.py`:

```python
async def process_audio_task(task_params: dict) -> dict:
    settings = get_settings()
    result = {}
    
    # Always run: preprocessing + ASR
    audio_path = preprocess_audio(task_params["audio_path"])
    result["transcript"] = await run_asr(audio_path, task_params)
    
    # Conditional: diarization
    if settings.features.enable_diarization and task_params.get("enable_diarization"):
        if HAS_DIARIZATION:
            result["diarization"] = await run_diarization(audio_path)
        else:
            logger.warning("Diarization requested but not available")
    
    # Conditional: LLM enhancement
    if settings.features.enable_llm and "summarize" in task_params.get("features", []):
        if HAS_VLLM:
            result["summary"] = await run_llm_enhancement(result["transcript"])
        else:
            logger.warning("LLM enhancement requested but not available")
    
    return result
```

---

## 4. Simplified Edge Server Setup

### 4.1 Edge Server Design (`src/api/edge_main.py`)

A minimal, synchronous HTTP server for Jetson:

```python
"""
Simplified Edge API for Jetson Nano.

Features:
- Single-process, synchronous execution
- No Redis/RQ dependency
- ASR-only processing
- Application-level lock for single-task semantics
"""
import asyncio
from pathlib import Path
from uuid import uuid4

from litestar import Litestar, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body

from src.config import get_settings
from src.worker.pipeline import process_audio_task
from src.api.schemas import ProcessResponse, HealthResponse

# Global lock for single-task processing
_process_lock = asyncio.Lock()


@get("/health")
async def health_check() -> HealthResponse:
    """Simple health check endpoint."""
    return HealthResponse(status="healthy", version="jetson-1.0")


@post("/v1/transcribe")
async def transcribe_audio(
    data: UploadFile = Body(media_type=RequestEncodingType.MULTI_PART),
    asr_backend: str = "chunkformer",
) -> ProcessResponse:
    """
    Synchronous ASR endpoint.
    
    Accepts audio file, returns transcript directly.
    Only one request processed at a time.
    """
    settings = get_settings()
    task_id = str(uuid4())
    
    # Save uploaded file
    audio_dir = settings.paths.audio_dir / task_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"input{Path(data.filename).suffix}"
    
    async with _process_lock:
        # Write file
        content = await data.read()
        audio_path.write_bytes(content)
        
        # Process synchronously
        task_params = {
            "task_id": task_id,
            "audio_path": str(audio_path),
            "asr_backend": asr_backend,
            "features": ["transcribe"],  # ASR only
        }
        
        result = await process_audio_task(task_params)
    
    return ProcessResponse(
        task_id=task_id,
        status="completed",
        transcript=result.get("transcript"),
    )


# Create minimal app
app = Litestar(
    route_handlers=[health_check, transcribe_audio],
    debug=False,
)
```

### 4.2 Running the Edge Server

```bash
# Set environment
export ENVIRONMENT=jetson

# Run with uvicorn (single worker)
uvicorn src.api.edge_main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 4.3 Systemd Service (Optional)

Create `/etc/systemd/system/maie-edge.service`:

```ini
[Unit]
Description=MAIE Edge ASR Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/maie
Environment="ENVIRONMENT=jetson"
ExecStart=/home/jetson/maie/.pixi/envs/default/bin/uvicorn src.api.edge_main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 5. Jetson-Specific Dependencies

### 5.1 Minimal `pyproject.toml` for Jetson

```toml
[project]
name = "maie"
version = "0.1.0-jetson"
description = "MAIE - Jetson Nano ASR Edition"
requires-python = ">=3.10"

# Minimal dependencies - ASR only
dependencies = [
    "pydantic>=2.12.0,<3",
    "pydantic-settings>=2.12.0,<3",
    "numpy<2",
    "faster-whisper>=1.2.1,<2",
    "litestar[standard]",
    "loguru",
    "aiofiles",
]

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-aarch64"]

[tool.pixi.pypi-options]
extra-index-urls = ["https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"]
index-strategy = "unsafe-best-match"

[tool.pixi.dependencies]
python = "3.10.*"
ffmpeg = "*"

[tool.pixi.pypi-dependencies]
maie = { path = ".", editable = true }
chunkformer = "*"
torch = "== 2.8.0"
torchaudio = "== 2.8.0"
# Custom wheel for ARM64
ctranslate2 = { path = "./wheels/ctranslate2-4.6.1-cp310-cp310-linux_aarch64.whl" }

[tool.pixi.feature.dev.tasks]
serve = "uvicorn src.api.edge_main:app --host 0.0.0.0 --port 8000"
test = "pytest tests/unit"
```

### 5.2 Removed Dependencies (vs main)

| Package | Reason |
|---------|--------|
| `vllm` | No ARM64 wheels, requires flashinfer |
| `flashinfer` | x86_64 only |
| `pyannote-audio` | Depends on torchcodec (no ARM64) |
| `torchcodec` | No ARM64 wheels |
| `redis`, `rq`, `rq-scheduler` | Not needed for sync edge API |
| `rq-dashboard` | Not needed |

### 5.3 Custom Wheels Location

Store ARM64 wheels in the repo or a known location:

```
/home/jetson/jetsonpack/wheels/
├── ctranslate2-4.6.1-cp310-cp310-linux_aarch64.whl
├── chunkformer-x.x.x-cp310-cp310-linux_aarch64.whl  # if needed
└── README.md  # Document wheel sources
```

---

## 6. Merge Strategy Script

Create `scripts/sync-from-main.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Sync jetson branch with main while preserving jetson-specific files

JETSON_SPECIFIC_FILES=(
    "pyproject.toml"
    "pixi.lock"
    "Dockerfile.jetson"
    "src/api/edge_main.py"
    "docs/JETSON_NANO_BRANCH_PLAN.md"
    ".env.jetson.example"
)

echo "=== Syncing jetson branch with main ==="

# Ensure we're on jetson branch
current_branch=$(git branch --show-current)
if [[ "$current_branch" != "jetson" ]]; then
    echo "Error: Must be on 'jetson' branch. Currently on '$current_branch'"
    exit 1
fi

# Fetch latest main
echo "Fetching latest main..."
git fetch origin main

# Start merge without committing
echo "Starting merge..."
git merge origin/main --no-commit --no-ff || true

# Preserve jetson-specific files
echo "Preserving jetson-specific files..."
for file in "${JETSON_SPECIFIC_FILES[@]}"; do
    if git ls-files --error-unmatch "$file" &>/dev/null 2>&1; then
        echo "  Keeping: $file"
        git checkout --ours "$file" 2>/dev/null || true
    fi
done

# Show status
echo ""
echo "=== Merge Status ==="
git status --short

echo ""
echo "=== Next Steps ==="
echo "1. Review changes: git diff --cached"
echo "2. Resolve any remaining conflicts"
echo "3. Commit: git commit -m 'chore: sync with main branch'"
echo ""
echo "To abort: git merge --abort"
```

Make executable:
```bash
chmod +x scripts/sync-from-main.sh
```

---

## 7. Implementation Checklist

### Phase 1: Branch Setup ✅
- [x] Create `jetson` branch from `main`
- [x] Initial `pyproject.toml` modifications for ARM64
- [x] Document branch strategy (this file)

### Phase 2: Dependency Cleanup ✅
- [x] Remove vllm, flashinfer from jetson `pyproject.toml`
- [x] Remove pyannote-audio, torchcodec
- [x] Remove redis, rq, rq-scheduler, rq-dashboard
- [x] Verify custom ctranslate2 wheel works
- [ ] Test `pixi install` on Jetson hardware

### Phase 3: Code Refactoring
- [x] Add `FeatureFlags` to `src/config/model.py`
- [x] Create `JETSON_PROFILE` in `src/config/profiles.py`
- [x] Guard optional imports in `src/processors/`
- [ ] Add feature-flag checks in `src/worker/pipeline.py`
- [ ] Ensure tests pass with features disabled

### Phase 4: Edge Server
- [x] Create/update `src/api/edge_main.py` for ASR-only
- [x] Add `/v1/transcribe` synchronous endpoint
- [x] Add `/health` endpoint
- [ ] Test single-task locking behavior
- [ ] Create systemd service file

### Phase 5: Testing & Validation
- [ ] Unit tests pass on jetson branch
- [ ] Integration test: upload audio → get transcript
- [ ] Performance test: measure latency on Jetson hardware
- [ ] Memory test: ensure 4GB RAM is sufficient

### Phase 6: Documentation
- [ ] Update README with Jetson quickstart
- [ ] Document API differences (vs main)
- [ ] Add troubleshooting section
- [x] Create `.env.jetson.example`

### Phase 7: Maintenance Scripts
- [x] Create `scripts/sync-from-main.sh`
- [ ] Create `scripts/build-jetson.sh`
- [ ] Add CI workflow for jetson branch (optional)

---

## Quick Reference

### Switch Between Branches
```bash
git checkout main      # Full-featured version
git checkout jetson    # Jetson ASR-only version
```

### Sync Jetson with Main
```bash
./scripts/sync-from-main.sh
```

### Run on Jetson
```bash
export ENVIRONMENT=jetson
pixi run serve
# Or directly:
uvicorn src.api.edge_main:app --host 0.0.0.0 --port 8000
```

### Test ASR Endpoint
```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "data=@test.mp3" \
  -F "asr_backend=chunkformer"
```

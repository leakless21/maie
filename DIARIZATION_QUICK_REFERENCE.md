# Diarization Implementation - Quick Reference Card

## 📍 You Are Here

```
PHASE: Planning & Review ✅ COMPLETE

Next: Phase 1 (Foundation)
→ Start with: DIARIZATION_CHECKLIST.md
```

---

## 📚 Where to Find Everything

| Need                     | Document                            | Time |
| ------------------------ | ----------------------------------- | ---- |
| **Quick Status**         | DIARIZATION_READY.md                | 5m   |
| **Architecture**         | DIARIZATION_VISUAL_SUMMARY.md       | 15m  |
| **Implementation Steps** | DIARIZATION_CHECKLIST.md            | Ref  |
| **Design Details**       | DIARIZATION_IMPLEMENTATION_GUIDE.md | 20m  |
| **Gap Analysis**         | DIARIZATION_REVIEW.md               | 20m  |
| **Navigation**           | DIARIZATION_INDEX.md                | 5m   |

---

## 🎯 Executive Summary

| Aspect                     | Status        | Confidence |
| -------------------------- | ------------- | ---------- |
| Plan Soundness             | ✅ Verified   | 95%        |
| API Compatibility          | ✅ Verified   | 100%       |
| Graceful Fallback          | ✅ Designed   | 95%        |
| Implementation Feasibility | ✅ Feasible   | 95%        |
| Timeline Estimate          | ✅ 7-10 hours | 85%        |
| Test Strategy              | ✅ Clear      | 90%        |
| **Overall GO/NO-GO**       | **✅ GO**     | **95%**    |

---

## 🔧 5-Phase Implementation Map

```
Phase 1: Foundation (2-3h)
├── device.py ................... torch device helpers
├── timestamp_utils.py .......... timestamp normalization
└── Tests ........................ device + timestamp tests
   Status: Not started
   Effort: MEDIUM

Phase 2: Core Logic (3-4h)
├── diarizer.py ................. main processor
├── Alignment functions ......... IoU + split logic
└── Tests ........................ alignment + error handling
   Status: Not started
   Effort: HARD

Phase 3: Config (30m)
├── DiarizationSettings ......... in config/model.py
└── AppSettings ................. add diarization field
   Status: Not started
   Effort: EASY

Phase 4: Integration (1-2h)
├── Worker init ................. in worker/main.py
├── Pipeline step ............... in worker/pipeline.py
└── E2E testing ................. verify end-to-end
   Status: Not started
   Effort: MEDIUM

Phase 5: Documentation (30m)
├── README ...................... update with feature
└── Example script .............. examples/infer_diarization.py
   Status: Not started
   Effort: EASY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL: 7-10 hours
```

---

## 🔑 Key Implementation Points

### 1. Device Handling

```python
from src.utils.device import get_torch_device, require_cuda

# Get device (fallback to CPU if needed)
device = get_torch_device(prefer_cuda=True)

# Require CUDA (raises RuntimeError if missing)
require_cuda("diarization")
```

### 2. Timestamp Normalization

```python
from src.processors.audio.timestamp_utils import normalize_timestamp

# Convert any format to float seconds
ts = normalize_timestamp("[00:00:12.340]")  # → 12.34
ts = normalize_timestamp(12.34)             # → 12.34
```

### 3. Diarization

```python
from src.processors.audio.diarizer import SpeakerDiarizer

# Initialize (lazy-loads model)
diarizer = SpeakerDiarizer(settings.diarization)

# Run diarization
diar_segs = diarizer.diarize("audio.wav")
# → [{"start": 10.0, "end": 12.0, "speaker": "S1"}, ...]

# Align with ASR
aligned = diarizer.align_diarization_with_asr(diar_segs, asr_segments)
# → [{"start": 10.0, "end": 12.0, "text": "hello", "speaker": "S1"}, ...]
```

### 4. Worker Integration

```python
# In worker/pipeline.py, after ASR:
if settings.diarization.enabled:
    diarizer = get_diarizer()
    if diarizer:
        diar_segs = diarizer.diarize(audio_path)
        if diar_segs:
            segments = diarizer.align_diarization_with_asr(diar_segs, segments)
```

---

## ⚠️ Critical Design Decisions

### 1. Graceful Fallback (Most Important)

```
IF diarization fails OR CUDA missing OR model missing:
  THEN skip diarization (return None)
  AND log warning/error
  AND continue to LLM processing

NOT: fail entire job
```

### 2. Alignment Strategy

```
Single speaker      → assign
Multiple speakers   → split (word-level if available, else proportional)
No speakers        → speaker = None
```

### 3. Word-Level Split (Preferred)

```
IF ASR backend provides word timestamps:
  THEN assign each word to overlapping speaker
  ELSE use proportional split (fallback)
```

### 4. Proportional Split (Fallback)

```
Duration of speaker S1 = 40% of segment
Number of words in segment = 10
Words for S1 = round(10 * 0.40) = 4 words
Remaining 6 words → last speaker (ensure no loss)
```

---

## 🧪 Test Categories

### Unit Tests (Fast, Mocked)

```
tests/unit/utils/test_device.py
├── test_get_torch_device_cuda_available()
├── test_get_torch_device_cuda_unavailable()
└── test_require_cuda_missing()

tests/unit/processors/audio/test_timestamp_utils.py
├── test_normalize_timestamp_float()
├── test_normalize_timestamp_string_hms()
├── test_iou_overlap()
└── test_iou_no_overlap()

tests/unit/processors/audio/test_diarizer.py
├── test_align_single_speaker()
├── test_align_no_overlap()
├── test_align_multiple_speakers()
├── test_merge_adjacent()
└── [+5 edge cases]
```

### Integration Tests (GPU Required)

```
tests/integration/processors/audio/test_diarizer_integration.py
├── @pytest.mark.gpu
├── test_diarizer_e2e_with_real_model()
└── Requires: real audio + real model
```

---

## 🎁 Output Format

### Segment Structure (With Diarization)

```json
{
  "start": 12.5,
  "end": 14.3,
  "text": "What do you think?",
  "speaker": "S1",
  "speaker_confidence": 0.92
}
```

### Speaker Statistics

```python
speakers = set(seg.get("speaker") for seg in segments if seg.get("speaker"))
print(f"Detected speakers: {len(speakers)}")  # → "Detected speakers: 3"
```

---

## 🚨 Error Scenarios (All Graceful)

| Scenario         | Action      | Log                                               |
| ---------------- | ----------- | ------------------------------------------------- |
| CUDA unavailable | Skip        | Warning: "CUDA unavailable; skipping diarization" |
| Model not found  | Skip        | Error: "Diarization model not found"              |
| Load fails       | Skip        | Error: "Failed to load diarization model"         |
| Inference fails  | Skip        | Error: "Diarization inference failed"             |
| Alignment fails  | Best effort | Warning: "Alignment degraded; partial results"    |

**Result in all cases:** Continue to LLM processing with best available data.

---

## 📊 Performance Impact

| Metric                              | Expected      |
| ----------------------------------- | ------------- |
| RTF (Real-Time Factor) overhead     | <10%          |
| GPU memory (typical)                | <2GB          |
| CPU fallback                        | Usable (slow) |
| Model load time (first use)         | ~10-30s       |
| Inference time (typical 5min audio) | 5-15s         |

---

## ✅ Success Criteria

Before calling implementation "done":

1. ✅ All tests pass (>85% coverage)
2. ✅ Graceful fallback works
3. ✅ Segments include speaker info
4. ✅ <10% RTF overhead
5. ✅ Comprehensive logging
6. ✅ README updated
7. ✅ Example script works

---

## 🎬 Getting Started (Right Now)

1. **Read this card** ← You are here (5 min)
2. **Read DIARIZATION_READY.md** (5 min)
3. **Read DIARIZATION_IMPLEMENTATION_GUIDE.md** (20 min)
4. **Open DIARIZATION_CHECKLIST.md** (keep open while coding)
5. **Start Phase 1** (device.py with tests)

---

## 🆘 Troubleshooting Quick Links

- "CUDA error" → See DIARIZATION_VISUAL_SUMMARY.md error handling
- "Model not found" → See DIARIZATION_IMPLEMENTATION_GUIDE.md Phase 4
- "Tests failing" → See DIARIZATION_CHECKLIST.md troubleshooting
- "Alignment weird" → See DIARIZATION_VISUAL_SUMMARY.md data flow

---

## 📞 Contact Info for Questions

- **Design questions** → See DIARIZATION_IMPLEMENTATION_GUIDE.md
- **Gap concerns** → See DIARIZATION_REVIEW.md
- **Architecture** → See DIARIZATION_VISUAL_SUMMARY.md
- **Task checklist** → See DIARIZATION_CHECKLIST.md

---

## ⏱️ Estimated Velocity

| Phase     | Effort    | Velocity                                     |
| --------- | --------- | -------------------------------------------- |
| Phase 1   | 2-3h      | Coding: 1.5h, Testing: 1h, Integration: 0.5h |
| Phase 2   | 3-4h      | Coding: 2h, Testing: 1.5h, Debug: 0.5h       |
| Phase 3   | 0.5h      | Coding: 0.25h, Testing: 0.25h                |
| Phase 4   | 1-2h      | Coding: 0.5h, Testing: 1h, Debugging: 0.5h   |
| Phase 5   | 0.5h      | Documentation: 0.5h                          |
| **Total** | **7-10h** | **Average: 1.5h per phase**                  |

---

## 🏁 Definition of Done

```
✅ Code Quality
  ├─ Black formatted
  ├─ isort applied
  ├─ Type hints on all functions
  ├─ Docstrings on classes/public methods
  └─ >85% test coverage

✅ Functionality
  ├─ Diarization runs after ASR
  ├─ Segments have speaker info
  ├─ Graceful fallback works
  └─ No test failures

✅ Performance
  ├─ <10% RTF overhead
  └─ GPU memory <2GB typical

✅ Documentation
  ├─ README updated
  ├─ Example script runnable
  └─ Code is self-documenting
```

---

## 🎯 Next Action

```
→ Read: DIARIZATION_READY.md (5 min)
→ Read: DIARIZATION_IMPLEMENTATION_GUIDE.md (20 min)
→ Open: DIARIZATION_CHECKLIST.md (keep handy)
→ Start: Phase 1, Task 1.1 (device.py)
```

---

**Status: ✅ READY TO IMPLEMENT**  
**Confidence: 95%**  
**Timeline: 7-10 hours**  
**Go/No-Go: GO** 🚀

---

_Quick Reference Card v1.0_  
_Last Updated: 2025-10-24_  
_For detailed info, see DIARIZATION_INDEX.md_

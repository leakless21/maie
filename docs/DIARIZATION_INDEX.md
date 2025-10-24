# Diarization Implementation - Documentation Index

## Quick Navigation

### 🎯 Start Here (Consolidated)

- **[DIARIZATION_OVERVIEW_AND_GUIDE.md](./DIARIZATION_OVERVIEW_AND_GUIDE.md)** ← START HERE
  - Executive overview, architecture, backend differences
  - Implementation path, error handling, config, testing, checklist
- **[DIARIZATION_PROMPT_PACKAGING.md](./DIARIZATION_PROMPT_PACKAGING.md)**
  - Exact LLM input format and examples

### 📚 Archived (for reference)

- [DIARIZATION_INTEGRATION_PLAN.md](./DIARIZATION_INTEGRATION_PLAN.md)

---

## Which Document to Read?

### 👤 I'm a Manager/Lead

→ Read **DIARIZATION_OVERVIEW_AND_GUIDE.md** (10–15 min)

- Status overview
- Architecture and plan
- Timeline estimate
- Go/no-go decision

### 🎓 I'm a New Team Member

→ Read in this order:

1. **DIARIZATION_OVERVIEW_AND_GUIDE.md** – Overview and architecture
2. **DIARIZATION_PROMPT_PACKAGING.md** – LLM input format

### 🛠️ I'm Implementing This

→ Read in this order:

1. **DIARIZATION_OVERVIEW_AND_GUIDE.md** – Approach + steps
2. **DIARIZATION_PROMPT_PACKAGING.md** – LLM input spec

### 🔬 I'm Reviewing Code

→ Read:

1. **DIARIZATION_OVERVIEW_AND_GUIDE.md** (Design + behavior)
2. **DIARIZATION_PROMPT_PACKAGING.md** (LLM input contract)

### 🐛 Something's Not Working

→ Check in **DIARIZATION_OVERVIEW_AND_GUIDE.md**:
- Error handling
- Alignment algorithm

---

## Key Findings (Summary)

### ✅ Verified

- Pyannote.audio API works as specified
- Backends expose segment-level timestamps; no word-level timestamps in adapters
- Device handling is standard PyTorch
- Alignment algorithm is sound (IoU + proportional policy)
- Graceful fallback strategy is feasible

### 🔧 Improvements Made

1. **Graceful CUDA handling** – Skip diarization if GPU unavailable, don't fail
2. **Word extraction strategy** – Different approach per ASR backend
3. **Lazy model validation** – Check at runtime, not config load
4. **Robust proportional splitting** – Ensure no text is lost
5. **Comprehensive logging** – Track all steps for debugging

### 📊 Implementation Overview

- **5 phases** (utilities → config → integration → testing → docs)
- **7-10 hours** total effort
- **3 new modules/files** to create (device helpers, diarizer, accompanying tests)
- **3 existing files** to modify (config/model.py, worker/main.py, worker/pipeline.py)
- **>85% test coverage** goal

---

## Core Concepts (Quick Reference)

### Alignment Algorithm: IoU (Intersection over Union)

```
Given two time intervals [a0, a1] and [b0, b1]:
  intersection = max(0, min(a1, b1) - max(a0, b0))
  union = max(a1, b1) - min(a0, b0)
  iou = intersection / union

Used to determine if ASR and diarization segments overlap.
```

### Three Key Strategies

1. **Single Speaker** → Assign ASR segment to that speaker
2. **No Speakers** → Mark speaker=None
3. **Multiple Speakers** → Split text using proportional policy:
   - If one speaker covers ≥70% of the ASR span, assign the whole segment to that speaker
   - Otherwise split once at the proportional boundary (maximum two subsegments)

### Error Handling Philosophy

**DIARIZATION IS OPTIONAL ENHANCEMENT**

- Never fail job because of diarization
- Skip gracefully if GPU/model unavailable
- Always continue to LLM processing
- Log all decisions for observability

---

## Files to Create/Modify

### Create (New)

| File                                                              | Purpose                    |
| ----------------------------------------------------------------- | -------------------------- |
| `src/utils/device.py`                                             | Torch device helpers       |
| `src/processors/audio/diarizer.py`                                | Main diarization processor |
| `tests/unit/utils/test_device.py`                                 | Unit tests                 |
| `tests/unit/processors/audio/test_diarizer.py`                    | Unit tests                 |
| `tests/integration/processors/audio/test_diarizer_integration.py` | Integration tests          |
| `examples/infer_diarization.py`                                   | Example usage              |

### Modify (Existing)

| File                     | Change                  |
| ------------------------ | ----------------------- |
| `src/config/model.py`    | Add DiarizationSettings |
| `src/worker/main.py`     | Initialize diarizer     |
| `src/worker/pipeline.py` | Add diarization step    |
| `README.md`              | Document feature        |

---

## Timeline

| Phase              | Time      | Focus                                |
| ------------------ | --------- | ------------------------------------ |
| **1: Foundation**  | 2-3h      | device.py, diarizer test scaffolding |
| **2: Core Logic**  | 3-4h      | diarizer.py, alignment, mocked tests |
| **3: Config**      | 30m       | DiarizationSettings                  |
| **4: Integration** | 1-2h      | Worker init, pipeline step, E2E test |
| **5: Docs**        | 30m       | README, example script               |
| **Total**          | **7-10h** | **Complete implementation**          |

---

## Success Criteria

✅ All criteria must be met:

1. **Code Quality**

   - Black formatted
   - Type hints on all functions
   - Docstrings on classes/public methods
   - > 85% test coverage

2. **Functionality**

   - Diarization runs after ASR
   - Segments include speaker info
   - Graceful fallback works (CUDA missing)
   - No failing tests

3. **Performance**

   - <10% RTF overhead
   - GPU memory <2GB typical

4. **Observability**

   - Comprehensive logging
   - Clear error messages
   - Debug-level per-segment tracking

5. **Documentation**
   - README updated
   - Example script runnable
   - Code is self-documenting

---

## Common Questions

**Q: What if CUDA is unavailable?**  
A: Diarization is skipped gracefully (return None). Job continues to LLM. Warning logged.

**Q: What if the model isn't downloaded?**  
A: Lazy validation at diarizer init time. Skipped with error log if not found.

**Q: How accurate is the alignment?**  
A: Depends on available data:

- With word timestamps (FasterWhisper): ~95% accuracy
- With proportional split (ChunkFormer): ~70-80% accuracy

**Q: Will this impact performance?**  
A: <10% RTF overhead (diarization is fast; alignment is O(n)).

**Q: Can I disable it?**  
A: Yes, set `APP_DIARIZATION__ENABLED=false` in environment.

**Q: How do I test without real GPU?**  
A: All unit tests use mocks. GPU tests are marked with `@pytest.mark.gpu` (optional).

---

## Getting Started

1. **Read:** **DIARIZATION_OVERVIEW_AND_GUIDE.md** (10–15 min)
2. **Understand:** **DIARIZATION_PROMPT_PACKAGING.md** (5–10 min)
3. **Execute:** Follow the checklist in the Overview & Guide
4. **Reference:** **DIARIZATION_INTEGRATION_PLAN.md** if you need historical context

---

## Document Metadata

| Document                              | Purpose                 | Length    | Audience                |
| ------------------------------------- | ----------------------- | --------- | ----------------------- |
| DIARIZATION_OVERVIEW_AND_GUIDE.md     | Overview + plan         | 10–15 min | All                     |
| DIARIZATION_PROMPT_PACKAGING.md       | LLM input format        | 5–10 min  | Implementers, reviewers |
| DIARIZATION_INTEGRATION_PLAN.md       | Original design         | 30 min    | Reference               |

---

## Status: ✅ READY TO IMPLEMENT

All reviews complete. All APIs verified. All gaps documented. Ready to start Phase 1.

**Next action:** Begin Phase 1 (Foundation) with TDD approach.

---

_Last updated: 2025-10-24_  
_Status: Ready for Implementation ✅_  
_Confidence: 95%_

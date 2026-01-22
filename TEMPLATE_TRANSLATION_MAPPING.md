# Template Translation Mapping (MAIE Constraints Compliant)

## Constraint Summary

- **Keep in English:** Canonical keys (`title`, `summary`, `tags`, `key_topics`, `action_items`, `attendees`, `decisions`)
- **Keep in English:** API envelope (task_id, status, results, template_id, timestamps, metrics)
- **Keep in English:** Enum/status values (PENDING, COMPLETE, FAILED, etc.)
- **Keep in English:** Transcript keys (`raw_transcript`, `clean_transcript`)
- **Can translate:** Template-specific "extra" fields only

---

## Per-Template Translation Map

### 1. meeting_notes_v2

**Canonical fields (KEEP English):**

```
title, summary, tags, decisions, action_items
```

**Template-specific fields (TRANSLATE to Vietnamese):**

```
meeting_date          → ngày_họp
participants          → người_tham_gia
agenda                → chương_trình_nghị_sự
```

**Schema changes:**

```json
{
  "properties": {
    "title": { ... },                    // KEEP
    "ngày_họp": { ... },                 // ← CHANGED from meeting_date
    "người_tham_gia": { ... },           // ← CHANGED from participants
    "summary": { ... },                  // KEEP
    "chương_trình_nghị_sự": { ... },    // ← CHANGED from agenda
    "decisions": { ... },                // KEEP
    "action_items": { ... },             // KEEP
    "tags": { ... }                      // KEEP
  },
  "required": ["title", "ngày_họp", "người_tham_gia", "summary", "chương_trình_nghị_sự", "decisions", "action_items", "tags"]
}
```

---

### 2. interview_transcript_v2

**Canonical fields (KEEP English):**

```
title (implied), summary (implied), tags
```

**Template-specific fields (TRANSLATE to Vietnamese):**

```
interview_date          → ngày_phỏng_vấn
interview_summary       → tóm_tắt_phỏng_vấn
key_insights            → những_hiểu_biết_chính
participant_sentiment   → cảm_tính_người_tham_gia
```

**Schema changes:**

```json
{
  "properties": {
    "ngày_phỏng_vấn": { ... },           // ← CHANGED from interview_date
    "tóm_tắt_phỏng_vấn": { ... },       // ← CHANGED from interview_summary
    "những_hiểu_biết_chính": { ... },   // ← CHANGED from key_insights
    "cảm_tính_người_tham_gia": { ... }, // ← CHANGED from participant_sentiment
    "tags": { ... }                      // KEEP
  },
  "required": ["tóm_tắt_phỏng_vấn", "những_hiểu_biết_chính", "cảm_tính_người_tham_gia", "tags"]
}
```

---

### 3. text_enhancement_v1

**Canonical fields (KEEP English):**

```
title (implied), summary (implied), tags
```

**Template-specific fields (TRANSLATE to Vietnamese):**

```
original_text       → văn_bản_gốc
enhanced_text       → văn_bản_cải_thiện
quality_score       → điểm_chất_lượng
language            → ngôn_ngữ
```

**Schema changes:**

```json
{
  "properties": {
    "văn_bản_gốc": { ... },              // ← CHANGED from original_text
    "văn_bản_cải_thiện": { ... },        // ← CHANGED from enhanced_text
    "điểm_chất_lượng": { ... },          // ← CHANGED from quality_score
    "ngôn_ngữ": { ... },                 // ← CHANGED from language
    "tags": { ... }                      // KEEP
  },
  "required": ["văn_bản_gốc", "văn_bản_cải_thiện", "điểm_chất_lượng", "ngôn_ngữ", "tags"]
}
```

---

### 4. generic_summary_v2

**All fields are canonical (KEEP English):**

```
title, summary, key_topics, tags
```

✅ **No translation needed**

---

### 5. generic_summary_en_v2

**All fields are canonical + intentionally English (KEEP English):**

```
title, summary, key_topics, tags
```

✅ **No translation needed** (language-specific template)

---

### 6. structured_analysis_v1

**Canonical fields (KEEP English):**

```
title, summary, tags
```

**Template-specific sections (ALREADY TRANSLATED ✓):**

```
mở_đầu, báo_cáo, thảo_luận, kết_luận, giao_việc
```

✅ **Already complete**

---

## Benefits of This Approach

| Aspect                   | Benefit                                                                   |
| ------------------------ | ------------------------------------------------------------------------- |
| **Server compatibility** | Canonical keys untouched; server uses them without code changes           |
| **Frontend safe**        | UI/exporters don't break on extra translated fields; they ignore unknowns |
| **Gradual rollout**      | Update frontend selectors for translations as time permits                |
| **Backward compatible**  | Existing clients that expect only canonical keys still work               |
| **Vietnamese UX**        | Better display names for Vietnamese users in template editors/docs        |
| **No enum risk**         | Status, sentiment, language codes stay English (international standard)   |

---

## Implementation Order

1. **meeting_notes_v2** (low risk: 3 field renames)
2. **interview_transcript_v2** (low risk: 4 field renames)
3. **text_enhancement_v1** (low risk: 4 field renames)

Each update follows the same pattern:

- Rename field keys in schema.json
- Update example.json with new keys
- Update prompt.jinja examples with new keys
- Update README.md documentation
- Validate JSON

---

## Frontend/Client Integration

Once templates are updated, frontend can be updated on its own timeline:

**Current state (server returns Vietnamese keys, UI expects old English keys):**

- UI sees unfamiliar keys → ignored/dropped
- Core keys (title, summary, tags) still work
- Template output partially functional

**After frontend update (selector mapping):**

- UI updated to look for Vietnamese keys in template-specific contexts
- Full functionality restored
- Dual support for fallback if needed

**No API changes needed** – just template schema updates and frontend selector updates.

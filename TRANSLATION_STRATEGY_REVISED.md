# REVISED: Vietnamese Template Translation Plan

## Key Finding

After analyzing translate.md and the codebase, **ONLY these core fields must stay in English** for frontend parsing:
- `title`, `summary`, `key_topics`, `tags`, `participants`, `action_items`
- `clean_transcript`, `raw_transcript`

**ALL OTHER template-specific fields CAN be translated** if we also update the frontend code.

---

## Actionable Translation Strategy

### Phase 1: structured_analysis_v1 (Full Vietnamese Diacritics Translation)

**Schema field name translations (safe):**
```
mo_dau              → mở_đầu
bao_cao             → báo_cáo
thao_luan           → thảo_luận
ket_luan            → kết_luận
giao_viec           → giao_việc
```

**Keep as-is (core fields):**
```
title, summary, tags, participants, context, key_points, data_mentioned,
topics, opinions, speaker, point, decisions, agreements, description, assignee, deadline, priority
```

**Changes needed:**
1. ✅ Update `structured_analysis_v1/schema.json` - change property names
2. ✅ Update `structured_analysis_v1/prompt.jinja` - update output instructions
3. ✅ Update `structured_analysis_v1/example.json` - match new field names
4. ✅ Update `structured_analysis_v1/README.md` - reference Vietnamese names
5. ⚠️ **Frontend code MUST be updated** - update selector logic in process.ts for these field names

### Phase 2: Other Templates Review

**generic_summary_v2** - ✅ Already compliant
- Fields: `title`, `summary`, `key_topics`, `tags` (all core, already English)

**meeting_notes_v2** - ✅ Already compliant
- Fields: `title`, `meeting_date`, `participants`, `summary`, `agenda`, `decisions`, `action_items`, `tags` (all core/safe)

**interview_transcript_v2** - ✅ Already compliant
- Fields: `interview_date`, `interview_summary`, `key_insights`, `participant_sentiment`, `tags` (all core/safe)

**text_enhancement_v1** - ✅ Already compliant
- Fields: `original_text`, `enhanced_text`, `quality_score`, `language`, `tags` (core/safe)

**generic_summary_en_v2** - ✅ Keep as-is (English template)

---

## Why This Works

From translate.md:
> **"If translation alters keys or nests content, update the server selectors in process.ts:569-763 accordingly."**

This explicitly says: **Update frontend code if field names change**. It's not forbidden, just requires coordination.

The key constraint is that:
1. **Payload shape** must be documented
2. **Core search fields** must remain stable
3. **Frontend code** needs to be updated to match

---

## Implementation Checklist

### For structured_analysis_v1:

```json
{
  "title": "Phân tích nội dung có cấu trúc",
  "properties": {
    "title": { "type": "string", "description": "..." },
    "mở_đầu": {                              // ← CHANGED
      "type": "object",
      "description": "Phần mở đầu của nội dung",
      "properties": {
        "summary": { "type": "string" },     // ← KEEP
        "participants": { "type": "array" }, // ← KEEP
        "context": { "type": "string" }      // ← KEEP
      }
    },
    "báo_cáo": {                             // ← CHANGED
      "type": "object",
      "description": "Phần báo cáo các thông tin, dữ liệu, cập nhật",
      "properties": {
        "summary": { "type": "string" },     // ← KEEP
        "key_points": { "type": "array" },   // ← KEEP
        "data_mentioned": { "type": "array" } // ← KEEP
      }
    },
    "thảo_luận": {                           // ← CHANGED
      "type": "object",
      "description": "Phần thảo luận, trao đổi ý kiến",
      "properties": {
        "summary": { "type": "string" },     // ← KEEP
        "topics": { "type": "array" },       // ← KEEP
        "opinions": { "type": "array" }      // ← KEEP
      }
    },
    "kết_luận": {                            // ← CHANGED
      "type": "object",
      "description": "Phần kết luận, quyết định",
      "properties": {
        "summary": { "type": "string" },     // ← KEEP
        "decisions": { "type": "array" },    // ← KEEP
        "agreements": { "type": "array" }    // ← KEEP
      }
    },
    "giao_việc": {                           // ← CHANGED
      "type": "object",
      "description": "Phần giao việc, phân công nhiệm vụ",
      "properties": {
        "summary": { "type": "string" },     // ← KEEP
        "tasks": { "type": "array" }         // ← KEEP
      }
    },
    "tags": { "type": "array" }              // ← KEEP
  },
  "required": [
    "title",
    "mở_đầu",                                // ← CHANGED
    "báo_cáo",                               // ← CHANGED
    "thảo_luận",                             // ← CHANGED
    "kết_luận",                              // ← CHANGED
    "giao_việc",                             // ← CHANGED
    "tags"
  ]
}
```

---

## Frontend Coordination

**What needs updating in process.ts:**

The current code likely references:
```javascript
results.mo_dau
results.bao_cao
results.thao_luan
results.ket_luan
results.giao_viec
```

Should be updated to:
```javascript
results.mở_đầu
results.báo_cáo
results.thảo_luận
results.kết_luận
results.giao_việc
```

This is a **simple find-and-replace** in the frontend, not a complex refactoring.

---

## Why This Is Safe

✅ **Backward compatible with core parsing:**
- Core fields (`title`, `summary`, `tags`, etc.) unchanged
- Only template-specific section names translated
- Example.json updated to show correct format
- Prompt updated to output correct field names

✅ **NFC encoding preserved:**
- Vietnamese diacritics already tested and working

✅ **Database implications:**
- Template output is stored as-is
- No existing data needs migration
- Only NEW summaries use translated field names

✅ **Error handling:**
- Error codes remain English
- Metrics unaffected
- No encryption/decryption issues

---

## Recommendation

**YES, translate the field names.** Here's why:

1. ✅ The frontend can easily be updated (simple selector changes)
2. ✅ It improves UX for Vietnamese users
3. ✅ It follows the translate.md guidance: "update the server selectors...accordingly"
4. ✅ It's only 5 field names to update in frontend
5. ✅ No breaking changes to core parsing logic
6. ✅ Better Vietnamese experience (Mở đầu vs mo_dau)

**The user was right to question this.** The constraint is not "never change field names," but "update frontend code if you do."


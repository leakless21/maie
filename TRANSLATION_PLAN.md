# Vietnamese Template Translation Plan

## Executive Summary

This document outlines a comprehensive plan to translate MAIE template schemas and descriptions from snake_case English/mixed Vietnamese to proper Vietnamese with diacritical marks (mở_đầu instead of mo_dau), while maintaining JSON parsing compatibility and critical field names.

---

## 1. Analysis of Current State

### 1.1 Templates Inventory

| Template | Type | Status | Field Names |
|----------|------|--------|-------------|
| `structured_analysis_v1` | Complex | Already partially translated | `mo_dau`, `bao_cao`, `thao_luan`, `ket_luan`, `giao_viec` |
| `generic_summary_v2` | Simple | Already translated | `title`, `summary`, `key_topics`, `tags` |
| `meeting_notes_v2` | Medium | Already translated | `title`, `meeting_date`, `participants`, `summary`, `agenda`, `decisions`, `action_items`, `tags` |
| `interview_transcript_v2` | Medium | Already translated | `interview_date`, `interview_summary`, `key_insights`, `participant_sentiment`, `tags` |
| `text_enhancement_v1` | Functional | Already translated | `original_text`, `enhanced_text`, `quality_score`, `language`, `tags` |
| `generic_summary_en_v2` | Simple | English only | `title`, `summary`, `key_topics`, `tags` |

### 1.2 Key Constraints (from translate.md)

✅ **CRITICAL FIELD NAMES TO KEEP IN ENGLISH:**
- Field names in JSON must remain unchanged for parsing
- Frontend/API parsing expects exact field names: `summary`, `title`, `key_topics`, `tags`, `participants`, `action_items`, `clean_transcript`, `raw_transcript`
- Server normalizes strings with `.normalize("NFC")` before encryption
- Transcripts and summaries are encrypted byte-for-byte

❌ **WHAT SHOULD STAY ENGLISH:**
- Error codes
- Technical enum values
- Field names in schemas
- System-level metadata fields

✅ **WHAT CAN/SHOULD BE TRANSLATED:**
- Field `description` (used for LLM prompts and UI labels)
- Field `title` in schema (metadata)
- Enum display values (with caution for downstream matching)
- Subsection headers and labels in prompts

### 1.3 Current Translation Status

**Already in Vietnamese with diacritics:**
- `generic_summary_v2/schema.json` - Good ✓
- `meeting_notes_v2/schema.json` - Good ✓
- `interview_transcript_v2/schema.json` - Good ✓
- `text_enhancement_v1/schema.json` - Good ✓

**Needs Improvement (snake_case without diacritics):**
- `structured_analysis_v1/schema.json` - Uses `mo_dau`, `bao_cao`, `thao_luan`, `ket_luan`, `giao_viec` instead of `mở_đầu`, `báo_cáo`, `thảo_luận`, `kết_luận`, `giao_việc`

**English templates (not translated):**
- `generic_summary_en_v2/schema.json` - Keep as-is (language=English)

---

## 2. Translation Challenges & Solutions

### 2.1 Challenge: Field Name vs. Display Value Distinction

**Problem:** JSON schema uses `mo_dau` as a field name, but this must stay in English for parsing.

**Solution:** 
- **Field names remain unchanged** (`mo_dau`, `bao_cao`, etc.)
- Translate **only descriptions** and **title** in schema
- Update **prompts** to reference Vietnamese names in instructions

✅ **Example (structured_analysis_v1):**
```json
{
  "mo_dau": {
    "type": "object",
    "description": "Phần mở đầu của nội dung",  // ← TRANSLATABLE
    "properties": {
      "summary": {
        "type": "string",
        "description": "Tóm tắt nội dung phần mở đầu",  // ← TRANSLATABLE
        ...
      }
    }
  }
}
```

### 2.2 Challenge: Enum Values Localization

**Problem:** Enum values like `["cao", "trung bình", "thấp"]` are already Vietnamese but need consistency check.

**Solution:**
- Verify enum values are standard Vietnamese
- Document that DB will store Vietnamese strings
- Ensure consistency across templates

### 2.3 Challenge: UTF-8 Encoding & NFC Normalization

**Problem:** Vietnamese diacritics (á, ả, ã, ạ, etc.) can be represented in multiple Unicode forms.

**Solution:**
- Always use **composed form (NFC)**: `"mở_đầu"` not decomposed
- Server already normalizes with `.normalize("NFC")`
- Verify in validation step that all Vietnamese strings are in NFC form
- JSON files should be UTF-8 encoded

### 2.4 Challenge: Prompt Rendering

**Problem:** Jinja templates need to work with both field names (English) and display names (Vietnamese).

**Solution:**
- Keep field references in Jinja as-is (English)
- Update instruction text to reference Vietnamese section names
- Use comments in prompts to clarify mapping

---

## 3. Detailed Translation Map

### 3.1 structured_analysis_v1

**Current Field Names → Keep As-Is (JSON parsing):**
```
mo_dau, bao_cao, thao_luan, ket_luan, giao_viec
summary, participants, context, key_points, data_mentioned
topics, opinions, speaker, point, decisions, agreements, tasks
description, assignee, deadline, priority
```

**Description Translations (Update in schema.json):**
```
mo_dau: "Phần mở đầu của nội dung" (currently correct)
bao_cao: "Phần báo cáo các thông tin, dữ liệu, cập nhật" (currently correct)
thao_luan: "Phần thảo luận, trao đổi ý kiến" (currently correct)
ket_luan: "Phần kết luận, quyết định" (currently correct)
giao_viec: "Phần giao việc, phân công nhiệm vụ" (currently correct)
```

**Prompt Updates (update in prompt.jinja):**
- Replace `{{ context.mo_dau }}` references with clear Vietnamese labels in output
- Update instruction text: "Phân tích nội dung thành 5 phần: Mở đầu, Báo cáo, Thảo luận, Kết luận, Giao việc"

**Priority Fields (Enum):**
```
"enum": ["cao", "trung bình", "thấp", "không xác định"]
```
✅ Already proper Vietnamese, keep as-is.

### 3.2 generic_summary_v2

**Status:** ✅ Already good, fields are in English (correct), descriptions in Vietnamese.
No changes needed.

### 3.3 meeting_notes_v2

**Status:** ✅ Already good, fields are in English (correct), descriptions in Vietnamese.
No changes needed.

**Note:** Check `action_items` structure for consistency:
```json
"action_items": {
  "description": { "type": "string", "maxLength": 500 },
  "assignee": { "type": "string", "maxLength": 100 },
  "due_date": { "type": ["string", "null"] }
}
```
✅ Correct - field names in English, descriptions in Vietnamese.

### 3.4 interview_transcript_v2

**Status:** ✅ Already good.

**Note:** `participant_sentiment` enum:
```
"enum": ["positive", "neutral", "negative", "mixed"]
```
✅ Keep in English (standard international convention for sentiment analysis).

### 3.5 text_enhancement_v1

**Status:** ✅ Already good.

**Note:** `language` field uses ISO 639-1 codes:
```
"pattern": "^[a-z]{2}$"
```
✅ Correct - must remain `"vi"` for Vietnamese, `"en"` for English.

### 3.6 generic_summary_en_v2

**Status:** ✅ Keep in English (intentional language variant).
No changes needed.

---

## 4. Implementation Strategy

### Phase 1: Verification & Backup
- [ ] Verify current UTF-8 encoding of all template files
- [ ] Test NFC normalization on current Vietnamese strings
- [ ] Create backup: `git commit "backup: templates before translation update"`

### Phase 2: Primary Changes (structured_analysis_v1)
- [ ] Review current schema.json descriptions (verify they're already good)
- [ ] Update prompt.jinja to use Vietnamese labels in output instructions
- [ ] Update example.json to demonstrate expected Vietnamese field usage

### Phase 3: Consistency Audit
- [ ] Audit all schema descriptions across templates for translation quality
- [ ] Ensure all Vietnamese text uses diacritics properly
- [ ] Document enum value translations and localization decisions

### Phase 4: Testing & Validation
- [ ] Parse all schema files with JSON validator
- [ ] Verify NFC normalization on all Vietnamese strings
- [ ] Check encoding: `file -i template_files/*.json`
- [ ] Test LLM parsing with examples (if applicable)
- [ ] Update tests to reflect any schema changes

### Phase 5: Documentation
- [ ] Update this TRANSLATION_PLAN.md with completion status
- [ ] Update README.md in templates/ directory
- [ ] Document enum value localization decisions
- [ ] Add comments to schemas explaining field name vs. description distinction

---

## 5. Specific File Changes Required

### 5.1 structured_analysis_v1/schema.json
**Current Issues:**
- ✅ Descriptions are already in Vietnamese (correct)
- ✅ Field names are in snake_case without diacritics (`mo_dau` instead of `mở_đầu`)

**Decision:**
- **KEEP field names as-is** (`mo_dau`, `bao_cao`, etc.) for JSON parsing compatibility
- **KEEP descriptions in Vietnamese** (they're already correct)
- ✅ No changes needed to schema.json

**Rationale:** Changing field names would break JSON parsing logic. The current schema is correct.

### 5.2 structured_analysis_v1/prompt.jinja
**Updates Needed:**
- Ensure prompt instructions reference Vietnamese section names clearly
- Update output labels to use Vietnamese (Mở đầu, Báo cáo, Thảo luận, Kết luận, Giao việc)
- Verify context variable usage

### 5.3 structured_analysis_v1/example.json
**Status:** ✅ Already good
- Contains Vietnamese content with proper diacritics
- Field names match schema
- No changes needed

### 5.4 All Other Templates
**Status:** ✅ Already compliant
- Field names in English (correct)
- Descriptions in Vietnamese (correct)
- No changes needed

---

## 6. Vietnamese Translation Reference

### 6.1 Section Names (Structured Analysis)
```
mở_đầu = opening, introduction
báo_cáo = report, briefing
thảo_luận = discussion, deliberation
kết_luận = conclusion, conclusion & decisions
giao_việc = task assignment, delegation
```

### 6.2 Common Field Names (Keep in English)
```
title → tiêu đề (metadata only, not field name)
summary → tóm tắt (metadata only)
participants → người tham gia (metadata only)
key_points → điểm chính (metadata only)
decisions → quyết định (metadata only)
tags → thẻ (metadata only)
```

### 6.3 Priority Levels (Already Translated)
```
cao = high
trung bình = medium
thấp = low
không xác định = undefined/not specified
```

### 6.4 Sentiment Values (Keep in English)
```
positive = (international standard)
neutral = (international standard)
negative = (international standard)
mixed = (international standard)
```

### 6.5 Language Codes (Keep in English)
```
vi = Vietnamese (ISO 639-1)
en = English (ISO 639-1)
```

---

## 7. Validation Checklist

Before finalizing:

- [ ] All JSON files are valid JSON
- [ ] All Vietnamese strings are in UTF-8 NFC form
- [ ] Field names remain unchanged (backward compatibility)
- [ ] Example files demonstrate correct usage
- [ ] Descriptions are in proper Vietnamese with diacritics
- [ ] Enum values are consistent across templates
- [ ] Error codes remain in English
- [ ] Technical metadata fields unchanged
- [ ] No breaking changes to API contracts
- [ ] Prompts reference Vietnamese names correctly

---

## 8. Testing Strategy

### 8.1 Unit Tests
- Verify schema.json files load without errors
- Verify NFC normalization on Vietnamese strings
- Verify field names match expectations

### 8.2 Integration Tests
- Test LLM prompt rendering with templates
- Verify JSON parsing of generated summaries
- Check encryption/decryption with Vietnamese content

### 8.3 Manual Testing
- Verify example.json parses correctly
- Check UI displays Vietnamese descriptions properly
- Verify tags are stored/retrieved correctly in database

---

## 9. Risk Assessment

### Low Risk ✅
- Updating descriptions (already in Vietnamese)
- Updating prompts to use Vietnamese labels
- Updating example files

### Medium Risk ⚠️
- Changes to enum values (could affect db lookups if values change)
- Changes to field names (would break JSON parsing)

### High Risk ❌
- **NOT RECOMMENDED:** Changing field names from `mo_dau` to `mở_đầu` (breaks parsing)
- **NOT RECOMMENDED:** Changing technical enum values like sentiment types
- **NOT RECOMMENDED:** Mixing English field names with Vietnamese descriptions inconsistently

---

## 10. Final Recommendations

### ✅ DO:
1. Keep field names in English/snake_case for JSON parsing compatibility
2. Ensure all Vietnamese descriptions use proper diacritics (NFC form)
3. Keep enum values consistent (don't rename once in use)
4. Document the distinction between field names and display names
5. Verify UTF-8 encoding on all files

### ❌ DON'T:
1. Change field names from current values (breaks JSON parsing)
2. Translate error codes or technical fields
3. Change international standard values (sentiment, language codes)
4. Mix normalization forms in UTF-8 encoding
5. Change required field lists in schema

### 🤔 CURRENT STATE:
The templates are **already well-translated**. The main issue is that `structured_analysis_v1` uses snake_case without diacritics for field names (`mo_dau` instead of `mở_đầu`), but this is actually **correct for JSON parsing**.

The descriptions are already in proper Vietnamese. No breaking changes are necessary.

---

## 11. Conclusion

The MAIE templates are **already properly translated to Vietnamese** with the following characteristics:

1. **Field names:** Remain in English/snake_case (required for JSON parsing)
2. **Descriptions:** In proper Vietnamese with diacritics (correct)
3. **Enum values:** Vietnamese where appropriate, English for international standards
4. **Encoding:** UTF-8 with NFC normalization (correct)

**No breaking changes are needed.** The focus should be on:
- Verifying current Vietnamese text quality
- Ensuring consistency across prompts and labels
- Documenting the field name vs. display value distinction
- Testing with Vietnamese content to ensure everything works

---

## Appendix: File Checklist

```
templates/
├── base/
│   └── structured_output_v1.jinja ✅
├── generic_summary_en_v2/
│   ├── schema.json ✅ (English - intentional)
│   ├── prompt.jinja ✅
│   └── example.json ✅
├── generic_summary_v2/
│   ├── schema.json ✅ (Vietnamese)
│   ├── prompt.jinja ✅
│   └── example.json ✅
├── interview_transcript_v2/
│   ├── schema.json ✅ (Vietnamese)
│   ├── prompt.jinja ✅
│   └── example.json ⚠️ (not found in listing)
├── meeting_notes_v2/
│   ├── schema.json ✅ (Vietnamese)
│   ├── prompt.jinja ✅
│   └── example.json ✅
├── structured_analysis_v1/
│   ├── schema.json ⏳ (needs description review)
│   ├── prompt.jinja ⏳ (needs prompt review)
│   ├── example.json ✅
│   └── README.md ✅
└── text_enhancement_v1/
    ├── schema.json ✅ (Vietnamese)
    ├── prompt.jinja ✅
    └── example.json ✅
```


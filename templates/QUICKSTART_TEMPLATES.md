# Templates Quick Reference Guide

## Template Selection Flowchart

```
START
  ↓
Do you need BOTH English & Vietnamese?
  ├─ YES → Need bullet points?
  │         ├─ YES → Use: bilingual_summary_bullets_v1 ⭐
  │         └─ NO → Create custom bilingual template
  └─ NO → Vietnamese only?
           ├─ YES → Need bullet points?
           │        ├─ YES → Use: generic_summary_bullets_v2 ⭐
           │        └─ NO → Use: generic_summary_v2
           └─ NO → English only?
                    ├─ YES → Need bullet points?
                    │        ├─ YES → Create custom English bullets template
                    │        └─ NO → Use: generic_summary_en_v2
                    └─ NO → Choose specialized template
                             (interview_transcript_v2, meeting_notes_v2, etc.)
```

---

## Template Overview Table

| Template                                | Language      | Format     | Best For                         | Fields                                                       |
| --------------------------------------- | ------------- | ---------- | -------------------------------- | ------------------------------------------------------------ |
| **generic_summary_v2**                  | 🇻🇳 Vietnamese | Paragraph  | General Vietnamese summaries     | title, summary, key_topics, tags                             |
| **generic_summary_en_v2**               | 🇬🇧 English    | Paragraph  | General English summaries        | title, summary, key_topics, tags                             |
| **generic_summary_bullets_v2** ⭐ NEW   | 🇻🇳 Vietnamese | Bullets    | Vietnamese structured summaries  | title, summary (with bullets), key_topics, tags              |
| **bilingual_summary_bullets_v1** ⭐ NEW | 🇻🇳🇬🇧 Both     | Bullets    | Bilingual professional summaries | title_en, title_vi, summary_en, summary_vi, key_topics, tags |
| interview_transcript_v2                 | 🇻🇳 Vietnamese | Structured | Interview analysis               | Specialized interview fields                                 |
| meeting_notes_v2                        | 🇻🇳 Vietnamese | Structured | Meeting documentation            | Meeting-specific fields                                      |

---

## Key Differences: New vs. Original Templates

### generic_summary_v2 vs. generic_summary_bullets_v2

**generic_summary_v2 (Original)**

```json
{
  "summary": "Comprehensive overview paragraph that synthesizes all content in one flowing narrative without specific point breakdown."
}
```

**generic_summary_bullets_v2 (New with Bullets)**

```json
{
  "summary": "Opening overview statement setting context.\n\n• Key point 1\n• Key point 2\n• Key point 3\n• ...(up to 10 bullets)"
}
```

**When to upgrade:**

- Content needs better scannability → Use bullets
- Action items are important → Use bullets
- Audience prefers quick reference → Use bullets
- Narrative flow is critical → Keep original

---

### New: Bilingual Summary with Bullets

**Feature**: Simultaneous English & Vietnamese output

```json
{
  "title_en": "Strategic Quarterly Review Meeting",
  "title_vi": "Cuộc họp rà soát chiến lược hàng quý",

  "summary_en": "Meeting overview in English...\n\n• Bullet 1 (English)\n• Bullet 2 (English)",
  "summary_vi": "Tóm tắt cuộc họp tiếng Việt...\n\n• Điểm 1 (Tiếng Việt)\n• Điểm 2 (Tiếng Việt)",

  "key_topics": ["Strategic Planning", "Quarterly Review", ...],
  "tags": ["meeting", "strategy", ...]
}
```

**Advantages:**

- ✅ Guaranteed translation consistency
- ✅ No separate translation step needed
- ✅ Single output for international teams
- ✅ Professional bilingual documentation

---

## Output Examples

### Example 1: generic_summary_bullets_v2

**Input**: Lecture about AI security

**Output**:

```json
{
  "title": "Các nguyên tắc bảo mật trong phát triển ứng dụng AI",
  "summary": "Bài giảng về các nguyên tắc cơ bản để bảo vệ ứng dụng AI khỏi các lỗ hổng bảo mật.\n\n• Xác thực đầu vào của người dùng trước khi xử lý dữ liệu\n• Sử dụng các truy vấn được tham số hóa để ngăn SQL injection\n• Triển khai cơ chế xác thực và ủy quyền mạnh mẽ\n• Mã hóa dữ liệu nhạy cảm cả trong quá trình truyền lẫn khi lưu trữ\n• Thiết lập ghi nhật ký và giám sát cho các sự kiện bảo mật",
  "key_topics": ["Bảo mật", "Xác thực", "Mã hóa", "Monitoring"],
  "tags": ["security", "ai", "best practices"]
}
```

---

### Example 2: bilingual_summary_bullets_v1

**Input**: International product launch meeting

**Output**:

```json
{
  "title_en": "Global Product Launch Strategy Meeting",
  "title_vi": "Cuộc họp chiến lược ra mắt sản phẩm toàn cầu",

  "summary_en": "Strategic planning session for coordinating product launch across North America, Europe, and Asia-Pacific regions. Meeting covered market analysis, timeline, resource allocation, and success metrics.\n\n• Market research indicates strong demand in three key regions\n• Launch timeline: prototype (Month 1-2), beta testing (Month 3-4), full launch (Month 5)\n• Budget allocation: $5M for marketing, $3M for infrastructure\n• Team assignments defined by region and function\n• Success metrics: 10K+ users in first month",

  "summary_vi": "Phiên họp lập kế hoạch chiến lược để điều phối ra mắt sản phẩm trên các khu vực Bắc Mỹ, Châu Âu và châu Á-Thái Bình Dương. Cuộc họp bao gồm phân tích thị trường, tiến độ, phân bổ tài nguyên và các chỉ số thành công.\n\n• Nghiên cứu thị trường cho thấy nhu cầu mạnh trong ba khu vực chính\n• Tiến độ ra mắt: tạo mẫu (Tháng 1-2), thử nghiệm beta (Tháng 3-4), ra mắt toàn bộ (Tháng 5)\n• Phân bổ ngân sách: 5 triệu đô la cho marketing, 3 triệu đô la cho cơ sở hạ tầng\n• Gán nhiệm vụ cho đội theo khu vực và chức năng\n• Chỉ số thành công: 10K+ người dùng trong tháng đầu",

  "key_topics": [
    "Product Launch",
    "Market Analysis",
    "Timeline",
    "Budget",
    "Regional Strategy"
  ],
  "tags": ["strategy", "product", "launch", "global", "planning"]
}
```

---

## Bullet Point Formatting Rules

### DO ✅

- Use "• " prefix consistently
- Start each bullet with a capital letter
- Keep bullets to 1-2 sentences maximum
- Maintain parallel structure
- Order logically (chronological, importance, etc.)
- Include supporting details when needed

### DON'T ❌

- Mix bullet styles (•, -, \*, etc.)
- Create sub-bullets (not in current schema)
- Write multiple sentences as one bullet
- Use inconsistent grammatical structures
- Create bullets that are too vague
- Include unrelated information in a bullet

### Good Bullet Examples

```
• Successfully migrated 100K records to new database system
• Implemented automated backup with 99.9% reliability
• Reduced query response time from 5s to 500ms
• Established monitoring alerts for critical thresholds
```

### Poor Bullet Examples

```
✗ • The database and system were migrated and it was successful
✗ • Query optimization
✗ • Did performance tuning and fixed issues and improved everything
✗ • Something about backups and also monitoring
```

---

## Implementation Checklist

### For Using generic_summary_bullets_v2

- [ ] Input text prepared (Vietnamese or mixed content)
- [ ] Understand the content's main topics
- [ ] Identify 5-10 key points to highlight
- [ ] Schema validation ready
- [ ] Example format reviewed
- [ ] Output checked against schema
- [ ] Bullet formatting verified

### For Using bilingual_summary_bullets_v1

- [ ] Input text ready (any language or mixed)
- [ ] Both English and Vietnamese language proficiency available
- [ ] Bilingual consistency guidelines understood
- [ ] Title translation pairs prepared
- [ ] Bullet point parallelism maintained across languages
- [ ] Schema requires both language fields
- [ ] Output validated for both languages
- [ ] key_topics and tags in English only

---

## Common Use Cases

### Use Case 1: Executive Summary

**Recommended Template**: bilingual_summary_bullets_v1

- Senior leaders from multiple countries need quick understanding
- Bullet format aids quick scanning
- Bilingual support reaches wider audience

### Use Case 2: Training Material

**Recommended Template**: generic_summary_bullets_v2

- Students benefit from structured bullet points
- Easy to convert to slide format
- Quick reference for study

### Use Case 3: Meeting Notes

**Recommended Template**: generic_summary_bullets_v1

- Action items are clear
- Decisions highlighted
- Archive for future reference

### Use Case 4: Documentation

**Recommended Template**: bilingual_summary_bullets_v1

- Supports international documentation
- Clear structure aids navigation
- Maintainable across teams

---

## Migration Path

**From generic_summary_v2 → generic_summary_bullets_v2**

1. Copy existing summary text
2. Extract 5-10 main points
3. Convert each point to a bullet
4. Keep opening summary line
5. Verify no information lost
6. Format with "• " prefix

**From generic_summary_v2 → bilingual_summary_bullets_v1**

1. Take original Vietnamese content
2. Create English version with bullets
3. Create Vietnamese version with matching bullets
4. Ensure titles match across languages
5. Use key_topics and tags in English
6. Validate both summaries are consistent

---

## Troubleshooting

| Issue                           | Solution                              |
| ------------------------------- | ------------------------------------- |
| Bullets too long                | Split into 2-3 shorter bullets        |
| Missing key points              | Review input, add missing bullet      |
| English/Vietnamese inconsistent | Review both summaries side-by-side    |
| Too many bullets                | Consolidate or remove least important |
| Title too long                  | Edit to max 15 words                  |
| Key topics not extracted        | Review content, identify main themes  |

---

## Quick Start

### Fastest Way to Get Started

1. **Choose your template**:

   - Vietnamese only? → `generic_summary_bullets_v2`
   - Bilingual needed? → `bilingual_summary_bullets_v1`

2. **Review the example**:

   - Open `example.json` in chosen template

3. **Understand the schema**:

   - Check `schema.json` for required fields

4. **Review the prompt**:

   - Check `prompt.jinja` for guidelines and examples

5. **Generate output**:
   - Feed your content to the template
   - Validate against schema
   - Use the structured output

---

## Resources

- **Full Analysis**: See `TEMPLATES_ANALYSIS.md` for detailed documentation
- **Template Files**: Located in `/maie/templates/` directory
- **Base Template**: `base/structured_output_v1.jinja`
- **Extended Templates**: Review examples in each template directory

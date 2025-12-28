# Bilingual Template V1 - Structure Update Summary

## Changes Made

The `bilingual_summary_bullets_v1` template has been restructured to make **Vietnamese the primary language** with **bullet points as a separate array field** for better parsing and display.

---

## New Schema Structure

### Required Fields (Vietnamese Primary)

```json
{
  "title": "string",              // Vietnamese only - for Android app parsing
  "summary": "string",            // Vietnamese paragraph (NO bullet points)
  "bullet_points": ["string"],    // Vietnamese array of key points
  "key_topics": ["string"],       // Vietnamese array of topics
  "tags": ["string"]              // Array of tags (Vietnamese or English)
}
```

### Optional Fields (English Secondary)

```json
{
  "summary_en": "string",         // English paragraph summary
  "bullet_points_en": ["string"]  // English array of key points
}
```

---

## Key Changes

### 1. **Single Title Field** ✅
**Before:** `title_en` and `title_vi` (two separate fields)
**After:** `title` (single Vietnamese field)

**Reason:** Android app parsing logic looks for a single `title` field. This ensures compatibility.

---

### 2. **Plain Text Summary** ✅
**Before:** Summary contained embedded bullet points with `\n\n• ` formatting
```json
"summary_vi": "Overview...\n\n• Point 1\n• Point 2"
```

**After:** Summary is pure paragraph text, bullet points in separate array
```json
"summary": "Overview paragraph without any bullets.",
"bullet_points": ["Point 1", "Point 2", "Point 3"]
```

**Reason:** Cleaner data structure, easier parsing, better display flexibility.

---

### 3. **Bullet Points as Array** ✅
**Before:** Bullet points embedded in summary string
**After:** Dedicated `bullet_points` array field

**Benefits:**
- Android app can display as formatted list (like `key_topics`)
- Easier to iterate and render with bullet character: `• Point 1`
- Better data structure for future processing

---

### 4. **Vietnamese as Primary Language** ✅
**Before:** English fields were required, Vietnamese secondary
**After:** Vietnamese fields are required, English optional

**Changed Fields:**
- `key_topics`: Now in Vietnamese (was English)
- `tags`: Now in Vietnamese or English (was English only)
- All required fields: Vietnamese

**English fields are now optional:**
- `summary_en`: Optional English summary
- `bullet_points_en`: Optional English bullet array

---

## Display in Android App

### How It Will Be Displayed

```
╔════════════════════════════════════════╗
║  Tác động của AI lên năng suất...      ║
╠════════════════════════════════════════╣
│                                        │
│  Summary:                              │
│  Bài trình bày toàn diện về cách      │
│  công nghệ AI biến đổi năng suất...   │
│                                        │
│  ──────────────────────────────────    │
│                                        │
│  Bullet Points:                        │
│  • Công cụ AI tự động hóa các tác vụ  │
│  • Tính năng gợi ý nội dung tăng      │
│  • Tối ưu hóa quy trình công việc     │
│  • Khung quản trị đảm bảo tiêu chuẩn  │
│  • Tích hợp AI yêu cầu quản lý thay đổi│
│  • Phân tích được hỗ trợ bởi AI       │
│                                        │
│  ──────────────────────────────────    │
│                                        │
│  Summary En:                           │  (if provided)
│  Comprehensive presentation on how...  │
│                                        │
│  ──────────────────────────────────    │
│                                        │
│  Bullet Points En:                     │  (if provided)
│  • AI tools automate repetitive tasks │
│  • Content suggestion features enhance │
│  • Smart workflow optimization...     │
│                                        │
│  ──────────────────────────────────    │
│                                        │
│  Key Topics:                           │
│  • Trí tuệ nhân tạo                   │
│  • Năng suất nơi làm việc             │
│  • Tự động hóa tác vụ                 │
│                                        │
│  ──────────────────────────────────    │
│                                        │
│  Tags:                                 │
│  • ai                                  │
│  • năng suất                           │
│  • tự động hóa                         │
│                                        │
╚════════════════════════════════════════╝
```

### Android Parsing Support

The Android app's `OverlayController.kt` already handles:
- ✅ Single `title` field (line 4086-4099)
- ✅ `summary` as plain text string (line 4072)
- ✅ `bullet_points` array displayed with `• ` prefix (line 3881-3884)
- ✅ `key_topics` array displayed with `• ` prefix
- ✅ Optional `summary_en` and `bullet_points_en` fields

**No code changes needed in Android app!**

---

## Example Output

### Minimal Example (Vietnamese Only)

```json
{
  "title": "Cuộc họp chiến lược Q3",
  "summary": "Cuộc họp tập trung vào phát triển sản phẩm mới với mục tiêu tăng thị phần.",
  "bullet_points": [
    "Nhu cầu thị trường tăng 35%",
    "Lộ trình phát triển 6 tháng",
    "Ngân sách 500 triệu đồng",
    "Mục tiêu tăng 25% thị phần"
  ],
  "key_topics": [
    "Chiến lược",
    "Phát triển sản phẩm",
    "Thị trường"
  ],
  "tags": ["chiến lược", "sản phẩm", "thị trường"]
}
```

### Full Example (Bilingual)

```json
{
  "title": "Thực hành lập trình bảo mật",
  "summary": "Buổi đào tạo về các thực hành lập trình bảo mật thiết yếu nhằm bảo vệ ứng dụng.",
  "bullet_points": [
    "Xác thực đầu vào người dùng",
    "Sử dụng truy vấn tham số hóa",
    "Triển khai xác thực mạnh mẽ",
    "Mã hóa dữ liệu nhạy cảm",
    "Ghi nhật ký bảo mật"
  ],
  "summary_en": "Training session on essential secure coding practices to protect applications.",
  "bullet_points_en": [
    "Validate user input",
    "Use parameterized queries",
    "Implement robust authentication",
    "Encrypt sensitive data",
    "Establish security logging"
  ],
  "key_topics": [
    "Bảo mật",
    "Lập trình",
    "Mã hóa"
  ],
  "tags": ["bảo mật", "lập trình", "mã hóa"]
}
```

---

## Migration Notes

### From Old Format to New Format

**Old format:**
```json
{
  "title_en": "Meeting",
  "title_vi": "Cuộc họp",
  "summary_vi": "Overview...\n\n• Point 1\n• Point 2",
  "key_topics": ["Topic A", "Topic B"]  // English
}
```

**New format:**
```json
{
  "title": "Cuộc họp",                    // Single Vietnamese title
  "summary": "Overview...",               // Plain text, no bullets
  "bullet_points": ["Point 1", "Point 2"], // Separate array
  "key_topics": ["Chủ đề A", "Chủ đề B"] // Vietnamese
}
```

---

## Advantages of New Structure

### 1. **Better Android Compatibility** ✅
- Single `title` field matches app's parsing logic
- Arrays display automatically with bullet formatting

### 2. **Cleaner Data Structure** ✅
- Separation of concerns: summary vs. bullet points
- No embedded formatting characters in strings
- Easier to validate and process

### 3. **Flexible Display** ✅
- App can choose to display summary only, bullets only, or both
- Easier to apply custom formatting per platform

### 4. **Vietnamese Priority** ✅
- Vietnamese is the primary language (required fields)
- English is secondary (optional enhancement)
- Aligns with target audience

### 5. **Better Database Storage** ✅
- Separate fields enable better querying
- Can search/filter by bullet points independently
- Structured data for analytics

---

## Usage Guidelines

### When to Include English Fields

**Include English fields when:**
- International audience needs access
- Documentation requires bilingual support
- Business stakeholders prefer English
- Export/sharing with English speakers

**Omit English fields when:**
- Vietnamese-only audience
- Internal use only
- Faster processing needed
- Storage optimization required

---

## Testing Checklist

- [ ] Generate JSON with new structure
- [ ] Verify `title` field is Vietnamese
- [ ] Verify `summary` has no bullet characters
- [ ] Verify `bullet_points` is an array
- [ ] Verify `key_topics` is in Vietnamese
- [ ] Upload to Android app
- [ ] Check display shows bullets correctly
- [ ] Test with English fields present
- [ ] Test with English fields omitted
- [ ] Verify copy/paste works

---

## Files Updated

1. **schema.json** - Updated field definitions and requirements
2. **prompt.jinja** - Updated instructions and examples
3. **example.json** - Updated to show new structure

All files are located in:
```
/maie/templates/bilingual_summary_bullets_v1/
```

---

## Backward Compatibility

⚠️ **Breaking Change**: This is NOT backward compatible with the old format.

**Old Android app behavior:**
- Would look for `title_en` or `title_vi` → Now looks for `title` ✅
- Would display `summary_vi` with embedded bullets → Now displays `summary` + `bullet_points` array ✅

**Server/MAIE changes needed:**
- Update template usage to generate new structure
- Existing stored results with old format will still display (app is flexible)

---

## Summary

The bilingual template now has:
- ✅ Vietnamese as primary language
- ✅ Single `title` field for app parsing
- ✅ Clean `summary` text without bullets
- ✅ Separate `bullet_points` array
- ✅ Vietnamese `key_topics`
- ✅ Optional English fields
- ✅ Better data structure for parsing and display

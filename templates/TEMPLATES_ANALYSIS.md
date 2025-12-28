# Template Analysis & New Templates Documentation

## Overview
This document provides analysis of existing templates and describes the two new templates created with bilingual support and bullet-point summaries.

---

## Existing Templates Analysis

### 1. **base/structured_output_v1.jinja**
The foundational template that all other templates extend. Provides:
- Standard structure for LLM prompts requiring JSON output
- Blocks for role definition, core task, step-by-step guides, guidelines, and examples
- Language configuration (defaults to Vietnamese)
- Extensible block system for customization

### 2. **generic_summary_v2/**
Vietnamese summary template for general content analysis:
- **Schema**: Title, Summary, Key Topics (array), Tags (array)
- **Key Features**: 
  - Concise titles (max 15 words)
  - Comprehensive summaries without bullet points
  - 1-15 key topics
  - 1-10 categorical tags
- **Use Cases**: General content summaries, meeting notes, lectures

### 3. **generic_summary_en_v2/**
English version of the generic summary template:
- Identical schema to generic_summary_v2
- English language output
- Same field constraints and requirements

### 4. **Other Templates**
- `interview_transcript_v2/`: Specialized for interview transcripts
- `meeting_notes_v2/`: Optimized for meeting summaries
- `structured_analysis_v1/`: For complex structured data extraction
- `text_enhancement_v1/`: For text improvement and refinement

---

## New Templates Created

### 1. **bilingual_summary_bullets_v1/** ⭐ NEW
**Purpose**: Bilingual (English & Vietnamese) summary with structured bullet points

#### Schema Structure
```json
{
  "title_en": "English title (max 15 words)",
  "title_vi": "Vietnamese title (max 15 words)",
  "summary_en": "English summary with 5-10 bullet points (max 3000 chars)",
  "summary_vi": "Vietnamese summary with 5-10 bullet points (max 3000 chars)",
  "key_topics": ["Main themes in English"],
  "tags": ["categorical tags in English"]
}
```

#### Key Features
- **Bilingual Output**: Complete summaries in both English and Vietnamese
- **Bullet-Point Format**: 5-10 structured bullet points per language
- **Parallel Structure**: Maintains consistency of meaning across languages
- **Professional Terminology**: Uses accurate technical terms in each language
- **Comprehensive Coverage**: Captures key ideas, arguments, and conclusions

#### Use Cases
- International business meetings and presentations
- Multilingual documentation projects
- Training sessions with diverse audiences
- Research projects requiring bilingual analysis
- Compliance documentation

#### Example Output Format
```json
{
  "title_en": "Q3 Strategic Meeting: New Product Development",
  "title_vi": "Cuộc họp chiến lược Q3: Phát triển sản phẩm mới",
  "summary_en": "Strategic meeting overview...\n\n• Bullet point one\n• Bullet point two\n• ...",
  "summary_vi": "Tóm tắt cuộc họp chiến lược...\n\n• Điểm chính một\n• Điểm chính hai\n• ...",
  "key_topics": ["Product Development", "Market Analysis", ...],
  "tags": ["strategy", "business", "product", ...]
}
```

---

### 2. **generic_summary_bullets_v2/** ⭐ NEW
**Purpose**: Vietnamese generic summary with structured bullet points (enhanced version of generic_summary_v2)

#### Schema Structure
```json
{
  "title": "Title in Vietnamese (max 15 words)",
  "summary": "Summary with 5-10 bullet points in Vietnamese (max 3000 chars)",
  "key_topics": ["Main themes"],
  "tags": ["categorical tags"]
}
```

#### Key Features
- **Bullet-Point Format**: Organizes summaries into clear, actionable bullet points
- **Improved Readability**: Easier to scan and understand compared to paragraph format
- **Standard Format**: Uses "• " prefix for consistent bullet styling
- **Concise Points**: Each bullet contains one main idea
- **Enhanced Structure**: Opening statement + 5-10 bullet points

#### Bullet Point Guidelines
- Each bullet focuses on one distinct idea
- Points follow a logical progression
- Include supporting details when relevant
- Avoid redundancy across points
- Total length max 3000 characters

#### Use Cases
- Meeting summaries with action items
- Lecture notes and training materials
- Project documentation
- Research summaries
- Quick reference documents
- Presentation takeaways

#### Example Output Format
```json
{
  "title": "Lịch sử Kinh tế Việt Nam giai đoạn Đổi Mới",
  "summary": "Bài giảng tóm lược quá trình phát triển...\n\n• Điểm chính 1\n• Điểm chính 2\n• Điểm chính 3\n• ...",
  "key_topics": ["Đổi Mới", "Kinh tế thị trường", ...],
  "tags": ["kinh tế", "lịch sử", "phát triển", ...]
}
```

---

## Comparison Matrix

| Feature | generic_summary_v2 | generic_summary_bullets_v2 | bilingual_summary_bullets_v1 |
|---------|-------------------|---------------------------|------------------------------|
| Language | Vietnamese | Vietnamese | English & Vietnamese |
| Format | Paragraph | Bullet Points | Bullet Points |
| Title Fields | 1 (title) | 1 (title) | 2 (title_en, title_vi) |
| Summary Fields | 1 | 1 | 2 (summary_en, summary_vi) |
| Max Summary Length | 2000 chars | 3000 chars | 3000 chars each |
| Key Topics | 1-15 | 1-15 | 1-15 (in English) |
| Tags | 1-10 | 1-10 | 1-10 (in English) |
| Best For | General Vietnamese summaries | Vietnamese summaries with structure | Bilingual content & international use |

---

## Implementation Recommendations

### Choose generic_summary_v2 when:
- Summarizing Vietnamese-only content
- Paragraph format is preferred
- Simple, flowing narrative is needed
- No specific bullet point requirements

### Choose generic_summary_bullets_v2 when:
- Vietnamese content needs structured breakdown
- Bullet-point format improves clarity
- Action items or key takeaways are important
- Quick-reference format is needed
- Training or documentation purposes

### Choose bilingual_summary_bullets_v1 when:
- Content must be available in both languages
- International audiences are involved
- Bullet-point structure is essential
- Consistency between languages is critical
- Professional bilingual documentation is required

---

## File Structure

All templates follow this directory structure:

```
template_name/
├── prompt.jinja          # Jinja2 template for LLM prompt
├── schema.json          # JSON Schema for output validation
└── example.json         # Example output showing expected format
```

### Files Included

**bilingual_summary_bullets_v1/**
- `prompt.jinja`: Bilingual prompt with examples in English and Vietnamese
- `schema.json`: Schema supporting title_en, title_vi, summary_en, summary_vi
- `example.json`: Example showing bilingual output with bullets

**generic_summary_bullets_v2/**
- `prompt.jinja`: Vietnamese prompt with bullet-point emphasis
- `schema.json`: Vietnamese schema with extended summary field
- `example.json`: Example showing bullet-point structure

---

## Formatting Guidelines

### Bullet Point Best Practices

1. **Consistency**: Use "• " prefix for all bullets
2. **Parallel Structure**: Maintain similar grammatical structure across bullets
3. **Conciseness**: Keep each bullet to 1-2 sentences
4. **Ordering**: Arrange bullets logically (chronological, importance, etc.)
5. **Clarity**: Each bullet should stand independently

### Example Well-Formatted Summary
```
Opening overview statement that sets context for the content...

• First key point with supporting context
• Second key point showing relationship to first
• Third key point with specific detail or implication
• Fourth key point adding depth
• Fifth key point leading to conclusion
• Final bullet point summarizing main takeaway
```

---

## Usage Examples

### Using bilingual_summary_bullets_v1

**Input**: Meeting transcript (English or Vietnamese or mixed)

**Output**: Structured bilingual summary with parallel bullet points
- Perfect for international teams
- Ensures message consistency across languages
- Easy to translate reference content

### Using generic_summary_bullets_v2

**Input**: Lecture notes, meeting minutes, training material

**Output**: Vietnamese summary with organized bullet points
- Improves readability and recall
- Facilitates documentation
- Enables quick reference lookup

---

## Future Enhancement Ideas

1. **Multilingual Expansion**: Add support for more languages (Chinese, Japanese, etc.)
2. **Hierarchical Bullets**: Sub-bullets for nested bullet points
3. **Interactive Format**: Add estimated read time, difficulty level indicators
4. **Action Items**: Dedicated field for action items with ownership and deadlines
5. **Confidence Scores**: Optional confidence rating for extracted information
6. **Citations**: Track which source text supports each bullet point

---

## Migration Guide

If you have existing summaries in generic_summary_v2 format and want to convert them to bullet-point format:

1. Take the existing summary paragraph
2. Identify 5-10 key concepts or ideas
3. Convert each into a single, clear bullet point
4. Arrange bullets in logical order
5. Verify all key information is captured
6. Test output against schema

---

## Questions & Support

For questions about:
- **Template Usage**: Refer to the example.json in each template directory
- **Schema Validation**: Check schema.json for field requirements and constraints
- **Customization**: Review the prompt.jinja blocks that can be extended
- **New Features**: Consider creating a new template variant based on existing ones

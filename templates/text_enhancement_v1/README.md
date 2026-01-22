# Text Enhancement Template (v1)

## Overview

The **Text Enhancement Template** is designed to fix clear mistakes in transcribed Vietnamese text while **preserving the original as much as possible**. This template prioritizes minimal changes - only correcting objective errors such as spelling mistakes, missing punctuation, and incorrect capitalization. It does NOT rephrase or "improve" text for style or clarity.

**Philosophy: Fix mistakes, don't change content.**

## Purpose

- **Primary Use Case**: Fix objective errors in Vietnamese transcribed text from audio/speech
- **Key Principle**: Make ONLY minimal corrections for clear mistakes
- **Secondary Use Cases**:
  - Quality control for transcription services
  - Text normalization for downstream processing
  - Error detection in ASR outputs

## Template Structure

This template follows the project's standard three-file bundle structure:

```
text_enhancement_v1/
├── prompt.jinja      # Jinja2 template with system instructions and examples
├── schema.json       # JSON Schema defining the output structure
└── example.json      # Sample output for reference
```

## Output Schema

### Fields

1. **original_text** (string, max 50000 chars)

   - The exact input text as provided
   - Used for reference and comparison

2. **enhanced_text** (string, max 50000 chars)

   - Text with ONLY necessary corrections applied
   - Minimal changes - preserves original structure and word choice
   - Fixes only: spelling, clear grammar errors, missing punctuation, capitalization, phonetic foreign words

3. **corrections** (array, max 200 items)

   - Detailed tracking of each correction made
   - Each correction includes:
     - **type**: Category (spelling, grammar, punctuation, capitalization, foreign_word, structure, filler_removal)
     - **location**: Where in text (e.g., "đầu câu", "giữa câu")
     - **original**: Text before correction
     - **corrected**: Text after correction
     - **reason**: Brief explanation

4. **quality_score** (float, 0.0-1.0)

   - Quality rating of original text
   - 0.9-1.0: Nearly perfect
   - 0.7-0.9: Good quality
   - 0.5-0.7: Average quality
   - 0.3-0.5: Poor quality
   - 0.0-0.3: Very poor quality

5. **language** (string, 2 chars)
   - ISO 639-1 language code (e.g., "vi")

## Correction Types

The template fixes ONLY these types of clear errors:

1. **spelling**: Typos and misspellings (e.g., "thế gới" → "thế giới")
2. **grammar**: Clear errors like subject-verb disagreement (NOT stylistic improvements)
3. **punctuation**: Missing periods at sentence ends, question marks (NOT stylistic commas)
4. **capitalization**: Sentence beginnings, proper nouns only
5. **foreign_word**: Phonetic transcriptions → original spelling (e.g., "phây búc" → "Facebook")
6. **structure**: ONLY when grammatically incorrect (NOT for "clarity" or "flow")
7. **filler_removal**: ONLY excessive repetition that impedes understanding

**What is NOT corrected:**

- Word choice (even if another word could be "better")
- Sentence structure (unless grammatically wrong)
- Style or tone
- Informal language
- Natural speech patterns

## Key Features

### 1. Maximum Preservation

**PRIMARY PRINCIPLE: Change as little as possible**

- Maintains original intent, meaning, word choice, and structure
- Makes ONLY minimal corrections for objective errors
- Does NOT rephrase, reword, or "improve" text
- Does NOT add or remove content for style/clarity
- Preserves speaker's tone, style, and natural speech patterns
- **When in doubt, keep the original**

### 2. Foreign Word Restoration

Automatically converts Vietnamese phonetic transcriptions back to original spelling:

- "lai trim" → "livestream"
- "phây búc" → "Facebook"
- "ây ai" → "AI"
- "ghít háp" → "GitHub"

### 3. Proper Noun Handling

Correctly capitalizes Vietnamese proper nouns:

- "hà nội" → "Hà Nội"
- "việt nam" → "Việt Nam"
- "hồ chí minh" → "Hồ Chí Minh"

### 4. Quality Assessment

Provides objective quality score to help prioritize review:

- High scores (>0.8): Minimal review needed
- Low scores (<0.4): Requires careful verification

### 5. Detailed Change Tracking

Every correction is documented with:

- What was changed
- Where it occurred
- Why it was changed
- Category of change

## Usage Examples

### Example 1: Moderate Quality Transcription

**Input:**

```
xin chào các bạn hôm nay tôi sẽ nói về trí tuệ nhân tạo ây ai mà hiện nay đang rất phổ biến ở việt nam
```

**Output:**

```json
{
  "original_text": "xin chào các bạn hôm nay tôi sẽ nói về trí tuệ nhân tạo ây ai mà hiện nay đang rất phổ biến ở việt nam",
  "enhanced_text": "Xin chào các bạn, hôm nay tôi sẽ nói về trí tuệ nhân tạo AI mà hiện nay đang rất phổ biến ở Việt Nam.",
  "corrections": [
    {
      "type": "capitalization",
      "location": "đầu câu",
      "original": "xin",
      "corrected": "Xin",
      "reason": "Viết hoa chữ cái đầu câu"
    },
    {
      "type": "foreign_word",
      "location": "thuật ngữ tiếng Anh",
      "original": "ây ai",
      "corrected": "AI",
      "reason": "Khôi phục từ viết tắt tiếng Anh"
    },
    {
      "type": "capitalization",
      "location": "tên địa danh",
      "original": "việt nam",
      "corrected": "Việt Nam",
      "reason": "Viết hoa tên riêng địa danh"
    }
  ],
  "quality_score": 0.68,
  "language": "vi"
}
```

### Example 2: Heavy Foreign Word Content

**Input:**

```
tôi đang xem một buổi lai trim trên phây búc về cách sử dụng ghít háp và đốc kờ để phát triển phần mềm
```

**Output:**

```json
{
  "enhanced_text": "Tôi đang xem một buổi livestream trên Facebook về cách sử dụng GitHub và Docker để phát triển phần mềm.",
  "corrections": [
    {
      "type": "foreign_word",
      "location": "từ tiếng Anh",
      "original": "lai trim",
      "corrected": "livestream",
      "reason": "Khôi phục từ tiếng Anh gốc"
    },
    {
      "type": "foreign_word",
      "location": "tên thương hiệu",
      "original": "phây búc",
      "corrected": "Facebook",
      "reason": "Khôi phục tên thương hiệu đúng"
    }
  ],
  "quality_score": 0.42,
  "language": "vi"
}
```

## Integration with MAIE Pipeline

### API Usage

The template can be used via the `/v1/transcribe` or `/v1/process` endpoints:

```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@input.wav" \
  -F "template_id=text_enhancement_v1"
```

### Python Integration

```python
from src.processors.prompt.template_loader import TemplateLoader
from src.processors.prompt.renderer import PromptRenderer
from pathlib import Path

# Initialize
loader = TemplateLoader(Path("templates"))
renderer = PromptRenderer(loader)

# Render prompt
prompt = renderer.render(
    "text_enhancement_v1",
    input_text="xin chào các bạn..."
)

# Send to LLM with schema validation
# (See src/processors/llm/ for schema validation utilities)
```

## Configuration

To enable this template system-wide, add it to your configuration:

```python
# src/config/settings.py
available_templates = [
    "generic_summary_v2",
    "meeting_notes_v2",
    "text_enhancement_v1",  # Add this line
]
```

## Best Practices

### When to Use This Template

✅ **Good for:**

- ASR/transcription outputs
- Text with phonetically transcribed foreign words
- Missing punctuation and capitalization
- Quality assessment of transcriptions

❌ **Not suitable for:**

- Creative writing enhancement
- Style/tone transformation
- Content summarization (use `generic_summary_v2`)
- Meeting structure extraction (use `meeting_notes_v2`)

### Handling Edge Cases

1. **Very Poor Quality (Quality Score < 0.3)**

   - Review enhanced text carefully
   - May require human verification
   - Consider re-transcribing audio

2. **Mixed Languages**

   - Template preserves code-switching
   - Corrects capitalization in all languages
   - Maintains technical terms

3. **Excessive Fillers**
   - Only removes when excessive
   - Preserves conversational tone
   - Documents all removals in corrections array

## Testing

Run template validation tests:

```bash
# Verify schema validity
python scripts/verify_schema_load.py text_enhancement_v1

# Test prompt rendering
python scripts/verify_prompts.py text_enhancement_v1

# Integration test
pytest tests/integration/test_text_enhancement_template.py
```

## Troubleshooting

### Common Issues

1. **Template Not Found**

   ```
   Error: Template text_enhancement_v1 not found
   ```

   **Solution**: Ensure all three files exist in `/templates/text_enhancement_v1/`

2. **Schema Validation Failure**

   ```
   Error: Output does not match schema
   ```

   **Solution**: Check LLM output format, ensure `required` fields are present

3. **Poor Correction Quality**
   **Solution**:
   - Add more examples to `prompt.jinja`
   - Tune LLM temperature (recommend 0.3-0.5)
   - Consider using larger model

## Future Enhancements

Potential improvements for v2:

- [ ] Add confidence scores per correction
- [ ] Support for dialectical variations
- [ ] Context-aware homophone disambiguation
- [ ] Integration with LLM hallucination filter
- [ ] Batch processing optimization
- [ ] Support for other languages (English, etc.)

## Related Templates

- **generic_summary_v2**: For content summarization
- **meeting_notes_v2**: For structured meeting extraction
- **interview_transcript_v2**: For interview-specific processing

## References

- Base template: `/templates/base/structured_output_v1.jinja`
- Template manager: `/src/utils/template_manager.py`
- Schema validator: `/src/processors/llm/schema_validator.py`
- LLM hallucinations: `/docs/LLM_HALLUCINATIONS.md`

---

**Version**: 1.0.0  
**Created**: 2025-12-26  
**Last Updated**: 2025-12-26  
**Maintainer**: MAIE Team

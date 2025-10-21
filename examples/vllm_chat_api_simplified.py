#!/usr/bin/env python
"""
Example: vLLM Chat API - The Simplified Approach

This demonstrates the SIMPLEST and most maintainable approach for using
vLLM's chat API in production:

✅ Inline message building (6 lines of code!)
✅ ONE system prompt template (no separate user template)
✅ No helper methods needed
✅ Automatic chat template & stop token handling

This is the exact approach used in src/processors/llm/processor.py.

Usage:
    python examples/vllm_chat_api_simplified.py
"""

from vllm import LLM, SamplingParams
from jinja2 import Template


def example_old_approach():
    """❌ OLD: Manual chat markers embedded in template."""
    print("\n" + "=" * 60)
    print("❌ OLD APPROACH (Maintenance Burden)")
    print("=" * 60)
    
    llm = LLM(model="data/models/qwen3-4b-instruct-2507-awq")
    
    # Manual chat markers - brittle and model-specific!
    manual_template = Template("""<|im_start|>system
You are an expert Vietnamese content analyst.<|im_end|>
<|im_start|>user
{{ transcript }}<|im_end|>
<|im_start|>assistant
""")
    
    prompt = manual_template.render(
        transcript="Hôm nay chúng ta họp về kế hoạch dự án Q4."
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=200,
        stop=["<|im_end|>"]  # Must remember to add this!
    )
    
    outputs = llm.generate([prompt], sampling_params)
    result = outputs[0].outputs[0].text
    
    print("\n⚠️  Problems:")
    print("  - Chat markers hardcoded (<|im_start|>, <|im_end|>)")
    print("  - Must manually specify stop tokens")
    print("  - Breaks if model uses different format")
    print("  - Template is harder to read/maintain")
    print(f"\n📊 Result:\n{result[:100]}...")


def example_simplified_inline():
    """✅ BEST: Inline message building (production approach)."""
    print("\n" + "=" * 60)
    print("✅ SIMPLIFIED APPROACH (Production)")
    print("=" * 60)
    
    llm = LLM(model="data/models/qwen3-4b-instruct-2507-awq")
    
    # 1. System prompt template (ONE file, no chat markers!)
    system_template = Template("""You are an expert Vietnamese content analyst.

Analyze the transcript and provide:
- Concise title
- Key points
- Action items

Output valid JSON matching this schema:
{
  "title": "string",
  "summary": "string",
  "action_items": ["array", "of", "strings"]
}""")
    
    # 2. Render system prompt
    system_prompt = system_template.render()
    
    # 3. Build messages inline (6 lines!)
    transcript = "Hôm nay chúng ta họp về kế hoạch dự án Q4. Cần hoàn thành trước ngày 31/12."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript to analyze:\n{transcript}"}
    ]
    
    # 4. Call chat API
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    
    result = outputs[0].outputs[0].text
    
    print("\n📝 Complete Implementation (6 lines):")
    print("```python")
    print("system_prompt = template.render()")
    print("messages = [")
    print('    {"role": "system", "content": system_prompt},')
    print('    {"role": "user", "content": f"Transcript:\\n{text}"}')
    print("]")
    print("outputs = llm.chat(messages=messages, sampling_params=sampling)")
    print("```")
    
    print("\n📂 Template Structure:")
    print("  templates/prompts/")
    print("    └── generic_summary_v2.jinja  (system prompt only)")
    print("  User message: Inlined in code")
    
    print("\n✅ Benefits:")
    print("  - Only 6 lines of code")
    print("  - ONE template file")
    print("  - No helper methods")
    print("  - Automatic stop tokens")
    print("  - Works with ANY chat model")
    print("  - 80+ lines of code deleted!")
    
    print(f"\n📊 Result:\n{result}")


def example_with_template_variables():
    """Show how to use template variables with the simplified approach."""
    print("\n" + "=" * 60)
    print("✅ WITH TEMPLATE VARIABLES")
    print("=" * 60)
    
    llm = LLM(model="data/models/qwen3-4b-instruct-2507-awq")
    
    # Template with variables
    system_template = Template("""You are an expert {{ language }} content analyst.

Focus on: {{ focus_area }}

Analyze the transcript and provide:
{{ instructions }}

Output valid JSON matching this schema:
{{ schema }}""")
    
    # Render with context
    system_prompt = system_template.render(
        language="Vietnamese",
        focus_area="business meetings",
        instructions="- Concise title\n- Key decisions\n- Next steps",
        schema='{\n  "title": "string",\n  "decisions": ["array"],\n  "next_steps": ["array"]\n}'
    )
    
    # Same inline message building
    transcript = "Quyết định: Triển khai hệ thống mới vào Q1 2024."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript to analyze:\n{transcript}"}
    ]
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    
    result = outputs[0].outputs[0].text
    
    print("\n📝 Template Variables Used:")
    print("  - language: Vietnamese")
    print("  - focus_area: business meetings")
    print("  - instructions: (custom)")
    print("  - schema: (custom)")
    
    print(f"\n📊 Result:\n{result}")


def example_with_guided_decoding():
    """Advanced: Guaranteed valid JSON with guided decoding."""
    print("\n" + "=" * 60)
    print("✅ ADVANCED: Guided Decoding (Production Best)")
    print("=" * 60)
    
    from vllm.sampling_params import GuidedDecodingParams
    
    llm = LLM(model="data/models/qwen3-4b-instruct-2507-awq")
    
    # Define JSON schema
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["title", "summary", "tags"]
    }
    
    # Simple system prompt
    system_template = Template("""You are an expert Vietnamese content analyst.
Analyze the transcript and output JSON only.""")
    
    system_prompt = system_template.render()
    
    # Build messages
    transcript = "Hôm nay họp về marketing. Quyết định tăng ngân sách 20%."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript:\n{transcript}"}
    ]
    
    # Guided decoding ensures valid JSON
    guided_decoding = GuidedDecodingParams(json=schema)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=200,
        guided_decoding=guided_decoding
    )
    
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    result = outputs[0].outputs[0].text
    
    print("\n📝 Schema Enforced:")
    print("  - title (required)")
    print("  - summary (required)")
    print("  - tags (required array)")
    
    print("\n✅ Guarantees:")
    print("  - Valid JSON syntax")
    print("  - All required fields present")
    print("  - Correct data types")
    print("  - No validation errors")
    
    print(f"\n📊 Result:\n{result}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("vLLM Chat API: The Simplified Approach")
    print("=" * 60)
    
    examples = [
        ("Old Approach (Manual)", example_old_approach),
        ("Simplified Inline (Recommended)", example_simplified_inline),
        ("With Template Variables", example_with_template_variables),
        ("With Guided Decoding (Best)", example_with_guided_decoding),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] {name}...")
        try:
            func()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Summary & Recommendations")
    print("=" * 60)
    print("""
🎯 Recommendation Order:

1. ✅ Simplified Inline + Guided Decoding (Best for production)
   - 6 lines of code
   - Guaranteed valid JSON
   - ONE template file
   - Zero helper methods

2. ✅ Simplified Inline (Best for simplicity)
   - Same 6 lines
   - No schema enforcement
   - Easy to maintain

3. ⚠️  Manual Templates (Avoid)
   - Brittle, model-specific
   - Maintenance burden

📊 Impact:
- Before: 80+ lines (helper methods + templates)
- After: 6 lines (inline approach)
- Reduction: ~90% less code!

🔧 Migration Steps:
1. Create new template: templates/prompts/your_task_v2.jinja
2. Remove chat markers from template
3. Use inline message building (see example_simplified_inline)
4. Test with chat API
5. Delete old template and helper methods

📚 References:
- Implementation: src/processors/llm/processor.py (lines 367-377)
- Tests: tests/unit/test_llm_chat_api.py
- Docs: docs/CHAT_API_IMPLEMENTATION_SUMMARY.md
- Template: templates/prompts/generic_summary_v2.jinja

✨ Key Insight:
You don't need abstractions for 6 lines of code.
Inline is clearer, simpler, and more maintainable.
    """)


if __name__ == "__main__":
    main()

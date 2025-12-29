#!/usr/bin/env python3
"""Test enhancement to see full LLM output"""
import sys
import json
sys.path.insert(0, '/home/cetech/UNV_AI/maie')

from src.processors.llm import LLMProcessor

# Test text
test_text = "xin chào tôi là một sinh viên"

# Initialize processor
processor = LLMProcessor()
processor._load_model()

# Run enhancement
result = processor.enhance_text(test_text)

print("=" * 80)
print("Enhancement Result:")
print("=" * 80)
print(json.dumps(result, indent=2, ensure_ascii=False))
print("\n" + "=" * 80)
print("Keys in result:")
print(list(result.keys()))
print("=" * 80)

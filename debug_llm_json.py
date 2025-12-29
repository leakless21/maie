#!/usr/bin/env python3
"""Check actual LLM JSON output for enhancement"""
import sys
import json
sys.path.insert(0, '/home/cetech/UNV_AI/maie')

from src.processors.llm import LLMProcessor

# Test text
test_text = "test nhanh"

# Initialize processor
processor = LLMProcessor()
processor._load_model()

# Run enhancement and capture the result
result = processor.execute(test_text, task="enhancement")

print("=" * 80)
print("Raw LLM Text Output:")
print("=" * 80)
print(result.text)  
print("\n" + "=" * 80)
print("Metadata:")
print("=" * 80)
print(json.dumps(result.metadata, indent=2, ensure_ascii=False))
print("=" * 80)

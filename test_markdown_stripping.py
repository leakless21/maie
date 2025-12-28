#!/usr/bin/env python3
"""Quick test for markdown code fence stripping and JSON parsing."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.json_utils import safe_parse_json, _strip_markdown_code_fence

# Test cases with markdown code fences
test_cases = [
    # Valid JSON with markdown fence
    (
        '```json\n{"title": "Test", "tags": ["tag1"]}\n```',
        {"title": "Test", "tags": ["tag1"]},
        "Basic markdown fence with json"
    ),
    # Valid JSON with just triple backticks
    (
        '```\n{"title": "Test", "tags": ["tag1"]}\n```',
        {"title": "Test", "tags": ["tag1"]},
        "Markdown fence without json language"
    ),
    # Valid JSON with 4 backticks
    (
        '````\n{"title": "Test", "tags": ["tag1"]}\n````',
        {"title": "Test", "tags": ["tag1"]},
        "Markdown fence with 4 backticks"
    ),
    # Plain JSON without fences
    (
        '{"title": "Test", "tags": ["tag1"]}',
        {"title": "Test", "tags": ["tag1"]},
        "Plain JSON without fences"
    ),
    # Vietnamese JSON with markdown fence
    (
        '```json\n{"title": "Giá rau tại Hà Nội", "tags": ["kinh tế"]}\n```',
        {"title": "Giá rau tại Hà Nội", "tags": ["kinh tế"]},
        "Vietnamese content with markdown fence"
    ),
]

print("Testing Markdown Fence Stripping & JSON Parsing\n" + "=" * 60)

all_passed = True
for input_str, expected, description in test_cases:
    print(f"\nTest: {description}")
    print(f"Input preview: {input_str[:50]}...")
    
    # Test stripping
    stripped = _strip_markdown_code_fence(input_str)
    print(f"Stripped preview: {stripped[:50]}...")
    
    # Test parsing
    parsed, error = safe_parse_json(input_str)
    
    if error:
        print(f"❌ FAILED: {error}")
        all_passed = False
    elif parsed == expected:
        print(f"✅ PASSED")
    else:
        print(f"❌ FAILED: Expected {expected}, got {parsed}")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("✅ All tests passed!")
    sys.exit(0)
else:
    print("❌ Some tests failed!")
    sys.exit(1)

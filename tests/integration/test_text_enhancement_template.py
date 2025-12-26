#!/usr/bin/env python3
"""
Test script for text_enhancement_v1 template.
Validates schema, prompt rendering, and example data.
"""

import json
from pathlib import Path

from src.processors.llm.schema_validator import (
    load_template_schema,
    validate_llm_output,
)
from src.processors.prompt.template_loader import TemplateLoader
from src.processors.prompt.renderer import PromptRenderer


def test_text_enhancement_template():
    """Test the text_enhancement_v1 template."""
    template_id = "text_enhancement_v1"
    templates_dir = Path("templates")

    print(f"Testing template: {template_id}")
    print("=" * 60)

    # 1. Test schema loading
    print("\n1. Testing schema loading...")
    try:
        schema = load_template_schema(template_id, templates_dir)
        print(f"   ✓ Schema loaded: {schema.get('title')}")
        print(f"   ✓ Properties: {list(schema.get('properties', {}).keys())}")
        print(f"   ✓ Required fields: {schema.get('required', [])}")
    except Exception as e:
        print(f"   ✗ Schema loading failed: {e}")
        return False

    # 2. Test example.json validation
    print("\n2. Testing example.json validation...")
    try:
        example_file = templates_dir / template_id / "example.json"
        with open(example_file, "r", encoding="utf-8") as f:
            example_data = json.load(f)

        # Convert to string and validate
        example_str = json.dumps(example_data)
        validated_data, error = validate_llm_output(example_str, schema)

        if error:
            print(f"   ✗ Example validation failed: {error}")
            return False

        print(f"   ✓ Example data is valid")
        print(
            f"   ✓ Original text length: {len(validated_data['original_text'])} chars"
        )
        print(
            f"   ✓ Enhanced text length: {len(validated_data['enhanced_text'])} chars"
        )
        print(f"   ✓ Corrections count: {len(validated_data['corrections'])}")
        print(f"   ✓ Quality score: {validated_data['quality_score']}")
        print(f"   ✓ Tags: {validated_data['tags']}")
    except Exception as e:
        print(f"   ✗ Example validation failed: {e}")
        return False

    # 3. Test prompt rendering
    print("\n3. Testing prompt rendering...")
    try:
        loader = TemplateLoader(templates_dir)
        renderer = PromptRenderer(loader)

        test_input = "xin chào các bạn hôm nay tôi sẽ nói về ây ai và lai trim"
        prompt = renderer.render(template_id, input_text=test_input)

        print(f"   ✓ Prompt rendered successfully")
        print(f"   ✓ Prompt length: {len(prompt)} characters")
        print(f"   ✓ Input text appears in prompt: {test_input in prompt}")

        # Check if key instructions are present
        key_phrases = [
            "Vietnamese text editor",
            "enhanced_text",
            "corrections",
            "quality_score",
            "tags",
        ]

        missing = [p for p in key_phrases if p not in prompt]
        if missing:
            print(f"   ✗ Missing key phrases: {missing}")
            return False

        print(f"   ✓ All key phrases present in prompt")

    except Exception as e:
        print(f"   ✗ Prompt rendering failed: {e}")
        return False

    # 4. Test all examples in schema
    print("\n4. Testing correction types...")
    correction_types = schema["properties"]["corrections"]["items"]["properties"][
        "type"
    ]["enum"]
    print(f"   ✓ Supported correction types: {correction_types}")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":
    import sys

    success = test_text_enhancement_template()
    sys.exit(0 if success else 1)

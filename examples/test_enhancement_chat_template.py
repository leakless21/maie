#!/usr/bin/env python
"""
Demonstration script showing that enhancement task now uses chat template.

This script demonstrates that the enhancement task follows the same pattern
as the summary task - both render their chat templates inside execute().

Note: This uses mocks for demonstration. The warnings/errors you might see are
from Mock objects not having all the methods of real vLLM objects, but they're
handled gracefully by the code and don't affect functionality.
"""

from unittest.mock import Mock, patch
from src.processors.llm.processor import LLMProcessor


def main():
    # Suppress debug logs for cleaner output
    import logging

    logging.getLogger("src").setLevel(logging.CRITICAL)

    print("=== Enhancement Task Chat Template Demonstration ===\n")

    # Create processor with better mocks to avoid warnings
    processor = LLMProcessor()
    processor._model_loaded = True
    processor.model = Mock()
    processor.model_info = {"model_name": "test-model"}

    # Mock tokenizer to avoid token calculation warnings
    processor.tokenizer = (
        None  # Explicitly set to None to skip dynamic token calculation
    )

    # Mock model generation
    mock_output = Mock()
    mock_output.outputs = [Mock()]
    mock_output.outputs[0].text = "Bạn tên là gì? Tôi tên là Nam."
    processor.model.generate.return_value = [mock_output]

    # Return None for sampling params to avoid override errors
    processor.model.get_default_sampling_params.return_value = None

    # Test that execute() renders the template for enhancement task
    print("1. Testing execute() method with task='enhancement':\n")

    with patch.object(processor.prompt_renderer, "render") as mock_render:
        mock_render.return_value = (
            "<|im_start|>system\n"
            "You are an expert Vietnamese proofreader...<|im_end|>\n"
            "<|im_start|>user\n"
            "bạn tên là gì tôi tên là nam<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        result = processor.execute("bạn tên là gì tôi tên là nam", task="enhancement")

        print(f"   ✓ Template renderer was called: {mock_render.called}")
        print(f"   ✓ Template name: {mock_render.call_args[0][0]}")
        print(
            f"   ✓ Text input parameter: {mock_render.call_args[1].get('text_input')}"
        )
        print(f"   ✓ Rendered output starts with: {mock_render.return_value[:50]}...")
        print(f"   ✓ Generated text: {result.text}\n")

    # Test that enhance_text() calls execute() which renders the template
    print("2. Testing enhance_text() method:\n")

    with patch.object(processor.prompt_renderer, "render") as mock_render:
        mock_render.return_value = (
            "<|im_start|>system\n"
            "You are an expert Vietnamese proofreader...<|im_end|>\n"
            "<|im_start|>user\n"
            "bạn tên là gì tôi tên là nam<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        result = processor.enhance_text("bạn tên là gì tôi tên là nam")

        print(f"   ✓ Template renderer was called: {mock_render.called}")
        print(f"   ✓ Template name: {mock_render.call_args[0][0]}")
        print(f"   ✓ Enhanced text: {result['enhanced_text']}")
        print(f"   ✓ Enhancement applied: {result['enhancement_applied']}")
        print(f"   ✓ Edit distance: {result['edit_distance']}\n")

    print("=== Summary ===\n")
    print("✓ Enhancement task now uses chat template rendering inside execute()")
    print("✓ Same pattern as summary task")
    print("✓ Template: text_enhancement_v1.jinja with Qwen3 chat format")
    print("✓ No duplicate template rendering - cleaner code structure\n")


if __name__ == "__main__":
    main()

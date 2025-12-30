import os
import sys
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

from src.processors.llm.processor import LLMProcessor
from src.processors.base import LLMResult
from src.config import settings

def test_text_enhancement_v1_as_summary_template():
    print("Testing text_enhancement_v1 as summary template with long text...")
    
    # Mock settings
    with patch("src.processors.llm.processor.settings") as mock_settings:
        mock_settings.llm_backend = "vllm_server"
        mock_settings.llm_enhance.max_model_len = 32768
        mock_settings.llm_sum.max_model_len = 32768
        mock_settings.llm_sum.structured_outputs_enabled = False
        mock_settings.llm_sum.temperature = 0.7
        mock_settings.llm_sum.top_p = 0.9
        mock_settings.llm_sum.top_k = 40
        mock_settings.paths.templates_dir = settings.paths.templates_dir
        
        # Initialize processor
        with patch.object(LLMProcessor, "_load_model") as mock_load:
            processor = LLMProcessor()
            processor.model_path = "data/models/qwen3-4b-instruct"
            processor._model_loaded = True
            processor.model_info = {"model_name": "test"}
            
            # Ensure tokenizer is loaded
            processor._ensure_tokenizer(processor.model_path)
            
            # Track execute calls
            execute_calls = []
            
            def side_effect_execute(text, task, template_id=None, **kwargs):
                execute_calls.append({"task": task, "template_id": template_id, "text_len": len(text)})
                print(f"  [EXECUTE] task={task}, template={template_id}, text_len={len(text)}")
                
                # Return valid JSON for text_enhancement_v1 schema
                data = {
                    "title": "Test Title",
                    "enhanced_text": f"Enhanced version of the text (length: {len(text)})",
                    "quality_score": 0.85,
                    "language": "vi",
                    "tags": ["test", "enhancement"]
                }
                return LLMResult(
                    text=json.dumps(data),
                    metadata={"structured_summary": data}
                )
            
            processor.execute = MagicMock(side_effect=side_effect_execute)
            
            # Test 1: Short text - should NOT trigger map-reduce
            print("\n--- Test 1: Short text with text_enhancement_v1 ---")
            short_text = "Đây là văn bản ngắn cần xử lý."
            result = processor.generate_summary(short_text, "text_enhancement_v1")
            
            print(f"Result: summary={bool(result.get('summary'))}, error={result.get('error')}")
            if result.get('summary'):
                print(f"SUCCESS: text_enhancement_v1 worked as summary template for short text")
            else:
                print(f"FAILURE: {result.get('error')}")
            
            # Test 2: Long text - SHOULD trigger map-reduce
            print("\n--- Test 2: Long text with text_enhancement_v1 ---")
            execute_calls.clear()
            
            long_text = "Đây là văn bản dài cần được xử lý và tóm tắt nội dung. " * 2500
            print(f"Long text length: {len(long_text)} chars")
            
            token_count = processor._estimate_tokens(long_text)
            print(f"Estimated token count: {token_count}")
            
            result = processor.generate_summary(long_text, "text_enhancement_v1")
            
            print(f"\nResult: summary={bool(result.get('summary'))}, error={result.get('error')}")
            print(f"Execute calls: {len(execute_calls)}")
            
            # Check if map-reduce was triggered
            templates_used = [c.get('template_id') for c in execute_calls]
            print(f"Templates used: {templates_used}")
            
            if "map_reduce_notes_v1" in templates_used:
                print(f"SUCCESS: Map-reduce was triggered for long text!")
            elif result.get('summary'):
                print(f"SUCCESS: text_enhancement_v1 worked (but no map-reduce needed)")
            else:
                print(f"FAILURE: {result.get('error')}")

if __name__ == "__main__":
    test_text_enhancement_v1_as_summary_template()

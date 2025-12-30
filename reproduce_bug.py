
import json
from unittest.mock import Mock, patch
from pathlib import Path
from src.processors.llm.processor import LLMProcessor
from src.processors.base import LLMResult

def test_reproduce_bug():
    # Mock settings
    with patch("src.processors.llm.processor.settings") as mock_settings:
        mock_settings.llm_enhance.model = "test-model"
        mock_settings.paths.templates_dir = Path("templates")
        mock_settings.llm_sum.temperature = 0.7
        mock_settings.llm_sum.top_p = 0.9
        mock_settings.llm_sum.top_k = 20
        mock_settings.llm_sum.max_tokens = 1000
        mock_settings.llm_sum.structured_outputs_enabled = False
        mock_settings.llm_sum.max_model_len = 32768

        processor = LLMProcessor()
        processor._model_loaded = True
        processor.client_enhance = Mock()
        processor.client_summary = processor.client_enhance

        # Mock enhance_text result
        enhanced_res = {
            "enhanced_text": "S0: Phỏng đoán 3N c...các video hấp dẫn.",
            "original_text": "raw text",
            "enhancement_applied": True,
            "edit_distance": 10,
            "edit_rate": 0.1,
            "model_info": {"model_name": "test"},
            "title": "Test Title",
            "quality_score": 0.9,
            "language": "vi",
            "tags": ["test"]
        }

        with patch.object(processor, "enhance_text", return_value=enhanced_res):
            result = processor.generate_summary("raw text", "text_enhancement_v1")
            
            print(f"Result summary type: {type(result['summary'])}")
            print(f"Result summary value: {result['summary']}")
            
            if isinstance(result['summary'], str):
                print("BUG REPRODUCED: Summary is a string!")
            else:
                print("BUG NOT REPRODUCED: Summary is not a string.")

if __name__ == "__main__":
    test_reproduce_bug()

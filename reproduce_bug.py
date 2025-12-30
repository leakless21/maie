
from src.processors.llm.processor import LLMProcessor
from unittest.mock import MagicMock

def test_strip_exact_hallu_bug():
    processor = LLMProcessor()
    # Mock hallucination phrases
    processor._llm_hallu_phrases = {"bad phrase"}
    
    data = {"summary": "This is a bad phrase"}
    
    # This should return the modified data, but will return None due to the bug
    result = processor._strip_exact_hallu_in_data(data)
    
    print(f"Input data: {data}")
    print(f"Result: {result}")
    
    if result is None:
        print("BUG CONFIRMED: _strip_exact_hallu_in_data returned None")
    else:
        print("Bug not reproduced: _strip_exact_hallu_in_data returned data")

if __name__ == "__main__":
    test_strip_exact_hallu_bug()

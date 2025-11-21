import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.processors.llm.processor import LLMProcessor
from src.config import settings

def verify_structured_output():
    print("Initializing LLMProcessor...")
    processor = LLMProcessor()
    
    text = "This is a test transcript. The user is talking about migrating to structured outputs in vLLM. It is a very important task."
    
    print(f"Executing summary task with text: {text}")
    try:
        # We need a valid template_id. Assuming 'meeting_summary_v1' exists or similar.
        # If not, we might need to check available templates.
        # Let's try a generic one or check templates dir first.
        
        # List templates to be safe
        templates_dir = settings.paths.templates_dir / "prompts"
        print(f"Checking templates in {templates_dir}")
        if templates_dir.exists():
            print("Available templates:", [f.stem for f in templates_dir.glob("*.yaml")])
        
        # Use a likely template ID
        template_id = "generic_summary_v1" 
        
        result = processor.execute(text, task="summary", template_id=template_id)
        
        print("\n--- Result ---")
        print(f"Model Info: {result.model_info}")
        print(f"Generated Text:\n{result.text}")
        
        # simple validation
        import json
        try:
            json_output = json.loads(result.text)
            print("\nSUCCESS: Output is valid JSON.")
            if "summary" in json_output or "title" in json_output:
                 print("SUCCESS: Output contains expected fields.")
            else:
                 print("WARNING: Output might not match schema exactly, but is JSON.")
        except json.JSONDecodeError:
            print("\nFAILURE: Output is NOT valid JSON.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_structured_output()

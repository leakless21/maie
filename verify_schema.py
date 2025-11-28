
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append("/home/cetech/maie")

from src.processors.llm.schema_validator import load_template_schema

def verify_schema():
    templates_dir = Path("/home/cetech/maie/templates")
    template_id = "meeting_notes_v2"
    
    try:
        print(f"Loading schema for {template_id}...")
        schema = load_template_schema(template_id, templates_dir)
        print("Schema loaded successfully!")
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"Failed to load schema: {e}")
        return False

if __name__ == "__main__":
    if verify_schema():
        sys.exit(0)
    else:
        sys.exit(1)

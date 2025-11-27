import shutil
from pathlib import Path
import os

TEMPLATES_DIR = Path("templates")
SCHEMAS_DIR = TEMPLATES_DIR / "schemas"
PROMPTS_DIR = TEMPLATES_DIR / "prompts"
EXAMPLES_DIR = TEMPLATES_DIR / "examples"

def migrate():
    # Ensure source directories exist
    if not SCHEMAS_DIR.exists():
        print("Schemas directory not found. Aborting.")
        return

    # Iterate over all schema files
    for schema_file in SCHEMAS_DIR.glob("*.json"):
        template_id = schema_file.stem
        print(f"Migrating {template_id}...")

        # Create bundle directory
        bundle_dir = TEMPLATES_DIR / template_id
        bundle_dir.mkdir(exist_ok=True)

        # Move Schema
        target_schema = bundle_dir / "schema.json"
        shutil.copy2(schema_file, target_schema)
        print(f"  Moved schema to {target_schema}")

        # Move Prompt
        prompt_file = PROMPTS_DIR / f"{template_id}.jinja"
        if prompt_file.exists():
            target_prompt = bundle_dir / "prompt.jinja"
            shutil.copy2(prompt_file, target_prompt)
            print(f"  Moved prompt to {target_prompt}")
        else:
            print(f"  Warning: No prompt found for {template_id}")

        # Move Example
        example_file = EXAMPLES_DIR / f"{template_id}.example.json"
        if example_file.exists():
            target_example = bundle_dir / "example.json"
            shutil.copy2(example_file, target_example)
            print(f"  Moved example to {target_example}")

    print("Migration complete.")

if __name__ == "__main__":
    migrate()

import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Add src to path to import TemplateLoader if needed, but here we can just use Jinja2 directly
# to simulate what TemplateLoader does, or import it.
# Let's import the actual TemplateLoader to test the full chain.
sys.path.append(str(Path(__file__).parent.parent))

from processors.prompt.template_loader import TemplateLoader
from config.model import AppSettings

def verify_prompts():
    print("Verifying prompt inheritance...")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    templates_dir = base_dir / "templates"
    
    print(f"Templates dir: {templates_dir}")
    
    loader = TemplateLoader(templates_dir)
    
    # Test cases
    test_templates = ["meeting_notes_v2", "interview_transcript_v2", "generic_summary_v2"]
    
    for template_id in test_templates:
        print(f"\n--- Testing {template_id} ---")
        try:
            # Try to load the template
            # The loader logic we updated looks for {id}/prompt.jinja
            template = loader.get_template(template_id)
            
            # Render with dummy input
            rendered = template.render(input_text="This is a test transcript.")
            
            # Checks
            checks = {
                "Base: Core Task": "**Core Task:**" in rendered,
                "Base: Step-by-Step": "**Step-by-Step Guide:**" in rendered,
                "Base: Key Guidelines": "**Key Guidelines:**" in rendered,
                "Base: Language Rule": "proper Vietnamese" in rendered,
                "Specific: Role Definition": "You are an expert" in rendered, # This might vary, checking generic
                "Input Text": "This is a test transcript." in rendered
            }
            
            all_passed = True
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"{status} {check_name}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                print(f"✅ {template_id} passed all checks.")
            else:
                print(f"❌ {template_id} failed some checks.")
                print("--- Rendered Output Preview ---")
                print(rendered[:500] + "...")
                
        except Exception as e:
            print(f"❌ Failed to load/render {template_id}: {e}")

if __name__ == "__main__":
    verify_prompts()

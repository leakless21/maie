
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path.cwd()))

from src.processors.llm.schema_validator import load_template_schema
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_schema_load():
    template_id = "meeting_notes_v2"
    logger.info(f"Attempting to load schema for {template_id}...")
    
    try:
        schema = load_template_schema(template_id, settings.paths.templates_dir)
        logger.info(f"Successfully loaded schema for {template_id}")
        logger.info(f"Schema type: {schema.get('type')}")
        return True
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        return False

if __name__ == "__main__":
    success = verify_schema_load()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

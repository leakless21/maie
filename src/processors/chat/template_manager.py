# Chat template manager for Modular Audio Intelligence Engine
import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
from jinja2 import Environment, FileSystemLoader, Template


class ChatTemplateManager:
    """Manages chat templates for vLLM integration"""
    
    def __init__(self, template_dir: str = "assets/chat-templates"):
        self.template_dir = Path(template_dir)
        self._templates: Dict[str, Template] = {}
        self._template_info: Dict[str, Dict[str, Any]] = {}
        
    def load_template(self, template_name: str) -> Optional[Template]:
        """Load a chat template by name"""
        template_path = self.template_dir / f"{template_name}.jinja"
        
        if not template_path.exists():
            return None
            
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
            
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.from_string(template_content)
        
        self._templates[template_name] = template
        self._template_info[template_name] = {
            'path': str(template_path),
            'name': template_name,
            'loaded_at': self._get_current_timestamp()
        }
        
        return template
    
    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists and is properly formatted"""
        template = self.load_template(template_name)
        if not template:
            return False
            
        try:
            # Try rendering with basic test data
            test_messages = [
                {'role': 'system', 'content': 'Test system message'},
                {'role': 'user', 'content': 'Test user message'},
                {'role': 'assistant', 'content': 'Test assistant message'}
            ]
            template.render(messages=test_messages)
            return True
        except Exception:
            return False
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded template"""
        if template_name not in self._template_info:
            self.load_template(template_name)
            
        return self._template_info.get(template_name)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def list_available_templates(self) -> list:
        """List all available template files in the template directory"""
        templates = []
        if self.template_dir.exists():
            for file_path in self.template_dir.glob("*.jinja"):
                template_name = file_path.stem
                templates.append(template_name)
        return templates
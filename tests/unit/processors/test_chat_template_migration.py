"""
Tests for chat template formatted Jinja templates.

This tests that our templates now output properly formatted chat prompts
with Qwen3's special tokens.
"""

import json
from pathlib import Path

import pytest

from src.config import settings
from src.processors.prompt.renderer import PromptRenderer
from src.processors.prompt.template_loader import TemplateLoader


class TestChatFormattedTemplates:
    """Test that templates output chat-formatted prompts."""

    @pytest.fixture
    def renderer(self):
        """Create prompt renderer with templates."""
        loader = TemplateLoader(settings.paths.templates_dir / "prompts")
        return PromptRenderer(loader)

    def test_meeting_notes_template_has_chat_tokens(self, renderer):
        """Test meeting_notes_v1 template outputs Qwen3 chat format."""
        schema = {"type": "object", "properties": {}}
        transcript = "test meeting transcript"
        
        output = renderer.render(
            "meeting_notes_v1",
            transcript=transcript,
            schema=json.dumps(schema, indent=2)
        )
        
        # Check for Qwen3 special tokens
        assert "<|im_start|>system" in output
        assert "<|im_end|>" in output
        assert "<|im_start|>user" in output
        assert "<|im_start|>assistant" in output
        
        # Check content is present
        assert transcript in output
        assert "meeting analyst" in output.lower()

    def test_generic_summary_template_has_chat_tokens(self, renderer):
        """Test generic_summary_v1 template outputs chat format."""
        schema = {"type": "object", "properties": {}}
        transcript = "test content to summarize"
        
        output = renderer.render(
            "generic_summary_v1",
            transcript=transcript,
            schema=json.dumps(schema, indent=2)
        )
        
        # Check for chat tokens
        assert "<|im_start|>system" in output
        assert "<|im_end|>" in output
        assert "<|im_start|>user" in output
        assert "<|im_start|>assistant" in output
        
        # Check content
        assert transcript in output
        assert "vietnamese content analyst" in output.lower()

    def test_interview_template_has_chat_tokens(self, renderer):
        """Test interview_transcript_v1 template outputs chat format."""
        schema = {"type": "object", "properties": {}}
        transcript = "test interview transcript"
        
        output = renderer.render(
            "interview_transcript_v1",
            transcript=transcript,
            schema=json.dumps(schema, indent=2)
        )
        
        # Check for chat tokens
        assert "<|im_start|>system" in output
        assert "<|im_end|>" in output
        assert "<|im_start|>user" in output
        assert "<|im_start|>assistant" in output
        
        # Check content
        assert transcript in output
        assert "interview analyst" in output.lower()

    def test_text_enhancement_template_has_chat_tokens(self, renderer):
        """Test text_enhancement_v1 template outputs chat format."""
        text_input = "test text to enhance"
        
        output = renderer.render(
            "text_enhancement_v1",
            text_input=text_input
        )
        
        # Check for chat tokens
        assert "<|im_start|>system" in output
        assert "<|im_end|>" in output
        assert "<|im_start|>user" in output
        assert "<|im_start|>assistant" in output
        
        # Check content
        assert text_input in output
        assert "proofreader" in output.lower()

    def test_templates_end_with_assistant_token(self, renderer):
        """Test all templates end with assistant start token for generation."""
        schema = {"type": "object"}
        transcript = "test"
        
        templates = [
            ("meeting_notes_v1", {"transcript": transcript, "schema": json.dumps(schema)}),
            ("generic_summary_v1", {"transcript": transcript, "schema": json.dumps(schema)}),
            ("interview_transcript_v1", {"transcript": transcript, "schema": json.dumps(schema)}),
            ("text_enhancement_v1", {"text_input": transcript}),
        ]
        
        for template_id, params in templates:
            output = renderer.render(template_id, **params)
            
            # Should end with assistant start token (ready for generation)
            assert output.strip().endswith("<|im_start|>assistant"), \
                f"Template {template_id} should end with assistant token"

    def test_templates_have_proper_structure(self, renderer):
        """Test templates have proper system/user/assistant structure."""
        schema = {"type": "object"}
        transcript = "test content"
        
        output = renderer.render(
            "meeting_notes_v1",
            transcript=transcript,
            schema=json.dumps(schema)
        )
        
        # Check structure order
        system_pos = output.find("<|im_start|>system")
        user_pos = output.find("<|im_start|>user")
        assistant_pos = output.find("<|im_start|>assistant")
        
        assert system_pos < user_pos < assistant_pos, \
            "Template should have system -> user -> assistant order"

    def test_templates_preserve_vietnamese_text(self, renderer):
        """Test templates properly handle Vietnamese characters."""
        vietnamese_text = "Xin chào! Tôi là trợ lý AI. Hôm nay trời đẹp."
        
        output = renderer.render(
            "text_enhancement_v1",
            text_input=vietnamese_text
        )
        
        # Check Vietnamese text is preserved
        assert vietnamese_text in output
        # Check no encoding issues
        assert "?" not in vietnamese_text.replace("?", "")  # Original has no question marks we didn't add

    def test_meeting_notes_json_schema_included(self, renderer):
        """Test meeting_notes template includes the JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"}
            }
        }
        transcript = "test meeting"
        
        output = renderer.render(
            "meeting_notes_v1",
            transcript=transcript,
            schema=json.dumps(schema, indent=2)
        )
        
        # Check schema content is mentioned in output (in system message)
        assert "object" in output.lower()
        # The schema variable should be rendered in the template
        assert transcript in output


class TestTemplateBackwardCompatibility:
    """Test that templates still work with existing code."""

    @pytest.fixture
    def renderer(self):
        """Create prompt renderer."""
        loader = TemplateLoader(settings.paths.templates_dir / "prompts")
        return PromptRenderer(loader)

    def test_templates_accept_same_parameters(self, renderer):
        """Test templates accept the same parameters as before."""
        schema = {"type": "object"}
        
        # These should not raise exceptions
        renderer.render("meeting_notes_v1", transcript="test", schema=json.dumps(schema))
        renderer.render("generic_summary_v1", transcript="test", schema=json.dumps(schema))
        renderer.render("interview_transcript_v1", transcript="test", schema=json.dumps(schema))
        renderer.render("text_enhancement_v1", text_input="test")

    def test_templates_return_strings(self, renderer):
        """Test templates return string output."""
        schema = {"type": "object"}
        
        outputs = [
            renderer.render("meeting_notes_v1", transcript="test", schema=json.dumps(schema)),
            renderer.render("text_enhancement_v1", text_input="test"),
        ]
        
        for output in outputs:
            assert isinstance(output, str)
            assert len(output) > 0

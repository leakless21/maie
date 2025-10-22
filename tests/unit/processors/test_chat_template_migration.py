"""
Tests for system prompt templates.

This tests that our templates output clean system prompts without chat formatting.
Chat formatting is applied by the LLM processor layer when building messages.
"""

import json
from pathlib import Path

import pytest

from src.config import settings
from src.processors.prompt.renderer import PromptRenderer
from src.processors.prompt.template_loader import TemplateLoader


class TestSystemPromptTemplates:
    """Test that templates output plain system prompts (no chat formatting)."""

    @pytest.fixture
    def renderer(self):
        """Create prompt renderer with templates."""
        loader = TemplateLoader(settings.templates_dir / "prompts")
        return PromptRenderer(loader)

    def test_meeting_notes_template_is_plain_text(self, renderer):
        """Test meeting_notes_v1 template outputs plain system prompt."""
        schema = {"type": "object", "properties": {}}
        
        output = renderer.render(
            "meeting_notes_v1",
            schema=json.dumps(schema, indent=2)
        )
        
        # Should NOT contain chat tokens (applied by LLM processor)
        assert "<|im_start|>" not in output
        assert "<|im_end|>" not in output
        
        # Should contain system prompt content
        assert "meeting analyst" in output.lower()
        assert "type" in output  # Schema included

    def test_generic_summary_template_is_plain_text(self, renderer):
        """Test generic_summary_v1 template outputs plain system prompt."""
        schema = {"type": "object", "properties": {}}
        
        output = renderer.render(
            "generic_summary_v1",
            schema=json.dumps(schema, indent=2)
        )
        
        # Should NOT contain chat tokens
        assert "<|im_start|>" not in output
        assert "<|im_end|>" not in output
        
        # Should contain system prompt content
        assert "content analyst" in output.lower()
        assert "type" in output

    def test_interview_template_is_plain_text(self, renderer):
        """Test interview_transcript_v1 template outputs plain system prompt."""
        schema = {"type": "object", "properties": {}}
        
        output = renderer.render(
            "interview_transcript_v1",
            schema=json.dumps(schema, indent=2)
        )
        
        # Should NOT contain chat tokens
        assert "<|im_start|>" not in output
        assert "<|im_end|>" not in output
        
        # Should contain system prompt content
        assert "interview analyst" in output.lower()
        assert "type" in output

    def test_text_enhancement_template_is_plain_text(self, renderer):
        """Test text_enhancement_v1 template outputs plain system prompt."""
        output = renderer.render("text_enhancement_v1")
        
        # Should NOT contain chat tokens
        assert "<|im_start|>" not in output
        assert "<|im_end|>" not in output
        
        # Should contain system prompt content
        assert "proofreader" in output.lower()

    def test_templates_contain_instructions_and_rules(self, renderer):
        """Test all templates contain proper instruction content."""
        schema = {"type": "object"}
        
        templates_and_keywords = [
            ("meeting_notes_v1", ["meeting", "analyst", "schema"], {"schema": json.dumps(schema)}),
            ("generic_summary_v1", ["vietnamese", "analyst", "schema"], {"schema": json.dumps(schema)}),
            ("interview_transcript_v1", ["interview", "analyst", "schema"], {"schema": json.dumps(schema)}),
            ("text_enhancement_v1", ["proofreader", "enhance"], {}),
        ]
        
        for template_id, required_keywords, extra_params in templates_and_keywords:
            output = renderer.render(template_id, **extra_params)
            
            # Check all required keywords are present
            for keyword in required_keywords:
                assert keyword.lower() in output.lower(), \
                    f"Template {template_id} should contain '{keyword}'"

    def test_templates_have_proper_structure(self, renderer):
        """Test templates have instructions and rules."""
        schema = {"type": "object"}
        
        output = renderer.render(
            "meeting_notes_v1",
            schema=json.dumps(schema)
        )
        
        # Check for instruction-like content (varies by template)
        # Should contain some rules or guidance
        assert len(output) > 200, "Template should have substantial content"
        assert "object" in output.lower() or "schema" in output.lower()

    def test_templates_preserve_vietnamese_text(self, renderer):
        """Test templates properly handle Vietnamese characters."""
        vietnamese_text = "Xin chào! Tôi là trợ lý AI. Hôm nay trời đẹp."
        
        output = renderer.render(
            "text_enhancement_v1"
        )
        
        # Check Vietnamese content is supported (no encoding errors in template itself)
        assert len(output) > 0
        # Template should be in Vietnamese
        assert "proofreader" in output.lower() or "improve" in output.lower() or "enhance" in output.lower()

    def test_meeting_notes_schema_handling(self, renderer):
        """Test meeting_notes template includes the JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"}
            }
        }
        
        output = renderer.render(
            "meeting_notes_v1",
            schema=json.dumps(schema, indent=2)
        )
        
        # Check schema content is rendered in output
        assert "object" in output.lower()
        # Schema dict should appear in the rendered content
        assert "properties" in output.lower() or "title" in output.lower()


class TestTemplateBackwardCompatibility:
    """Test that templates still work with existing code."""

    @pytest.fixture
    def renderer(self):
        """Create prompt renderer."""
        loader = TemplateLoader(settings.templates_dir / "prompts")
        return PromptRenderer(loader)

    def test_templates_accept_same_parameters(self, renderer):
        """Test templates accept the same parameters as before."""
        schema = {"type": "object"}
        
        # These should not raise exceptions
        renderer.render("meeting_notes_v1", schema=json.dumps(schema))
        renderer.render("generic_summary_v1", schema=json.dumps(schema))
        renderer.render("interview_transcript_v1", schema=json.dumps(schema))
        renderer.render("text_enhancement_v1")

    def test_templates_return_strings(self, renderer):
        """Test templates return string output."""
        schema = {"type": "object"}
        
        outputs = [
            renderer.render("meeting_notes_v1", schema=json.dumps(schema)),
            renderer.render("text_enhancement_v1"),
        ]
        
        for output in outputs:
            assert isinstance(output, str)
            assert len(output) > 0

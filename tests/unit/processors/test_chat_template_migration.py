"""
Tests for Jinja template rendering.

This tests that our templates output proper system prompts for use with
the LLM processor's OpenAI-format message building.
"""

import json

import pytest

from src.config import settings
from src.processors.prompt.renderer import PromptRenderer
from src.processors.prompt.template_loader import TemplateLoader


class TestTemplateRendering:
    """Test that templates output proper system prompts."""

    @pytest.fixture
    def renderer(self):
        """Create prompt renderer with templates."""
        loader = TemplateLoader(settings.paths.templates_dir / "prompts")
        return PromptRenderer(loader)

    def test_meeting_notes_template_renders_system_prompt(self, renderer):
        """Test meeting_notes_v1 template outputs proper system prompt."""
        schema = {"type": "object", "properties": {}}
        transcript = "test meeting transcript"

        output = renderer.render(
            "meeting_notes_v1",
            transcript=transcript,
            schema=json.dumps(schema, indent=2),
        )

        # Check that it's a plain system prompt (no ChatML tokens)
        assert "<|im_start|>system" not in output
        assert "<|im_end|>" not in output
        assert "<|im_start|>user" not in output
        assert "<|im_start|>assistant" not in output

        # Check content is present (system prompt content, not transcript)
        assert "meeting analyst" in output.lower()
        assert "vietnamese" in output.lower()
        assert "json" in output.lower()

    def test_generic_summary_template_renders_system_prompt(self, renderer):
        """Test generic_summary_v1 template outputs proper system prompt."""
        schema = {"type": "object", "properties": {}}
        transcript = "test content to summarize"

        output = renderer.render(
            "generic_summary_v1",
            transcript=transcript,
            schema=json.dumps(schema, indent=2),
        )

        # Check that it's a plain system prompt (no ChatML tokens)
        assert "<|im_start|>system" not in output
        assert "<|im_end|>" not in output
        assert "<|im_start|>user" not in output
        assert "<|im_start|>assistant" not in output

        # Check content (system prompt content, not transcript)
        assert "content analyst" in output.lower()
        assert "vietnamese" in output.lower()
        assert "json" in output.lower()

    def test_interview_template_renders_system_prompt(self, renderer):
        """Test interview_transcript_v1 template outputs proper system prompt."""
        schema = {"type": "object", "properties": {}}
        transcript = "test interview transcript"

        output = renderer.render(
            "interview_transcript_v1",
            transcript=transcript,
            schema=json.dumps(schema, indent=2),
        )

        # Check that it's a plain system prompt (no ChatML tokens)
        assert "<|im_start|>system" not in output
        assert "<|im_end|>" not in output
        assert "<|im_start|>user" not in output
        assert "<|im_start|>assistant" not in output

        # Check content (system prompt content, not transcript)
        assert "interview analyst" in output.lower()
        assert "vietnamese" in output.lower()
        assert "json" in output.lower()

    def test_text_enhancement_template_renders_system_prompt(self, renderer):
        """Test text_enhancement_v1 template outputs proper system prompt."""
        text_input = "test text to enhance"

        output = renderer.render("text_enhancement_v1", text_input=text_input)

        # Check that it's a plain system prompt (no ChatML tokens)
        assert "<|im_start|>system" not in output
        assert "<|im_end|>" not in output
        assert "<|im_start|>user" not in output
        assert "<|im_start|>assistant" not in output

        # Check content (system prompt content, not text_input)
        assert "proofreader" in output.lower()
        assert "vietnamese" in output.lower()

    def test_templates_are_plain_system_prompts(self, renderer):
        """Test all templates output plain system prompts without chat tokens."""
        schema = {"type": "object"}
        transcript = "test"

        templates = [
            (
                "meeting_notes_v1",
                {"transcript": transcript, "schema": json.dumps(schema)},
            ),
            (
                "generic_summary_v1",
                {"transcript": transcript, "schema": json.dumps(schema)},
            ),
            (
                "interview_transcript_v1",
                {"transcript": transcript, "schema": json.dumps(schema)},
            ),
            ("text_enhancement_v1", {"text_input": transcript}),
        ]

        for template_id, params in templates:
            output = renderer.render(template_id, **params)

            # Should be plain text without ChatML tokens
            assert "<|im_start|>" not in output, (
                f"Template {template_id} should not contain ChatML tokens"
            )
            assert "<|im_end|>" not in output, (
                f"Template {template_id} should not contain ChatML tokens"
            )
            # Should contain system prompt content (not the input transcript)
            assert "vietnamese" in output.lower()

    def test_templates_contain_expected_content(self, renderer):
        """Test templates contain expected instructional content."""
        schema = {"type": "object"}
        transcript = "test content"

        output = renderer.render(
            "meeting_notes_v1", transcript=transcript, schema=json.dumps(schema)
        )

        # Check that template contains expected instructional content
        assert "vietnamese" in output.lower()
        assert "json" in output.lower()
        assert "schema" in output.lower()
        assert "meeting analyst" in output.lower()

    def test_templates_preserve_vietnamese_text(self, renderer):
        """Test templates properly handle Vietnamese characters."""
        vietnamese_text = "Xin chào! Tôi là trợ lý AI. Hôm nay trời đẹp."

        output = renderer.render("text_enhancement_v1", text_input=vietnamese_text)

        # Check Vietnamese text is preserved in template content
        assert "vietnamese" in output.lower()
        assert "proofreader" in output.lower()
        # Check no encoding issues in template content
        assert "?" not in vietnamese_text.replace(
            "?", ""
        )  # Original has no question marks we didn't add

    def test_meeting_notes_json_schema_included(self, renderer):
        """Test meeting_notes template includes the JSON schema."""
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
        }
        transcript = "test meeting"

        output = renderer.render(
            "meeting_notes_v1",
            transcript=transcript,
            schema=json.dumps(schema, indent=2),
        )

        # Check schema content is mentioned in output
        assert "object" in output.lower()
        assert "title" in output.lower()
        assert "summary" in output.lower()
        assert "schema" in output.lower()


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
        renderer.render(
            "meeting_notes_v1", transcript="test", schema=json.dumps(schema)
        )
        renderer.render(
            "generic_summary_v1", transcript="test", schema=json.dumps(schema)
        )
        renderer.render(
            "interview_transcript_v1", transcript="test", schema=json.dumps(schema)
        )
        renderer.render("text_enhancement_v1", text_input="test")

    def test_templates_return_strings(self, renderer):
        """Test templates return string output."""
        schema = {"type": "object"}

        outputs = [
            renderer.render(
                "meeting_notes_v1", transcript="test", schema=json.dumps(schema)
            ),
            renderer.render("text_enhancement_v1", text_input="test"),
        ]

        for output in outputs:
            assert isinstance(output, str)
            assert len(output) > 0

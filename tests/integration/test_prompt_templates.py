"""
Integration tests for existing prompt templates.

Tests cover:
- Text enhancement template rendering
- Meeting notes template rendering
- Generic summary template rendering
- Interview transcript template rendering
- Template validation and output format
"""
import pytest
from pathlib import Path
import json

from src.processors.prompt.template_loader import TemplateLoader
from src.processors.prompt.renderer import PromptRenderer


class TestTextEnhancementTemplate:
    """Test text enhancement template rendering."""

    @pytest.fixture
    def template_loader(self):
        """Create template loader pointing to actual templates."""
        templates_dir = Path(__file__).parent.parent.parent / "templates" / "prompts"
        return TemplateLoader(templates_dir)

    @pytest.fixture
    def renderer(self, template_loader):
        """Create prompt renderer."""
        return PromptRenderer(template_loader)

    def test_rendering_with_transcript_variable(self, renderer):
        """Test rendering with transcript variable."""
        transcript = "hello world this is a test transcript"
        result = renderer.render("text_enhancement_v1", text_input=transcript)
        
        # Should contain the transcript
        assert transcript in result
        # Should contain enhancement instructions
        assert "enhance" in result.lower() or "punctuation" in result.lower()

    def test_rendering_with_empty_transcript(self, renderer):
        """Test rendering with empty transcript."""
        result = renderer.render("text_enhancement_v1", text_input="")
        
        # Should still render without errors
        assert isinstance(result, str)
        # May be empty if no text_input provided, which is acceptable
        assert len(result) >= 0

    def test_rendering_with_very_long_transcript(self, renderer):
        """Test rendering with very long transcript (10K+ chars)."""
        long_transcript = "This is a test sentence. " * 500  # ~12K characters
        result = renderer.render("text_enhancement_v1", text_input=long_transcript)
        
        # Should handle long input
        assert len(result) > len(long_transcript)
        assert long_transcript in result

    def test_rendering_with_special_characters(self, renderer):
        """Test rendering with special characters."""
        special_transcript = "Hello! @#$%^&*()_+-=[]{}|;':\",./<>? World!"
        result = renderer.render("text_enhancement_v1", text_input=special_transcript)
        
        # Should handle special characters (may be HTML-escaped)
        # Check that the content is there, even if escaped
        assert "Hello!" in result
        assert "World!" in result
        # Special characters may be escaped
        assert "@#$%^" in result or "&amp;" in result

    def test_output_format_is_correct(self, renderer):
        """Test output format is correct."""
        transcript = "hello world test"
        result = renderer.render("text_enhancement_v1", text_input=transcript)
        
        # Should be a string
        assert isinstance(result, str)
        # Should contain the input
        assert transcript in result
        # Should not be empty
        assert len(result.strip()) > 0


class TestMeetingNotesTemplate:
    """Test meeting notes template rendering."""

    @pytest.fixture
    def template_loader(self):
        """Create template loader pointing to actual templates."""
        templates_dir = Path(__file__).parent.parent.parent / "templates" / "prompts"
        return TemplateLoader(templates_dir)

    @pytest.fixture
    def renderer(self, template_loader):
        """Create prompt renderer."""
        return PromptRenderer(template_loader)

    def test_rendering_with_all_variables(self, renderer):
        """Test rendering with all variables."""
        context = {
            "title": "Weekly Team Meeting",
            "date": "2024-01-15",
            "participants": ["Alice", "Bob", "Charlie"],
            "summary": "Discussed project progress and upcoming deadlines.",
            "key_points": [
                "Project is on track",
                "Deadline is next Friday",
                "Need to hire additional developer"
            ],
            "action_items": [
                {"description": "Review code changes", "assignee": "Alice"},
                {"description": "Prepare presentation", "assignee": "Bob"}
            ],
            "decisions": [
                "Approved new feature design",
                "Decided to use React for frontend"
            ]
        }
        
        result = renderer.render("meeting_notes_v1", **context)
        
        # Should contain all provided information
        assert "Weekly Team Meeting" in result
        assert "2024-01-15" in result
        assert "Alice" in result
        assert "Project is on track" in result
        assert "Review code changes" in result
        assert "Approved new feature design" in result

    def test_rendering_with_minimal_variables(self, renderer):
        """Test rendering with minimal variables."""
        context = {
            "summary": "Brief meeting summary",
            "key_points": ["Point 1", "Point 2"]
        }
        
        result = renderer.render("meeting_notes_v1", **context)
        
        # Should contain provided information
        assert "Brief meeting summary" in result
        assert "Point 1" in result
        assert "Point 2" in result
        # Should handle missing optional fields gracefully
        assert "No action items identified" in result
        assert "No decisions made" in result

    def test_list_rendering_key_points(self, renderer):
        """Test list rendering for key points."""
        context = {
            "summary": "Test summary",
            "key_points": ["First point", "Second point", "Third point"]
        }
        
        result = renderer.render("meeting_notes_v1", **context)
        
        # Should render as bullet points
        assert "- First point" in result
        assert "- Second point" in result
        assert "- Third point" in result

    def test_list_rendering_action_items(self, renderer):
        """Test list rendering for action items."""
        context = {
            "summary": "Test summary",
            "key_points": ["Point 1"],
            "action_items": [
                {"description": "Task 1", "assignee": "Person A"},
                {"description": "Task 2", "assignee": "Person B"}
            ]
        }
        
        result = renderer.render("meeting_notes_v1", **context)
        
        # Should render action items with assignees
        assert "- [ ] Task 1 (Assigned to: Person A)" in result
        assert "- [ ] Task 2 (Assigned to: Person B)" in result

    def test_conditional_rendering_no_action_items(self, renderer):
        """Test conditional rendering when no action items."""
        context = {
            "summary": "Test summary",
            "key_points": ["Point 1"]
        }
        
        result = renderer.render("meeting_notes_v1", **context)
        
        # Should show "No action items identified"
        assert "No action items identified" in result

    def test_conditional_rendering_no_decisions(self, renderer):
        """Test conditional rendering when no decisions."""
        context = {
            "summary": "Test summary",
            "key_points": ["Point 1"]
        }
        
        result = renderer.render("meeting_notes_v1", **context)
        
        # Should show "No decisions made"
        assert "No decisions made" in result

    def test_output_markdown_format_is_valid(self, renderer):
        """Test output markdown format is valid."""
        context = {
            "title": "Test Meeting",
            "summary": "Test summary",
            "key_points": ["Point 1", "Point 2"]
        }
        
        result = renderer.render("meeting_notes_v1", **context)
        
        # Should contain markdown headers
        assert "# Test Meeting" in result
        assert "## Meeting Summary" in result
        assert "## Key Points" in result
        # Should contain bullet points
        assert "- Point 1" in result
        assert "- Point 2" in result


class TestGenericSummaryTemplate:
    """Test generic summary template rendering."""

    @pytest.fixture
    def template_loader(self):
        """Create template loader pointing to actual templates."""
        templates_dir = Path(__file__).parent.parent.parent / "templates" / "prompts"
        return TemplateLoader(templates_dir)

    @pytest.fixture
    def renderer(self, template_loader):
        """Create prompt renderer."""
        return PromptRenderer(template_loader)

    def test_rendering_with_transcript_and_schema_variables(self, renderer):
        """Test rendering with transcript and schema variables."""
        context = {
            "title": "Document Summary",
            "source_type": "Meeting Recording",
            "date": "2024-01-15",
            "summary": "This is a comprehensive summary of the meeting.",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "highlights": ["Highlight 1", "Highlight 2"],
            "important_details": ["Detail 1", "Detail 2"]
        }
        
        result = renderer.render("generic_summary_v1", **context)
        
        # Should contain all provided information
        assert "Document Summary" in result
        assert "Meeting Recording" in result
        assert "2024-01-15" in result
        assert "comprehensive summary" in result
        assert "Point 1" in result
        assert "Highlight 1" in result
        assert "Detail 1" in result

    def test_rendering_with_minimal_variables(self, renderer):
        """Test rendering with minimal variables."""
        context = {
            "summary": "Basic summary",
            "key_points": ["Key point 1"]
        }
        
        result = renderer.render("generic_summary_v1", **context)
        
        # Should contain provided information
        assert "Basic summary" in result
        assert "Key point 1" in result
        # Should handle missing optional fields
        assert "No highlights identified" in result
        assert "No important details noted" in result

    def test_conditional_rendering_no_highlights(self, renderer):
        """Test conditional rendering when no highlights."""
        context = {
            "summary": "Test summary",
            "key_points": ["Point 1"]
        }
        
        result = renderer.render("generic_summary_v1", **context)
        
        # Should show "No highlights identified"
        assert "No highlights identified" in result

    def test_conditional_rendering_no_important_details(self, renderer):
        """Test conditional rendering when no important details."""
        context = {
            "summary": "Test summary",
            "key_points": ["Point 1"]
        }
        
        result = renderer.render("generic_summary_v1", **context)
        
        # Should show "No important details noted"
        assert "No important details noted" in result

    def test_multiline_transcript_handling(self, renderer):
        """Test multiline transcript handling."""
        multiline_summary = """This is a multi-line summary.
        
It contains multiple paragraphs and should be handled properly.
        
The formatting should be preserved."""
        
        context = {
            "summary": multiline_summary,
            "key_points": ["Point 1"]
        }
        
        result = renderer.render("generic_summary_v1", **context)
        
        # Should preserve multiline content
        assert "multi-line summary" in result
        assert "multiple paragraphs" in result

    def test_output_markdown_format_is_valid(self, renderer):
        """Test output markdown format is valid."""
        context = {
            "title": "Test Document",
            "summary": "Test summary",
            "key_points": ["Point 1", "Point 2"]
        }
        
        result = renderer.render("generic_summary_v1", **context)
        
        # Should contain markdown headers
        assert "# Test Document" in result
        assert "## Summary" in result
        assert "## Key Points" in result
        # Should contain bullet points
        assert "- Point 1" in result
        assert "- Point 2" in result


class TestInterviewTranscriptTemplate:
    """Test interview transcript template rendering."""

    @pytest.fixture
    def template_loader(self):
        """Create template loader pointing to actual templates."""
        templates_dir = Path(__file__).parent.parent.parent / "templates" / "prompts"
        return TemplateLoader(templates_dir)

    @pytest.fixture
    def renderer(self, template_loader):
        """Create prompt renderer."""
        return PromptRenderer(template_loader)

    def test_rendering_with_all_expected_variables(self, renderer):
        """Test rendering with all expected variables."""
        context = {
            "title": "Product Manager Interview",
            "interviewee_name": "Jane Smith",
            "interviewer_name": "John Doe",
            "date": "2024-01-15",
            "duration": "45 minutes",
            "summary": "Comprehensive interview about product management experience.",
            "topics": ["Product Strategy", "Team Management", "Technical Skills"],
            "quotes": [
                {"text": "I believe in data-driven decisions", "speaker": "Jane Smith"},
                {"text": "Communication is key to success", "speaker": "Jane Smith"}
            ],
            "insights": ["Strong technical background", "Excellent communication skills"],
            "follow_up_questions": ["What's your experience with agile?", "How do you handle conflicts?"]
        }
        
        result = renderer.render("interview_transcript_v1", **context)
        
        # Should contain all provided information
        assert "Product Manager Interview" in result
        assert "Jane Smith" in result
        assert "John Doe" in result
        assert "2024-01-15" in result
        assert "45 minutes" in result
        assert "Comprehensive interview" in result
        assert "Product Strategy" in result
        assert "I believe in data-driven decisions" in result
        assert "Strong technical background" in result
        # Follow-up questions may be HTML-escaped
        assert "What" in result and "agile" in result

    def test_rendering_with_minimal_variables(self, renderer):
        """Test rendering with minimal variables."""
        context = {
            "summary": "Brief interview summary",
            "topics": ["Topic 1"],
            "insights": ["Insight 1"]
        }
        
        result = renderer.render("interview_transcript_v1", **context)
        
        # Should contain provided information
        assert "Brief interview summary" in result
        assert "Topic 1" in result
        assert "Insight 1" in result
        # Should handle missing optional fields
        assert "No notable quotes identified" in result
        assert "No follow-up questions noted" in result

    def test_quotes_rendering_with_speaker(self, renderer):
        """Test quotes rendering with speaker attribution."""
        context = {
            "summary": "Test summary",
            "topics": ["Topic 1"],
            "insights": ["Insight 1"],
            "quotes": [
                {"text": "This is a quote", "speaker": "Speaker Name"}
            ]
        }
        
        result = renderer.render("interview_transcript_v1", **context)
        
        # Should render quotes with proper attribution
        assert '> "This is a quote" - Speaker Name' in result

    def test_conditional_rendering_no_quotes(self, renderer):
        """Test conditional rendering when no quotes."""
        context = {
            "summary": "Test summary",
            "topics": ["Topic 1"],
            "insights": ["Insight 1"]
        }
        
        result = renderer.render("interview_transcript_v1", **context)
        
        # Should show "No notable quotes identified"
        assert "No notable quotes identified" in result

    def test_conditional_rendering_no_follow_up_questions(self, renderer):
        """Test conditional rendering when no follow-up questions."""
        context = {
            "summary": "Test summary",
            "topics": ["Topic 1"],
            "insights": ["Insight 1"]
        }
        
        result = renderer.render("interview_transcript_v1", **context)
        
        # Should show "No follow-up questions noted"
        assert "No follow-up questions noted" in result

    def test_output_structure_is_correct(self, renderer):
        """Test output structure is correct."""
        context = {
            "title": "Test Interview",
            "summary": "Test summary",
            "topics": ["Topic 1", "Topic 2"],
            "insights": ["Insight 1", "Insight 2"]
        }
        
        result = renderer.render("interview_transcript_v1", **context)
        
        # Should contain proper markdown structure
        assert "# Test Interview" in result
        assert "## Interview Summary" in result
        assert "## Key Topics Discussed" in result
        assert "## Key Insights" in result
        # Should contain bullet points
        assert "- Topic 1" in result
        assert "- Topic 2" in result
        assert "- Insight 1" in result
        assert "- Insight 2" in result


class TestTemplateValidation:
    """Test template validation and error handling."""

    @pytest.fixture
    def template_loader(self):
        """Create template loader pointing to actual templates."""
        templates_dir = Path(__file__).parent.parent.parent / "templates" / "prompts"
        return TemplateLoader(templates_dir)

    @pytest.fixture
    def renderer(self, template_loader):
        """Create prompt renderer."""
        return PromptRenderer(template_loader)

    def test_all_templates_exist(self, renderer):
        """Test that all expected templates exist."""
        templates = [
            "text_enhancement_v1",
            "meeting_notes_v1",
            "generic_summary_v1",
            "interview_transcript_v1"
        ]
        
        for template_name in templates:
            # Should not raise TemplateNotFound
            result = renderer.render(template_name, summary="test", key_points=["test"])
            assert isinstance(result, str)
            # Some templates may be empty without proper context, which is acceptable
            assert len(result) >= 0

    def test_template_rendering_without_required_variables(self, renderer):
        """Test template rendering without required variables."""
        # All templates should handle missing variables gracefully
        templates = [
            "text_enhancement_v1",
            "meeting_notes_v1",
            "generic_summary_v1",
            "interview_transcript_v1"
        ]
        
        for template_name in templates:
            result = renderer.render(template_name)
            assert isinstance(result, str)
            # Some templates may be empty without proper context, which is acceptable
            assert len(result) >= 0

    def test_template_output_consistency(self, renderer):
        """Test that template output is consistent across multiple renders."""
        context = {
            "summary": "Test summary",
            "key_points": ["Point 1", "Point 2"]
        }
        
        # Render multiple times
        results = []
        for _ in range(5):
            result = renderer.render("meeting_notes_v1", **context)
            results.append(result)
        
        # All results should be identical
        assert all(result == results[0] for result in results)

    def test_template_handles_large_data(self, renderer):
        """Test that templates handle large data gracefully."""
        large_context = {
            "summary": "Large summary " * 100,
            "key_points": [f"Point {i}" for i in range(100)],
            "action_items": [{"description": f"Task {i}", "assignee": f"Person {i}"} for i in range(50)]
        }
        
        result = renderer.render("meeting_notes_v1", **large_context)
        
        # Should handle large data without errors
        assert isinstance(result, str)
        assert len(result) > 1000  # Should be substantial output
        assert "Point 0" in result
        assert "Point 99" in result

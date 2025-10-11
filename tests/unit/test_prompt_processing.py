"""
Consolidated unit tests for prompt processing layer.

Tests cover:
- TemplateLoader: initialization, loading, caching, environment
- PromptRenderer: initialization, rendering, variables, errors
- Security: XSS prevention, template injection, context sanitization
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from jinja2 import TemplateNotFound, TemplateSyntaxError, Environment

from src.processors.prompt.template_loader import TemplateLoader
from src.processors.prompt.renderer import PromptRenderer


# ============================================================================
# TEMPLATE LOADER TESTS
# ============================================================================

class TestTemplateLoader:
    """Comprehensive tests for TemplateLoader class."""

    def test_initialization_with_valid_directory(self, tmp_path):
        """Test successful initialization with valid template directory."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "test.jinja").write_text("Hello {{ name }}!")
        
        loader = TemplateLoader(template_dir)
        
        assert loader.template_dir == template_dir
        assert loader.env is not None
        assert isinstance(loader.env, Environment)

    def test_autoescape_enabled_by_default(self, tmp_path):
        """Test autoescape is enabled by default for XSS prevention."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        loader = TemplateLoader(template_dir)
        
        assert loader.env.autoescape is True

    def test_load_existing_template(self, tmp_path):
        """Test loading existing template."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        template_file = template_dir / "test.jinja"
        template_file.write_text("Hello {{ name }}!")
        
        loader = TemplateLoader(template_dir)
        template = loader.get_template("test")
        
        assert template is not None
        assert template.render(name="World") == "Hello World!"

    def test_load_nonexistent_template_raises_error(self, tmp_path):
        """Test loading non-existent template raises TemplateNotFound."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        loader = TemplateLoader(template_dir)
        
        with pytest.raises(TemplateNotFound):
            loader.get_template("nonexistent")

    def test_template_name_normalization(self, tmp_path):
        """Test .jinja extension is handled correctly."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "test.jinja").write_text("Test template")
        
        loader = TemplateLoader(template_dir)
        
        # Should work with or without .jinja extension
        template1 = loader.get_template("test")
        template2 = loader.get_template("test.jinja")
        
        assert template1.render() == template2.render()

    def test_lru_cache_working(self, tmp_path):
        """Test that lru_cache is working."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "test.jinja").write_text("Cached")
        
        loader = TemplateLoader(template_dir)
        
        # Load template multiple times
        template1 = loader.get_template("test")
        template2 = loader.get_template("test")
        template3 = loader.get_template("test")
        
        # Should return cached instance
        assert template1 is template2
        assert template2 is template3

    def test_cache_size_limit(self, tmp_path):
        """Test cache size limit (maxsize=128)."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        # Create 130 templates
        for i in range(130):
            (template_dir / f"template_{i}.jinja").write_text(f"Template {i}")
        
        loader = TemplateLoader(template_dir)
        
        # Load all templates
        templates = []
        for i in range(130):
            templates.append(loader.get_template(f"template_{i}"))
        
        # All templates should be loaded successfully
        assert len(templates) == 130
        assert all(t is not None for t in templates)


# ============================================================================
# PROMPT RENDERER TESTS
# ============================================================================

class TestPromptRenderer:
    """Comprehensive tests for PromptRenderer class."""

    def test_initialization_with_template_loader(self, tmp_path):
        """Test successful initialization with TemplateLoader instance."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        assert renderer.template_loader is template_loader

    def test_initialization_fails_without_template_loader(self):
        """Test initialization fails without TemplateLoader."""
        with pytest.raises(TypeError):
            PromptRenderer(None)

    def test_render_simple_template(self, tmp_path):
        """Test rendering simple template with variables."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "hello.jinja").write_text("Hello {{ name }}!")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        result = renderer.render("hello", name="World")
        assert result == "Hello World!"

    def test_render_with_multiple_variables(self, tmp_path):
        """Test rendering template with multiple variables."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "multi.jinja").write_text("Hello {{ first }} {{ last }}!")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        result = renderer.render("multi", first="John", last="Doe")
        assert result == "Hello John Doe!"

    def test_render_with_nested_context(self, tmp_path):
        """Test rendering template with nested context."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "nested.jinja").write_text("Name: {{ user.name }}, Age: {{ user.age }}")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        result = renderer.render("nested", user={"name": "Alice", "age": 30})
        assert result == "Name: Alice, Age: 30"

    def test_render_nonexistent_template(self, tmp_path):
        """Test rendering non-existent template raises error."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        with pytest.raises(TemplateNotFound):
            renderer.render("nonexistent")

    def test_extra_variables_ignored(self, tmp_path):
        """Test that extra variables are ignored gracefully."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "simple.jinja").write_text("Hello World!")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        # Should work with extra variables
        result = renderer.render("simple", extra_var="ignored", another_var=123)
        assert result == "Hello World!"


# ============================================================================
# SECURITY TESTS
# ============================================================================

class TestPromptSecurity:
    """Comprehensive security tests for prompt processing layer."""

    def test_html_injection_escaped(self, tmp_path):
        """Test HTML injection is escaped."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "xss.jinja").write_text("Content: {{ content }}")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        malicious = "<script>alert('xss')</script>"
        result = renderer.render("xss", content=malicious)
        
        # Should be escaped
        assert "&lt;script&gt;" in result
        assert "<script>" not in result

    def test_javascript_injection_escaped(self, tmp_path):
        """Test JavaScript injection is escaped."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "js.jinja").write_text("Message: {{ message }}")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        js_injection = "javascript:alert('xss')"
        result = renderer.render("js", message=js_injection)
        
        # Quotes should be escaped
        assert "&#39;" in result

    def test_ssti_prevention(self, tmp_path):
        """Test Server-Side Template Injection prevention."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "ssti.jinja").write_text("User input: {{ user_input }}")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        # Try to inject template syntax
        malicious = "{{ 7*7 }}"
        result = renderer.render("ssti", user_input=malicious)
        
        # Should be treated as literal text, not executed
        assert result == "User input: {{ 7*7 }}"
        assert "49" not in result

    def test_dangerous_objects_cannot_execute(self, tmp_path):
        """Test that dangerous objects cannot be executed."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "dangerous.jinja").write_text("Object: {{ obj }}")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        # Try to pass dangerous objects
        for dangerous_obj in [open, __import__, eval, exec]:
            result = renderer.render("dangerous", obj=dangerous_obj)
            # Should render as string representation, not execute
            assert "built-in function" in result or "method" in result
            assert "&lt;" in result  # HTML-escaped

    def test_template_path_traversal_prevented(self, tmp_path):
        """Test path traversal attacks are prevented."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "secure.jinja").write_text("Secure content")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        # Try path traversal
        with pytest.raises(TemplateNotFound):
            renderer.render("../../../etc/passwd")
        
        # Try absolute path
        with pytest.raises(TemplateNotFound):
            renderer.render("/etc/passwd")

    def test_context_isolation(self, tmp_path):
        """Test that context is isolated between renders."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "isolated.jinja").write_text("Value: {{ value }}")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        # Render with different contexts
        result1 = renderer.render("isolated", value="first")
        result2 = renderer.render("isolated", value="second")
        
        # Should be isolated
        assert result1 == "Value: first"
        assert result2 == "Value: second"

    def test_large_context_handling(self, tmp_path):
        """Test large context handling (memory limits)."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "large.jinja").write_text("Count: {{ count }}")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        # Create large context
        large_data = "x" * 10000  # 10KB string
        large_list = list(range(1000))
        
        result = renderer.render("large", count=len(large_list), data=large_data)
        
        # Should handle large context without issues
        assert "Count: 1000" in result


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPromptIntegration:
    """Integration tests for complete prompt processing workflow."""

    def test_end_to_end_template_rendering(self, tmp_path):
        """Test complete workflow from template loading to rendering."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        # Create a realistic template
        template_content = """
# {{ title }}

## Summary
{{ summary }}

## Key Points
{% for point in key_points %}
- {{ point }}
{% endfor %}
"""
        (template_dir / "document.jinja").write_text(template_content)
        
        # Initialize components
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        # Render with context
        context = {
            "title": "Test Document",
            "summary": "This is a test summary.",
            "key_points": ["Point 1", "Point 2", "Point 3"]
        }
        
        result = renderer.render("document", **context)
        
        # Verify output
        assert "# Test Document" in result
        assert "This is a test summary" in result
        assert "- Point 1" in result
        assert "- Point 2" in result
        assert "- Point 3" in result

    def test_multiple_templates_caching(self, tmp_path):
        """Test caching works across multiple templates."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        # Create multiple templates
        (template_dir / "template1.jinja").write_text("Template 1: {{ value }}")
        (template_dir / "template2.jinja").write_text("Template 2: {{ value }}")
        (template_dir / "template3.jinja").write_text("Template 3: {{ value }}")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        # Render each template multiple times
        for i in range(10):
            result1 = renderer.render("template1", value=i)
            result2 = renderer.render("template2", value=i)
            result3 = renderer.render("template3", value=i)
            
            assert f"Template 1: {i}" in result1
            assert f"Template 2: {i}" in result2
            assert f"Template 3: {i}" in result3

    def test_output_consistency(self, tmp_path):
        """Test that output is consistent across multiple renders."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "consistent.jinja").write_text("Value: {{ value }}")
        
        template_loader = TemplateLoader(template_dir)
        renderer = PromptRenderer(template_loader)
        
        # Render multiple times
        results = []
        for _ in range(10):
            result = renderer.render("consistent", value="test")
            results.append(result)
        
        # All results should be identical
        assert all(result == results[0] for result in results)
        assert results[0] == "Value: test"


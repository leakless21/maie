from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class TemplateLoader:
    """
    Template loader for Jinja2 templates with caching and security features.

    This class provides a secure, cached interface for loading Jinja2 templates
    from the filesystem. It includes automatic HTML escaping for XSS prevention
    and LRU caching for performance optimization.

    Args:
        template_dir: Path to directory containing .jinja template files

    Example:
        >>> loader = TemplateLoader(Path("templates"))
        >>> template = loader.get_template("my_template")
        >>> result = template.render(name="World")
    """

    def __init__(self, template_dir: Path):
        """
        Initialize TemplateLoader with template directory.

        Args:
            template_dir: Path to directory containing .jinja template files
        """
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

    @lru_cache(maxsize=128)
    def get_template(self, template_name: str):
        """
        Loads and caches a Jinja2 template.

        Args:
            template_name: Name of template (with or without .jinja extension)

        Returns:
            Jinja2 Template object

        Raises:
            TemplateNotFound: If template file doesn't exist

        Example:
            >>> template = loader.get_template("my_template")
            >>> template = loader.get_template("my_template.jinja")  # Also works
        """
        # Handle .jinja extension - if already present, don't add it again
        if template_name.endswith(".jinja"):
            return self.env.get_template(template_name)
        else:
            return self.env.get_template(f"{template_name}.jinja")

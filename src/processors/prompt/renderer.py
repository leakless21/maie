from .template_loader import TemplateLoader


class PromptRenderer:
    """
    Prompt renderer for Jinja2 templates with context validation.
    
    This class provides a secure interface for rendering Jinja2 templates
    with proper context validation and error handling.
    
    Args:
        template_loader: TemplateLoader instance for loading templates
        
    Example:
        >>> loader = TemplateLoader(Path("templates"))
        >>> renderer = PromptRenderer(loader)
        >>> result = renderer.render("my_template", name="World")
    """
    
    def __init__(self, template_loader: TemplateLoader):
        """
        Initialize PromptRenderer with TemplateLoader.
        
        Args:
            template_loader: TemplateLoader instance
            
        Raises:
            TypeError: If template_loader is None
        """
        if template_loader is None:
            raise TypeError("template_loader cannot be None")
        self.template_loader = template_loader

    def render(self, template_name: str, **context) -> str:
        """
        Renders a prompt template with the given context.
        
        Args:
            template_name: Name of template to render
            **context: Context variables for template rendering
            
        Returns:
            Rendered template as string
            
        Raises:
            TemplateNotFound: If template doesn't exist
            UndefinedError: If required template variable is missing
            TemplateSyntaxError: If template has syntax errors
        """
        template = self.template_loader.get_template(template_name)
        return template.render(**context)

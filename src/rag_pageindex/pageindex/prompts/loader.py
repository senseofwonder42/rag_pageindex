from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATE_DIR = Path(__file__).parent

_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    trim_blocks=False,
    lstrip_blocks=False,
    autoescape=False,
)


def render(name: str, /, **context: Any) -> str:  # noqa: ANN401
    """Render a Jinja2 template file with the given context.

    Args:
        name: Template file name (e.g., 'check_toc.j2').
        **context: Variables to pass to the template.

    Returns:
        Rendered template string.

    Raises:
        jinja2.TemplateNotFound: If template file does not exist.
        jinja2.UndefinedError: If template references undefined variable.
    """
    template = _env.get_template(name)
    return template.render(**context)


def list_templates() -> list[str]:
    """List all available Jinja2 templates in the prompts directory.

    Returns:
        Sorted list of .j2 template file names.
    """
    return sorted(p.name for p in _TEMPLATE_DIR.glob("*.j2"))

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
    """Render a Jinja template by file name (with .j2 extension)."""
    template = _env.get_template(name)
    return template.render(**context)


def list_templates() -> list[str]:
    """Return all .j2 template names available."""
    return sorted(p.name for p in _TEMPLATE_DIR.glob("*.j2"))

"""Lightweight `@observe` shim that's a no-op unless tracing is enabled.

Importing `langfuse.observe` directly at module load time triggers an
authentication warning when no Langfuse credentials are configured. That
makes test runs and CI noisy. This module checks `settings.tracing_enabled`
once at import time and either re-exports the real decorator or substitutes
a transparent passthrough.
"""

from collections.abc import Callable
from typing import Any, TypeVar, cast

from rag_pageindex.core.config import settings

_F = TypeVar("_F", bound=Callable[..., Any])


def _noop_observe(
    func: _F | None = None,
    *,
    name: str | None = None,  # noqa: ARG001
    as_type: str | None = None,  # noqa: ARG001
    **_kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Same call shape as `langfuse.observe` but does nothing."""
    if func is not None and callable(func):
        return func

    def decorator(f: _F) -> _F:
        return f

    return decorator


if settings.tracing_enabled:
    from langfuse import observe as _real_observe

    observe = cast(Any, _real_observe)
else:
    observe = cast(Any, _noop_observe)

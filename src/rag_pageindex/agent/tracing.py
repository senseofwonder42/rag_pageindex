from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from rag_pageindex.core.config import settings

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler

_initialized = False


def _ensure_langfuse_client() -> None:
    """Register the Langfuse global client once per process.

    Idempotent: subsequent calls are no-ops. Raises if tracing is enabled
    but credentials are missing — same contract as the indexer pipeline.
    """
    global _initialized
    if _initialized:
        return

    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        raise RuntimeError(
            "tracing_enabled=True but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set."
        )

    from langfuse import Langfuse

    Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    _initialized = True
    logger.info("Langfuse tracing initialised for agent (host={})", settings.langfuse_host)


def langchain_callbacks() -> list[BaseCallbackHandler]:
    """Return the Langfuse LangChain callback handlers, or [] if tracing is off.

    The returned handlers can be attached to a ChatModel via
    `model.with_config({"callbacks": [...]})` or passed in
    `RunnableConfig.callbacks` at invocation time. Callbacks propagate to
    child runnables (tools, nested chains), so binding once at the top is
    enough to capture the whole agent trace.
    """
    if not settings.tracing_enabled:
        return []

    _ensure_langfuse_client()
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]

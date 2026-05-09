from rag_pageindex.core.config import Settings
from rag_pageindex.pageindex.llm.openai_compatible_client import OpenAICompatibleClient
from rag_pageindex.pageindex.llm.protocol import LLMClient


def _build_base_client(settings: Settings) -> LLMClient:
    """Build an OpenAI-compatible LLM client from settings.

    Args:
        settings: Configuration settings with llm_* fields.

    Returns:
        OpenAICompatibleClient instance.

    Raises:
        RuntimeError: If LLM_API_KEY is not set.
    """
    if settings.llm_api_key is None:
        raise RuntimeError("LLM_API_KEY is not set; cannot build OpenAICompatibleClient.")
    return OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        max_retries=settings.llm_max_retries,
        retry_delay_s=settings.llm_retry_delay_s,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout=settings.llm_timeout,
    )


def get_default_client(settings: Settings) -> LLMClient:
    """Build the configured `LLMClient` from `Settings`.

    When `settings.tracing_enabled`, wraps the client in `TracingLLMClient`
    so every call is recorded as a Langfuse generation. Importing the
    Langfuse SDK is deferred to keep the no-tracing path cheap.
    """
    client = _build_base_client(settings)
    if not settings.tracing_enabled:
        return client

    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        raise RuntimeError(
            "tracing_enabled=True but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set."
        )

    from langfuse import get_client

    from rag_pageindex.pageindex.llm.tracing_client import TracingLLMClient

    return TracingLLMClient(client, langfuse=get_client())

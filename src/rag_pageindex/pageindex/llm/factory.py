from rag_pageindex.core.config import Settings
from rag_pageindex.pageindex.llm.anthropic_client import AnthropicClient
from rag_pageindex.pageindex.llm.openai_compatible_client import OpenAICompatibleClient
from rag_pageindex.pageindex.llm.protocol import LLMClient


def _build_base_client(settings: Settings) -> LLMClient:
    """Build an LLM client based on settings.llm_provider.

    Args:
        settings: Configuration settings with llm_provider and llm_model.

    Returns:
        LLMClient instance (AnthropicClient or OpenAICompatibleClient).

    Raises:
        RuntimeError: If the required API key is not set or provider is unknown.
    """
    if settings.llm_provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot build AnthropicClient.")
        return AnthropicClient(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.llm_model,
            max_retries=settings.llm_max_retries,
            retry_delay_s=settings.llm_retry_delay_s,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    if settings.llm_provider == "openai_compatible":
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
    raise RuntimeError(f"Unknown llm_provider: {settings.llm_provider}")


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

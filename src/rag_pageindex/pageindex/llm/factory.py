from rag_pageindex.core.config import Settings
from rag_pageindex.pageindex.llm.anthropic_client import AnthropicClient
from rag_pageindex.pageindex.llm.protocol import LLMClient


def get_default_client(settings: Settings) -> LLMClient:
    """Build the configured `LLMClient` from `Settings`."""
    if settings.llm_provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; cannot build AnthropicClient."
            )
        return AnthropicClient(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.llm_model,
            max_retries=settings.llm_max_retries,
            retry_delay_s=settings.llm_retry_delay_s,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    raise RuntimeError(f"Unknown llm_provider: {settings.llm_provider}")

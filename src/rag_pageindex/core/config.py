from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load environment variables as settings."""

    environment: Literal["local", "test", "dev", "preprod", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )

    # LLM client
    llm_provider: Literal["anthropic"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: SecretStr | None = None
    llm_max_retries: int = 10
    llm_retry_delay_s: float = 1.0
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 4096

    # PageIndex pipeline tuning (was config.yaml upstream)
    pageindex_toc_check_page_num: int = 20
    pageindex_max_pages_per_node: int = 10
    pageindex_max_tokens_per_node: int = 20_000
    pageindex_token_ceiling: int = 110_000
    pageindex_add_node_id: bool = True
    pageindex_add_node_summary: bool = False
    pageindex_add_node_text: bool = False
    pageindex_add_doc_description: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return settings."""
    return Settings()


settings = get_settings()

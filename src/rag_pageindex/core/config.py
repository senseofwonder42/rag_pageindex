from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

_CONFIG_YAML = Path(__file__).parent.parent.parent.parent / "config.yaml"


class Settings(BaseSettings):
    """Load settings from config.yaml then .env (env vars win)."""

    environment: Literal["local", "test", "dev", "preprod", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # LLM client
    llm_provider: Literal["anthropic", "openai_compatible"] = "openai_compatible"
    llm_model: str = "google/gemini-2.5-flash-preview"
    anthropic_api_key: SecretStr | None = None
    # OpenAI-compatible backend (OpenRouter, vLLM, etc.)
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: SecretStr | None = None
    llm_max_retries: int = 10
    llm_retry_delay_s: float = 1.0
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 4096
    llm_timeout: float = 120.0

    # PageIndex pipeline tuning
    pageindex_toc_check_page_num: int = 20
    pageindex_max_pages_per_node: int = 10
    pageindex_max_tokens_per_node: int = 20_000
    pageindex_token_ceiling: int = 110_000
    pageindex_toc_max_output_tokens: int = 16_000
    pageindex_add_node_id: bool = True
    pageindex_add_node_summary: bool = False
    pageindex_add_node_text: bool = False
    pageindex_add_doc_description: bool = False

    # VLM fallback: render page images when text verification fails
    pageindex_vision_mode: Literal["off", "fallback"] = "off"
    pageindex_vision_dpi: int = 144
    pageindex_vision_fallback_threshold: float = 0.6
    # Some providers cap images-per-prompt (e.g. OpenRouter→Nvidia: 10).
    pageindex_vision_max_images_per_call: int = 1

    # Langfuse tracing (off by default)
    tracing_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Priority: init > env vars > .env file > config.yaml > defaults
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if _CONFIG_YAML.exists():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=_CONFIG_YAML))
        return tuple(sources)

    _yaml_path: ClassVar[Path] = _CONFIG_YAML


@lru_cache
def get_settings() -> Settings:
    """Return settings."""
    return Settings()


settings = get_settings()

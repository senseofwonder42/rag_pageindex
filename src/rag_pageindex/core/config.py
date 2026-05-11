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

from rag_pageindex.core.constants import PROJECT_ROOT

_CONFIG_YAML = Path(__file__).parent.parent.parent.parent / "config.yaml"


class Settings(BaseSettings):
    """Pydantic settings model for the PageIndex RAG application.

    Loads configuration from multiple sources in priority order:
    init arguments > environment variables > .env file > config.yaml > defaults.
    """

    environment: Literal["local", "test", "dev", "preprod", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # LLM client (OpenAI-compatible backend: OpenRouter, vLLM, etc.)
    llm_model: str = "google/gemini-2.5-flash-preview"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: SecretStr | None = None
    llm_max_retries: int = 10
    llm_retry_delay_s: float = 1.0
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 4096
    llm_timeout: float = 120.0

    # PageIndex pipeline tuning
    pageindex_toc_check_page_num: int = 20
    pageindex_toc_index_max_tokens: int = 6000
    pageindex_toc_resolve_max_tokens: int = 6000
    pageindex_max_pages_per_node: int = 10
    pageindex_max_tokens_per_node: int = 20_000
    pageindex_token_ceiling: int = 110_000
    pageindex_toc_max_output_tokens: int = 16_000
    pageindex_add_node_id: bool = True
    pageindex_add_node_summary: bool = False
    pageindex_add_node_text: bool = False
    pageindex_add_doc_description: bool = False

    # Where indexed structure JSONs live, with their source PDFs alongside.
    pageindex_results_dir: Path = PROJECT_ROOT / "examples" / "results"

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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings loading sources and their priority order.

        Priority: init args > env vars > .env file > config.yaml > defaults.

        Args:
            settings_cls: The Settings class.
            init_settings: Settings from initialization.
            env_settings: Settings from environment variables.
            dotenv_settings: Settings from .env file.
            file_secret_settings: Unused; kept for compatibility.

        Returns:
            Tuple of settings sources in priority order.
        """
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
    """Return a cached singleton instance of Settings.

    Returns:
        Settings instance loaded from configuration sources.
    """
    return Settings()


settings = get_settings()

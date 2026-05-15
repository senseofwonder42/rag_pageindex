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
    pageindex_add_node_id: bool = True
    pageindex_add_node_summary: bool = False
    pageindex_add_node_text: bool = False
    pageindex_add_doc_description: bool = False

    # Where indexed structure JSONs live, with their source PDFs alongside.
    pageindex_results_dir: Path = PROJECT_ROOT / "examples" / "results"

    # VLM batch indexing: render every page and send batches of images to a
    # vision model that returns (headings, description) per page.
    pageindex_vlm_dpi: int = 144
    pageindex_vlm_pages_per_batch: int = 8
    # Provider cap on images per prompt (OpenRouter→Nvidia: 10). The actual
    # batch size used is min(pages_per_batch, max_images_per_call).
    pageindex_vlm_max_images_per_call: int = 8

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

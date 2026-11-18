"""Runtime configuration, read from environment variables (prefix IELTS_) or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IELTS_", env_file=".env", extra="ignore")

    exam: str = "ielts"
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "var" / "indexes"

    # LLM: "anthropic" uses the Anthropic SDK; "openai" covers OpenAI and any
    # OpenAI-compatible server (Ollama, vLLM) via llm_base_url.
    llm_provider: Literal["anthropic", "openai", "fake"] = "anthropic"
    llm_model: str = "claude-opus-5-5"
    llm_base_url: str | None = None
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 16000

    # API keys may sit in .env under their usual names; they are passed to the SDK
    # explicitly because pydantic-settings does not export .env into os.environ.
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "IELTS_ANTHROPIC_API_KEY")
    )
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "IELTS_OPENAI_API_KEY")
    )

    # Embeddings: "sentence-transformers" loads embedding_model (semantic search);
    # "hashing" needs no download and only matches surface words (tests, offline demos).
    embedder: Literal["hashing", "sentence-transformers"] = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Query expansion glossary (relative to exam_dir); set use_glossary=false to disable.
    use_glossary: bool = True
    glossary_file: str = "glossary.yaml"

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # The project's .env wins over variables exported in the shell, so a stale
        # global OPENAI_API_KEY / ANTHROPIC_API_KEY cannot shadow the project key.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    @property
    def exam_dir(self) -> Path:
        return self.data_dir / self.exam


def get_settings() -> Settings:
    return Settings()

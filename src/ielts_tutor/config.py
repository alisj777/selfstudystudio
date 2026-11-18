"""Runtime configuration, read from environment variables (prefix IELTS_) or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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

    # Embeddings: "hashing" needs no download (good for tests and demos);
    # "sentence-transformers" loads embedding_model (e.g. BAAI/bge-m3).
    embedder: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_model: str = "BAAI/bge-m3"

    @property
    def exam_dir(self) -> Path:
        return self.data_dir / self.exam


def get_settings() -> Settings:
    return Settings()

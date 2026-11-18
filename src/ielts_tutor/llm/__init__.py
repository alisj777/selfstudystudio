"""Switchable LLM layer: pick the provider with IELTS_LLM_PROVIDER."""

from __future__ import annotations

from ..config import Settings, get_settings
from .base import LLMClient, LLMError, LLMRefusal, Message
from .providers import AnthropicClient, FakeLLM, OpenAICompatibleClient

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMRefusal",
    "Message",
    "AnthropicClient",
    "OpenAICompatibleClient",
    "FakeLLM",
    "get_llm",
]


def get_llm(settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    if s.llm_provider == "anthropic":
        return AnthropicClient(model=s.llm_model, effort=s.llm_effort, max_tokens=s.llm_max_tokens)
    if s.llm_provider == "openai":
        return OpenAICompatibleClient(model=s.llm_model, base_url=s.llm_base_url, max_tokens=s.llm_max_tokens)
    return FakeLLM()

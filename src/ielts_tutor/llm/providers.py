"""Concrete LLM clients. SDKs are imported lazily so only the chosen one must be installed."""

from __future__ import annotations

import json
from collections.abc import Callable

from .base import LLMError, LLMRefusal, Message, T, parse_json_output, strict_json_schema


def _as_dicts(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


class AnthropicClient:
    """Claude via the official Anthropic SDK.

    Uses adaptive thinking (the model default) with an explicit effort level, and
    server-side refusal fallbacks (`fallbacks="default"`).
    """

    name = "anthropic"
    _FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str, effort: str = "medium", max_tokens: int = 16000):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise LLMError("Install the anthropic extra: uv sync --extra anthropic") from exc
        self._client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def _create(self, messages: list[Message], system: str | None, output_format: dict | None):
        output_config: dict = {"effort": self.effort}
        if output_format is not None:
            output_config["format"] = output_format
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=_as_dicts(messages),
            output_config=output_config,
            betas=[self._FALLBACK_BETA],
            fallbacks="default",
        )
        if system:
            kwargs["system"] = system
        response = self._client.beta.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            raise LLMRefusal(str(getattr(response, "stop_details", None)))
        if response.stop_reason == "max_tokens":
            raise LLMError("Response hit max_tokens; raise IELTS_LLM_MAX_TOKENS")
        return "".join(block.text for block in response.content if block.type == "text")

    def complete(self, messages: list[Message], *, system: str | None = None) -> str:
        return self._create(messages, system, None)

    def complete_json(self, messages: list[Message], schema: type[T], *, system: str | None = None) -> T:
        fmt = {"type": "json_schema", "schema": strict_json_schema(schema)}
        return parse_json_output(self._create(messages, system, fmt), schema)


class OpenAICompatibleClient:
    """OpenAI, or any OpenAI-compatible server (Ollama, vLLM) through `base_url`."""

    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, max_tokens: int = 16000):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise LLMError("Install the openai extra: uv sync --extra openai") from exc
        self._client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def _chat(self, messages: list[Message], system: str | None, **extra) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + _as_dicts(messages)
        resp = self._client.chat.completions.create(
            model=self.model, messages=msgs, max_tokens=self.max_tokens, **extra
        )
        return resp.choices[0].message.content or ""

    def complete(self, messages: list[Message], *, system: str | None = None) -> str:
        return self._chat(messages, system)

    def complete_json(self, messages: list[Message], schema: type[T], *, system: str | None = None) -> T:
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": strict_json_schema(schema), "strict": True},
        }
        return parse_json_output(self._chat(messages, system, response_format=fmt), schema)


class FakeLLM:
    """Deterministic stand-in for tests: `responder(messages, system)` returns the text."""

    name = "fake"

    def __init__(self, responder: Callable[[list[Message], str | None], str] | None = None):
        self.responder = responder or (lambda messages, system: "{}")
        self.calls: list[tuple[list[Message], str | None]] = []

    def complete(self, messages: list[Message], *, system: str | None = None) -> str:
        self.calls.append((messages, system))
        return self.responder(messages, system)

    def complete_json(self, messages: list[Message], schema: type[T], *, system: str | None = None) -> T:
        out = self.complete(messages, system=system)
        if not isinstance(out, str):
            out = json.dumps(out)
        return parse_json_output(out, schema)

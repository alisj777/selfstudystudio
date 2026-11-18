"""Provider-neutral LLM interface used by every agent and builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    """The provider declined the request."""


class LLMClient(Protocol):
    name: str

    def complete(self, messages: list[Message], *, system: str | None = None) -> str: ...

    def complete_json(
        self, messages: list[Message], schema: type[T], *, system: str | None = None
    ) -> T: ...


def strict_json_schema(schema: type[BaseModel]) -> dict:
    """Pydantic JSON schema with additionalProperties=false on every object,
    as structured-output APIs require."""

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        return node

    return walk(schema.model_json_schema())


def parse_json_output(text: str, schema: type[T]) -> T:
    """Parse model text into `schema`, tolerating a surrounding ```json fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        return schema.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMError(f"Model output did not match {schema.__name__}: {exc}") from exc

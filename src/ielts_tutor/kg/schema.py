"""Knowledge-graph schema: node types, relation types and which endpoints each relation allows."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    SKILL = "Skill"
    CRITERION = "Criterion"
    BAND_DESCRIPTOR = "BandDescriptor"
    GRAMMAR_POINT = "GrammarPoint"
    TOPIC = "Topic"
    ESSAY_STRUCTURE = "EssayStructure"
    LISTENING_THEME = "ListeningTheme"
    QUESTION_TYPE = "QuestionType"
    VOCABULARY = "Vocabulary"
    ERROR_TYPE = "ErrorType"
    QUESTION = "Question"


class Rel(StrEnum):
    ASSESSED_BY = "ASSESSED_BY"
    HAS_DESCRIPTOR = "HAS_DESCRIPTOR"
    EVIDENCES = "EVIDENCES"
    PREREQUISITE_OF = "PREREQUISITE_OF"
    SUBTOPIC_OF = "SUBTOPIC_OF"
    RELATED_TO = "RELATED_TO"
    SUITS = "SUITS"
    APPEARS_IN = "APPEARS_IN"
    INDICATES = "INDICATES"
    AFFECTS = "AFFECTS"
    HAS_QUESTION_TYPE = "HAS_QUESTION_TYPE"
    OF_TYPE = "OF_TYPE"


N = NodeType

# relation -> (allowed source types, allowed target types)
EDGE_RULES: dict[Rel, tuple[set[NodeType], set[NodeType]]] = {
    Rel.ASSESSED_BY: ({N.SKILL}, {N.CRITERION}),
    Rel.HAS_DESCRIPTOR: ({N.CRITERION}, {N.BAND_DESCRIPTOR}),
    Rel.EVIDENCES: ({N.GRAMMAR_POINT, N.VOCABULARY, N.ESSAY_STRUCTURE}, {N.CRITERION}),
    Rel.PREREQUISITE_OF: ({N.GRAMMAR_POINT}, {N.GRAMMAR_POINT}),
    Rel.SUBTOPIC_OF: ({N.TOPIC}, {N.TOPIC}),
    Rel.RELATED_TO: ({N.TOPIC}, {N.TOPIC}),
    Rel.SUITS: ({N.ESSAY_STRUCTURE}, {N.QUESTION_TYPE}),
    Rel.APPEARS_IN: ({N.TOPIC, N.LISTENING_THEME, N.GRAMMAR_POINT, N.VOCABULARY}, {N.QUESTION}),
    Rel.INDICATES: ({N.ERROR_TYPE}, {N.GRAMMAR_POINT}),
    Rel.AFFECTS: ({N.ERROR_TYPE}, {N.CRITERION}),
    Rel.HAS_QUESTION_TYPE: ({N.SKILL, N.LISTENING_THEME}, {N.QUESTION_TYPE}),
    Rel.OF_TYPE: ({N.QUESTION}, {N.QUESTION_TYPE}),
}


class Node(BaseModel):
    id: str
    type: NodeType
    name: str
    exam: str = "ielts"
    attrs: dict[str, Any] = Field(default_factory=dict)

    def text(self) -> str:
        """Text used for embedding / keyword search."""
        parts = [self.name]
        for key in ("description", "descriptor", "pattern", "template", "scenario"):
            if self.attrs.get(key):
                parts.append(str(self.attrs[key]))
        for key in ("keywords", "key_features", "common_errors", "collocations"):
            if self.attrs.get(key):
                parts.append(", ".join(map(str, self.attrs[key])))
        return " | ".join(parts)


class Edge(BaseModel):
    source: str
    target: str
    rel: Rel
    attrs: dict[str, Any] = Field(default_factory=dict)

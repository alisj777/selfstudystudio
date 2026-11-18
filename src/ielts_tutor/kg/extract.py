"""LLM-assisted tagging of corpus items with knowledge-graph nodes.

The model may only choose ids from the candidate lists it is shown; anything else is
dropped. Proposed new topics and low-confidence results go to a review queue for a
human to accept, instead of being written into the graph directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import LLMClient, Message
from .graph import KnowledgeGraph
from .schema import NodeType

SYSTEM = (
    "You label IELTS practice material for a study system. "
    "Choose labels only from the candidate lists you are given, using their exact ids. "
    "Propose a new topic only when no candidate topic fits."
)


class TopicProposal(BaseModel):
    name: str
    parent_id: str | None = Field(description="Existing topic id this new topic belongs under, or null")
    keywords: list[str]


class QuestionTags(BaseModel):
    topic_ids: list[str] = Field(description="1-3 topic ids from the candidate list, most relevant first")
    question_type: str | None = Field(description="One question-type id from the candidate list, or null")
    grammar_ids: list[str] = Field(description="Grammar point ids the item particularly practises; may be empty")
    new_topics: list[TopicProposal] = Field(description="Only if no candidate topic fits; otherwise empty")
    confidence: float = Field(description="0-1, how sure you are about topic_ids and question_type")


class ReviewItem(BaseModel):
    item_id: str
    text: str
    tags: QuestionTags
    dropped_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def _for_skill(node, skill: str | None) -> bool:
    if skill is None:
        return True
    if "skill" in node.attrs:
        return node.attrs["skill"] == skill
    if "skills" in node.attrs:
        return skill in node.attrs["skills"]
    return True


def _catalogue(kg: KnowledgeGraph, node_type: NodeType, skill: str | None = None) -> list[str]:
    return [f"- {n.id}: {n.name}" for n in kg.nodes(node_type) if _for_skill(n, skill)]


def build_prompt(kg: KnowledgeGraph, text: str, skill: str) -> str:
    topics = "\n".join(_catalogue(kg, NodeType.TOPIC, skill))
    qtypes = "\n".join(_catalogue(kg, NodeType.QUESTION_TYPE, skill))
    grammar = "\n".join(_catalogue(kg, NodeType.GRAMMAR_POINT))
    return (
        f"Skill: {skill}\n\n<item>\n{text}\n</item>\n\n"
        f"Candidate topics:\n{topics}\n\n"
        f"Candidate question types:\n{qtypes}\n\n"
        f"Candidate grammar points:\n{grammar}\n\n"
        "Label the item."
    )


def tag_item(
    llm: LLMClient,
    kg: KnowledgeGraph,
    item_id: str,
    text: str,
    skill: str,
    min_confidence: float = 0.7,
) -> ReviewItem:
    tags = llm.complete_json([Message("user", build_prompt(kg, text, skill))], QuestionTags, system=SYSTEM)

    dropped: list[str] = []

    def keep(ids: list[str], node_type: NodeType) -> list[str]:
        valid = []
        for i in ids:
            if i in kg and kg.get(i).type == node_type:
                valid.append(i)
            else:
                dropped.append(i)
        return valid

    tags.topic_ids = keep(tags.topic_ids, NodeType.TOPIC)
    tags.grammar_ids = keep(tags.grammar_ids, NodeType.GRAMMAR_POINT)
    if tags.question_type is not None and not keep([tags.question_type], NodeType.QUESTION_TYPE):
        tags.question_type = None
    for proposal in tags.new_topics:
        if proposal.parent_id and proposal.parent_id not in kg:
            dropped.append(proposal.parent_id)
            proposal.parent_id = None

    reasons = []
    if tags.confidence < min_confidence:
        reasons.append(f"low confidence {tags.confidence:.2f}")
    if tags.new_topics:
        reasons.append("proposes new topics")
    if dropped:
        reasons.append("model returned unknown ids")
    if not tags.topic_ids:
        reasons.append("no topic assigned")
    return ReviewItem(item_id=item_id, text=text, tags=tags, dropped_ids=dropped, reasons=reasons)

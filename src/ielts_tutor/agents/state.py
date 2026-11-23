"""Data passed between the agents (design doc 5.2). Everything is a Pydantic model so it
can be validated, stored as JSON and replayed."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from ..rag.schema import Evidence

Task = Literal["task1", "task2"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PracticeTarget(BaseModel):
    exam: str = "ielts"
    skill: str = "writing"
    task: Task = "task2"
    question_type: str | None = None  # qt.writing.t2.opinion
    topic_id: str | None = None  # topic.education.online
    focus_criteria: list[str] = Field(default_factory=list)  # weakest criteria, e.g. ["CC"]
    grammar_focus: list[str] = Field(default_factory=list)  # gp.* nodes to practise
    source: Literal["real", "generate"] = "real"
    reason: str = ""


class Question(BaseModel):
    question_id: str
    task: Task
    question_type: str | None = None
    topic_ids: list[str] = Field(default_factory=list)
    prompt: str
    data_table: str | None = None  # Task 1: the chart data as a markdown table
    min_words: int = 250
    minutes: int = 40
    source: str = "generated"  # "generated" or the corpus chunk id of a real question
    outline: list[str] = Field(default_factory=list)  # what a good answer covers
    evidence_ids: list[str] = Field(default_factory=list)


class Response(BaseModel):
    text: str
    submitted_at: str = Field(default_factory=now)
    seconds_spent: int | None = None


class ErrorItem(BaseModel):
    text: str  # the wrong span, quoted from the essay
    correction: str
    error_type: str  # err.* node id from the knowledge graph
    explanation: str


class AnalysisReport(BaseModel):
    features: dict[str, float]
    errors: list[ErrorItem] = Field(default_factory=list)
    structure: dict[str, str] = Field(default_factory=dict)  # intro / body / conclusion notes
    task_notes: list[str] = Field(default_factory=list)  # covers all parts? on topic? position clear?
    focus: list[str] = Field(default_factory=list)  # criteria the scorer asked to re-check
    llm_used: bool = False


class CriterionScore(BaseModel):
    criterion: str  # TR / TA / CC / LR / GRA
    band: float
    model: float | None = None
    knn: float | None = None
    llm: float | None = None
    interval: tuple[float, float] | None = None
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class ScoreCard(BaseModel):
    criteria: dict[str, CriterionScore]
    overall: float
    model_version: str
    weights: dict[str, list[float]] = Field(default_factory=dict)  # per criterion: model, kNN, LLM
    needs_review: bool = False
    review_reason: str = ""


class FeedbackItem(BaseModel):
    criterion: str
    issue: str
    suggestion: str
    example: str = ""  # a rewrite of the student's own sentence
    evidence_ids: list[str] = Field(default_factory=list)


class Feedback(BaseModel):
    summary: str
    items: list[FeedbackItem]
    next_band_tips: list[str] = Field(default_factory=list)  # what the next band up requires
    review_nodes: list[str] = Field(default_factory=list)  # knowledge-graph nodes to revise
    dropped: int = 0  # items removed because they cited no evidence


class Mastery(BaseModel):
    value: float = 0.5  # 0-1
    attempts: int = 0
    updated: str = Field(default_factory=now)


class LearnerProfile(BaseModel):
    learner_id: str = "default"
    exam: str = "ielts"
    target_band: float = 7.0
    mastery: dict[str, Mastery] = Field(default_factory=dict)  # node id -> mastery
    history: list[dict] = Field(default_factory=list)  # {session_id, task, overall, criteria, at}


class StepLog(BaseModel):
    agent: str
    at: str = Field(default_factory=now)
    ms: int = 0
    summary: str = ""
    llm_calls: int = 0


class Session(BaseModel):
    """The shared state of one practice session (SessionState in the design doc)."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    exam: str = "ielts"
    created: str = Field(default_factory=now)
    learner: LearnerProfile = Field(default_factory=LearnerProfile)
    request: dict = Field(default_factory=dict)  # what the student asked for
    target: PracticeTarget | None = None
    question: Question | None = None
    response: Response | None = None
    analysis: AnalysisReport | None = None
    score: ScoreCard | None = None
    feedback: Feedback | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    trace: list[StepLog] = Field(default_factory=list)
    retries: dict[str, int] = Field(default_factory=dict)
    next: str | None = None  # an agent may ask to re-run an upstream agent

    def add_evidence(self, items: list[Evidence]) -> list[str]:
        have = {e.id for e in self.evidence}
        for e in items:
            if e.id not in have:
                self.evidence.append(e)
                have.add(e.id)
        return [e.id for e in items]

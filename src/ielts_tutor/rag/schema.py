"""Records stored in the three RAG sources, and the Evidence object agents receive."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceName = Literal["corpus", "rubric", "topic"]


class CorpusChunk(BaseModel):
    """Corpus RAG: exam questions, question-bank items, model answers, passages, transcripts."""

    chunk_id: str
    exam: str = "ielts"
    skill: Literal["writing", "speaking", "reading", "listening"]
    source_type: Literal["real_exam", "question_bank", "model_answer", "passage", "transcript"]
    question_id: str | None = None
    question_type: str | None = None  # QuestionType node id
    topic_ids: list[str] = Field(default_factory=list)
    grammar_ids: list[str] = Field(default_factory=list)
    cefr: str | None = None
    band: float | None = None
    text: str
    license: str = "internal_only"
    year: int | None = None


class ScoredSample(BaseModel):
    """Rubric RAG: a response with human scores, used to calibrate the scorer."""

    sample_id: str
    exam: str = "ielts"
    skill: Literal["writing", "speaking"]
    task: str  # task1 | task2 | part1 | part2 | part3
    prompt_id: str | None = None
    prompt_text: str | None = None
    response_text: str
    audio_uri: str | None = None
    scores: dict[str, float]  # criterion short name -> band, plus "overall"
    rater: Literal["official", "teacher", "model", "dataset"] = "dataset"
    comments: str | None = None
    source: str | None = None  # where the sample came from, e.g. "hf:<dataset id>"


class IndexDoc(BaseModel):
    id: str
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """One retrieved item, as handed to agents. Feedback cites evidence by `id`."""

    source: SourceName
    id: str
    text: str
    score: float
    meta: dict[str, Any] = Field(default_factory=dict)

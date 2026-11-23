"""Everything the agents may call: RAG, knowledge graph, LLM, score model. Built once per
process and passed to every agent, so tests can swap any part (FakeLLM, tiny indexes)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable

from ..config import Settings, get_settings
from ..importers.textquality import Vocab
from ..kg import KnowledgeGraph
from ..llm import LLMClient, Message
from ..rag.retrievers import MultiRAG
from .state import Session, StepLog


class CountingLLM:
    """Wraps an LLM client and counts calls, for the trace and for cost control."""

    def __init__(self, inner: LLMClient):
        self.inner, self.calls = inner, 0
        self.name = getattr(inner, "name", "llm")

    def complete(self, messages: list[Message], *, system: str | None = None) -> str:
        self.calls += 1
        return self.inner.complete(messages, system=system)

    def complete_json(self, messages, schema, *, system: str | None = None):
        self.calls += 1
        return self.inner.complete_json(messages, schema, system=system)


@dataclass
class Toolkit:
    kg: KnowledgeGraph
    rag: MultiRAG
    llm: CountingLLM | None  # None: run without LLM (model score only, no generated text)
    scorer: object | None  # writing.scorer.WritingScorer
    cache: object | None  # writing.scorer.EmbeddingCache
    vocab: Vocab
    settings: Settings
    lang: str = "zh"  # language of explanations shown to the student
    data: dict = field(default_factory=dict)  # sessions_dir, learners_dir

    @classmethod
    def load(cls, settings: Settings | None = None, llm: LLMClient | None = None, use_llm: bool = True) -> "Toolkit":
        from ..cli import _glossary, _kg_path
        from ..llm import get_llm
        from ..rag import get_embedder, load_multirag
        from ..writing.scorer import EmbeddingCache, WritingScorer

        s = settings or get_settings()
        kg = KnowledgeGraph.from_json(_kg_path())
        embedder = get_embedder(s)
        rag = load_multirag(s.index_dir, embedder, kg, _glossary())
        model_path = s.index_dir.parent / "models" / "writing_scorer.pkl"
        scorer = WritingScorer.load(model_path) if model_path.exists() else None
        cache = EmbeddingCache(embedder, None)
        client = (llm or get_llm(s)) if use_llm else None
        return cls(kg=kg, rag=rag, llm=CountingLLM(client) if client else None, scorer=scorer, cache=cache,
                   vocab=Vocab.load(), settings=s,
                   data={"sessions": s.index_dir.parent / "sessions", "learners": s.index_dir.parent / "learners"})


def agent(name: str) -> Callable:
    """Decorator: time the agent, count its LLM calls and append a StepLog to the trace."""

    def wrap(fn: Callable[[Session, Toolkit], str]) -> Callable[[Session, Toolkit], Session]:
        @wraps(fn)
        def run(session: Session, tk: Toolkit) -> Session:
            before = tk.llm.calls if tk.llm else 0
            t0 = time.perf_counter()
            summary = fn(session, tk) or ""
            session.trace.append(StepLog(agent=name, ms=int((time.perf_counter() - t0) * 1000), summary=summary,
                                         llm_calls=(tk.llm.calls if tk.llm else 0) - before))
            return session

        run.agent_name = name  # type: ignore[attr-defined]
        return run

    return wrap


def save_session(session: Session, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session.session_id}.json"
    path.write_text(session.model_dump_json(indent=1), encoding="utf-8")
    return path


def load_session(session_id: str, directory: Path) -> Session:
    return Session.model_validate_json((directory / f"{session_id}.json").read_text(encoding="utf-8"))

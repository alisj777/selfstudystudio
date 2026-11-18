"""Retrieval evaluation: hit@1, hit@3 and MRR@10 over a labelled query set."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .retrievers import MultiRAG
from .schema import Evidence, SourceName

DEPTH = 10


class EvalQuery(BaseModel):
    id: str
    source: SourceName
    query: str
    expected: list[str]
    skill: str | None = None
    lang: str = "en"


class QueryResult(BaseModel):
    query: EvalQuery
    rank: int | None  # 1-based rank of the first relevant result within DEPTH
    top: list[str] = Field(default_factory=list)


def load_queries(path: Path) -> list[EvalQuery]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [EvalQuery.model_validate(q) for q in doc["queries"]]


def _relevant(ev: Evidence, expected: list[str]) -> bool:
    return any(ev.id == e or ev.id.startswith(e + ".") for e in expected)


def _retrieve(rag: MultiRAG, q: EvalQuery) -> list[Evidence]:
    if q.source == "rubric":
        return rag.rubric.search_descriptors(q.query, DEPTH, skill=q.skill)
    if q.source == "topic":
        return rag.topic.retrieve(q.query, DEPTH, skill=q.skill, expand=False)
    return rag.corpus.retrieve(q.query, DEPTH, skill=q.skill)


def run(rag: MultiRAG, queries: list[EvalQuery]) -> list[QueryResult]:
    results = []
    for q in queries:
        hits = _retrieve(rag, q)
        rank = next((i + 1 for i, ev in enumerate(hits) if _relevant(ev, q.expected)), None)
        results.append(QueryResult(query=q, rank=rank, top=[ev.id for ev in hits[:3]]))
    return results


def summarise(results: list[QueryResult]) -> dict[str, dict[str, float]]:
    """Metrics overall, per source and per language."""
    groups: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        groups["all"].append(r)
        groups[r.query.source].append(r)
        if r.query.lang != "en":
            groups[f"lang:{r.query.lang}"].append(r)
    out = {}
    for name, rs in groups.items():
        n = len(rs)
        out[name] = {
            "n": n,
            "hit@1": sum(r.rank == 1 for r in rs) / n,
            "hit@3": sum(r.rank is not None and r.rank <= 3 for r in rs) / n,
            "mrr": sum(1 / r.rank for r in rs if r.rank) / n,
        }
    return out

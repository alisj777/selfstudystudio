"""Turn the knowledge graph and raw data files into the three indexes, and load them back."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from ..kg import KnowledgeGraph, NodeType, Rel
from .embed import Embedder
from .glossary import Glossary
from .index import HybridIndex
from .retrievers import CorpusRetriever, MultiRAG, RubricRetriever, TopicRetriever
from .schema import CorpusChunk, IndexDoc, ScoredSample

SOURCES = ("corpus", "rubric", "topic")


# ---------------------------------------------------------------- chunking
def chunk_text(text: str, max_words: int = 300) -> list[str]:
    """Split on blank lines, then pack paragraphs into chunks of at most ~max_words."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        words = len(para.split())
        if current and count + words > max_words:
            chunks.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += words
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def split_chunk(chunk: CorpusChunk, max_words: int = 300) -> list[CorpusChunk]:
    """Questions stay whole; passages, transcripts and model answers are split into paragraphs."""
    if chunk.source_type in ("real_exam", "question_bank"):
        return [chunk]
    parts = chunk_text(chunk.text, max_words)
    if len(parts) == 1:
        return [chunk]
    return [
        chunk.model_copy(update={"chunk_id": f"{chunk.chunk_id}#p{i + 1}", "text": part})
        for i, part in enumerate(parts)
    ]


# ---------------------------------------------------------------- loaders
def load_jsonl(path: Path, model):
    with path.open(encoding="utf-8") as fh:
        return [model.model_validate_json(line) for line in fh if line.strip()]


def load_corpus(directory: Path) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    for path in sorted(directory.glob("*.jsonl")):
        chunks.extend(load_jsonl(path, CorpusChunk))
    return chunks


def load_samples(directory: Path) -> list[ScoredSample]:
    samples: list[ScoredSample] = []
    if directory.exists():
        for path in sorted(directory.glob("*.jsonl")):
            samples.extend(load_jsonl(path, ScoredSample))
    return samples


def import_scored_csv(
    path: Path,
    *,
    text_col: str,
    score_cols: dict[str, str],
    prompt_col: str | None = None,
    task: str = "task2",
    skill: str = "writing",
    id_prefix: str = "csv",
) -> list[ScoredSample]:
    """Import a scored-essay CSV (e.g. a public IELTS essay dataset).

    `score_cols` maps our criterion names ("TR", "CC", "LR", "GRA", "overall") to CSV
    column names. Rows without a parseable overall band are skipped.
    """
    samples: list[ScoredSample] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            scores: dict[str, float] = {}
            for ours, col in score_cols.items():
                try:
                    scores[ours] = float(str(row.get(col, "")).strip())
                except ValueError:
                    continue
            text = (row.get(text_col) or "").strip()
            if "overall" not in scores or not text:
                continue
            samples.append(
                ScoredSample(
                    sample_id=f"{id_prefix}-{i:06d}",
                    skill=skill,  # type: ignore[arg-type]
                    task=task,
                    prompt_text=(row.get(prompt_col) or None) if prompt_col else None,
                    response_text=text,
                    scores=scores,
                    rater="dataset",
                )
            )
    return samples


def write_jsonl(path: Path, records: Iterable) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")
            n += 1
    return n


# ---------------------------------------------------------------- documents
def corpus_docs(chunks: list[CorpusChunk]) -> list[IndexDoc]:
    docs = []
    for chunk in chunks:
        for part in split_chunk(chunk):
            meta = part.model_dump(exclude={"text", "chunk_id"})
            docs.append(IndexDoc(id=part.chunk_id, text=part.text, meta=meta))
    return docs


def topic_docs(kg: KnowledgeGraph) -> list[IndexDoc]:
    docs = []
    for node in kg.nodes(NodeType.TOPIC):
        parents = kg.out(node.id, Rel.SUBTOPIC_OF)
        text = node.name
        if parents:
            text = f"{parents[0].name} > {text}"
        if node.attrs.get("keywords"):
            text += ". Keywords: " + ", ".join(node.attrs["keywords"])
        if node.attrs.get("collocations"):
            text += ". Useful collocations: " + ", ".join(node.attrs["collocations"])
        for arg in node.attrs.get("arguments", []):
            text += f". {arg.get('stance', '')}: {arg.get('point', '')}"
        docs.append(
            IndexDoc(
                id=node.id,
                text=text,
                meta={
                    "kind": "topic",
                    "exam": node.exam,
                    "name": node.name,
                    "parent": parents[0].id if parents else None,
                    "skills": node.attrs.get("skills", []),
                    "keywords": node.attrs.get("keywords", []),
                    "collocations": node.attrs.get("collocations", []),
                },
            )
        )
    return docs


def rubric_docs(kg: KnowledgeGraph, samples: list[ScoredSample]) -> list[IndexDoc]:
    docs = []
    for desc in kg.nodes(NodeType.BAND_DESCRIPTOR):
        crit = kg.get(desc.attrs["criterion"])
        features = desc.attrs.get("key_features", [])
        text = f"{crit.name} band {desc.attrs['band']}: {desc.attrs.get('descriptor', '')}"
        docs.append(
            IndexDoc(
                id=desc.id,
                text=text,
                meta={
                    "kind": "descriptor",
                    "exam": desc.exam,
                    "skill": crit.attrs.get("skill"),
                    "criterion": crit.id,
                    "criterion_short": crit.attrs.get("short"),
                    "band": desc.attrs["band"],
                    "key_features": features,
                    "source_doc": desc.attrs.get("source_doc"),
                    "page": desc.attrs.get("page"),
                },
            )
        )
    for s in samples:
        docs.append(
            IndexDoc(
                id=s.sample_id,
                text=s.response_text,
                meta={
                    "kind": "sample",
                    "exam": s.exam,
                    "skill": s.skill,
                    "task": s.task,
                    "prompt_id": s.prompt_id,
                    "prompt_text": s.prompt_text,
                    "scores": s.scores,
                    "overall": s.scores.get("overall"),
                    "rater": s.rater,
                    "comments": s.comments,
                    "source": s.source,
                },
            )
        )
    return docs


# ---------------------------------------------------------------- build / load
def build_indexes(
    kg: KnowledgeGraph,
    chunks: list[CorpusChunk],
    samples: list[ScoredSample],
    embedder: Embedder,
) -> dict[str, HybridIndex]:
    indexes = {name: HybridIndex(embedder) for name in SOURCES}
    indexes["corpus"].add(corpus_docs(chunks))
    indexes["rubric"].add(rubric_docs(kg, samples))
    indexes["topic"].add(topic_docs(kg))
    return indexes


def save_indexes(indexes: dict[str, HybridIndex], directory: Path) -> dict[str, int]:
    counts = {}
    for name, index in indexes.items():
        index.save(directory / name)
        counts[name] = len(index)
    (directory / "summary.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    return counts


def multirag_from(
    indexes: dict[str, HybridIndex], kg: KnowledgeGraph | None, glossary: Glossary | None = None
) -> MultiRAG:
    return MultiRAG(
        corpus=CorpusRetriever(indexes["corpus"], glossary),
        rubric=RubricRetriever(indexes["rubric"], glossary),
        topic=TopicRetriever(indexes["topic"], kg, glossary),
    )


def load_multirag(
    directory: Path, embedder: Embedder, kg: KnowledgeGraph | None, glossary: Glossary | None = None
) -> MultiRAG:
    indexes = {name: HybridIndex.load(directory / name, embedder) for name in SOURCES}
    return multirag_from(indexes, kg, glossary)

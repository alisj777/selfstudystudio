"""Hybrid (BM25 + dense) index with metadata filters and on-disk persistence.

Kept dependency-light on purpose: the `search` interface is what retrievers use,
so a Qdrant/Milvus-backed index can replace this class without touching them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from .embed import Embedder, tokenize
from .schema import IndexDoc

RRF_K = 60


class Hit(BaseModel):
    doc: IndexDoc
    score: float  # fused RRF score (higher is better)
    dense: float  # cosine similarity
    bm25: float


def _match(value: Any, cond: Any) -> bool:
    """Filter semantics: scalar -> equality (or membership if the doc value is a list);
    list -> any overlap; {"gte": x, "lte": y} -> numeric range."""
    if isinstance(cond, Mapping):
        if value is None:
            return False
        if "gte" in cond and value < cond["gte"]:
            return False
        if "lte" in cond and value > cond["lte"]:
            return False
        return True
    values = value if isinstance(value, list) else [value]
    wanted = cond if isinstance(cond, (list, tuple, set)) else [cond]
    return any(v in wanted for v in values)


class HybridIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.docs: list[IndexDoc] = []
        self._vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self.docs)

    def add(self, docs: list[IndexDoc]) -> None:
        if not docs:
            return
        seen = {d.id for d in self.docs}
        for d in docs:
            if d.id in seen:
                raise ValueError(f"Duplicate document id in index: {d.id}")
            seen.add(d.id)
        vecs = self.embedder.embed([d.text for d in docs])
        self.docs.extend(docs)
        self._vectors = np.vstack([self._vectors, vecs])
        self._bm25 = None  # rebuilt lazily

    def _bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(d.text) or ["_"] for d in self.docs])
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

    def filter_ids(self, filters: Mapping[str, Any] | None) -> np.ndarray:
        if not filters:
            return np.arange(len(self.docs))
        keep = [
            i for i, d in enumerate(self.docs)
            if all(_match(d.meta.get(key), cond) for key, cond in filters.items())
        ]
        return np.asarray(keep, dtype=int)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Mapping[str, Any] | None = None,
        *,
        bm25_query: str | None = None,
        symmetric: bool = False,
    ) -> list[Hit]:
        """`bm25_query` overrides the keyword-side text (e.g. glossary-expanded).
        `symmetric=True` embeds the query like a document (essay-to-essay similarity)
        instead of as a short search query."""
        if not self.docs:
            return []
        candidates = self.filter_ids(filters)
        if candidates.size == 0:
            return []
        embed = self.embedder.embed if symmetric else self.embedder.embed_query
        qvec = embed([query])[0]
        dense = self._vectors[candidates] @ qvec
        bm25 = self._bm25_scores(bm25_query or query)[candidates]

        dense_rank = np.empty_like(candidates)
        dense_rank[np.argsort(-dense, kind="stable")] = np.arange(candidates.size)
        bm25_rank = np.empty_like(candidates)
        bm25_rank[np.argsort(-bm25, kind="stable")] = np.arange(candidates.size)
        fused = 1.0 / (RRF_K + dense_rank + 1) + 1.0 / (RRF_K + bm25_rank + 1)
        # Documents with no lexical overlap get no BM25 credit.
        fused = np.where(bm25 > 0, fused, 1.0 / (RRF_K + dense_rank + 1))

        order = np.argsort(-fused, kind="stable")[:k]
        return [
            Hit(
                doc=self.docs[candidates[i]],
                score=float(fused[i]),
                dense=float(dense[i]),
                bm25=float(bm25[i]),
            )
            for i in order
        ]

    def vectors_for(self, ids: list[str]) -> np.ndarray:
        pos = {d.id: i for i, d in enumerate(self.docs)}
        return self._vectors[[pos[i] for i in ids]]

    # ---- persistence ----------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "docs.jsonl").open("w", encoding="utf-8") as fh:
            for d in self.docs:
                fh.write(d.model_dump_json() + "\n")
        np.save(directory / "vectors.npy", self._vectors)
        manifest = {"embedder": self.embedder.name, "dim": self.embedder.dim, "count": len(self.docs)}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> HybridIndex:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["embedder"] != embedder.name:
            raise ValueError(
                f"Index at {directory} was built with {manifest['embedder']}, "
                f"but the current embedder is {embedder.name}; rebuild the index."
            )
        index = cls(embedder)
        with (directory / "docs.jsonl").open(encoding="utf-8") as fh:
            index.docs = [IndexDoc.model_validate_json(line) for line in fh if line.strip()]
        index._vectors = np.load(directory / "vectors.npy")
        return index

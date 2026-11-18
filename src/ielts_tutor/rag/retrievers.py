"""The three RAG sources (corpus, rubric, topic) and the multi-source fusion used by agents."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..kg import KnowledgeGraph
from .glossary import Glossary
from .index import Hit, HybridIndex
from .schema import Evidence, SourceName

RRF_K = 60


def _evidence(source: SourceName, hit: Hit) -> Evidence:
    meta = dict(hit.doc.meta)
    meta.update(dense=round(hit.dense, 4), bm25=round(hit.bm25, 4))
    return Evidence(source=source, id=hit.doc.id, text=hit.doc.text, score=hit.score, meta=meta)


def rrf_fuse(rankings: Sequence[Sequence[Evidence]], k: int | None = None) -> list[Evidence]:
    """Reciprocal Rank Fusion of several ranked evidence lists (dedup by source+id)."""
    scores: dict[tuple[str, str], float] = {}
    first: dict[tuple[str, str], Evidence] = {}
    for ranking in rankings:
        for rank, ev in enumerate(ranking):
            key = (ev.source, ev.id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            first.setdefault(key, ev)
    fused = [first[key].model_copy(update={"score": s}) for key, s in scores.items()]
    fused.sort(key=lambda e: e.score, reverse=True)
    return fused[:k] if k else fused


class _Retriever:
    source: SourceName

    def __init__(self, index: HybridIndex, glossary: Glossary | None = None):
        self.index = index
        self.glossary = glossary

    def _search(self, query: str, k: int, filters: dict | None) -> list[Hit]:
        bm25_query = self.glossary.expand(query).query if self.glossary else None
        return self.index.search(query, k, filters, bm25_query=bm25_query)


class CorpusRetriever(_Retriever):
    """Exam questions, question-bank items, model answers and passages."""

    source: SourceName = "corpus"

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        skill: str | None = None,
        question_type: str | None = None,
        topic_ids: list[str] | None = None,
        source_types: list[str] | None = None,
        exclude_ids: Sequence[str] = (),
        mmr_lambda: float | None = None,
    ) -> list[Evidence]:
        """`mmr_lambda` (0-1) turns on Maximal Marginal Relevance to diversify results,
        e.g. when gathering seed questions for generating a new one."""
        filters = {
            key: value
            for key, value in {
                "skill": skill,
                "question_type": question_type,
                "topic_ids": topic_ids,
                "source_type": source_types,
            }.items()
            if value
        }
        excluded = set(exclude_ids)
        pool = k * 4 if mmr_lambda is not None else k + len(excluded)
        hits = [h for h in self._search(query, pool, filters) if h.doc.id not in excluded]
        if mmr_lambda is not None and len(hits) > k:
            hits = self._mmr(hits, k, mmr_lambda)
        return [_evidence(self.source, h) for h in hits[:k]]

    def _mmr(self, hits: list[Hit], k: int, lam: float) -> list[Hit]:
        vecs = self.index.vectors_for([h.doc.id for h in hits])
        relevance = np.array([h.dense for h in hits])
        chosen: list[int] = []
        remaining = list(range(len(hits)))
        while remaining and len(chosen) < k:
            if chosen:
                redundancy = (vecs[remaining] @ vecs[chosen].T).max(axis=1)
            else:
                redundancy = np.zeros(len(remaining))
            scores = lam * relevance[remaining] - (1 - lam) * redundancy
            pick = remaining[int(np.argmax(scores))]
            chosen.append(pick)
            remaining.remove(pick)
        return [hits[i] for i in chosen]

    def max_similarity(self, text: str, *, skill: str | None = None) -> tuple[float, Evidence | None]:
        """Highest cosine similarity between `text` and existing corpus items; used to
        reject generated questions that copy real ones too closely."""
        hits = self.index.search(text, 1, {"skill": skill} if skill else None, symmetric=True)
        if not hits:
            return 0.0, None
        return hits[0].dense, _evidence(self.source, hits[0])


class RubricRetriever(_Retriever):
    """Band descriptors plus scored samples for calibration."""

    source: SourceName = "rubric"

    def descriptors(self, criterion_id: str, band: float, window: int = 1) -> list[Evidence]:
        """Descriptors for `criterion_id` near `band`: 6 with window 1 -> bands 5-7;
        6.5 with window 1 -> bands 6-7 (the 'why 6.5 and not 7' comparison)."""
        if band % 1:
            lo, hi = int(np.floor(band)) - (window - 1), int(np.ceil(band)) + (window - 1)
        else:
            lo, hi = int(band) - window, int(band) + window
        docs = [
            d for d in self.index.docs
            if d.meta.get("kind") == "descriptor"
            and d.meta.get("criterion") == criterion_id
            and lo <= d.meta.get("band", -1) <= hi
        ]
        docs.sort(key=lambda d: d.meta["band"])
        return [Evidence(source=self.source, id=d.id, text=d.text, score=1.0, meta=d.meta) for d in docs]

    def search_descriptors(self, query: str, k: int = 5, *, skill: str | None = None) -> list[Evidence]:
        filters = {"kind": "descriptor", **({"skill": skill} if skill else {})}
        return [_evidence(self.source, h) for h in self._search(query, k, filters)]

    def similar_samples(
        self,
        response_text: str,
        k: int = 5,
        *,
        skill: str,
        task: str | None = None,
        around_band: float | None = None,
        band_window: float = 0.5,
    ) -> list[Evidence]:
        """Scored samples similar to a response, optionally within ±band_window of a predicted band."""
        filters: dict = {"kind": "sample", "skill": skill}
        if task:
            filters["task"] = task
        if around_band is not None:
            filters["overall"] = {"gte": around_band - band_window, "lte": around_band + band_window}
        hits = self.index.search(response_text, k, filters, symmetric=True)
        return [_evidence(self.source, h) for h in hits]

    def knn_band(self, samples: list[Evidence], criterion: str = "overall") -> float | None:
        """Similarity-weighted mean band of retrieved samples (the k-NN prior in the fusion formula)."""
        pairs = [
            (max(ev.meta.get("dense", 0.0), 0.0), ev.meta.get("scores", {}).get(criterion))
            for ev in samples
        ]
        pairs = [(w, b) for w, b in pairs if b is not None]
        if not pairs:
            return None
        total = sum(w for w, _ in pairs)
        if total == 0:
            return float(np.mean([b for _, b in pairs]))
        return float(sum(w * b for w, b in pairs) / total)


class TopicRetriever(_Retriever):
    """Topic entries with one-hop knowledge-graph expansion (GraphRAG)."""

    source: SourceName = "topic"

    def __init__(self, index: HybridIndex, kg: KnowledgeGraph | None = None, glossary: Glossary | None = None):
        super().__init__(index, glossary)
        self.kg = kg

    def retrieve(
        self, query: str, k: int = 3, *, skill: str | None = None, expand: bool = True
    ) -> list[Evidence]:
        hits = self._search(query, k, {"skills": skill} if skill else None)
        results = [_evidence(self.source, h) for h in hits]
        if not (expand and self.kg):
            return results
        seen = {ev.id for ev in results}
        by_id = {d.id: d for d in self.index.docs}
        expanded: list[Evidence] = []
        for ev in results:
            for neighbour in self.kg.topic_neighbourhood(ev.id):
                doc = by_id.get(neighbour.id)
                if doc is None or doc.id in seen:
                    continue
                seen.add(doc.id)
                meta = {**doc.meta, "expanded_from": ev.id}
                expanded.append(Evidence(source=self.source, id=doc.id, text=doc.text, score=ev.score * 0.5, meta=meta))
        return results + expanded


class MultiRAG:
    """Facade the agents use: query one or several sources and fuse the results."""

    def __init__(self, corpus: CorpusRetriever, rubric: RubricRetriever, topic: TopicRetriever):
        self.corpus = corpus
        self.rubric = rubric
        self.topic = topic

    def retrieve(
        self,
        query: str,
        sources: Sequence[SourceName] = ("corpus", "rubric", "topic"),
        k: int = 8,
        *,
        skill: str | None = None,
    ) -> list[Evidence]:
        rankings: list[list[Evidence]] = []
        if "corpus" in sources:
            rankings.append(self.corpus.retrieve(query, k, skill=skill))
        if "rubric" in sources:
            rankings.append(self.rubric.search_descriptors(query, k, skill=skill))
        if "topic" in sources:
            rankings.append(self.topic.retrieve(query, max(3, k // 2), skill=skill))
        return rrf_fuse(rankings, k)

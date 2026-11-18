"""Embedders. `HashingEmbedder` needs no model download; use sentence-transformers for real quality."""

from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np
import snowballstemmer

from ..config import Settings, get_settings

_WORD = re.compile(r"[a-z0-9]+")
_STEMMER = snowballstemmer.stemmer("english")

# Instruction prepended to short search queries (not to documents), per model card.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-m3": "",
}


def tokenize(text: str) -> list[str]:
    """Lower-case, split hyphenated words, and stem (overused / over-use -> over, use)."""
    return _STEMMER.stemWords(_WORD.findall(text.lower().replace("-", " ")))


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed documents: an (n, dim) float32 array of L2-normalised vectors."""
        ...

    def embed_query(self, texts: list[str]) -> np.ndarray:
        """Embed short search queries (may add a model-specific instruction)."""
        ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic feature hashing over words, word bigrams and character trigrams.

    It captures lexical overlap only (no synonyms), which is enough for tests,
    offline demos and as a fallback.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = tokenize(text)
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = zlib.crc32(feat.encode("utf-8"))
                weight = 0.5 if feat.startswith("c:") else 1.0
                mat[row, h % self.dim] += weight if (h >> 31) & 1 else -weight
        return _normalise(mat)

    def embed_query(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("Install the embeddings extra: uv sync --extra embeddings") from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model_name}"
        self.query_prefix = QUERY_PREFIXES.get(model_name, "") if query_prefix is None else query_prefix

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, texts: list[str]) -> np.ndarray:
        return self.embed([self.query_prefix + t for t in texts])


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    if s.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(s.embedding_model)
    return HashingEmbedder()

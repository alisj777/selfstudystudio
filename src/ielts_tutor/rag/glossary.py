"""Domain glossary for query expansion on the keyword (BM25) side of hybrid search.

The dense side keeps the original query: embedding models already handle general
synonyms, and padding their input with extra terms tends to blur the meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .embed import tokenize

_CJK = re.compile(r"[一-鿿]")


@dataclass
class Expansion:
    query: str  # text to send to BM25
    matched: list[str] = field(default_factory=list)  # glossary group names that fired
    english: str = ""  # the English glossary terms that were added

    @property
    def dense_query(self) -> str | None:
        """For a Chinese query, the English terms are a better input for an English
        embedding model than the Chinese text; otherwise keep the original query."""
        if self.english and _CJK.search(self.query):
            return self.english
        return None


class Glossary:
    def __init__(self, groups: dict[str, list[str]]):
        self.groups = groups
        # Pre-tokenise English terms; Chinese terms are matched as substrings.
        self._en: list[tuple[tuple[str, ...], str]] = []
        self._zh: list[tuple[str, str]] = []
        for name, terms in groups.items():
            for term in terms:
                if _CJK.search(term):
                    self._zh.append((term, name))
                else:
                    tokens = tuple(tokenize(term))
                    if tokens:
                        self._en.append((tokens, name))

    @classmethod
    def load(cls, path: Path) -> Glossary:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls({g["name"]: list(g["terms"]) for g in doc.get("groups", [])})

    def match(self, query: str) -> list[str]:
        tokens = tokenize(query)
        hits: list[str] = []
        for term_tokens, name in self._en:
            n = len(term_tokens)
            if any(tuple(tokens[i:i + n]) == term_tokens for i in range(len(tokens) - n + 1)):
                if name not in hits:
                    hits.append(name)
        for term, name in self._zh:
            if term in query and name not in hits:
                hits.append(name)
        return hits

    def expand(self, query: str) -> Expansion:
        matched = self.match(query)
        if not matched:
            return Expansion(query=query)
        # Add each new word once: BM25 counts repeated query terms, so "linking words,
        # transition words" would otherwise triple the weight of the generic "word".
        seen = set(tokenize(query))
        extra: list[str] = []
        for name in matched:
            for term in self.groups[name]:
                if _CJK.search(term):
                    continue
                # Emit original words (not stems): the index stems the query itself,
                # and stemming a stem again can change it ("cohes" -> "cohe").
                for word in term.replace("-", " ").split():
                    stems = tokenize(word)
                    if stems and not set(stems) <= seen:
                        seen.update(stems)
                        extra.append(word)
        english = " ".join(extra)
        return Expansion(query=f"{query} {english}", matched=matched, english=english)

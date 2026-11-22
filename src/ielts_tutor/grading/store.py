"""Load answer keys and explanation sections written by the importers."""

from __future__ import annotations

from pathlib import Path

from ..importers.answer_keys import AnswerKey
from ..importers.explanations import ExplanationSection
from ..rag.build import load_jsonl


def _load(directory: Path, model) -> list:
    out: list = []
    if directory.exists():
        for path in sorted(directory.rglob("*.jsonl")):
            out.extend(load_jsonl(path, model))
    return out


def load_keys(exam_dir: Path) -> list[AnswerKey]:
    """All answer files; for the same question a later file wins, so a hand-made
    `answers/private/manual.jsonl` can fill or correct what the importer extracted."""
    keys: dict[tuple, AnswerKey] = {}
    for k in _load(exam_dir / "answers", AnswerKey):
        keys[(k.book, k.test.upper(), k.skill, k.question)] = k
    return list(keys.values())


def load_explanations(exam_dir: Path) -> list[ExplanationSection]:
    return _load(exam_dir / "explanations", ExplanationSection)


def keys_for(keys: list[AnswerKey], book: int, test: str, skill: str) -> list[AnswerKey]:
    return sorted((k for k in keys if k.book == book and k.test.upper() == test.upper() and k.skill == skill),
                  key=lambda k: k.question)

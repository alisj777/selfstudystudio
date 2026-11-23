"""Listening and Reading practice on the user's Cambridge books (MVP).

The questions are shown as page images of the book (tables, maps and diagrams survive that
way), the audio is played per part, and the answers are marked against the extracted keys,
with the Chinese explanation for each wrong answer. Results go into the learner profile
next to the writing history, so the planner can see all four skills.

Only files listed in the test manifest are ever served; nothing is read from a path the
browser sends.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

from ..agents.learner import load_learner, save_learner
from ..agents.state import Mastery, now
from ..importers.cambridge import PracticeTest
from ..importers.explanations import ExplanationSection, find_explanation
from ..rag.build import load_jsonl
from .objective import mark
from .store import keys_for, load_explanations, load_keys


class PracticeService:
    def __init__(self, exam_dir: Path, var_dir: Path):
        self.exam_dir, self.var_dir = exam_dir, var_dir
        manifest = exam_dir / "tests" / "private" / "cambridge.jsonl"
        self.tests = {t.test_id: t for t in load_jsonl(manifest, PracticeTest)} if manifest.exists() else {}
        self.keys = load_keys(exam_dir)
        self._explanations: list[ExplanationSection] | None = None

    @property
    def explanations(self) -> list[ExplanationSection]:
        if self._explanations is None:
            self._explanations = load_explanations(self.exam_dir)
        return self._explanations

    # -- catalogue
    def catalogue(self, learner_id: str = "default") -> list[dict]:
        done = {h.get("test_id"): h for h in load_learner(learner_id, self.var_dir / "learners").history
                if h.get("test_id")}
        out = []
        for t in sorted(self.tests.values(), key=lambda t: (t.skill, t.book, int(t.test))):
            n = len(keys_for(self.keys, t.book, t.test, t.skill))
            out.append({"test_id": t.test_id, "book": t.book, "test": t.test, "skill": t.skill,
                        "pages": len(t.pages), "audio_parts": sorted(t.audio), "keys": n,
                        "done": done.get(t.test_id, {}).get("band")})
        return out

    def get(self, test_id: str) -> PracticeTest:
        if test_id not in self.tests:
            raise KeyError(test_id)
        return self.tests[test_id]

    # -- files
    def page_png(self, test_id: str, index: int, dpi: int = 110) -> Path:
        """Render page `index` (0-based within the test) once and cache it."""
        import pymupdf

        t = self.get(test_id)
        if not 0 <= index < len(t.pages):
            raise IndexError(index)
        out = self.var_dir / "cache" / "pages" / f"{test_id}-{index}-{dpi}.png"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            pymupdf.open(t.pdf)[t.pages[index] - 1].get_pixmap(dpi=dpi).save(out)
        return out

    def audio_file(self, test_id: str, part: int) -> Path:
        t = self.get(test_id)
        if part not in t.audio:
            raise KeyError(part)
        return Path(t.audio[part])

    # -- marking
    def mark(self, test_id: str, answers: dict[int, str], learner_id: str = "default",
             seconds: int | None = None, explain: bool = True) -> dict:
        t = self.get(test_id)
        keys = keys_for(self.keys, t.book, t.test, t.skill)
        if not keys:
            raise KeyError(f"no answer keys for {test_id}")
        m = mark(keys, answers)
        by_q = {k.question: k for k in keys}
        rows = []
        for r in m.results:
            row = asdict(r)
            if explain and r.correct is False and r.question in by_q:
                k = by_q[r.question]
                hit = find_explanation(self.explanations, t.book, k.test, t.skill, r.question, k.answers or k.all_of or [])
                if hit:
                    row["explanation"] = {"section": hit[0].section_id, "pages": hit[0].pages, "text": hit[1]}
            rows.append(row)
        result = {"result_id": uuid.uuid4().hex[:12], "test_id": test_id, "book": t.book, "test": t.test,
                  "skill": t.skill, "raw": m.raw, "keyed": m.keyed, "band": m.band, "seconds": seconds,
                  "at": now(), "results": rows}
        self._record(learner_id, result)
        return result

    def _record(self, learner_id: str, result: dict) -> None:
        directory = self.var_dir / "learners"
        p = load_learner(learner_id, directory)
        node = f"skill.{result['skill']}"
        m = p.mastery.setdefault(node, Mastery())
        value = result["band"] / 9
        m.value = round(value if m.attempts == 0 else 0.7 * m.value + 0.3 * value, 3)
        m.attempts += 1
        m.updated = now()
        p.history.append({"session_id": result["result_id"], "task": result["skill"], "test_id": result["test_id"],
                          "band": result["band"], "overall": result["band"],
                          "raw": f"{result['raw']}/{result['keyed']}", "at": result["at"]})
        save_learner(p, directory)
        out = self.var_dir / "sessions" / f"{result['result_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


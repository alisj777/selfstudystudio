"""Mark Listening and Reading answers against the answer keys and convert raw scores to bands.

Answer notation in the keys:
  "(the/a) pianist/piano player"  words in brackets are optional; "/" separates accepted answers
  "drink(s)"                      a bracketed suffix is optional
  group [15, 16]                  answers may be given in either order
  all_of ["coal", "firewood"]     every item is needed for the one mark, in any order
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..importers.answer_keys import AnswerKey

# Raw score (out of 40) -> band. Indicative conversion tables published by IELTS; the exact
# cut-offs vary a little from test to test, so the band is an estimate.
_LISTENING = [(39, 9.0), (37, 8.5), (35, 8.0), (32, 7.5), (30, 7.0), (26, 6.5), (23, 6.0), (18, 5.5), (16, 5.0),
              (13, 4.5), (10, 4.0), (8, 3.5), (6, 3.0), (4, 2.5)]
_ACADEMIC_READING = [(39, 9.0), (37, 8.5), (35, 8.0), (33, 7.5), (30, 7.0), (27, 6.5), (23, 6.0), (19, 5.5),
                     (15, 5.0), (13, 4.5), (10, 4.0), (8, 3.5), (6, 3.0), (4, 2.5)]
_GENERAL_READING = [(40, 9.0), (39, 8.5), (37, 8.0), (36, 7.5), (34, 7.0), (32, 6.5), (30, 6.0), (27, 5.5),
                    (23, 5.0), (19, 4.5), (15, 4.0), (12, 3.5), (9, 3.0), (6, 2.5)]


def raw_to_band(raw: int, skill: str, module: str = "academic") -> float:
    table = _LISTENING if skill == "listening" else _GENERAL_READING if module == "general" else _ACADEMIC_READING
    return next((band for cut, band in table if raw >= cut), 0.0 if raw == 0 else 2.0)


# ---------------------------------------------------------------- matching

_TFNG = {"t": "true", "f": "false", "ng": "not given", "y": "yes", "n": "no", "notgiven": "not given"}


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("–", "-")
    text = re.sub(r"[-‐]", " ", text)
    text = re.sub(r"[^\w%.'& ]+", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)  # keep decimal points ("10.50")
    text = " ".join(text.split())
    return _TFNG.get(text.replace(" ", ""), text) if len(text) <= 9 else text


def _split_top(text: str, sep: str = "/") -> list[str]:
    """Split on `sep` outside brackets."""
    out, depth, cur = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return [x.strip() for x in out + [cur] if x.strip()]


def _pattern(answer: str) -> re.Pattern:
    """'(the/a) pianist' -> optional words; 'drink(s)' -> optional suffix."""
    parts, pos = [], 0
    for m in re.finditer(r"\(([^()]*)\)", answer):
        parts.append(re.escape(_norm(answer[pos:m.start()])).replace(r"\ ", r"\s*"))
        inner = "|".join(re.escape(_norm(x)) for x in m.group(1).split("/") if _norm(x))
        glued = m.start() > 0 and answer[m.start() - 1].isalnum()
        parts.append(f"(?:{inner})?" if glued else rf"(?:\s*(?:{inner})\s*)?")
        pos = m.end()
    parts.append(re.escape(_norm(answer[pos:])).replace(r"\ ", r"\s*"))
    body = "".join(parts)
    return re.compile(rf"^\s*{body}\s*$")


def answer_matches(given: str, key: str) -> bool:
    g = _norm(given)
    if not g:
        return False
    for alt in _split_top(key):
        if _norm(alt) == g or _pattern(alt).match(g):
            return True
    return False


# ---------------------------------------------------------------- marking


@dataclass
class QuestionResult:
    question: int
    given: str
    expected: list[str]
    correct: bool | None  # None: no answer key for this question
    note: str = ""


@dataclass
class Marking:
    book: int
    test: str
    skill: str
    module: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def raw(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def keyed(self) -> int:
        return sum(1 for r in self.results if r.correct is not None)

    @property
    def band(self) -> float:
        """Band from the raw score; questions without a key are scaled out of the 40."""
        raw = self.raw if self.keyed in (0, 40) else round(self.raw * 40 / self.keyed)
        return raw_to_band(raw, self.skill, self.module)


def _items(text: str) -> list[str]:
    return [x for x in re.split(r"\s*(?:,|/|;|&|\band\b)\s*", text) if x.strip()]


def mark(keys: list[AnswerKey], answers: dict[int, str], n_questions: int = 40) -> Marking:
    """Mark `answers` ({question: text}) against the keys of one test and skill."""
    if not keys:
        raise ValueError("no answer keys")
    k0 = keys[0]
    module = "general" if k0.test in ("A", "B") else "academic"
    marking = Marking(k0.book, k0.test, k0.skill, module)
    by_q = {k.question: k for k in keys}
    done: set[int] = set()
    for q in range(1, n_questions + 1):
        if q in done:
            continue
        key = by_q.get(q)
        given = answers.get(q, "").strip()
        if key is None:
            marking.results.append(QuestionResult(q, given, [], None, "no answer key"))
            continue
        if key.all_of:
            items = _items(given)
            ok = len(items) >= len(key.all_of) and all(
                any(answer_matches(i, want) for i in items) for want in key.all_of)
            marking.results.append(QuestionResult(q, given, [" + ".join(key.all_of)], ok, "both required"))
            continue
        if key.group and len(key.group) > 1:
            qs = [x for x in key.group if x in by_q]
            pool = [alt for x in qs for alt in [by_q[x].answers]]
            left = list(pool)
            for x in qs:
                g = answers.get(x, "").strip()
                hit = next((alts for alts in left if any(answer_matches(g, a) for a in alts)), None)
                if hit:
                    left.remove(hit)
                marking.results.append(QuestionResult(x, g, ["/".join(a) for a in pool], hit is not None,
                                                      "either order"))
                done.add(x)
            continue
        ok = any(answer_matches(given, a) for a in key.answers)
        marking.results.append(QuestionResult(q, given, key.answers, ok))
    marking.results.sort(key=lambda r: r.question)
    return marking


def parse_answer_text(text: str) -> dict[int, str]:
    """Read answers as "1 litter" / "1. litter" / "1) A" lines, or one answer per line in order."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numbered = [re.match(r"^(\d{1,2})\s*[.):=\-]?\s+(.*)$", ln) for ln in lines]
    if lines and all(numbered):
        return {int(m.group(1)): m.group(2).strip() for m in numbered if m}
    if len(lines) == 1 and "," in lines[0]:
        lines = [x.strip() for x in lines[0].split(",")]
    return {i: ans for i, ans in enumerate(lines, start=1)}

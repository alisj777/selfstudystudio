"""Chinese explanation books (解析/精讲) for the Cambridge tests: OCR, then cut into sections.

The books are scanned, so every page is OCRed (see ocr.py) and cached. Each test is laid
out in the same order - Listening Section/Part 1-4, Reading Passage 1-3, Writing Task 1-2,
Speaking Part 1-3 - so the pages are cut with a small state machine over those headings.
Newer books also print a running header ("Test 1 Reading"), which overrides the guess.

The result is one record per (book, test, skill, part). Explanations are not indexed for
semantic search (the embedder is English-only); the marker looks them up directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

LICENSE = "Chinese explanation book for Cambridge IELTS; user's own copy, internal use only"
SKILLS = ("listening", "reading", "writing", "speaking")


class ExplanationSection(BaseModel):
    section_id: str  # cam17-t1-reading-p2
    book: int
    test: str
    skill: str
    part: int | None = None  # listening part, reading passage, writing task, speaking part
    pages: list[int]  # 1-based pages of the explanation PDF
    questions: list[list[int]] = []  # "Questions 14~19" ranges seen in the section
    text: str
    source: str
    license: str = LICENSE


def explanation_books(name: str) -> list[int]:
    """Book numbers covered by one explanation PDF, from its file name ("4-5-6.pdf", "剑19解析")."""
    stem = Path(name).stem
    if m := re.fullmatch(r"(\d{1,2})(?:-(\d{1,2}))*", stem):
        return [int(x) for x in stem.split("-")]
    nums = [int(x) for x in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", stem) if 4 <= int(x) <= 21]
    return nums[:1]


# ---------------------------------------------------------------- headings

_N = r"([1-4IlL!|7])"  # OCR reads 1 as I, l, !, | and sometimes 7
RE_HEADER = re.compile(r"^\s*Test\s*([1-8])\s*(Listening|Reading|Writing|Speaking)\b", re.I)
RE_LISTEN = re.compile(rf"^\s*(?:L\S*ste\S*\s+)?(?:Section|Part)\s*{_N}(?=\s*[>»›’'”\s]|$)", re.I)
RE_SECTION = re.compile(rf"^\s*Section\s*{_N}(?=\s*[>»›’'”\s]|$)", re.I)
RE_PASSAGE = re.compile(rf"^\s*Reading\s*Passage\s*{_N}(?=\s*[>»›’'”\s]|$)", re.I)
RE_TASK = re.compile(rf"^\s*(?:Writing\s*)?Task\s*{_N}(?=\s*[>»›’'”\s]|$)", re.I)
RE_READING = re.compile(  # "Reading 和" (a garbled "Reading Passage 1"), or headings only reading sections have
    r"^\s*Reading\b(?!\s*Passage\s*[23])\W*\S?\s*$|^\W{0,3}\S?\s*(难度系数|解题顺序|篇章结构|体裁)")
RE_SCENE = re.compile(r"^\W{0,3}\S?\s*场景介绍\s*$")  # scene introduction: only in Listening
ORDER = {"listening": 0, "reading": 1, "writing": 2, "speaking": 3}
RE_QRANGE = re.compile(r"Questions?\s*(\d{1,2})\s*[~～\-–—一]\s*(\d{1,2})")
RE_CONTENTS = re.compile(r"目\s*录|CONTENTS", re.I)


def _n(tok: str, top: int) -> int:
    v = {"I": 1, "l": 1, "L": 1, "!": 1, "|": 1}.get(tok)
    if v is None:
        v = int(tok)
        if v > top:  # "Part 7"
            v = 1
    return v


@dataclass
class _Sec:
    test: int
    skill: str
    part: int | None
    pages: list[int] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


def segment(pages: dict[int, str]) -> list[_Sec]:
    """Cut OCRed pages ({0-based page: text}) into sections, in reading order."""
    secs: list[_Sec] = []
    cur: _Sec | None = None
    test, skill, part = 0, "", None

    def go(sk: str, p: int | None) -> None:
        """Move to skill `sk`; going back in the L-R-W-S order means the next test has begun."""
        t = test or 1
        if skill and ORDER[sk] < ORDER[skill]:
            t = test + 1
        start(t, sk, p)

    def start(t: int, sk: str, p: int | None) -> None:
        nonlocal cur, test, skill, part
        test, skill, part = t, sk, p
        if cur and (cur.test, cur.skill, cur.part) == (t, sk, p):
            return
        cur = _Sec(t, sk, p)
        secs.append(cur)

    for pno in sorted(pages):
        text = pages[pno]
        head = "\n".join(text.splitlines()[:6])
        lines = [x.strip() for x in text.splitlines()]
        n_heads = sum(1 for x in lines if RE_PASSAGE.match(x) or RE_TASK.match(x) or RE_LISTEN.match(x))
        if RE_CONTENTS.search(head) or len(set(re.findall(r"Test\s*(\d)", text))) >= 2 or n_heads >= 5:
            continue  # table of contents
        for line in text.splitlines():
            s = line.strip()
            if not s:
                if cur:
                    cur.lines.append("")
                continue
            if m := RE_HEADER.match(s):
                t, sk = int(m.group(1)), m.group(2).lower()
                if (t, sk) != (test, skill):
                    start(t, sk, part if sk == skill else None)
                rest = s[m.end():].strip()
                if not rest:
                    continue
                s = rest
            if m := RE_PASSAGE.match(s):
                go("reading", _n(m.group(1), 3))
            elif skill == "listening" and part == 4 and RE_READING.match(s):
                go("reading", 1)
            elif m := RE_TASK.match(s):
                go("writing", _n(m.group(1), 2))
            elif m := RE_SECTION.match(s):
                go("listening", _n(m.group(1), 4))  # "Section n" is only used for Listening
            elif RE_SCENE.match(s) and skill not in ("", "listening"):
                go("listening", 1)
            elif m := RE_LISTEN.match(s):  # "Part n": Listening (new books) or Speaking
                n = _n(m.group(1), 4)
                if skill == "listening" or (skill in ("", "speaking") and n == 1):
                    go("listening", n)
                elif skill in ("writing", "speaking") and n <= 3:
                    go("speaking", n)
            if cur and test:
                cur.lines.append(s)
                if not cur.pages or cur.pages[-1] != pno + 1:
                    cur.pages.append(pno + 1)
    return [s for s in secs if s.test and any(x.strip() for x in s.lines)]


def to_records(secs: list[_Sec], books: list[int], source: str) -> list[ExplanationSection]:
    """Number the sections by book and test. A file covering several books restarts at Test 1
    for each; book 12 numbers its Academic tests 5-8."""
    out: list[ExplanationSection] = []
    # Number the segmenter's running test count per book. A test with no Reading Passage is
    # General Training (its Reading is set in "Section 1-3", which looked like Listening).
    # Each book has up to four Academic tests followed by its General Training ones, so an
    # Academic test after a GT test - or a fifth Academic test - starts the next book.
    # A "test" with fewer than three sections is front matter.
    sizes: dict[int, int] = {}
    for s in secs:
        sizes[s.test] = sizes.get(s.test, 0) + 1
    has_passage = {s.test for s in secs if s.skill == "reading"}
    label: dict[int, tuple[int, str]] = {}
    book_i, n_acad, n_gt = 0, 0, 0
    for t in sorted(sizes):
        if sizes[t] < 3:
            continue
        academic = t in has_passage
        if academic and (n_gt or n_acad == 4) and book_i + 1 < len(books):
            book_i, n_acad, n_gt = book_i + 1, 0, 0
        book = books[book_i]
        if academic:
            n_acad += 1
            label[t] = (book, str(n_acad + 4 if book == 12 else n_acad))
        else:
            label[t] = (book, "AB"[n_gt % 2])
            n_gt += 1
    for s in secs:
        if s.test not in label:
            continue
        book, test = label[s.test]
        if test in ("A", "B") and s.skill == "listening":
            s.skill = "reading"
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(s.lines)).strip()
        part = f"-p{s.part}" if s.part else ""
        out.append(ExplanationSection(
            section_id=f"cam{book}-t{test}-{s.skill}{part}",
            book=book, test=str(test), skill=s.skill, part=s.part, pages=s.pages,
            questions=[[int(a), int(b)] for a, b in RE_QRANGE.findall(text) if int(a) <= int(b) <= 40],
            text=text, source=source,
        ))
    # Merge repeats of the same id (a heading misread mid-section starts it again).
    merged: dict[str, ExplanationSection] = {}
    for r in out:
        if r.section_id in merged:
            m = merged[r.section_id]
            m.text += "\n\n" + r.text
            m.pages = sorted(set(m.pages + r.pages))
            m.questions += r.questions
        else:
            merged[r.section_id] = r
    return list(merged.values())


def import_explanations(
    root: Path, cache_dir: Path, books: list[int] | None = None, workers: int = 6, progress=None
) -> list[ExplanationSection]:
    """OCR (cached) and segment the explanation PDFs under `root` (folders whose name has 解析)."""
    from .ocr import ocr_pdf

    records: list[ExplanationSection] = []
    for pdf in sorted(root.rglob("*.pdf")):
        if "解析" not in pdf.parent.name:
            continue
        covered = explanation_books(pdf.name)
        if not covered or (books and not set(covered) & set(books)):
            continue
        pages = ocr_pdf(pdf, cache_dir / f"{pdf.stem}.json", workers=workers,
                        progress=(lambda d, n, name=pdf.name: progress(name, d, n)) if progress else None)
        records += to_records(segment(pages), covered, f"{pdf.parent.name}/{pdf.name}")
    return [r for r in records if not books or r.book in books]


# ---------------------------------------------------------------- lookup


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def find_explanation(
    sections: list[ExplanationSection], book: int, test: str, skill: str, question: int,
    answers: list[str], max_chars: int = 500,
) -> tuple[ExplanationSection, str] | None:
    """The paragraph that explains `question`: it names the question number and/or the answer."""
    cands = [s for s in sections if s.book == book and s.test == test and s.skill == skill]
    if skill == "listening":
        part = (question - 1) // 10 + 1
        cands = [s for s in cands if s.part == part] or cands
    elif skill == "reading":
        ranged = [s for s in cands if any(a <= question <= b for a, b in s.questions)]
        cands = ranged or cands
    best: tuple[float, ExplanationSection, str] | None = None
    q = str(question)
    for s in cands:
        for para in _paragraphs(s.text):
            if "答案速查" in para or len(re.findall(r"(?<!\d)\d{1,2}\s*[.、]\s*\w+", para[:200])) >= 4:
                continue  # the answer list at the top of a section explains nothing
            low = para.lower()
            score = 0.0
            if re.match(rf"^\W{{0,3}}{q}\s*[.、)）]|^\(\s*{q}\s*\)", para):
                score += 3
            for a in answers:
                a = re.sub(r"\(.*?\)", "", a).strip().lower()
                if len(a) > 2 and a in low:
                    score += 2
                elif len(a) <= 2 and re.search(rf"答案[为是应]*\s*{re.escape(a)}\b|{re.escape(a)}\s*选项", para, re.I):
                    score += 2
            if "答案" in para:
                score += 1
            if not re.search(r"[一-鿿]", para):
                score -= 2  # English-only paragraphs are script quotes, not explanations
            if score >= 3 and (best is None or score > best[0]):
                best = (score, s, para)
    if best is None:
        return None
    _, sec, para = best
    return sec, para if len(para) <= max_chars else para[:max_chars] + "…"

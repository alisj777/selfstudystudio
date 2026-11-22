"""Listening and Reading answer keys from the Cambridge books, plus a grader.

The key pages are set in two or three columns, and the old books' text layer is itself
OCR, so plain text extraction interleaves the columns ("12 local craftsmen 35 dangerous").
The page is therefore rebuilt from word positions: words are grouped into rows, rows are
cut into column segments, and segments are read column by column. Each question number
is paired with the words to its right; "IN EITHER ORDER" groups take the unnumbered
lines that follow them.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- model


class AnswerKey(BaseModel):
    """One question's accepted answers. `group` lists questions whose answers may come in any order."""

    key_id: str  # cam17-t1-listening-q15
    book: int
    test: str  # "1".."8", "A", "B"
    module: str  # academic | general | both (listening)
    skill: str  # listening | reading
    question: int
    answers: list[str]  # alternatives; "(the) pianist/piano player" style kept verbatim
    group: list[int] | None = None  # questions whose answers may be given in either order
    all_of: list[str] | None = None  # "BOTH REQUIRED FOR ONE MARK": every item needed, any order
    page: int | None = None


# ---------------------------------------------------------------- page layout

_DIGITS = re.compile(r"^\d{1,2}$")
_HEADING_WORDS = {"TEST", "PASSAGE", "SECTION", "PART", "QUESTIONS", "QUESTION"}
_OCR_NUM = {"I": "1", "l": "1", "O": "0", "o": "0", "S": "5"}


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    words: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.words)


def page_segments(words: list[tuple], col_gap: float = 40.0) -> list[Segment]:
    """Group PyMuPDF words (x0, y0, x1, y1, text, ...) into rows, then cut rows at wide gaps."""
    ws = sorted((w for w in words if w[4].strip()), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    for w in ws:
        yc, h = (w[1] + w[3]) / 2, w[3] - w[1]
        if rows:
            r = rows[-1]
            ryc = sum((x[1] + x[3]) / 2 for x in r) / len(r)
            if abs(yc - ryc) < max(2.5, 0.45 * h):
                r.append(w)
                continue
        rows.append([w])
    segs: list[Segment] = []
    for r in rows:
        r.sort(key=lambda w: w[0])
        cur: Segment | None = None
        for w in r:
            gap = w[0] - cur.x1 if cur else 1e9
            # a question number after a visible gap starts a new segment ("FOR ONE MARK   31 cities")
            new_q = (gap > 8 and _DIGITS.match(w[4]) and not _DIGITS.match(cur.words[-1])
                     and cur.words[-1].upper().strip(",") not in _HEADING_WORDS) if cur else False
            if cur is None or gap > col_gap or new_q:
                cur = Segment(w[0], w[1], w[2], [w[4]])
                segs.append(cur)
            elif _DIGITS.match(cur.words[-1]) and _DIGITS.match(w[4]) and gap < 3 and len(cur.words[-1] + w[4]) <= 2:
                cur.words[-1] += w[4]  # a two-digit number extracted as two glyphs
                cur.x1 = w[2]
            else:
                cur.words.append(w[4])
                cur.x1 = w[2]
    return segs


def _columns(segs: list[Segment], min_gap: float = 60.0) -> list[list[Segment]]:
    """Split segments into columns by their left edge; each column is read top to bottom."""
    if not segs:
        return []
    starts = sorted({round(s.x0) for s in segs})
    bounds = [starts[0]]
    for a, b in zip(starts, starts[1:]):
        if b - a > min_gap:
            bounds.append(b)
    cols: list[list[Segment]] = [[] for _ in bounds]
    for s in segs:
        i = max(j for j, b in enumerate(bounds) if s.x0 >= b - 1)
        cols[i].append(s)
    return [sorted(c, key=lambda s: s.y0) for c in cols]


# ---------------------------------------------------------------- parsing

RE_TEST = re.compile(r"^TEST\s*([1-8IVl]{1,3}|[AB])\b", re.I)
RE_GT_TEST = re.compile(r"GENERAL\s*TRA\S*\s*(?:READING\s*)?(?:AND WRITING\s*)?TEST\s*([AB])", re.I)
RE_LISTENING = re.compile(r"^LISTENING\b")
RE_READING = re.compile(r"^(ACADEMIC |GENERAL TRAINING )?READING\b", re.I)
RE_ORDER = re.compile(r"^(\d{1,2})\s*(?:&|-|–|—|and)?\s*(\d{1,2})?\s*,?\s*IN\s*(?:EITHER|ANY)\s*ORDER", re.I)
RE_ANSWER = re.compile(r"^([1-9IlO]\d?|\d{2})[.,°'’]?\s+(?=\S)(.+)$")
RE_STOP = re.compile(r"^(if you score|you are|you may|each question|section|part|reading passage|questions?|"
                     r"\d{1,2}\s*[-–]\s*\d{1,2}$|answer key|listening and reading|for one mark|both required)", re.I)
_HEADING = re.compile(r"^(TEST\s*\S+|LISTENING|(ACADEMIC |GENERAL TRAINING )?READING|General Training.*|"
                      r"Listening and Reading|Answer Key)", re.I)
_LETTER = re.compile(r"^[A-Za-z]$|^(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi)$|^(TRUE|FALSE|YES|NO|NOT ?GIVEN)$", re.I)


def _infer_numbers(col: list[Segment]) -> dict[int, int]:
    """Number unnumbered one-line answers from the numbered rows around them in a column."""
    rows = [(s.y0, q) for s in col if (q := _qnum(s.words[0])) and len(s.words) > 1]
    steps = sorted((y2 - y1) / (q2 - q1) for (y1, q1), (y2, q2) in zip(rows, rows[1:]) if 0 < q2 - q1 <= 2)
    if len(steps) < 3:
        return {}
    pitch = steps[len(steps) // 2]
    out = {}
    for s in col:
        if _qnum(s.words[0]) or RE_STOP.match(s.text) or RE_ORDER.match(s.text):
            continue
        y_ref, q_ref = min(rows, key=lambda r: abs(r[0] - s.y0))
        k = (s.y0 - y_ref) / pitch
        if abs(k) <= 3 and abs(k - round(k)) < 0.3 and 1 <= q_ref + round(k) <= 40 and round(k):
            out[id(s)] = q_ref + round(k)
    return out


def _qnum(token: str) -> int | None:
    t = "".join(_OCR_NUM.get(c, c) for c in token)
    return int(t) if t.isdigit() and 1 <= int(t) <= 40 else None


@dataclass
class _Raw:
    test: str
    skill: str
    module: str
    q: int | None  # None: an unnumbered line inside an either-order group
    text: str
    page: int
    group: tuple[int, int] | None = None
    both: bool = False  # all items of the group answer one question
    inferred: bool = False  # number lost in OCR, inferred from the row position


def _test_label(token: str) -> str:
    t = token.upper()
    return {"I": "1", "II": "2", "III": "3", "IV": "4", "L": "1"}.get(t, t)


def parse_key_pages(pages: list[tuple[int, list[tuple]]], base: int = 1) -> list[_Raw]:
    """`pages` = [(page_no, pymupdf words)]. Returns answers in reading order with their context."""
    raws: list[_Raw] = []
    test, skill, module = "", "", "academic"
    group: tuple[int, int] | None = None
    group_left = 0
    both = False
    auto = 0  # Academic tests counted by their LISTENING heading when "TEST n" is unreadable
    explicit = False
    # A "Test 4" printed on most pages is the book's running header, not a key heading.
    per_page = [{_test_label(m.group(1)) for s in page_segments(w) if (m := RE_TEST.match(s.text.strip()))}
                for _, w in pages]
    running = {t for t in set().union(*per_page) if sum(t in x for x in per_page) > 2} if per_page else set()
    for pno, words in pages:
        segs = page_segments(words)
        # Headings above the first numbered row span all columns: read them first.
        numbered = [s.y0 for s in segs if _qnum(s.words[0]) and len(s.words) > 1]
        top = min(numbered) if numbered else 1e9
        is_head = [s.y0 < top - 2 and bool(_HEADING.match(s.text)) for s in segs]
        head = sorted((s for s, h in zip(segs, is_head) if h), key=lambda s: (s.y0, s.x0))
        cols = _columns([s for s, h in zip(segs, is_head) if not h])
        body = [s for col in cols for s in col]
        guess = {id(s): q for col in cols for s, q in _infer_numbers(col).items()}
        for s in head + body:
            text = s.text.strip()
            if m := RE_GT_TEST.search(text):
                test, module, explicit = m.group(1).upper(), "general", True
                continue
            if (m := RE_TEST.match(text)) and _test_label(m.group(1)) in running:
                continue
            if m := RE_TEST.match(text):
                test = _test_label(m.group(1))
                module = "general" if test in "AB" else "academic"
                explicit = True
                if test.isdigit():
                    auto = int(test) - 1
                continue
            if RE_LISTENING.match(text):
                if not explicit:
                    auto += 1
                    test, module = str(auto), "academic"
                elif test.isdigit():
                    auto = int(test)
                skill, explicit = "listening", False
                continue
            if m := RE_READING.match(text):
                skill = "reading"
                if (m.group(1) or "").upper().startswith("GENERAL") and not explicit:
                    test, module = ("B" if test == "A" else "A"), "general"
                explicit = False
                continue
            if not test or not skill:
                continue
            if m := RE_ORDER.match(text):
                a = int(m.group(1))
                rest = text[m.end():].strip(" ,")
                both = not m.group(2) and bool(re.search(r"BOTH|REQUIRED", rest, re.I))
                b = a if both else int(m.group(2)) if m.group(2) else a + 1
                if a <= b <= a + 3:
                    group = (a, b)
                    group_left = 2 if both else b - a + 1
                rest = re.sub(r"(?i)BOTH REQUIRED( FOR ONE MARK)?", "", rest)
                for tok in rest.split():  # "15&16 IN EITHER ORDER A D"
                    if group_left and _LETTER.match(tok):
                        raws.append(_Raw(test, skill, module, None, tok, pno, group, both))
                        group_left -= 1
                continue
            if RE_STOP.match(text):
                continue
            if m := RE_ANSWER.match(text):
                q = _qnum(m.group(1))
                if q:
                    group_left = 0
                    raws.append(_Raw(test, skill, module, q, m.group(2).strip(), pno))
                    continue
            if group_left and len(text.split()) <= 4:
                raws.append(_Raw(test, skill, module, None, text, pno, group, both))
                group_left -= 1
            elif id(s) in guess and len(text.split()) <= 4:
                raws.append(_Raw(test, skill, module, guess[id(s)], text, pno, inferred=True))
    if base > 1:
        for r in raws:
            if r.test.isdigit() and int(r.test) < base:
                r.test = str(int(r.test) + base - 1)
    return raws


def _clean(ans: str) -> list[str]:
    ans = re.sub(r"\s+", " ", ans).strip(" .,;")
    ans = re.sub(r"^[^\w(]+", "", ans)  # OCR debris before the answer: "~~ market", "= low"
    ans = re.sub(r"^_\w{0,2}(?=[a-z]{3})", "", ans)  # "_soprivate"
    if re.search(r"IN\s*(EITHER|ANY)\s*ORDER|BOTH REQUIRED", ans, re.I):
        return []  # a heading merged into the row; better no key than a wrong one
    ans = re.sub(r"^0\s+(?=[a-z])", "", ans)  # "I 0 climb the tower" -> question 10
    ans = re.sub(r"(\s+[I|·•])+$", "", ans)  # a "/" read as "I" at the end of a wrapped line
    if m := re.fullmatch(r"([A-Za-z])\s+[A-Z]", ans):  # "c E": the next column's letter joined on
        ans = m.group(1)
    if m := re.fullmatch(r"\d?([A-Za-z])\1?", ans, re.I) or re.fullmatch(r"\d([A-K])", ans):
        ans = m.group(1).upper()  # "c", "Cc", "8B", "4H" -> the letter
    ans = re.sub(r"(?i)^NOT\s?GIVEN$", "NOT GIVEN", ans)
    parts = re.split(r"\s+/\s+|\s+I\s+(?=[a-z(])|\s+OR\s+", ans)
    return [p.strip() for p in parts if p.strip()]


def assemble(book: int, raws: list[_Raw]) -> list[AnswerKey]:
    """Turn raw rows into one key per question, repairing misread numbers from their neighbours."""
    by_set: dict[tuple[str, str], list[_Raw]] = defaultdict(list)
    for r in raws:
        by_set[(r.test, r.skill)].append(r)
    keys: list[AnswerKey] = []
    for (test, skill), rows in by_set.items():
        numbered = [r for r in rows if r.q and not r.inferred]
        # "1 Bristol" between 6 and 8 is 7: fix a number that repeats when its slot is free.
        seen = defaultdict(int)
        for r in numbered:
            seen[r.q] += 1
        for i, r in enumerate(numbered):
            prev = numbered[i - 1].q if i else None
            nxt = numbered[i + 1].q if i + 1 < len(numbered) else None
            if prev and nxt and nxt == prev + 2 and r.q != prev + 1 and not seen.get(prev + 1):
                seen[r.q] -= 1
                r.q = prev + 1
                seen[r.q] += 1
        out: dict[int, AnswerKey] = {}
        groups = defaultdict(list)
        for r in rows:
            if r.q is None and r.group:
                groups[(r.group, r.both)].append(r)
        for ((a, b), both), members in groups.items():
            qs = list(range(a, b + 1))
            if both:
                out[a] = AnswerKey(
                    key_id=f"cam{book}-t{test.lower()}-{skill}-q{a}", book=book, test=test,
                    module="both" if skill == "listening" else members[0].module, skill=skill, question=a,
                    answers=[], all_of=[_clean(r.text)[0] for r in members if _clean(r.text)], page=members[0].page)
                continue
            for q, r in zip(qs, members):
                out.setdefault(q, AnswerKey(
                    key_id=f"cam{book}-t{test.lower()}-{skill}-q{q}", book=book, test=test,
                    module="both" if skill == "listening" else r.module, skill=skill, question=q,
                    answers=_clean(r.text), group=qs, page=r.page))
        for r in numbered + [r for r in rows if r.inferred]:
            if r.q in out or not _clean(r.text):
                continue
            out[r.q] = AnswerKey(
                key_id=f"cam{book}-t{test.lower()}-{skill}-q{r.q}", book=book, test=test,
                module="both" if skill == "listening" else r.module, skill=skill, question=r.q,
                answers=_clean(r.text), page=r.page)
        keys += [out[q] for q in sorted(out)]
    return keys


def _readable(words: list[tuple]) -> bool:
    texts = [w[4] for w in words]
    ok = sum(1 for t in texts if re.fullmatch(r"[A-Za-z0-9().,/&'’\-–%]+", t))
    return len(texts) > 30 and ok / len(texts) > 0.8


def ocr_words(page, dpi: int = 300, psm: int = 6) -> list[tuple]:
    """Word boxes for a scanned page from tesseract's TSV output, in PDF points."""
    import subprocess
    import tempfile

    import pymupdf

    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "p.png"
        page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY).save(img)
        tsv = subprocess.run(["tesseract", str(img), "-", "-l", "eng", "--psm", str(psm), "tsv"],
                             capture_output=True, text=True).stdout
    scale = 72 / dpi
    out = []
    for line in tsv.splitlines()[1:]:
        f = line.split("\t")
        if len(f) == 12 and f[11].strip() and float(f[10]) > 10:
            x, y, w, h = (int(v) * scale for v in f[6:10])
            out.append((x, y, x + w, y + h, f[11]))
    return out


def extract_keys(pdf: Path, pages: list[int], book: int, base: int = 1, ocr: bool | None = None) -> list[AnswerKey]:
    """Read the answer-key pages (1-based numbers, as in `ParsedBook.key_text`) of one book.

    Pages whose text layer is unreadable are OCRed with tesseract (`ocr=None` decides per
    page; True forces OCR, False never OCRs). When OCR runs, the better of the two readings
    is kept for each test and skill.
    """
    import pymupdf

    doc = pymupdf.open(pdf)
    text_words = [(p, doc[p - 1].get_text("words")) for p in pages]
    keys = assemble(book, parse_key_pages(text_words, base))
    cov = coverage(keys)
    need = ocr if ocr is not None else (
        not cov or any(not _readable(w) for _, w in text_words) or min(cov.values()) < 40
    )
    if not need or ocr is False:
        return keys
    from .ocr import tesseract_available

    if tesseract_available("eng"):
        log.warning("%s: answer keys incomplete and tesseract is unavailable; keeping the text layer only", pdf.name)
        return keys
    ocr_keys = assemble(book, parse_key_pages([(p, ocr_words(doc[p - 1])) for p in pages], base))
    # Per test and skill, start from the reading that found more questions and fill its gaps
    # from the other one.
    merged: dict[tuple[str, str], dict[int, AnswerKey]] = defaultdict(dict)
    cov_t, cov_o = coverage(keys), coverage(ocr_keys)
    for ts in set(cov_t) | set(cov_o):
        first, second = (keys, ocr_keys) if cov_t.get(ts, 0) >= cov_o.get(ts, 0) else (ocr_keys, keys)
        for source in (first, second):
            for k in source:
                if (k.test, k.skill) == ts:
                    merged[ts].setdefault(k.question, k)
    return [merged[ts][q] for ts in sorted(merged) for q in sorted(merged[ts])]


def coverage(keys: list[AnswerKey]) -> dict[tuple[str, str], int]:
    """Questions found per (test, skill); a complete set has 40."""
    out: dict[tuple[str, str], int] = defaultdict(int)
    for k in keys:
        out[(k.test, k.skill)] += 1
    return dict(out)

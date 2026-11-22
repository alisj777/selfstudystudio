"""Import Cambridge IELTS books (4-20) from the user's own PDFs and audio.

The books are © Cambridge University Press & Assessment, so everything is written to
git-ignored folders and kept for local, internal use:
  data/<exam>/corpus/private/cambridge.jsonl    writing and speaking prompts, reading passages,
                                                listening audioscripts, examiner model answers
  data/<exam>/scoring/private/cambridge.jsonl   candidate writing with the examiner's band and comment
  data/<exam>/answers/private/cambridge.jsonl   Listening and Reading answer keys (see answer_keys.py)

Only PDFs with a text layer are parsed. Scanned books (no text layer) are reported so
they can be OCR'd first. Every item carries `quality`, the share of recognisable English
words; low-quality items are kept on disk but left out of the indexes.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..rag.schema import CorpusChunk, ScoredSample
from .answer_keys import AnswerKey, extract_keys
from .ielts_org import _clean
from .textquality import Vocab

LICENSE = "© Cambridge University Press & Assessment; user's own copy, internal use only"
MIN_TEXT_CHARS_PER_PAGE = 200  # below this the PDF is treated as scanned
CLEAN = 0.9
MAX_ESSAY_WORDS = 600
MAX_COMMENT_WORDS = 450  # quality at or above which candidate writing is treated as reliable text

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "l": 1}


def _num(token: str) -> int | str:
    token = token.strip()
    if token.isdigit():
        return int(token)
    if token in ("A", "B"):
        return token
    return _ROMAN.get(token.upper() if token != "l" else token, 0)


# ---------------------------------------------------------------- headings
RE_TEST = re.compile(r"^test\s+([1-8]|I{1,3}|IV|[AB])$", re.I)
RE_TEST_UPPER = re.compile(r"^TEST\s+([1-8]|I{1,3}|IV|[AB])$")
RE_RP = re.compile(r"^READING PASSAGE\s*([123Il])$")
RE_WT = re.compile(r"^WRITING TASK\s*([12I])$")
RE_SECTION_START = re.compile(r"^(LISTENING|READING|WRITING)$")
RE_QUESTIONS = re.compile(r"^Q\s?uestions?\s*\d", re.I)  # a listening/reading question block
RE_PART = re.compile(r"^part\s*([123])$", re.I)
RE_SCRIPT_PART = re.compile(r"^(?:SECTION|PART)\s*([1-4])$")
RE_SCRIPTS = re.compile(r"^(audio ?scripts|tape ?scripts)$", re.I)
RE_KEYS = re.compile(r"answer keys?$", re.I)
RE_SAMPLES = re.compile(r"^(model and )?sample (writing )?answers( for writing( tasks)?)?$", re.I)
RE_END = re.compile(r"^(acknowledgements|sample answer sheets?)$", re.I)
RE_SAMPLE_HEAD = re.compile(r"TEST\s*([1-8]|I{1,3}|IV|[AB])\s*[,~.]?\s*WRITING TASK\s*([12I])", re.I)  # OCR: "TEST 1~"
RE_BAND = re.compile(r"achieved a Band\s*(\d(?:\.\d)?)\s*score\.?", re.I)
RE_COMMENT = re.compile(r"Here is the examiner['’]?s\s*comment\s*:?", re.I)
RE_MODEL = re.compile(r"This model has been prepared by an examiner as an example of a very good answer\.?", re.I)
RE_WORD_LIMIT = re.compile(r"W\S{0,2}ite\s*at\s*least\s*\d+\s*words\.?", re.I)  # OCR: "Writeatleast", "W「ite"
RE_GT = re.compile(r"^General Training:? Reading and Writing(?: Test ([AB]))?$", re.I)
RE_NOISE = re.compile(
    r"^(\d{1,3}|test\s+\S+|reading|writing|speaking|listening|audio ?scripts|tape ?scripts|"
    r"(model and )?sample (writing )?answers( for writing tasks)?|sample answ ?er|model answ ?er|"
    r"=====PAGE.*|[＿_\W]*)$",
    re.I,
)


def _norm(line: str) -> str:
    line = line.replace("，", ", ").replace("：", ": ").replace(" ", " ").replace(" ", " ")
    return " ".join(line.split())


# ---------------------------------------------------------------- PDF text
def book_number(path: Path) -> int | None:
    name = path.stem
    if any(k in name for k in ("精讲", "解析", "抢先")):
        return None
    m = re.search(r"【(\d+)】", name) or re.search(r"(?:剑桥雅思|剑雅|剑)\s*(\d+)", name)
    return int(m.group(1)) if m else None


def pdf_pages(path: Path, cache_dir: Path | None = None) -> list[str]:
    cache = cache_dir / f"{path.stem}.json" if cache_dir else None
    if cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    pages = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages


def has_text_layer(pages: list[str]) -> bool:
    return sum(len(p.strip()) for p in pages) / max(len(pages), 1) >= MIN_TEXT_CHARS_PER_PAGE


# ---------------------------------------------------------------- classification
def writing_type(task: int, text: str, module: str) -> str | None:
    t = text.lower()
    if task == 1:
        if module == "general":
            return None
        kinds = []
        if re.search(r"\bmaps?\b|\bplans?\b|\blayout\b", t):
            kinds.append("map")
        if re.search(r"\bprocess\b|\bdiagrams? (below )?(shows?|illustrates?) how\b|\bstages\b", t):
            kinds.append("process")
        if "table" in t:
            kinds.append("table")
        if "pie chart" in t:
            kinds.append("pie_chart")
        if re.search(r"bar (chart|graph)", t):
            kinds.append("bar_chart")
        if re.search(r"line graph|\bgraph\b", t) and "bar_chart" not in kinds:
            kinds.append("line_graph")
        if len(kinds) > 1:
            return "qt.writing.t1.mixed"
        return f"qt.writing.t1.{kinds[0]}" if kinds else None
    if re.search(r"discuss both (these )?views", t):
        return "qt.writing.t2.discussion"
    if re.search(r"advantages?.{0,40}disadvantages?|outweigh", t):
        return "qt.writing.t2.adv_disadv"
    if re.search(r"(problems?|causes?|reasons?).{0,120}(solutions?|solve|measures|tackle)", t, re.S):
        return "qt.writing.t2.problem_solution"
    if re.search(r"agree or disagree|to what extent", t):
        return "qt.writing.t2.opinion"
    prompt = RE_WORD_LIMIT.split(text)[0].split("Write about the following topic:")[-1]
    if prompt.count("?") >= 2:
        return "qt.writing.t2.two_part"
    return None


# ---------------------------------------------------------------- parsing
@dataclass
class _Item:
    kind: str  # reading | writing | speaking | transcript
    test: int | str
    n: int
    module: str
    page: int
    lines: list[str] = field(default_factory=list)


@dataclass
class ParsedBook:
    book: int
    items: list[_Item]
    sample_text: list[tuple[int, str]]  # (page, line) in the sample-answers section
    key_text: list[tuple[int, str]] = field(default_factory=list)  # (page, line) in the answer keys
    base: int = 1  # number of the first Academic test (5 in book 12)


def _test_module(t: int | str) -> str:
    return "general" if t in ("A", "B") else "academic"


def parse_book(pages: list[str], book: int) -> ParsedBook:
    items: list[_Item] = []
    sample_lines: list[tuple[int, str]] = []
    key_lines: list[tuple[int, str]] = []
    cur: _Item | None = None
    test: int | str = 0
    zone = "front"
    seen_wt2 = False
    # transcript numbering in the back matter
    script_test: int | str = 0
    explicit_test: int | str | None = None
    last_part = 9

    def close():
        nonlocal cur
        if cur and cur.lines:
            items.append(cur)
        cur = None

    # Academic tests are counted by their first reading passage, because running headers
    # ("Test 1") are often extracted after the headings they belong to. Book 12 numbers its
    # Academic tests 5-8, so the first test number is taken from those headers.
    numbered = Counter(
        t for page in pages for raw in page.splitlines()
        if (m := RE_TEST.match(_norm(raw))) and isinstance(t := _num(m.group(1)), int) and t
    )
    base = 5 if sum(numbered[t] for t in (5, 6, 7, 8)) > sum(numbered[t] for t in (1, 2, 3, 4)) else 1
    academic_count = 0

    for pno, page in enumerate(pages, start=1):
        for raw in page.splitlines():
            line = _norm(raw)
            if not line:
                continue
            if zone != "end" and seen_wt2 and RE_END.match(line):
                close()
                zone = "end"
                continue
            if seen_wt2 and (RE_SCRIPTS.match(line) or RE_KEYS.search(line) or RE_SAMPLES.match(line)):
                new = "scripts" if RE_SCRIPTS.match(line) else "keys" if RE_KEYS.search(line) else "samples"
                if new != zone:
                    close()
                    zone = new
                continue
            if zone == "front":
                if m := RE_TEST.match(line):
                    t = _num(m.group(1))
                    if t in ("A", "B") and t != test:  # General Training tests are lettered
                        close()
                        test = t
                    continue
                if m := RE_GT.match(line):
                    if not isinstance(test, str):  # first General Training page
                        close()
                        test = m.group(1) or "A"
                    continue
                if m := RE_RP.match(line):
                    close()
                    if m.group(1) in ("1", "I", "l") and not isinstance(test, str):
                        test = base + academic_count
                        academic_count += 1
                    cur = _Item("reading", test, int(_num(m.group(1))), _test_module(test), pno)
                    continue
                if m := RE_WT.match(line):
                    close()
                    n = 1 if m.group(1) in ("1", "I") else 2
                    seen_wt2 |= n == 2
                    # Books without "Test A/B" headings: the second GT Task 1 starts Test B.
                    if n == 1 and test == "A" and any(i.kind == "writing" and i.test == "A" for i in items):
                        test = "B"
                    cur = _Item("writing", test, n, _test_module(test), pno)
                    continue
                if RE_SECTION_START.match(line) or (cur and cur.kind == "speaking" and RE_QUESTIONS.match(line)):
                    close()
                    continue
                if m := RE_PART.match(line):
                    n = int(m.group(1))
                    starts_speaking = n == 1 and cur and cur.kind == "writing" and cur.n == 2
                    if starts_speaking or (cur and cur.kind == "speaking" and n == cur.n + 1):
                        close()
                        cur = _Item("speaking", test, n, _test_module(test), pno)
                        continue
                if cur:
                    cur.lines.append(line)
            elif zone == "scripts":
                if m := RE_TEST_UPPER.match(line):
                    explicit_test = _num(m.group(1))
                    continue
                if m := RE_SCRIPT_PART.match(line):
                    close()
                    part = int(m.group(1))
                    if explicit_test is not None:
                        script_test, explicit_test = explicit_test, None
                    elif part <= last_part:
                        script_test = (script_test + 1) if isinstance(script_test, int) else script_test
                    last_part = part
                    cur = _Item("transcript", script_test, part, _test_module(script_test), pno)
                    continue
                if cur:
                    cur.lines.append(line)
            elif zone == "samples":
                sample_lines.append((pno, line))
            elif zone == "keys":
                key_lines.append((pno, line))
    close()

    # Transcripts numbered by counting start from 1; shift them for books numbered 5-8.
    if base > 1:
        for i in items:
            if i.kind == "transcript" and isinstance(i.test, int) and i.test and i.test < base:
                i.test += base - 1
    return ParsedBook(book, items, sample_lines, key_lines, base)


def _body(lines: list[str], vocab: Vocab) -> str:
    kept = [vocab.respace(ln) for ln in lines if not RE_NOISE.match(ln)]
    return _clean("\n".join(kept))


# ---------------------------------------------------------------- writing samples
@dataclass
class RawSample:
    book: int
    test: int | str
    task: int
    band: float | None  # None = examiner model answer
    essay: str
    comment: str
    layout: str  # "essay_then_comment" | "comment_then_essay" | "model"


def parse_samples(book: int, sample_lines: list[tuple[int, str]], vocab: Vocab) -> list[RawSample]:
    if not sample_lines:
        return []
    text_parts, starts, pos = [], [], 0
    for pno, line in sample_lines:
        starts.append((pos, pno))
        text_parts.append(line)
        pos += len(line) + 1
    text = "\n".join(text_parts)

    def page_at(offset: int) -> int:
        page = starts[0][1]
        for p, pno in starts:
            if p > offset:
                break
            page = pno
        return page

    heads = [(m.start(), page_at(m.start()), _num(m.group(1)), 1 if m.group(2) in ("1", "I") else 2)
             for m in RE_SAMPLE_HEAD.finditer(text)]
    entries = sorted(
        [(m.start(), m.end(), float(m.group(1))) for m in RE_BAND.finditer(text)]
        + [(m.start(), m.end(), None) for m in RE_MODEL.finditer(text)]
    )
    out: list[RawSample] = []
    for idx, (start, end, band) in enumerate(entries):
        block_end = entries[idx + 1][0] if idx + 1 < len(entries) else len(text)
        page = page_at(start)
        # The heading is usually just before the entry, but some layouts print it at the
        # bottom of the page; prefer a heading on the same page.
        same_page = [h for h in heads if h[1] == page]
        same_before = [h for h in same_page if h[0] < start]
        if same_before:
            head = same_before[-1]
        elif same_page:
            head = same_page[0]
        else:
            earlier = [h for h in heads if h[0] < start]
            head = earlier[-1] if earlier else None
        if head is None:
            continue
        block = text[end:block_end]
        if band is None:
            essay, comment, layout = block, "", "model"
        elif (cm := RE_COMMENT.search(block)) and len(block[: cm.start()].split()) > 40:
            essay, comment, layout = block[: cm.start()], block[cm.end():], "essay_then_comment"
        else:
            rest = block[cm.end():] if cm else block
            comment, essay = _split_comment_first(rest, vocab)
            layout = "comment_then_essay"
        if layout == "model":
            essay_text = _BOILERPLATE.sub("", _body(essay.splitlines(), vocab)).strip()
        else:
            essay_text = _clean("\n".join(l for l in essay.splitlines() if not _is_noise(l)))
        comment_text = _clean("\n".join(l for l in comment.splitlines() if not _is_noise(l)))
        # Safety cap: a missed boundary must not pull the rest of the book into one record.
        essay_text = " ".join(essay_text.split(" ")[:MAX_ESSAY_WORDS]) if len(essay_text.split()) > MAX_ESSAY_WORDS else essay_text
        if len(comment_text.split()) > MAX_COMMENT_WORDS:
            continue
        out.append(RawSample(book, head[2], head[3], band, essay_text, comment_text, layout))
    return out


def _is_noise(line: str) -> bool:
    return bool(RE_NOISE.match(line) or RE_SAMPLE_HEAD.fullmatch(line.strip()))


_ARTEFACT = re.compile(r"[a-z][-^~](?=[\s.,]|$)")


def _split_comment_first(rest: str, vocab: Vocab) -> tuple[str, str]:
    """Older books print the examiner comment first, then the scanned handwritten essay.

    The comment is typeset (word ratio ~1.0); the essay OCR is noisier and shows
    handwriting artefacts such as "the-" or "par^". Split before the first line from which
    the next three lines score clearly lower, provided the comment so far ends a sentence.
    """
    lines = [l for l in rest.splitlines() if not _is_noise(l)]
    if len(lines) < 4:
        return rest, ""
    scores = [vocab.ratio(l) - 0.05 * len(_ARTEFACT.findall(l)) for l in lines]
    for i in range(2, len(lines) - 1):
        window = scores[i : i + 3]
        if (
            sum(window) / len(window) < 0.85
            and min(scores[max(0, i - 2) : i]) >= 0.8
            and re.search(r"[.!?'\"’)\]]$", lines[i - 1])
        ):
            # Clean lines without handwriting artefacts still belong to the comment.
            j = i
            while j < len(lines) - 1 and scores[j] >= 0.9 and not _ARTEFACT.search(lines[j]):
                j += 1
            if j > i and re.search(r"[.!?'\"’)\]]$", lines[j - 1]):
                i = j
            return "\n".join(lines[:i]), "\n".join(lines[i:])
    return "\n".join(lines), ""  # essay could not be told apart: keep the comment, drop the essay


# ---------------------------------------------------------------- audio
def audio_manifest(audio_root: Path) -> dict[tuple[int, int, int], Path]:
    """(book, test, part) -> mp3, from folder names like 剑13 and file names like Test1.Section2."""
    found: dict[tuple[int, int, int], Path] = {}
    for mp3 in sorted(audio_root.rglob("*.mp3")):
        m = re.search(r"剑(?:雅)?(\d+)", str(mp3.parent.relative_to(audio_root)))
        if not m:
            continue
        book = int(m.group(1))
        stem = re.sub(rf"(?i)^(ELT_)?(IELTS\s*{book}|C{book}|{book}-)", "", mp3.stem)
        if re.search(r"(?i)track", stem):
            continue  # track-numbered files do not say which test/part they belong to
        nums = [int(x) for x in re.findall(r"\d+", stem)]
        if len(nums) >= 2:
            test, part = nums[-2], nums[-1]
            if book == 12 and test >= 5:
                pass  # book 12 numbers its Academic tests 5-8, same as the PDF
            found.setdefault((book, test, part), mp3)
    return found


# ---------------------------------------------------------------- assembly
@dataclass
class BookResult:
    book: int
    path: Path
    chunks: list[CorpusChunk] = field(default_factory=list)
    samples: list[ScoredSample] = field(default_factory=list)
    keys: list[AnswerKey] = field(default_factory=list)
    skipped: str | None = None


def _src(book: int, test, extra: str = "") -> str:
    label = f"General Training Test {test}" if test in ("A", "B") else f"Test {test}"
    return f"Cambridge IELTS {book}, {label}{extra}"


# Instructions printed with every task. They carry no topic, so they are removed before
# indexing; otherwise every Task 1 prompt matches any query containing "chart".
_BOILERPLATE = re.compile(
    r"You should spend about\s*\d+\s*minutes on this task\.?|"
    r"Write about the following topic\s*:?|"
    r"Summari[sz]e the information by selecting and reporting the main features,?\s*and make comparisons where relevant\.?|"
    r"Give reasons for your answer and include any relevant examples from your own knowledge or experience\.?|"
    r"W\S{0,2}ite\s*at\s*least\s*\d+\s*words\.?|"
    r"You will have to talk about the topic for one to two minutes\.\s*You have one minute to think about what "
    r"you are going to say\.\s*You can make some notes to help you if you wish\.?|"
    r"However, please note that this is just one example out of many possible approaches\.?",
    re.I,
)


def strip_boilerplate(text: str) -> str:
    text = _BOILERPLATE.sub(" ", " ".join(text.split()))
    return " ".join(text.split())


_CUE_CARD = re.compile(r"Describe .*?(?:and )?(?:explain|say) .*?[.?](?=\s|$)", re.S)


def _move_cue_cards(items: list[_Item]) -> None:
    """Some layouts extract the Part 2 cue card after Part 3; move it back to Part 2."""
    speaking = {(i.test, i.n): i for i in items if i.kind == "speaking"}
    for (test, n), part2 in speaking.items():
        if n != 2 or any(l.startswith("Describe") for l in part2.lines):
            continue
        for other_n in (3, 1):
            other = speaking.get((test, other_n))
            if not other:
                continue
            joined = "\n".join(other.lines)
            if m := _CUE_CARD.search(joined):
                part2.lines += m.group(0).splitlines()
                other.lines = (joined[: m.start()] + joined[m.end():]).splitlines()
                break


def _unique_ids(res: BookResult) -> None:
    """A mis-read heading can give two records the same id; suffix the later ones."""
    for records, attr in ((res.chunks, "chunk_id"), (res.samples, "sample_id")):
        seen: Counter = Counter()
        for rec in records:
            rid = getattr(rec, attr)
            seen[rid] += 1
            if seen[rid] > 1:
                setattr(rec, attr, f"{rid}-dup{seen[rid]}")


def convert(parsed: ParsedBook, vocab: Vocab, audio: dict, path: Path) -> BookResult:
    _move_cue_cards(parsed.items)
    res = BookResult(parsed.book, path)
    b = parsed.book
    prompts: dict[tuple, str] = {}
    counts: Counter = Counter()
    for it in parsed.items:
        if not it.test:
            continue
        key = (it.kind, it.test, it.n)
        counts[key] += 1
        if counts[key] > 1:
            continue  # a stray repeated heading; keep the first
        text = _body(it.lines, vocab)
        if it.kind == "writing":
            m = RE_WORD_LIMIT.search(text)
            text = strip_boilerplate(text[: m.end()] if m else " ".join(text.split()[:120]))
            prompts[(it.test, it.n)] = text
            cid, skill, stype, qtype = f"w{it.n}", "writing", "real_exam", writing_type(it.n, text, it.module)
        elif it.kind == "speaking":
            cid, skill, stype, qtype = f"s{it.n}", "speaking", "real_exam", f"qt.speaking.part{it.n}"
            text = strip_boilerplate(" ".join(text.split()[:350]))
        elif it.kind == "reading":
            cid, skill, stype, qtype = f"r{it.n}", "reading", "passage", None
        else:
            cid, skill, stype, qtype = f"l{it.n}", "listening", "transcript", None
        if len(text.split()) < 15:
            continue
        test_id = f"t{it.test}".lower()
        media = audio.get((b, it.test, it.n)) if it.kind == "transcript" else None
        res.chunks.append(
            CorpusChunk(
                chunk_id=f"cam{b}-{test_id}-{cid}",
                skill=skill,  # type: ignore[arg-type]
                source_type=stype,  # type: ignore[arg-type]
                question_id=f"cam{b}-{test_id}-{cid}",
                question_type=qtype,
                text=text,
                license=LICENSE,
                source=_src(b, it.test),
                module=it.module,  # type: ignore[arg-type]
                media_uri=str(media) if media else None,
                quality=round(vocab.ratio(text), 3),
            )
        )
    for n, s in enumerate(parse_samples(b, parsed.sample_text, vocab), start=1):
        test_id = f"t{s.test}".lower()
        task = f"task{s.task}"
        if s.band is None:
            if len(s.essay.split()) >= 80:
                res.chunks.append(
                    CorpusChunk(
                        chunk_id=f"cam{b}-{test_id}-w{s.task}-model",
                        skill="writing",
                        source_type="model_answer",
                        question_id=f"cam{b}-{test_id}-w{s.task}",
                        text=s.essay,
                        license=LICENSE,
                        source=_src(b, s.test, f", Writing Task {s.task} examiner model answer"),
                        module=_test_module(s.test),  # type: ignore[arg-type]
                        quality=round(vocab.ratio(s.essay), 3),
                    )
                )
            continue
        if len(s.comment.split()) < 20:
            continue
        res.samples.append(
            ScoredSample(
                sample_id=f"cam{b}-{test_id}-w{s.task}-sample{n}",
                skill="writing",
                task=task,
                prompt_id=f"cam{b}-{test_id}-w{s.task}",
                prompt_text=prompts.get((s.test, s.task)),
                response_text=s.essay,
                scores={"overall": s.band},
                rater="official",
                comments=s.comment,
                source=_src(b, s.test, f", Writing Task {s.task} sample answer"),
                quality=round(vocab.ratio(s.essay), 3) if s.essay else 0.0,
            )
        )
    _unique_ids(res)
    return res


def find_books(root: Path) -> dict[int, Path]:
    books: dict[int, Path] = {}
    for pdf in sorted(root.rglob("*.pdf")):
        if "解析" in str(pdf.parent.name):
            continue
        n = book_number(pdf)
        if n and n not in books:
            books[n] = pdf
    return books


def import_books(
    root: Path, cache_dir: Path | None = None, books: list[int] | None = None, keys: bool = True
) -> list[BookResult]:
    vocab = Vocab.load()
    audio = audio_manifest(root)
    results = []
    for n, path in sorted(find_books(root).items()):
        if books and n not in books:
            continue
        pages = pdf_pages(path, cache_dir)
        if not has_text_layer(pages):
            results.append(BookResult(n, path, skipped="scanned PDF without a text layer (needs OCR)"))
            continue
        parsed = parse_book(pages, n)
        res = convert(parsed, vocab, audio, path)
        key_pages = sorted({p for p, _ in parsed.key_text})
        if keys and key_pages:
            res.keys = extract_keys(path, key_pages, n, parsed.base)
        results.append(res)
    return results

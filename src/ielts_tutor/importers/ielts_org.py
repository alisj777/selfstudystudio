"""Import the free official samples published on ielts.org.

Writes to git-ignored folders (the material is © the IELTS partners):
  data/<exam>/corpus/private/ielts_org.jsonl    speaking sample questions
  data/<exam>/scoring/private/ielts_org.jsonl   writing responses with examiner bands
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from ..rag.schema import CorpusChunk, ScoredSample

WRITING_URL = (
    "https://ielts.org/cdn/computer-delivered-sample-tests-academic-writing/"
    "ielts-academic-writing-example-responses-to-parts-1-and-2-with-band-scores-and-examiner-comments.pdf"
)
SPEAKING_URL = "https://ielts.org/cdn/ielts-downloadable-assets/ielts-sample-tests/ielts-speaking-sample-tasks-2023.pdf"
SOURCE = "ielts.org free sample test"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        # The CDN rejects urllib's default User-Agent with 403.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ielts-tutor)"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https URLs
            dest.write_bytes(resp.read())
    return dest


def pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def _clean(text: str) -> str:
    """Drop whitespace-only lines and re-join wrapped lines into paragraphs."""
    lines = [ln.strip() for ln in text.splitlines()]
    paragraphs, current = [], []
    for ln in lines:
        if not ln:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(ln)
        if re.search(r"[.!?]$", ln) and len(ln) < 75:  # short final line ends a paragraph
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(p for p in paragraphs if p)


_WRITING = re.compile(
    r"Sample Academic Writing Part (?P<part>[12])\s*\n\s*Candidate Response (?P<n>\d)\s*\n"
    r"(?:WRITING TASK 2\s*\n)?(?P<essay>.*?)\n\s*Examiner comment\s*\n\s*Band (?P<band>\d(?:\.5)?)\s*\n(?P<comment>.*)",
    re.S,
)


def parse_writing(pages: list[str]) -> list[ScoredSample]:
    samples = []
    for page in pages:
        m = _WRITING.search(page)
        if not m:
            continue
        task = "task1" if m["part"] == "1" else "task2"
        samples.append(
            ScoredSample(
                sample_id=f"ielts-org-w{m['part']}-r{m['n']}",
                skill="writing",
                task=task,
                prompt_text=None,  # the task prompts are only shown in the online player
                response_text=_clean(m["essay"]),
                scores={"overall": float(m["band"])},
                rater="official",
                comments=" ".join(m["comment"].split()),
                source=f"{SOURCE}: {WRITING_URL}",
            )
        )
    if len(samples) != 4:
        raise ValueError(f"Expected 4 examiner-scored responses, parsed {len(samples)}; the PDF layout may have changed")
    return samples


def parse_speaking(pages: list[str]) -> list[CorpusChunk]:
    text = "\n".join(pages)

    def questions(start: str, end: str) -> list[str]:
        block = text.split(start, 1)[1].split(end, 1)[0]
        return [" ".join(q.split()) for q in re.findall(r"^\s*\d+\.\s*(.+?)\s*$", block, re.M)]

    part1 = questions("Part 1 – Introduction and interview: Examiner Frame", "Part 1 – Introduction and interview: Recording")
    part3 = questions("Part 3 – Two-way Discussion: Examiner Frame", "Part 3 – Two-way Discussion: Recording")
    card = re.search(r"(Describe .*?explain why .*?\.)", text, re.S)
    if not (part1 and part3 and card):
        raise ValueError("Could not find the speaking sample questions; the PDF layout may have changed")
    cue_card = " ".join(card.group(1).split())

    def chunk(cid: str, qtype: str, body: str) -> CorpusChunk:
        return CorpusChunk(
            chunk_id=f"ielts-org-{cid}",
            skill="speaking",
            source_type="official_sample",
            question_id=f"ielts-org-{cid}",
            question_type=qtype,
            text=body,
            license="© IELTS partners; internal use only",
            source=f"{SOURCE}: {SPEAKING_URL}",
        )

    return [
        chunk("s1", "qt.speaking.part1", "Part 1 questions: " + " ".join(part1)),
        chunk("s2", "qt.speaking.part2", cue_card),
        chunk("s3", "qt.speaking.part3", "Part 3 questions: " + " ".join(part3)),
    ]


def import_all(download_dir: Path) -> tuple[list[CorpusChunk], list[ScoredSample]]:
    writing = pdf_pages(download(WRITING_URL, download_dir / "ielts_org_writing_samples.pdf"))
    speaking = pdf_pages(download(SPEAKING_URL, download_dir / "ielts_org_speaking_samples.pdf"))
    return parse_speaking(speaking), parse_writing(writing)

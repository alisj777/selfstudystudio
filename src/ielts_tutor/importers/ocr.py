"""OCR scanned PDF pages with tesseract, caching each page so runs can be resumed.

Needs the `ocr` extra (PyMuPDF renders the pages) and the `tesseract` binary with
the languages you ask for (`chi_sim` for the Chinese explanation books).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

DPI = 300
LANG = "chi_sim+eng"


def tesseract_available(lang: str = LANG) -> str | None:
    """Return None when tesseract and the languages are installed, else what is missing."""
    if not shutil.which("tesseract"):
        return "tesseract is not installed (brew install tesseract tesseract-lang)"
    langs = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout.split()
    missing = [x for x in lang.split("+") if x not in langs]
    return f"tesseract languages missing: {', '.join(missing)}" if missing else None


def _ocr_image(path: Path, lang: str, psm: int) -> str:
    out = subprocess.run(["tesseract", str(path), "-", "-l", lang, "--psm", str(psm)],
                         capture_output=True, text=True, check=False)
    return out.stdout


def ocr_pdf(
    pdf: Path,
    cache: Path,
    *,
    pages: Iterable[int] | None = None,
    lang: str = LANG,
    dpi: int = DPI,
    psm: int = 4,
    workers: int = 6,
    progress: Callable[[int, int], None] | None = None,
) -> dict[int, str]:
    """OCR `pages` (0-based, default all) of `pdf`; returns {page: text}.

    Results are stored in the JSON file `cache` after every batch, so an interrupted
    run picks up where it stopped.
    """
    import pymupdf

    done: dict[int, str] = {}
    if cache.exists():
        done = {int(k): v for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
    doc = pymupdf.open(pdf)
    wanted = sorted(set(range(doc.page_count) if pages is None else pages) & set(range(doc.page_count)))
    todo = [p for p in wanted if p not in done]
    cache.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps({str(k): done[k] for k in sorted(done)}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache)

    batch = workers * 4
    with tempfile.TemporaryDirectory() as tmpdir, ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            images = []
            for p in chunk:  # PyMuPDF is not thread-safe: render here, OCR in the pool
                img = Path(tmpdir) / f"{p}.png"
                doc[p].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY).save(img)
                images.append(img)
            for p, text in zip(chunk, pool.map(lambda i: _ocr_image(i, lang, psm), images)):
                done[p] = text
            for img in images:
                img.unlink()
            save()
            if progress:
                progress(len(wanted) - len(todo) + start + len(chunk), len(wanted))
    return {p: done[p] for p in wanted}

"""Importers for external scored-essay datasets.

Imported data stays local (data/<exam>/scoring and data/<exam>/holdout are git-ignored):
the upstream licensing and score provenance of public IELTS datasets is often unclear.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

from .rag.schema import ScoredSample

_MOJIBAKE = re.compile(r"[\xc2-\xf4][\x80-\xbf]+")
_VERSION_SUFFIX = re.compile(r"\s+v\.\d+\s*$")


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1 (e.g. 'â\\x80\\x99' -> '’')."""

    def repair(m: re.Match) -> str:
        try:
            return m.group().encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group()

    return _MOJIBAKE.sub(repair, text)


def official_overall(scores: list[float]) -> float:
    """IELTS overall from criterion bands: mean rounded to the nearest half band
    (.25 rounds up to .5, .75 rounds up to the next whole band)."""
    mean = sum(scores) / len(scores)
    whole = int(mean)
    rest = mean - whole
    return whole + (0.0 if rest < 0.25 else 0.5 if rest < 0.75 else 1.0)


def load_btnotpt(directory: Path) -> tuple[list[ScoredSample], dict]:
    """Hugging Face `btnotpt/ielts_task_2` (train + test parquet files).

    Duplicated essays are merged; essays whose copies disagree on the overall band are
    dropped. Returns the samples and a small report.
    """
    import pyarrow.parquet as pq

    rows = []
    for name in ("train", "test"):
        path = directory / f"{name}.parquet"
        if path.exists():
            rows.extend(pq.read_table(path).to_pylist())

    by_essay: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        essay = fix_mojibake((r.get("essay") or "").strip())
        if essay and None not in (r.get(k) for k in ("TR", "CC", "LR", "GRA", "Overall")):
            by_essay[essay].append(r)

    samples, conflicts = [], 0
    for i, (essay, copies) in enumerate(sorted(by_essay.items())):
        if len({c["Overall"] for c in copies}) > 1:
            conflicts += 1
            continue
        r = copies[0]
        topic = _VERSION_SUFFIX.sub("", fix_mojibake(r.get("topic") or "")).strip()
        samples.append(
            ScoredSample(
                sample_id=f"btnotpt-{i:05d}",
                skill="writing",
                task="task2",
                prompt_text=topic or None,
                response_text=essay,
                scores={k: float(r[k]) for k in ("TR", "CC", "LR", "GRA")} | {"overall": float(r["Overall"])},
                rater="dataset",
                comments=fix_mojibake(r.get("feedback") or "") or None,
                source="hf:btnotpt/ielts_task_2",
            )
        )
    report = {
        "rows": len(rows),
        "unique_essays": len(by_essay),
        "dropped_conflicting": conflicts,
        "kept": len(samples),
    }
    return samples, report


def stratified_holdout(
    samples: list[ScoredSample], fraction: float = 0.15, seed: int = 13
) -> tuple[list[ScoredSample], list[ScoredSample]]:
    """Split by overall band so the held-out set keeps the band distribution."""
    rng = random.Random(seed)
    by_band: dict[float, list[ScoredSample]] = defaultdict(list)
    for s in samples:
        by_band[s.scores["overall"]].append(s)
    keep, hold = [], []
    for band in sorted(by_band):
        group = by_band[band][:]
        rng.shuffle(group)
        n_hold = round(len(group) * fraction)
        hold += group[:n_hold]
        keep += group[n_hold:]
    return keep, hold


def band_distribution(samples: list[ScoredSample]) -> dict[float, int]:
    counts: dict[float, int] = defaultdict(int)
    for s in samples:
        counts[s.scores["overall"]] += 1
    return dict(sorted(counts.items()))

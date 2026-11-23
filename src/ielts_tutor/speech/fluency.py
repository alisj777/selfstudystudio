"""Fluency and pronunciation signals from a timed transcript (design doc 6.2, first version).

  speech rate     words per minute over the speaking time (long silences excluded)
  pauses          gaps >= 0.5 s; "mid-clause" if the previous word does not end a sentence
  fillers         um / uh / er / erm / hmm, "you know", "I mean"
  repetitions     the same word twice in a row (a common self-correction pattern)
  mean run        words between pauses (longer runs read as more fluent)
  ASR confidence  mean word probability: a rough stand-in for intelligibility until a
                  pronunciation model (GOP, MFA) is added

The rule-based bands below are uncalibrated heuristics, labelled as such in the score card.
"""

from __future__ import annotations

import re

from .asr import Transcript

FILLERS = {"um", "uh", "er", "erm", "hmm", "mm", "ah", "eh", "uhm", "umm"}
PAUSE = 0.5
LONG_PAUSE = 1.0


def _norm(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


def fluency_features(t: Transcript) -> dict[str, float]:
    words = t.words
    n = len(words)
    if n == 0:
        return {"spoken_words": 0, "duration_s": round(t.duration, 1), "wpm": 0.0}
    gaps = [(words[i].start - words[i - 1].end, words[i - 1].text) for i in range(1, n)]
    pauses = [(g, prev) for g, prev in gaps if g >= PAUSE]
    mid = [g for g, prev in pauses if not prev.rstrip().endswith((".", "?", "!"))]
    speaking = sum(w.end - w.start for w in words) + sum(g for g, _ in gaps if g < LONG_PAUSE)
    tokens = [_norm(w.text) for w in words]
    fillers = sum(1 for x in tokens if x in FILLERS)
    text = " ".join(tokens)
    fillers += text.count("you know") + text.count("i mean")
    reps = sum(1 for a, b in zip(tokens, tokens[1:]) if a and a == b and a not in FILLERS)
    runs, cur = [], 0
    for i in range(n):
        cur += 1
        if i == n - 1 or words[i + 1].start - words[i].end >= 0.3:
            runs.append(cur)
            cur = 0
    minutes = max(t.duration, 1e-6) / 60
    return {
        "spoken_words": n,
        "duration_s": round(t.duration, 1),
        "wpm": round(n / max(speaking, 1e-6) * 60, 1),
        "pauses_per_min": round(len(pauses) / minutes, 2),
        "mid_clause_pauses_per_min": round(len(mid) / minutes, 2),
        "long_pauses": sum(1 for g, _ in pauses if g >= LONG_PAUSE),
        "mean_pause_s": round(sum(g for g, _ in pauses) / len(pauses), 2) if pauses else 0.0,
        "fillers_per_100w": round(100 * fillers / n, 2),
        "repetitions_per_100w": round(100 * reps / n, 2),
        "mean_run_words": round(sum(runs) / len(runs), 1),
        "asr_confidence": round(sum(w.prob for w in words) / n, 3),
        "low_confidence_ratio": round(sum(1 for w in words if w.prob < 0.5) / n, 3),
    }


def _band(value: float, cuts: list[tuple[float, float]], floor: float) -> float:
    return next((b for c, b in cuts if value >= c), floor)


def rule_bands(f: dict[str, float]) -> dict[str, float]:
    """Heuristic Fluency (FC) and Pronunciation (P) bands from the features. Uncalibrated."""
    if not f.get("spoken_words"):
        return {"FC": 0.0, "P": 0.0}
    fc = _band(f["wpm"], [(150, 7.5), (130, 7.0), (115, 6.5), (100, 6.0), (85, 5.5), (70, 5.0)], 4.5)
    fc -= 0.5 * (f["mid_clause_pauses_per_min"] > 6) + 0.5 * (f["fillers_per_100w"] > 6) \
        + 0.5 * (f["mean_run_words"] < 4)
    p = _band(f["asr_confidence"], [(0.93, 7.5), (0.9, 7.0), (0.86, 6.5), (0.82, 6.0), (0.77, 5.5), (0.7, 5.0)], 4.5)
    return {"FC": max(fc, 3.0), "P": p}

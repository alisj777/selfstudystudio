"""Evaluate the writing scorer on held-out essays and fit the model/kNN/LLM fusion weights.

LLM judgements are cached one essay per line, so an interrupted run resumes without paying
twice. Fusion weights are reported with 2-fold cross-validation on the judged sample, then
refitted on the whole sample and written to var/models/fusion.json for the Scorer agent.
"""

from __future__ import annotations

import itertools
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..rag.schema import ScoredSample
from .judge import judge
from .scorer import CRITERIA, metrics, overall_band, round_half


def _grid(step: float = 0.1):
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2)
    for a, b in itertools.product(vals, vals):
        if a + b <= 1 + 1e-9:
            yield float(a), float(b), max(0.0, float(round(1 - a - b, 2)))


def fit_weights(m: np.ndarray, k: np.ndarray, l: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    return min(_grid(), key=lambda w: float(np.mean((w[0] * m + w[1] * k + w[2] * l - y) ** 2)))


def judge_evidence(rag, h: ScoredSample, crits: list[str], base: dict[str, float]) -> list:
    """Descriptors near the estimate plus similar scored essays (never the essay itself).
    Run this outside worker threads: it embeds the essay."""
    ev = [e for c in crits for e in rag.rubric.descriptors(f"crit.writing.{c}", base[c], window=1)]
    return ev + [e for e in rag.rubric.similar_samples(h.response_text, 4, skill="writing", task=h.task)
                 if e.id != h.sample_id][:3]


def run_judge(llm, ev: list, h: ScoredSample, crits: list[str], base: dict[str, float], lang: str = "en") -> dict:
    """One LLM judgement of a scored essay, as a cacheable record."""
    try:
        out = judge(llm, h.task, h.prompt_text or "", h.response_text, crits, base, ev, lang=lang)
        return {"id": h.sample_id, "llm": {c: j.band for c, j in out.items()}, "error": ""}
    except Exception as exc:  # keep going; the essay is simply left out
        return {"id": h.sample_id, "llm": {}, "error": str(exc)[:200]}


def official_check(scorer, samples: list[ScoredSample], cache, vocab, rag, llm, judge_cache: Path | None = None,
                   workers: int = 4) -> list[dict]:
    """Model and LLM overall bands against real examiners' bands (Cambridge / ielts.org samples)."""
    done: dict[str, dict] = {}
    if judge_cache and judge_cache.exists():
        done = {r["id"]: r for r in map(json.loads, judge_cache.read_text().splitlines())}

    prepared = []  # model scores first, in this thread (embedding is not thread-safe)
    for h in samples:
        crits = ["TA" if h.task == "task1" else "TR", "CC", "LR", "GRA"]
        model, base = None, {c: 6.0 for c in crits}
        if h.task == "task2":
            p = scorer.predict(h.response_text, h.prompt_text, h.task, cache, vocab)
            model = overall_band({c: round_half(v) for c, v in p.fused.items()})
            base = {c: round_half(p.fused[c]) for c in crits}
        prepared.append((h, crits, model, base, judge_evidence(rag, h, crits, base)))
    lock = threading.Lock()

    def one(item) -> dict:
        h, crits, model, base, ev = item
        rec = done.get(h.sample_id) or run_judge(llm, ev, h, crits, base)
        if judge_cache and h.sample_id not in done:
            with lock, judge_cache.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        llm_overall = overall_band({("TR" if k == "TA" else k): v for k, v in rec["llm"].items()}) \
            if len(rec["llm"]) == 4 else None
        return {"id": h.sample_id, "task": h.task, "examiner": h.scores["overall"], "model": model, "llm": llm_overall}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, prepared))


def evaluate(scorer, holdout: list[ScoredSample], cache, vocab, *, rag=None, llm=None, n_llm: int = 0,
             judge_cache: Path | None = None, seed: int = 0, lang: str = "en", progress=None,
             workers: int = 4) -> dict:
    preds = [scorer.predict(h.response_text, h.prompt_text, h.task, cache, vocab) for h in holdout]
    cache.flush()
    report: dict = {"n": len(holdout), "sources": {}}
    for c in CRITERIA:
        y = [h.scores[c] for h in holdout]
        report["sources"][c] = {"model": metrics(y, [p.model[c] for p in preds]),
                                "knn": metrics(y, [p.knn[c] for p in preds]),
                                "model+knn": metrics(y, [p.fused[c] for p in preds])}
    report["overall"] = {"model+knn": metrics([h.scores["overall"] for h in holdout],
                                              [overall_band({c: round_half(p.fused[c]) for c in CRITERIA}) for p in preds])}
    if not (llm and rag and n_llm):
        return report

    # ---- LLM judge on a fixed random sample
    idx = sorted(random.Random(seed).sample(range(len(holdout)), min(n_llm, len(holdout))))
    done: dict[str, dict] = {}
    if judge_cache and judge_cache.exists():
        done = {r["id"]: r for r in map(json.loads, judge_cache.read_text().splitlines())}
    todo = [i for i in idx if holdout[i].sample_id not in done]
    lock = threading.Lock()

    crits = list(CRITERIA)
    bases = {i: {c: round_half(preds[i].fused[c]) for c in crits} for i in todo}
    evid = {i: judge_evidence(rag, holdout[i], crits, bases[i]) for i in todo}

    def one(i: int) -> None:
        h = holdout[i]
        rec = run_judge(llm, evid[i], h, crits, bases[i], lang)
        with lock:
            done[h.sample_id] = rec
            if judge_cache:
                judge_cache.parent.mkdir(parents=True, exist_ok=True)
                with judge_cache.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            if progress:
                progress(len(done), len(idx))

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))

    rows = [(holdout[i], preds[i], done[holdout[i].sample_id]["llm"]) for i in idx
            if len(done.get(holdout[i].sample_id, {}).get("llm", {})) == 4]
    report["llm_judged"] = len(rows)
    if len(rows) < 10:
        return report
    weights, cv = {}, {}
    half = np.arange(len(rows)) % 2
    for c in CRITERIA:
        y = np.array([h.scores[c] for h, _, _ in rows])
        m = np.array([p.model[c] for _, p, _ in rows])
        k = np.array([p.knn[c] for _, p, _ in rows])
        l = np.array([j[c] for _, _, j in rows])
        oof = np.zeros_like(y)
        for f in (0, 1):  # weights fitted on one half, scored on the other
            w = fit_weights(m[half != f], k[half != f], l[half != f], y[half != f])
            oof[half == f] = w[0] * m[half == f] + w[1] * k[half == f] + w[2] * l[half == f]
        weights[c] = fit_weights(m, k, l, y)
        cv[c] = {"model+knn": metrics(y, (scorer.weights[c][0] * m + scorer.weights[c][1] * k).tolist()),
                 "llm": metrics(y, l.tolist()), "fused (2-fold)": metrics(y, oof.tolist()),
                 "weights": weights[c]}
    report["with_llm"] = cv
    report["weights"] = weights
    return report

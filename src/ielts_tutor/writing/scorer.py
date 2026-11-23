"""Writing score model (design doc 6.1, light version).

  s_model  regression on [hand-made features, essay embedding, prompt-essay similarity]
  s_kNN    similarity-weighted mean score of the k most similar scored essays
  s_LLM    the LLM judge's independent score (writing/judge.py), when available

The final score per criterion is a weighted sum of the sources; weights are fitted on
out-of-fold predictions (stacking), so they reflect how well each source predicts the
held-out human score. Confidence intervals come from the out-of-fold residuals.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..rag.schema import ScoredSample
from .features import EssayFeatures, extract

CRITERIA = ("TR", "CC", "LR", "GRA")
MODEL_VERSION = "light-1"


# ---------------------------------------------------------------- band helpers


def round_half(x: float) -> float:
    return float(min(9.0, max(0.0, round(x * 2) / 2)))


def overall_band(criteria: dict[str, float]) -> float:
    """IELTS rule: average the four criteria; .25 rounds up to .5 and .75 up to the next band."""
    return _ielts_round(sum(criteria[c] for c in CRITERIA) / 4)


def _ielts_round(mean: float) -> float:
    whole = math.floor(mean)
    frac = mean - whole
    if frac < 0.25 - 1e-9:
        return float(whole)
    if frac < 0.75 - 1e-9:
        return whole + 0.5
    return float(whole + 1)


def qwk(y_true: list[float], y_pred: list[float]) -> float:
    """Quadratic weighted kappa on half-band classes."""
    a = np.round(np.asarray(y_true) * 2).astype(int)
    b = np.round(np.asarray(y_pred) * 2).astype(int)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    k = hi - lo + 1
    obs = np.zeros((k, k))
    for x, y in zip(a - lo, b - lo):
        obs[x, y] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / max((k - 1) ** 2, 1)
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    return float(1 - (w * obs).sum() / (w * exp).sum()) if (w * exp).sum() else 1.0


def metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    t, p = np.asarray(y_true), np.asarray([round_half(x) for x in y_pred])
    return {
        "qwk": round(qwk(t, p), 3),
        "pearson": round(float(np.corrcoef(t, np.asarray(y_pred))[0, 1]), 3) if len(t) > 1 else 0.0,
        "exact": round(float(np.mean(t == p)), 3),
        "within_0.5": round(float(np.mean(np.abs(t - p) <= 0.5)), 3),
        "mae": round(float(np.mean(np.abs(t - np.asarray(y_pred)))), 3),
    }


# ---------------------------------------------------------------- embeddings cache


class EmbeddingCache:
    """Embeds texts once and keeps the vectors on disk (keyed by model and text hash)."""

    def __init__(self, embedder, path: Path | None):
        self.embedder, self.path = embedder, path
        self.store: dict[str, list[float]] = {}
        self._dirty = False
        if path and path.exists():
            self.store = json.loads(path.read_text())

    def flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.store))
        self._dirty = False

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.embedder.name}|{text}".encode()).hexdigest()

    def embed(self, texts: list[str], batch: int = 64) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self.store]
        for i in range(0, len(missing), batch):
            part = missing[i:i + batch]
            for t, v in zip(part, self.embedder.embed(part)):
                self.store[self._key(t)] = [round(float(x), 5) for x in v]
        if len(missing) >= 50:
            self.flush()  # large batches only; call flush() after many small calls
        self._dirty |= bool(missing)
        return np.asarray([self.store[self._key(t)] for t in texts], dtype=np.float32)


# ---------------------------------------------------------------- model


@dataclass
class Prediction:
    model: dict[str, float]
    knn: dict[str, float]
    fused: dict[str, float]  # before the LLM source is mixed in
    interval: dict[str, tuple[float, float]]
    neighbours: list[tuple[str, float]]  # (sample_id, similarity)
    features: EssayFeatures


@dataclass
class WritingScorer:
    """Fit with `fit`, then `predict`. Pickled to disk with `save` / `load`."""

    embedder_name: str = ""
    k: int = 15
    alpha: float = 1.0  # small ridge penalty is enough: features are standardised
    regressor: str = "gbr"  # gbr (gradient boosting) | ridge
    # fitted state
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    coef: dict = field(default_factory=dict)
    gbr: dict = field(default_factory=dict)
    train_emb: np.ndarray | None = None
    train_y: np.ndarray | None = None
    train_ids: list[str] = field(default_factory=list)
    weights: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # model, kNN, LLM
    residual_q: dict[str, tuple[float, float]] = field(default_factory=dict)
    cv_metrics: dict = field(default_factory=dict)
    version: str = MODEL_VERSION

    # -- design matrix
    @staticmethod
    def _rows(samples: list[ScoredSample], emb: np.ndarray, prompt_emb: np.ndarray, vocab) -> np.ndarray:
        feats = np.asarray([extract(s.response_text, s.prompt_text, s.task, vocab).vector() for s in samples])
        sim = (emb * prompt_emb).sum(1, keepdims=True)
        return np.hstack([feats, sim, emb])

    def _standardise(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        return (x - self.mu) / self.sd

    def _fit_regression(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.regressor == "gbr":
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.gbr = {c: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                         l2_regularization=1.0).fit(x, y[:, i])
                        for i, c in enumerate(CRITERIA)}
        else:
            xb = np.hstack([x, np.ones((len(x), 1))])
            reg = self.alpha * np.eye(xb.shape[1])
            reg[-1, -1] = 0
            beta = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
            self.coef = {"beta": beta}

    def _regress(self, x: np.ndarray) -> np.ndarray:
        if self.regressor == "gbr":
            return np.stack([self.gbr[c].predict(x) for c in CRITERIA], 1)
        return np.hstack([x, np.ones((len(x), 1))]) @ self.coef["beta"]

    def _knn(self, emb: np.ndarray, exclude_self: bool = False) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
        sims = emb @ self.train_emb.T
        if exclude_self:
            np.fill_diagonal(sims, -1)
        idx = np.argsort(-sims, 1)[:, : self.k]
        out, neigh = [], []
        for r, row in enumerate(idx):
            w = np.clip(sims[r, row], 0, None) ** 4 + 1e-9
            out.append((w[:, None] * self.train_y[row]).sum(0) / w.sum())
            neigh.append([(int(j), float(sims[r, j])) for j in row])
        return np.asarray(out), neigh

    # -- training
    def fit(self, samples: list[ScoredSample], cache: EmbeddingCache, vocab, folds: int = 5, seed: int = 0) -> dict:
        samples = [s for s in samples if all(c in s.scores for c in CRITERIA)]
        self.embedder_name = cache.embedder.name
        emb = cache.embed([s.response_text for s in samples])
        pemb = cache.embed([s.prompt_text or "" for s in samples])
        x = self._rows(samples, emb, pemb, vocab)
        y = np.asarray([[s.scores[c] for c in CRITERIA] for s in samples])

        # out-of-fold predictions for both sources -> stacking weights and residuals
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, folds, len(samples))
        oof_model, oof_knn = np.zeros_like(y), np.zeros_like(y)
        for f in range(folds):
            tr, te = fold != f, fold == f
            sub = WritingScorer(k=self.k, alpha=self.alpha, regressor=self.regressor)
            xs = sub._standardise(x[tr], fit=True)
            sub._fit_regression(xs, y[tr])
            oof_model[te] = sub._regress(sub._standardise(x[te]))
            sub.train_emb, sub.train_y = emb[tr], y[tr]
            oof_knn[te] = sub._knn(emb[te])[0]
        for i, c in enumerate(CRITERIA):
            w = _best_weight(oof_model[:, i], oof_knn[:, i], y[:, i])
            self.weights[c] = (w, 1 - w, 0.0)
            fused = w * oof_model[:, i] + (1 - w) * oof_knn[:, i]
            res = y[:, i] - fused
            self.residual_q[c] = (float(np.quantile(res, 0.1)), float(np.quantile(res, 0.9)))
            self.cv_metrics[c] = {"model": metrics(y[:, i], oof_model[:, i]), "knn": metrics(y[:, i], oof_knn[:, i]),
                                  "fused": metrics(y[:, i], fused), "w_model": round(w, 2)}

        # final fit on everything
        self._fit_regression(self._standardise(x, fit=True), y)
        self.train_emb, self.train_y, self.train_ids = emb, y, [s.sample_id for s in samples]
        return self.cv_metrics

    # -- inference
    def predict(self, essay: str, prompt: str | None, task: str, cache: EmbeddingCache, vocab,
                grammar_errors: int | None = None) -> Prediction:
        s = ScoredSample(sample_id="_", skill="writing", task=task, prompt_text=prompt, response_text=essay,
                         scores={}, rater="model")
        emb = cache.embed([essay])
        pemb = cache.embed([prompt or ""])
        x = self._rows([s], emb, pemb, vocab)
        model = self._regress(self._standardise(x))[0]
        knn, neigh = self._knn(emb)
        out_m, out_k, out_f, iv = {}, {}, {}, {}
        for i, c in enumerate(CRITERIA):
            wm, wk, _ = self.weights.get(c, (0.5, 0.5, 0))
            m, k = float(np.clip(model[i], 0, 9)), float(knn[0][i])
            f = (wm * m + wk * k) / (wm + wk)
            out_m[c], out_k[c], out_f[c] = round(m, 2), round(k, 2), round(f, 2)
            lo, hi = self.residual_q.get(c, (-0.5, 0.5))
            iv[c] = (round_half(f + lo), round_half(f + hi))
        return Prediction(out_m, out_k, out_f, iv,
                          [(self.train_ids[j], round(sim, 3)) for j, sim in neigh[0][:5]],
                          extract(essay, prompt, task, vocab, grammar_errors))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "WritingScorer":
        with path.open("rb") as fh:
            return pickle.load(fh)


def _best_weight(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    """Weight on source `a` (vs `b`) that minimises squared error, searched on a grid."""
    grid = np.linspace(0, 1, 21)
    errs = [float(np.mean((w * a + (1 - w) * b - y) ** 2)) for w in grid]
    return float(grid[int(np.argmin(errs))])

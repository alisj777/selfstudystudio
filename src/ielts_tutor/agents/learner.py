"""Learner Modeler: update mastery on knowledge-graph nodes after each scored attempt.

A simple exponential moving average for now (M4 replaces it with BKT/DKT):
  criteria          mastery <- EMA of band / 9
  question type, topic  attempts += 1, mastery <- EMA of overall / 9
  grammar points    linked (via ErrorType -INDICATES->) to errors found: mastery drops per error
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .state import LearnerProfile, Mastery, Session, now
from .toolkit import Toolkit, agent

RATE = 0.3
ERROR_PENALTY = 0.08


def _update(profile: LearnerProfile, node: str, value: float) -> None:
    m = profile.mastery.setdefault(node, Mastery())
    m.value = value if m.attempts == 0 else (1 - RATE) * m.value + RATE * value
    m.value = round(min(1.0, max(0.0, m.value)), 3)
    m.attempts += 1
    m.updated = now()


@agent("learner_modeler")
def update(session: Session, tk: Toolkit) -> str:
    p, sc, q = session.learner, session.score, session.question
    for c, s in sc.criteria.items():
        _update(p, f"crit.writing.{c}", s.band / 9)
    for node in [q.question_type, *q.topic_ids]:
        if node:
            _update(p, node, sc.overall / 9)
    counts: Counter = Counter()
    for e in session.analysis.errors:
        if e.error_type in tk.kg:
            for g in tk.kg.grammar_for_error(e.error_type):
                counts[g.id] += 1
    for g, n in counts.items():
        m = p.mastery.setdefault(g, Mastery())
        m.value = round(max(0.0, m.value - ERROR_PENALTY * min(n, 3)), 3)
        m.attempts += 1
        m.updated = now()
    p.history.append({"session_id": session.session_id, "task": q.task, "question_source": q.source,
                      "question_type": q.question_type, "overall": sc.overall,
                      "criteria": {c: s.band for c, s in sc.criteria.items()}, "at": now()})
    save_learner(p, tk.data["learners"])
    weakest = sorted(counts, key=lambda g: -counts[g])[:3]
    return f"history {len(p.history)}; grammar to revise: {', '.join(weakest) or '-'}"


def load_learner(learner_id: str, directory: Path) -> LearnerProfile:
    path = directory / f"{learner_id}.json"
    return LearnerProfile.model_validate_json(path.read_text()) if path.exists() else LearnerProfile(learner_id=learner_id)


def save_learner(profile: LearnerProfile, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{profile.learner_id}.json").write_text(profile.model_dump_json(indent=1), encoding="utf-8")


def summary(profile: LearnerProfile) -> dict:
    crit = {k.rsplit(".", 1)[1]: round(v.value * 9, 1) for k, v in profile.mastery.items() if k.startswith("crit.")}
    weak = sorted(((k, v.value) for k, v in profile.mastery.items() if k.startswith("gp.")), key=lambda x: x[1])[:5]
    return {"attempts": len(profile.history), "criteria_band_estimate": crit,
            "weak_grammar": [k for k, _ in weak], "last": profile.history[-3:]}


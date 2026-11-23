"""Planner: choose the next practice target from the learner's mastery (no LLM).

Request keys honoured: task (task1|task2), topic (id or words), question_type, source (real|generate).
Otherwise: the least-practised question type, a topic the learner has done least and scored
lowest on, and the weakest criteria / grammar points as the focus.
"""

from __future__ import annotations

import random

from ..kg.schema import NodeType
from .state import PracticeTarget, Session
from .toolkit import Toolkit, agent


def _mastery(session: Session, node_id: str) -> tuple[int, float]:
    m = session.learner.mastery.get(node_id)
    return (m.attempts, m.value) if m else (0, 0.5)


def _match_topic(tk: Toolkit, wanted: str) -> str | None:
    if wanted in tk.kg:
        return wanted
    words = [w for w in wanted.lower().replace("-", " ").split() if len(w) > 2]
    best, score = None, (0, 0, 0)
    for node in tk.kg.nodes(NodeType.TOPIC):
        name, text = node.name.lower(), node.text().lower()
        key = (sum(2 for w in words if w in name) + sum(1 for w in words if w in text), node.id.count("."), 0)
        if key > score:
            best, score = node.id, key
    return best


@agent("planner")
def plan(session: Session, tk: Toolkit) -> str:
    req = session.request
    rnd = random.Random(session.session_id)
    task = req.get("task", "task2")
    prefix = "qt.writing.t1." if task == "task1" else "qt.writing.t2."

    qtypes = [n.id for n in tk.kg.nodes(NodeType.QUESTION_TYPE) if n.id.startswith(prefix)]
    qtype = req.get("question_type") or min(qtypes, key=lambda q: (_mastery(session, q)[0], rnd.random()))

    if req.get("topic"):
        topic = _match_topic(tk, req["topic"])
    else:
        topics = [n.id for n in tk.kg.nodes(NodeType.TOPIC) if n.id.count(".") == 2]  # specific subtopics
        topic = min(topics, key=lambda t: (_mastery(session, t)[0], _mastery(session, t)[1], rnd.random()))

    crits = [f"crit.writing.{c}" for c in ("TA" if task == "task1" else "TR", "CC", "LR", "GRA")]
    practised = [c for c in crits if _mastery(session, c)[0]]
    focus = sorted(practised, key=lambda c: _mastery(session, c)[1])[:2]
    weak_grammar = sorted((k for k, m in session.learner.mastery.items() if k.startswith("gp.") and m.attempts),
                          key=lambda k: session.learner.mastery[k].value)[:3]
    source = req.get("source") or ("generate" if task == "task1" else "real")  # real Task 1 needs the chart image

    session.target = PracticeTarget(
        exam=session.exam, task=task, question_type=qtype, topic_id=topic,
        focus_criteria=[c.rsplit(".", 1)[1] for c in focus], grammar_focus=weak_grammar, source=source,
        reason=("requested" if req.get("topic") or req.get("question_type") else "least practised")
        + (f"; weakest criteria {', '.join(c.rsplit('.', 1)[1] for c in focus)}" if focus else ""),
    )
    return f"{task} {qtype} on {topic} ({source})"

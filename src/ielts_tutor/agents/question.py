"""Question Generator: pick a real question, or generate a new one from retrieved seeds.

Generated questions are self-checked (design doc 5.3): too similar to a real question
(cosine >= SIMILARITY_LIMIT) or missing the question type's wording -> regenerate, at most
MAX_RETRIES times.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..kg.schema import NodeType
from ..llm import Message
from .state import Question, Session
from .toolkit import Toolkit, agent

SIMILARITY_LIMIT = 0.92
MAX_RETRIES = 2


class _Generated(BaseModel):
    prompt: str = Field(description="The full task wording the candidate sees")
    data_table: str = Field(description="Task 1 only: the chart data as a markdown table; empty for Task 2")
    outline: list[str] = Field(description="3-5 points a Band 7+ answer would cover")


def _done_questions(session: Session) -> set[str]:
    return {h.get("question_source", "") for h in session.learner.history}


def _real(session: Session, tk: Toolkit) -> Question | None:
    t = session.target
    node = tk.kg.get(t.topic_id) if t.topic_id and t.topic_id in tk.kg else None
    topic = " ".join(filter(None, [session.request.get("topic", ""), node.name if node else "",
                                   ", ".join(node.attrs.get("keywords", [])[:8]) if node else ""]))
    qt = tk.kg.get(t.question_type) if t.question_type and t.question_type in tk.kg else None
    query = f"{topic} {qt.name if qt else ''}".strip() or "essay"
    kinds = ["real_exam", "official_sample", "question_bank"]
    seen = _done_questions(session)
    speaking = t.skill == "speaking"
    strict = bool(session.request.get("question_type")) or speaking  # asked for this type / a speaking part
    filt = {"question_type": t.question_type} if strict else {}
    hits = tk.rag.corpus.retrieve(query if strict else (topic or query), 15, skill=t.skill, source_types=kinds, **filt)
    is_t2 = lambda h: (h.meta.get("question_type") or "qt.writing.t2").startswith("qt.writing.t2")  # noqa: E731
    hits = [h for h in hits if h.id not in seen and h.meta.get("module") in (None, "academic")
            and (speaking or is_t2(h) == (t.task == "task2"))]
    if not hits:
        return None
    # The topic decides; among questions about as relevant as the best one, prefer the planned type.
    close = [x for x in hits[:5] if x.score >= 0.85 * hits[0].score]
    h = next((x for x in close if x.meta.get("question_type") == t.question_type), hits[0])
    session.add_evidence([h])
    t.question_type = h.meta.get("question_type") or t.question_type
    min_words, minutes = _length(t.task)
    return Question(question_id=f"q-{h.id}", task=t.task, question_type=t.question_type,
                    topic_ids=h.meta.get("topic_ids") or ([t.topic_id] if t.topic_id else []),
                    prompt=h.text, min_words=min_words, minutes=minutes, source=h.id, evidence_ids=[h.id])


def _length(task: str) -> tuple[int, int]:
    """Minimum words and minutes: writing task limits; for speaking, a sensible answer length."""
    return {"task1": (150, 20), "task2": (250, 40), "part1": (60, 4), "part2": (150, 2), "part3": (120, 5)}[task]


def _cues(tk: Toolkit, qtype: str | None) -> list[str]:
    if qtype and qtype in tk.kg:
        return [c.lower() for c in tk.kg.get(qtype).attrs.get("prompt_cues", [])]
    return []


def _generate(session: Session, tk: Toolkit) -> tuple[Question | None, str]:
    t = session.target
    topic = tk.kg.get(t.topic_id) if t.topic_id and t.topic_id in tk.kg else None
    qt = tk.kg.get(t.question_type) if t.question_type and t.question_type in tk.kg else None
    query = f"{topic.name if topic else ''} {qt.name if qt else ''}"
    seeds = tk.rag.corpus.retrieve(query, 4, skill="writing", question_type=t.question_type, mmr_lambda=0.5) \
        or tk.rag.corpus.retrieve(query, 4, skill="writing", mmr_lambda=0.5)
    topic_ev = tk.rag.topic.retrieve(topic.name if topic else query, 2, skill="writing", expand=False)
    ids = session.add_evidence(seeds + topic_ev)
    cues = _cues(tk, t.question_type)
    task1 = t.task == "task1"
    system = ("You write IELTS Academic Writing questions. Write a NEW question in the official style; "
              "do not copy the examples. " + ("Task 1: describe a chart in words and give its data as a markdown "
                                               "table with realistic numbers (4-6 rows), ask the candidate to "
                                               "summarise the main features." if task1 else
                                               "Task 2: an essay question on the topic."))
    base = (f"Question type: {qt.name if qt else 'any'}\n"
            + (f"Use one of these closing instructions: {cues}\n" if cues and not task1 else "")
            + f"Topic: {topic.text() if topic else query}\n\nExamples of real questions (style only):\n"
            + "\n".join(f"- {e.text[:400]}" for e in seeds)
            + "\n\nTopic notes:\n" + "\n".join(f"- {e.text[:300]}" for e in topic_ev))
    note = ""
    for attempt in range(MAX_RETRIES + 1):
        out = tk.llm.complete_json([Message("user", base + note)], _Generated, system=system)
        sim, nearest = tk.rag.corpus.max_similarity(out.prompt, skill="writing")
        problems = []
        if sim >= SIMILARITY_LIMIT:
            problems.append(f"too close to an existing question ({nearest.id if nearest else '?'}, {sim:.2f})")
        if cues and not any(c in out.prompt.lower() for c in cues):
            problems.append(f"the closing instruction must be one of {cues}")
        if task1 and "|" not in out.data_table:
            problems.append("Task 1 needs the data as a markdown table")
        session.retries["generate"] = attempt
        if not problems:
            return Question(question_id=f"gen-{session.session_id}", task=t.task, question_type=t.question_type,
                            topic_ids=[t.topic_id] if t.topic_id else [], prompt=out.prompt.strip(),
                            data_table=out.data_table.strip() or None, min_words=150 if task1 else 250,
                            minutes=20 if task1 else 40, source="generated", outline=out.outline,
                            evidence_ids=ids), f"generated (attempt {attempt + 1}, similarity {sim:.2f})"
        note = "\n\nYour previous question was rejected: " + "; ".join(problems) + ". Write a different one."
    return None, "generation failed self-check: " + note


@agent("question_generator")
def generate(session: Session, tk: Toolkit) -> str:
    t = session.target
    if t.source == "generate" and tk.llm and t.skill == "writing":
        q, msg = _generate(session, tk)
        if q:
            session.question = q
            return msg
    q = _real(session, tk)
    if q is None:
        raise RuntimeError("no question available: build the indexes or enable the LLM")
    session.question = q
    return f"real question {q.source}"

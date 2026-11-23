"""Response Analyzer: features always; with an LLM, error spans, structure and task notes.

Errors are only kept when the quoted span really occurs in the essay and the error type
is a knowledge-graph ErrorType node, so invented errors are dropped (evidence-driven).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..kg.schema import NodeType
from ..llm import Message
from ..writing.features import extract
from .state import AnalysisReport, ErrorItem, Session, skill_of
from .toolkit import Toolkit, agent

MAX_ERRORS = 25


class _Error(BaseModel):
    text: str = Field(description="Exact wrong span copied from the essay (3-12 words)")
    correction: str
    error_type: str = Field(description="One of the given error type ids")
    explanation: str


class _Analysis(BaseModel):
    errors: list[_Error]
    introduction: str = Field(description="Does it paraphrase the question and state a position/overview?")
    body: str = Field(description="Paragraph roles, topic sentences, development and examples")
    conclusion: str
    task_notes: list[str] = Field(description="Every part of the question answered? On topic? Position clear "
                                              "and consistent? (Task 1: overview, key features, data accuracy)")


def _llm_analysis(session: Session, tk: Toolkit, focus: list[str]) -> tuple[list[ErrorItem], dict, list[str]]:
    q, essay = session.question, session.response.text
    err_nodes = tk.kg.nodes(NodeType.ERROR_TYPE)
    types = "\n".join(f"- {n.id}: {n.name}" for n in err_nodes)
    if skill_of(q.task) == "speaking":
        system = ("You are an experienced IELTS speaking examiner's assistant. The answer is an automatic transcript "
                  "of speech: ignore punctuation and capitalisation, and do not count fillers (um, er) as grammar "
                  "errors. Find grammar and word-choice errors, and describe how the answer opens, develops and "
                  "ends (use the introduction/body/conclusion fields for that) and whether it answers the "
                  "question fully. Quote error spans exactly as transcribed. Do not score.")
        head = f"IELTS Speaking {q.task.replace('part', 'Part ')}:\n{q.prompt}\n\nTranscript:\n<<<\n{essay}\n>>>\n"
    else:
        system = ("You are an experienced IELTS writing examiner's assistant. Find language errors and describe the "
                  "essay's organisation and task response. Quote error spans exactly as written. Do not score.")
        head = (f"IELTS Writing {q.task.upper()} question:\n{q.prompt}\n"
                + (f"\nData:\n{q.data_table}\n" if q.data_table else "")
                + f"\nEssay:\n<<<\n{essay}\n>>>\n")
    user = (head + f"\nError types:\n{types}\n"
            + (f"\nThe scorer asked for a closer look at: {', '.join(focus)}.\n" if focus else "")
            + f"\nList up to {MAX_ERRORS} of the most important errors. Write explanations in "
            + ("Simplified Chinese" if tk.lang == "zh" else "English") + "; keep quoted text and corrections in English.")
    out = tk.llm.complete_json([Message("user", user)], _Analysis, system=system)
    valid = {n.id for n in err_nodes}
    lowered = " ".join(essay.split()).lower()
    errors = []
    for e in out.errors[:MAX_ERRORS]:
        span = " ".join(e.text.split())
        if not span or span.lower() not in lowered:
            continue  # not in the essay: drop
        errors.append(ErrorItem(text=span, correction=e.correction,
                                error_type=e.error_type if e.error_type in valid else "err.other",
                                explanation=e.explanation))
    structure = {"introduction": out.introduction, "body": out.body, "conclusion": out.conclusion}
    return errors, structure, out.task_notes


@agent("response_analyzer")
def analyze(session: Session, tk: Toolkit) -> str:
    q, essay = session.question, session.response.text
    focus = session.request.get("_focus", []) if session.next == "analyze" else []
    errors, structure, notes = [], {}, []
    feats = extract(essay, q.prompt + " " + (q.data_table or ""), q.task, tk.vocab)
    if tk.llm:
        errors, structure, notes = _llm_analysis(session, tk, focus)
        feats.grammar_errors_per_100w = 100 * len(errors) / max(feats.words, 1)
    speaking = skill_of(q.task) == "speaking"
    if feats.words < q.min_words:
        notes.insert(0, f"short answer: {feats.words} words (aim for {q.min_words}+)" if speaking
                     else f"under length: {feats.words} words (minimum {q.min_words})")
    values = {k: round(v, 4) for k, v in zip(feats.names(), feats.vector())}
    if speaking and session.response.transcript:
        from ..speech.asr import Transcript
        from ..speech.fluency import fluency_features

        values.update(fluency_features(Transcript.model_validate(session.response.transcript)))
    session.analysis = AnalysisReport(features=values,
                                      errors=errors, structure=structure, task_notes=notes, focus=focus,
                                      llm_used=bool(tk.llm))
    session.next = None
    return f"{feats.words} words, {len(errors)} errors" + (f", re-check {focus}" if focus else "")

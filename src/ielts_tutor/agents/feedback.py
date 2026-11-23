"""Feedback Coach: concrete, evidence-backed advice (design doc 5.3 "feedback consistency").

Every item must name a criterion and cite at least one evidence id that the session
actually retrieved; items that do not are dropped. Without an LLM, feedback is built from
the analysis (errors, length, linking words) and the next band's descriptor.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import Message
from .state import Feedback, FeedbackItem, Session, skill_of
from .toolkit import Toolkit, agent


class _Item(BaseModel):
    criterion: str
    issue: str
    suggestion: str
    example: str = Field(description="Rewrite one of the student's own sentences; empty if not applicable")
    evidence_ids: list[str]


class _Feedback(BaseModel):
    summary: str
    items: list[_Item]
    next_band_tips: list[str]


def _next_band_evidence(session: Session, tk: Toolkit):
    ev = []
    for c, s in session.score.criteria.items():
        target = int(s.band) + 1
        if target <= 9:
            ev += tk.rag.rubric.descriptors(f"crit.{skill_of(session.question.task)}.{c}", target, window=0)
    q = session.question
    if skill_of(q.task) == "writing":  # model answers and examiner comments exist for writing only
        ev += tk.rag.corpus.retrieve(q.prompt, 2, skill="writing", source_types=["model_answer"])
        ev += tk.rag.rubric.examiner_comments(q.prompt, 2, task=q.task,
                                              around_band=min(9.0, session.score.overall + 1))
    return ev


def _review_nodes(session: Session, tk: Toolkit) -> list[str]:
    nodes: list[str] = []
    for e in session.analysis.errors:
        if e.error_type in tk.kg:
            nodes += [g.id for g in tk.kg.grammar_for_error(e.error_type)]
    nodes += session.question.topic_ids
    if session.question.question_type:
        nodes.append(session.question.question_type)
    return list(dict.fromkeys(nodes))[:8]


def _rule_based(session: Session, ev_ids: dict[str, list[str]]) -> list[FeedbackItem]:
    a, q = session.analysis, session.question
    items = []
    f = a.features
    if skill_of(q.task) == "speaking":
        if f.get("wpm", 999) < 110:
            items.append(FeedbackItem(criterion="FC", issue=f"语速偏慢（每分钟 {f['wpm']:.0f} 词）",
                                      suggestion="先用熟悉的短语把句子说连贯，再逐步加长；7 分左右的语速通常在每分钟 130 词以上。",
                                      evidence_ids=ev_ids.get("FC", [])))
        if f.get("mid_clause_pauses_per_min", 0) > 6:
            items.append(FeedbackItem(criterion="FC", issue="句中停顿较多",
                                      suggestion="停顿尽量放在意群和句子之间，用 well / I mean / what I'd say is 等表达争取思考时间。",
                                      evidence_ids=ev_ids.get("FC", [])))
        if f.get("fillers_per_100w", 0) > 6:
            items.append(FeedbackItem(criterion="FC", issue=f"填充词偏多（每 100 词 {f['fillers_per_100w']:.0f} 个）",
                                      suggestion="把 um / er 换成有意义的过渡语，或者短暂停顿后直接接着说。",
                                      evidence_ids=ev_ids.get("FC", [])))
        for e in a.errors[:3]:
            items.append(FeedbackItem(criterion="GRA", issue=e.explanation, suggestion=f"'{e.text}' → '{e.correction}'",
                                      evidence_ids=ev_ids.get("GRA", [])))
        return items
    if a.features["words"] < q.min_words:
        items.append(FeedbackItem(criterion="TA" if q.task == "task1" else "TR", issue="字数不足",
                                  suggestion=f"至少写 {q.min_words} 词；字数不足会直接压低任务完成分。",
                                  evidence_ids=ev_ids.get("TA" if q.task == "task1" else "TR", [])))
    if a.features["linker_types"] < 4:
        items.append(FeedbackItem(criterion="CC", issue="连接手段单一", suggestion="使用更多样的衔接方式（指代、替换、"
                                  "不同的连接词），并让每段有清楚的主题句。", evidence_ids=ev_ids.get("CC", [])))
    for e in a.errors[:3]:
        items.append(FeedbackItem(criterion="GRA", issue=e.explanation, suggestion=f"'{e.text}' → '{e.correction}'",
                                  evidence_ids=ev_ids.get("GRA", [])))
    return items


@agent("feedback_coach")
def coach(session: Session, tk: Toolkit) -> str:
    evidence = _next_band_evidence(session, tk)
    session.add_evidence(evidence)
    known = {e.id for e in session.evidence}
    by_crit: dict[str, list[str]] = {}
    for e in evidence:
        c = str(e.meta.get("criterion", "")).rsplit(".", 1)[-1]
        if c:
            by_crit.setdefault(c, []).append(e.id)
    for c, s in session.score.criteria.items():
        by_crit.setdefault(c, []).extend(s.evidence_ids)

    if not tk.llm:
        items = _rule_based(session, by_crit)
        if not items:  # nothing tripped a rule: point at the weakest criterion's next band
            weakest = min(session.score.criteria.values(), key=lambda x: x.band)
            nxt = [e for e in evidence if str(e.meta.get("criterion", "")).endswith(f".{weakest.criterion}")]
            if nxt:
                items = [FeedbackItem(criterion=weakest.criterion, issue=f"{weakest.criterion} 是目前最低的一项（{weakest.band}）",
                                      suggestion="对照下一档评分描述逐条检查：" + nxt[0].text[:200],
                                      evidence_ids=[nxt[0].id])]
        summary, tips = f"总分 {session.score.overall}。", []
    else:
        sc = session.score
        speaking = skill_of(session.question.task) == "speaking"
        label = (f"IELTS Speaking {session.question.task.replace('part', 'Part ')}" if speaking
                 else f"IELTS Writing {session.question.task.upper()}")
        fl = session.analysis.features
        if speaking:
            answer = (f"Transcript of the spoken answer:\n<<<\n{session.response.text}\n>>>\n"
                      f"Speech: {fl.get('wpm')} words/min, {fl.get('mid_clause_pauses_per_min')} mid-clause "
                      f"pauses/min, fillers {fl.get('fillers_per_100w')}/100 words.\n")
            measured = f"Measured: {int(fl['words'])} words (aim {session.question.min_words}+)"
        else:
            answer = f"Essay:\n<<<\n{session.response.text}\n>>>\n"
            measured = (f"Measured: {int(fl['words'])} words (minimum {session.question.min_words}), "
                        f"{int(fl['paragraphs'])} paragraphs")
        parts = [
            f"{label} question:\n{session.question.prompt}\n",
            answer,
            measured + "; use these numbers, do not estimate.",
            f"Scores: overall {sc.overall}; " + "; ".join(f"{c} {x.band} ({x.rationale[:300]})"
                                                         for c, x in sc.criteria.items()),
            "Errors found:\n" + "\n".join(f"- '{e.text}' -> '{e.correction}' [{e.error_type}]"
                                           for e in session.analysis.errors[:15]),
            "Evidence (next-band descriptors, model answers, examiner comments):\n"
            + "\n".join(f"[{e.id}] {e.text[:500]}" for e in evidence),
            f"Give 3-6 feedback items, most important first. Each item: the criterion ({'/'.join(sc.criteria)}), "
            "the problem in this answer, an actionable suggestion, a rewrite of one of the student's own sentences "
            "when useful, and the evidence ids that support it (required). Then 2-3 tips on what the next band up "
            "requires. Write in " + ("Simplified Chinese; keep English examples in English." if tk.lang == "zh"
                                     else "English."),
        ]
        user = "\n\n".join(parts)
        out = tk.llm.complete_json([Message("user", user)], _Feedback,
                                   system=f"You are an IELTS {'speaking' if speaking else 'writing'} coach. "
                                          "Be specific to this answer; no generic advice.")
        items = [FeedbackItem(criterion=i.criterion.upper().replace("CRIT.WRITING.", "").replace("CRIT.SPEAKING.", ""),
                              issue=i.issue,
                              suggestion=i.suggestion, example=i.example,
                              evidence_ids=[x for x in i.evidence_ids if x in known]) for i in out.items]
        summary, tips = out.summary, out.next_band_tips
    kept = [i for i in items if i.evidence_ids and i.criterion in session.score.criteria]
    session.feedback = Feedback(summary=summary, items=kept, next_band_tips=tips,
                                review_nodes=_review_nodes(session, tk), dropped=len(items) - len(kept))
    return f"{len(kept)} items ({len(items) - len(kept)} dropped without evidence)"

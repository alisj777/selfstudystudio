"""Scorer: model + k-NN + LLM judge, fused per criterion, with review (design doc 5.3 / 6.1).

  band_c = w_model * model_c + w_kNN * kNN_c + w_LLM * LLM_c

Default weights (Task 2): model 0.5, LLM 0.5, k-NN 0. Weights fitted on the btnotpt labels
gave the LLM almost no weight, but those labels look machine-made, while against real
examiners' bands the LLM did no worse than the model (README, M2). Fitted weights are only
used if written to var/models/fusion.json (eval-scorer --llm N --apply-weights).

If the LLM and the model disagree by more than REVIEW_GAP on any criterion, the scorer asks
the analyzer for a closer look once (session.next = "analyze"); if they still disagree the
score card is marked for human review. Task 1 has no trained model yet, so it leans on the
LLM judge and is always flagged as low-confidence.
"""

from __future__ import annotations

import json

from ..writing.judge import judge
from ..writing.scorer import overall_band, round_half
from .state import CriterionScore, ScoreCard, Session, criteria_for, skill_of
from .toolkit import Toolkit, agent

REVIEW_GAP = 1.0
DEFAULT_WEIGHTS = (0.5, 0.0, 0.5)  # model, k-NN, LLM (chosen 2026-11-23, see module docstring)


def _criteria(task: str) -> list[str]:
    return criteria_for(task)


def fusion_weights(tk: Toolkit, criterion: str, task: str, has_llm: bool) -> tuple[float, float, float]:
    path = tk.settings.index_dir.parent / "models" / "fusion.json"
    if path.exists():
        fitted = json.loads(path.read_text()).get(task, {}).get(criterion)
        if fitted and (has_llm or fitted[2] == 0):
            return tuple(fitted)  # type: ignore[return-value]
    if task == "task1":  # no trained Task 1 model yet
        return (0.0, 0.0, 1.0) if has_llm else (1.0, 0.0, 0.0)
    if not has_llm:
        return (1.0, 0.0, 0.0)
    return DEFAULT_WEIGHTS


def _notes(session: Session) -> str:
    a = session.analysis
    f = a.features
    lines = [f"words {int(f['words'])}, paragraphs {int(f['paragraphs'])}, mean sentence length "
             f"{f['mean_sentence_len']:.1f}, MTLD {f['mtld']:.0f}, linker types {int(f['linker_types'])}"]
    if "wpm" in f:
        lines.append(f"speech: {f['duration_s']} s, {f['wpm']} words/min, {f['pauses_per_min']} pauses/min "
                     f"({f['mid_clause_pauses_per_min']} mid-clause), {f['long_pauses']} pauses over 1 s, "
                     f"fillers {f['fillers_per_100w']}/100 words, repetitions {f['repetitions_per_100w']}/100 words, "
                     f"mean run {f['mean_run_words']} words")
    lines += [f"structure: {k}: {v}" for k, v in a.structure.items()]
    lines += [f"task: {n}" for n in a.task_notes]
    lines += [f"error: '{e.text}' -> '{e.correction}' ({e.error_type})" for e in a.errors[:15]]
    return "\n".join(lines)


# Speaking has no trained model yet: (rule-based, LLM) weights per criterion. The LLM reads only
# the transcript, so Pronunciation comes from the ASR-confidence rule alone.
SPEAKING_WEIGHTS = {"FC": (0.3, 0.7), "LR": (0.0, 1.0), "GRA": (0.0, 1.0), "P": (1.0, 0.0)}


def _score_speaking(session: Session, tk: Toolkit) -> str:
    from ..speech.fluency import rule_bands

    q, text, f = session.question, session.response.text, session.analysis.features
    crits = criteria_for(q.task)
    rules = rule_bands(f) if "wpm" in f else {}
    if tk.scorer is not None:  # rough LR/GRA proxy from the writing model when there is no LLM
        pred = tk.scorer.predict(text, q.prompt, "task2", tk.cache, tk.vocab)
        rules.setdefault("LR", round_half(pred.model["LR"]))
        rules.setdefault("GRA", round_half(pred.model["GRA"]))
    base = {c: round_half(rules.get(c, 6.0)) for c in crits}
    evidence = [e for c in crits for e in tk.rag.rubric.descriptors(f"crit.speaking.{c}", base[c], window=1)]
    session.add_evidence(evidence)
    llm = {}
    if tk.llm:
        llm = judge(tk.llm, q.task, q.prompt, text, ["FC", "LR", "GRA"], {c: base[c] for c in ("FC", "LR", "GRA")},
                    evidence, _notes(session), tk.lang)
    out = {}
    for c in crits:
        wr, wl = SPEAKING_WEIGHTS[c]
        parts = [(wr, rules.get(c)), (wl, llm[c].band if c in llm else None)]
        parts = [(w, v) for w, v in parts if v is not None and w > 0] or [(1.0, rules.get(c, 6.0))]
        band = round_half(sum(w * v for w, v in parts) / sum(w for w, _ in parts))
        j = llm.get(c)
        ids = j.evidence_ids if j and j.evidence_ids else [e.id for e in evidence if e.meta.get("criterion") ==
                                                           f"crit.speaking.{c}" and e.meta.get("band") == int(band)][:1]
        out[c] = CriterionScore(criterion=c, band=band, model=rules.get(c), llm=j.band if j else None,
                                rationale=j.rationale if j else ("由识别置信度估计" if c == "P" else
                                                                 "由语速、停顿、填充词估计" if c == "FC" else ""),
                                evidence_ids=ids)
    session.score = ScoreCard(
        criteria=out, overall=overall_band({"TR": out["FC"].band, "CC": out["LR"].band, "LR": out["GRA"].band,
                                            "GRA": out["P"].band}),
        model_version="speaking-rules-0" + ("+llm" if llm else ""),
        weights={c: [SPEAKING_WEIGHTS[c][0], 0.0, SPEAKING_WEIGHTS[c][1]] for c in crits},
        needs_review=True,
        review_reason="口语还没有训练好的评分模型：流利度和发音由规则估计（发音只看识别置信度），词汇和语法由 LLM 根据转写稿判断",
    )
    return f"overall {session.score.overall} " + " ".join(f"{c}={x.band}" for c, x in out.items()) + " (needs review)"


@agent("scorer")
def score(session: Session, tk: Toolkit) -> str:
    q, essay = session.question, session.response.text
    if skill_of(q.task) == "speaking":
        return _score_speaking(session, tk)
    crits = _criteria(q.task)
    model, knn, interval, neighbours = {}, {}, {}, []
    if tk.scorer is not None:
        pred = tk.scorer.predict(essay, q.prompt, q.task, tk.cache, tk.vocab)
        for c in crits:
            src = "TR" if c == "TA" else c
            model[c], knn[c], interval[c] = pred.model[src], pred.knn[src], pred.interval[src]
        neighbours = pred.neighbours
    base = {c: round_half((model[c] + knn[c]) / 2) if c in model else 6.0 for c in crits}

    as4 = {("TR" if k == "TA" else k): v for k, v in base.items()}
    evidence = []
    for c in crits:
        evidence += tk.rag.rubric.descriptors(f"crit.writing.{c}", base[c], window=1)
    evidence += tk.rag.rubric.similar_samples(essay, 3, skill="writing", task=q.task)
    evidence += tk.rag.rubric.examiner_comments(q.prompt, 2, task=q.task, around_band=overall_band(as4))
    session.add_evidence(evidence)
    llm_scores = {}
    if tk.llm:
        llm_scores = judge(tk.llm, q.task, q.prompt + ("\n" + q.data_table if q.data_table else ""), essay, crits,
                           base, evidence, _notes(session), tk.lang)

    out, gaps = {}, []
    for c in crits:
        wm, wk, wl = fusion_weights(tk, c, q.task, c in llm_scores)
        parts = [(wm, model.get(c)), (wk, knn.get(c)), (wl, llm_scores[c].band if c in llm_scores else None)]
        parts = [(w, v) for w, v in parts if v is not None and w > 0]
        band = round_half(sum(w * v for w, v in parts) / sum(w for w, _ in parts)) if parts else 6.0
        j = llm_scores.get(c)
        if j and c in model and abs(j.band - (model[c] + knn[c]) / 2) > REVIEW_GAP:
            gaps.append(c)
        if j:
            ids = j.evidence_ids
        else:  # no judge: cite the descriptor of the awarded band and the nearest scored essays
            ids = [e.id for e in evidence if e.meta.get("criterion") == f"crit.writing.{c}"
                   and e.meta.get("band") == int(band)][:1] + [n for n, _ in neighbours[:2]]
        out[c] = CriterionScore(criterion=c, band=band, model=model.get(c), knn=knn.get(c),
                                llm=j.band if j else None, interval=interval.get(c),
                                rationale=j.rationale if j else "", evidence_ids=ids)

    if gaps and tk.llm and session.retries.get("analyze", 0) < 1:
        session.retries["analyze"] = session.retries.get("analyze", 0) + 1
        session.request["_focus"] = gaps
        session.next = "analyze"
        return f"model and LLM disagree on {gaps}: asking the analyzer to re-check"

    vals = {("TR" if c == "TA" else c): s.band for c, s in out.items()}
    review = bool(gaps) or q.task == "task1"
    session.score = ScoreCard(
        criteria=out, overall=overall_band(vals),
        model_version=getattr(tk.scorer, "version", "none") + ("+llm" if llm_scores else ""),
        weights={c: list(fusion_weights(tk, c, q.task, c in llm_scores)) for c in crits},
        needs_review=review,
        review_reason=(f"model and LLM still disagree on {', '.join(gaps)}" if gaps else
                       "Task 1 has no trained score model yet" if q.task == "task1" else ""),
    )
    return f"overall {session.score.overall} " + " ".join(f"{c}={s.band}" for c, s in out.items()) \
        + (" (needs review)" if review else "")

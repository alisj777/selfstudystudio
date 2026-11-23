"""LLM judge: an independent band per criterion, explained with cited evidence (design doc 6.1 step 5).

The judge sees the band descriptors near the model's estimate, similar scored essays and
official examiner comments. It must cite evidence ids for each criterion; citations that
are not in the evidence given are removed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import Message
from ..rag.schema import Evidence


class _Judged(BaseModel):
    criterion: str
    band: float = Field(description="0-9 in steps of 0.5")
    rationale: str = Field(description="Why this band and not the next one up, quoting the essay")
    evidence_ids: list[str] = Field(description="Ids of the descriptors / samples this judgement relies on")


class JudgeOutput(BaseModel):
    criteria: list[_Judged]


SYSTEM = ("You are a certified IELTS writing examiner. Score each criterion against the official band "
          "descriptors provided. Judge independently: the model estimate is only a starting point. "
          "Bands are 0-9 in steps of 0.5. Cite the evidence ids you relied on.")


def judge(llm, task: str, prompt: str, essay: str, criteria: list[str], estimate: dict[str, float],
          evidence: list[Evidence], analysis_notes: str = "", lang: str = "zh") -> dict[str, _Judged]:
    ev_text = "\n".join(f"[{e.id}] {e.text[:600]}" for e in evidence)
    user = (f"IELTS Writing {task.upper()}\nQuestion:\n{prompt}\n\nEssay:\n<<<\n{essay}\n>>>\n\n"
            f"Model estimate (may be wrong): {estimate}\n\nAnalysis notes:\n{analysis_notes}\n\n"
            f"Evidence (band descriptors, scored samples, examiner comments):\n{ev_text}\n\n"
            f"Score these criteria: {', '.join(criteria)}. Write each rationale in "
            + ("Simplified Chinese, quoting the essay in English." if lang == "zh" else "English."))
    out = llm.complete_json([Message("user", user)], JudgeOutput, system=SYSTEM)
    known = {e.id for e in evidence}
    result = {}
    for j in out.criteria:
        c = j.criterion.upper().replace("CRIT.WRITING.", "")
        if c in criteria:
            j.band = float(min(9.0, max(0.0, round(j.band * 2) / 2)))
            j.evidence_ids = [i for i in j.evidence_ids if i in known]
            result[c] = j
    return result

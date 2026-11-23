import json
import re

import pytest

pytest.importorskip("langgraph")

from ielts_tutor.agents.orchestrator import WritingTutor
from ielts_tutor.agents.toolkit import CountingLLM, Toolkit
from ielts_tutor.config import get_settings
from ielts_tutor.importers.textquality import Vocab
from ielts_tutor.llm import FakeLLM
from ielts_tutor.rag import HashingEmbedder, ScoredSample
from ielts_tutor.writing.scorer import EmbeddingCache, WritingScorer

ESSAY = ("Some people believe that online learning is better than studying at school. In my opinion, online "
         "courses is useful, but they can not replace teachers.\n\nFirstly, online learning is flexible. Students "
         "can study at home and save time. However, they may feel lonely because they do not meet classmates.\n\n"
         "In conclusion, I think online learning should be combined with classroom teaching for best results.")


def _essay(band: float, i: int) -> ScoredSample:
    words = ["moreover", "however", "consequently", "nevertheless", "flexibility", "collaboration"]
    text = " ".join(["Students learn online at home and it is good."] * (3 + int(band))
                    + words[: int(band) - 3] + [f"essay{i}"])
    return ScoredSample(sample_id=f"syn-{i}", skill="writing", task="task2", prompt_text="Online learning",
                        response_text=text, scores={c: band for c in ("TR", "CC", "LR", "GRA", "overall")})


@pytest.fixture(scope="module")
def scorer_parts():
    vocab = Vocab({w for w in ESSAY.lower().replace(".", " ").replace(",", " ").split()})
    cache = EmbeddingCache(HashingEmbedder(), None)
    scorer = WritingScorer(k=5)
    scorer.fit([_essay(b, i) for i, b in enumerate([5, 5.5, 6, 6.5, 7, 7.5, 8] * 4)], cache, vocab, folds=3)
    return scorer, cache, vocab


class Script:
    """FakeLLM responder: answers each agent by its system prompt; knobs change the behaviour."""

    def __init__(self, judge_bands=(6.0, 6.0, 6.0, 6.0), copy_first_question=None):
        self.judge_bands = list(judge_bands)
        self.copy_first = copy_first_question
        self.question_calls = 0
        self.judge_calls = 0

    def __call__(self, messages, system):
        user = messages[-1].content
        ids = re.findall(r"^\[([^\]]+)\]", user, re.M)
        if "write IELTS Academic Writing questions" in system:
            self.question_calls += 1
            prompt = self.copy_first if self.question_calls == 1 and self.copy_first else (
                "Some say robots will run every hospital by 2050. To what extent do you agree or disagree?")
            return {"prompt": prompt, "data_table": "", "outline": ["position", "examples"]}
        if "examiner's assistant" in system:
            return {"errors": [
                {"text": "online courses is useful", "correction": "online courses are useful",
                 "error_type": "err.verb_sva", "explanation": "主谓一致"},
                {"text": "this sentence is not in the essay", "correction": "x", "error_type": "err.det",
                 "explanation": "invented"},
                {"text": "can not", "correction": "cannot", "error_type": "err.made_up", "explanation": "拼写"}],
                "introduction": "clear position", "body": "two ideas", "conclusion": "restates",
                "task_notes": ["position clear"]}
        if "certified IELTS" in system:
            self.judge_calls += 1
            crits = re.search(r"Score these criteria: ([A-Z, ]+)\.", user).group(1).split(", ")
            return {"criteria": [{"criterion": c, "band": b, "rationale": f"{c} reason",
                                  "evidence_ids": ids[:1] + ["not-an-id"]} for c, b in zip(crits, self.judge_bands)]}
        if "coach" in system:
            return {"summary": "整体清楚", "next_band_tips": ["develop ideas"], "items": [
                {"criterion": "GRA", "issue": "主谓一致", "suggestion": "check verbs", "example": "courses are",
                 "evidence_ids": ids[:1]},
                {"criterion": "CC", "issue": "no evidence", "suggestion": "x", "example": "", "evidence_ids": ["nope"]}]}
        raise AssertionError(f"unexpected prompt: {system}")


def _tutor(tmp_path, kg, rag, scorer_parts, llm=True, script=None):
    scorer, cache, vocab = scorer_parts
    settings = get_settings().model_copy(update={"index_dir": tmp_path / "indexes"})
    client = CountingLLM(FakeLLM(script or Script())) if llm else None
    tk = Toolkit(kg=kg, rag=rag, llm=client, scorer=scorer, cache=cache, vocab=vocab, settings=settings,
                 data={"sessions": tmp_path / "sessions", "learners": tmp_path / "learners"})
    return WritingTutor(tk)


def test_start_picks_a_real_question_without_llm(tmp_path, kg, rag, scorer_parts):
    tutor = _tutor(tmp_path, kg, rag, scorer_parts, llm=False)
    s = tutor.start("amy", task="task2", topic="online education")
    assert s.question.source != "generated" and s.question.prompt
    assert s.target.topic_id.startswith("topic.") and [t.agent for t in s.trace] == ["planner", "question_generator"]
    assert (tmp_path / "sessions" / f"{s.session_id}.json").exists()


def test_generated_question_too_close_to_a_real_one_is_regenerated(tmp_path, kg, rag, scorer_parts, corpus):
    real = next(c.text for c in corpus if c.skill == "writing")
    script = Script(copy_first_question=real)
    s = _tutor(tmp_path, kg, rag, scorer_parts, script=script).start(task="task2", source="generate",
                                                                    question_type="qt.writing.t2.opinion")
    assert script.question_calls == 2 and s.question.source == "generated"
    assert "agree or disagree" in s.question.prompt and s.retries["generate"] == 1


def test_full_chain_scores_filters_and_updates_learner(tmp_path, kg, rag, scorer_parts):
    tutor = _tutor(tmp_path, kg, rag, scorer_parts)
    s = tutor.start("amy", question="Online learning is better than school. Discuss.")
    s = tutor.submit(s.session_id, ESSAY)
    assert [t.agent for t in s.trace] == ["response_analyzer", "scorer", "feedback_coach", "learner_modeler"]
    # only the error whose span is in the essay survives; an unknown type becomes err.other
    assert [(e.text, e.error_type) for e in s.analysis.errors] == [
        ("online courses is useful", "err.verb_sva"), ("can not", "err.other")]
    sc = s.score
    assert set(sc.criteria) == {"TR", "CC", "LR", "GRA"} and sc.criteria["TR"].llm == 6.0
    assert all("not-an-id" not in c.evidence_ids for c in sc.criteria.values())
    assert sc.overall in {x / 2 for x in range(0, 19)}
    assert len(s.feedback.items) == 1 and s.feedback.dropped == 1  # the item citing "nope" is dropped
    assert "gp.svagreement" in s.feedback.review_nodes
    profile = json.loads((tmp_path / "learners" / "amy.json").read_text())
    assert profile["history"][0]["overall"] == sc.overall and "crit.writing.TR" in profile["mastery"]
    with pytest.raises(ValueError):
        tutor.submit(s.session_id, ESSAY)  # already scored


def test_disagreement_triggers_one_reanalysis_then_review(tmp_path, kg, rag, scorer_parts):
    script = Script(judge_bands=(9.0, 9.0, 9.0, 9.0))  # far from the model's estimate
    tutor = _tutor(tmp_path, kg, rag, scorer_parts, script=script)
    s = tutor.submit(tutor.start(question="Discuss online learning.").session_id, ESSAY)
    agents = [t.agent for t in s.trace]
    assert agents.count("response_analyzer") == 2 and agents.count("scorer") == 2 and script.judge_calls == 2
    assert s.score.needs_review and "disagree" in s.score.review_reason


def test_chain_without_llm_uses_model_and_rule_based_feedback(tmp_path, kg, rag, scorer_parts):
    tutor = _tutor(tmp_path, kg, rag, scorer_parts, llm=False)
    s = tutor.submit(tutor.start(question="Discuss online learning.").session_id, ESSAY)
    assert all(c.llm is None and c.model is not None for c in s.score.criteria.values())
    assert s.feedback.items and all(i.evidence_ids for i in s.feedback.items)
    assert sum(t.llm_calls for t in s.trace) == 0


def test_web_api_runs_a_session(tmp_path, kg, rag, scorer_parts):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ielts_tutor.web.app import create_app

    client = TestClient(create_app(tutor=_tutor(tmp_path, kg, rag, scorer_parts)))
    assert "雅思自学练习" in client.get("/").text
    opts = client.get("/api/options").json()
    assert opts["llm"] and any(q["task"] == "task1" for q in opts["question_types"])
    s = client.post("/api/sessions", json={"learner": "web", "question": "Discuss online learning."}).json()
    assert client.post(f"/api/sessions/{s['session_id']}/submit", json={"text": "too short"}).status_code == 400
    done = client.post(f"/api/sessions/{s['session_id']}/submit", json={"text": ESSAY, "seconds": 600}).json()
    assert done["score"]["overall"] > 0 and done["feedback"]["items"]
    assert client.post(f"/api/sessions/{s['session_id']}/submit", json={"text": ESSAY}).status_code == 409
    assert client.get("/api/learners/web").json()["attempts"] == 1
    assert client.get("/api/sessions/nope").status_code == 404


def test_default_fusion_is_half_model_half_llm(tmp_path, kg, rag, scorer_parts):
    from ielts_tutor.agents.scorer import fusion_weights
    from ielts_tutor.writing.scorer import round_half

    tutor = _tutor(tmp_path, kg, rag, scorer_parts)
    assert fusion_weights(tutor.tk, "TR", "task2", True) == (0.5, 0.0, 0.5)
    assert fusion_weights(tutor.tk, "TR", "task2", False) == (1.0, 0.0, 0.0)
    assert fusion_weights(tutor.tk, "TA", "task1", True) == (0.0, 0.0, 1.0)
    s = tutor.submit(tutor.start(question="Discuss online learning.").session_id, ESSAY)
    for c in s.score.criteria.values():
        assert c.band == round_half(0.5 * c.model + 0.5 * c.llm)


def test_speaking_chain_from_a_recording(tmp_path, kg, rag, scorer_parts):
    from ielts_tutor.speech.asr import Transcript, Word

    words, t = [], 0.0
    for i, w in enumerate(ESSAY.replace("\n", " ").split()[:60]):
        t += 0.9 if i % 12 == 0 else 0.05  # a pause every 12 words
        words.append(Word(text=w, start=t, end=t + 0.3, prob=0.9))
        t += 0.3
    fake = Transcript(text=" ".join(w.text for w in words), words=words, duration=t + 0.5)
    tutor = _tutor(tmp_path, kg, rag, scorer_parts)
    s = tutor.start("sam", task="part2", question="Describe a teacher who helped you. You should say ...")
    with pytest.raises(ValueError):
        _tutor(tmp_path, kg, rag, scorer_parts).submit_audio(
            tutor.start(question="Discuss online learning.").session_id, b"x" * 2000)  # a writing session
    s = tutor.submit_audio(s.session_id, b"RIFF" + b"0" * 2000, ".wav", transcriber=lambda p: fake)
    assert set(s.score.criteria) == {"FC", "LR", "GRA", "P"} and s.score.needs_review
    f = s.analysis.features
    assert f["spoken_words"] == 60 and f["wpm"] > 0 and f["pauses_per_min"] > 0
    assert s.score.criteria["P"].llm is None and s.score.criteria["P"].model is not None  # rule only
    assert s.score.criteria["LR"].llm == 6.0
    assert (tmp_path / "audio" / f"{s.session_id}.wav").exists()
    assert json.loads((tmp_path / "learners" / "sam.json").read_text())["mastery"]["crit.speaking.FC"]

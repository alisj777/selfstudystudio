"""Orchestrator: the writing loop as a LangGraph state graph (design doc 5).

A session runs in two halves, because the student writes in between:

  start:   planner -> question_generator                                  (returns the question)
  submit:  response_analyzer -> scorer -(disagree, once)-> response_analyzer -> scorer
                                       -> feedback_coach -> learner_modeler

Agents share only the Session object; an agent asks for an upstream agent to run again by
setting `session.next` (the scorer does this to request a closer analysis).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .analyzer import analyze
from .feedback import coach
from .learner import load_learner, update
from .planner import plan
from .question import generate
from .scorer import score
from .state import Response, Session
from .toolkit import Toolkit, load_session, save_session


class GraphState(TypedDict):
    session: Session


def build_graph(tk: Toolkit):
    def node(fn):
        return lambda state: {"session": fn(state["session"], tk)}

    g = StateGraph(GraphState)
    for name, fn in [("planner", plan), ("question_generator", generate), ("response_analyzer", analyze),
                     ("scorer", score), ("feedback_coach", coach), ("learner_modeler", update)]:
        g.add_node(name, node(fn))
    g.add_conditional_edges(START, lambda s: "response_analyzer" if s["session"].response else "planner",
                            ["planner", "response_analyzer"])
    g.add_edge("planner", "question_generator")
    g.add_edge("question_generator", END)
    g.add_edge("response_analyzer", "scorer")
    g.add_conditional_edges("scorer", lambda s: "response_analyzer" if s["session"].next == "analyze"
                            else "feedback_coach", ["response_analyzer", "feedback_coach"])
    g.add_edge("feedback_coach", "learner_modeler")
    g.add_edge("learner_modeler", END)
    return g.compile()


class WritingTutor:
    """Entry point used by the CLI and the web API: writing tasks and speaking parts."""

    def __init__(self, tk: Toolkit):
        self.tk = tk
        self.graph = build_graph(tk)

    def start(self, learner_id: str = "default", **request) -> Session:
        session = Session(learner=load_learner(learner_id, self.tk.data["learners"]), request=request)
        if request.get("question"):  # the student brought their own question
            from .question import _length
            from .state import PracticeTarget, Question, skill_of

            task = request.get("task", "task2")
            min_words, minutes = _length(task)
            session.target = PracticeTarget(task=task, skill=skill_of(task), source="real",
                                            reason="student's own question")
            session.question = Question(question_id=f"own-{session.session_id}", task=task, prompt=request["question"],
                                        min_words=min_words, minutes=minutes, source="student")
        else:
            session = self.graph.invoke({"session": session})["session"]
        save_session(session, self.tk.data["sessions"])
        return session

    def submit(self, session_id: str, text: str, seconds: int | None = None, **extra) -> Session:
        session = load_session(session_id, self.tk.data["sessions"])
        if session.score:
            raise ValueError(f"session {session_id} was already scored")
        session.learner = load_learner(session.learner.learner_id, self.tk.data["learners"])
        session.response = Response(text=text, seconds_spent=seconds, **extra)
        session = self.graph.invoke({"session": session})["session"]
        save_session(session, self.tk.data["sessions"])
        return session

    def submit_audio(self, session_id: str, audio: bytes, suffix: str = ".webm", seconds: int | None = None,
                     transcriber=None) -> Session:
        """Speaking: keep the recording, transcribe it on this machine, then run the same chain."""
        session = load_session(session_id, self.tk.data["sessions"])
        if session.score:
            raise ValueError(f"session {session_id} was already scored")
        if not session.question or not session.question.task.startswith("part"):
            raise ValueError("not a speaking session")
        path = self.tk.data["sessions"].parent / "audio" / f"{session_id}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        if transcriber is None:
            from ..speech.asr import transcribe

            def transcriber(p):
                return transcribe(p, self.tk.settings.asr_model)
        t = transcriber(path)
        if not t.words:
            raise ValueError("no speech recognised in the recording")
        return self.submit(session_id, t.text, seconds or int(t.duration), audio_path=str(path),
                           transcript=t.model_dump())

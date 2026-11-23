"""Local web app: a FastAPI JSON API over WritingTutor plus one static page (static/index.html)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

STATIC = Path(__file__).parent / "static"


class StartRequest(BaseModel):
    learner: str = "default"
    task: str = "task2"
    topic: str | None = None
    question_type: str | None = None
    source: str | None = None
    question: str | None = None


class SubmitRequest(BaseModel):
    text: str
    seconds: int | None = None


def create_app(use_llm: bool = True, tutor=None) -> FastAPI:
    from ..agents.learner import load_learner, summary
    from ..agents.toolkit import load_session

    if tutor is None:
        from ..agents.orchestrator import WritingTutor
        from ..agents.toolkit import Toolkit

        tutor = WritingTutor(Toolkit.load(use_llm=use_llm))
        # The first search builds the keyword indexes (~25 s); pay that at startup, not on a student's essay.
        tutor.tk.rag.rubric.examiner_comments("warm up", 1)
        tutor.tk.rag.corpus.retrieve("warm up", 1)
    app = FastAPI(title="IELTS writing tutor")
    dirs = tutor.tk.data

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/options")
    def options():
        from ..kg.schema import NodeType

        kg = tutor.tk.kg
        return {
            "topics": [{"id": n.id, "name": n.name} for n in kg.nodes(NodeType.TOPIC) if n.id.count(".") == 2],
            "question_types": [{"id": n.id, "name": n.name, "task": n.attrs.get("task")}
                               for n in kg.nodes(NodeType.QUESTION_TYPE) if n.id.startswith("qt.writing.")],
            "llm": tutor.tk.llm is not None,
        }

    @app.post("/api/sessions")
    def start(req: StartRequest):
        params = {k: v for k, v in req.model_dump().items() if v and k != "learner"}
        try:
            return tutor.start(req.learner, **params).model_dump()
        except Exception as exc:  # surfaced to the page
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/sessions/{session_id}/submit")
    def submit(session_id: str, req: SubmitRequest):
        if len(req.text.split()) < 20:
            raise HTTPException(400, "essay too short to score")
        try:
            return tutor.submit(session_id, req.text, req.seconds).model_dump()
        except FileNotFoundError as exc:
            raise HTTPException(404, "unknown session") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        try:
            return load_session(session_id, dirs["sessions"]).model_dump()
        except FileNotFoundError as exc:
            raise HTTPException(404, "unknown session") from exc

    @app.get("/api/learners/{learner_id}")
    def learner(learner_id: str):
        return summary(load_learner(learner_id, dirs["learners"]))

    return app

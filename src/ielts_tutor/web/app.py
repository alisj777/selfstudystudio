"""Local web app: a FastAPI JSON API over WritingTutor plus one static page (static/index.html)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
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


class AnswersRequest(BaseModel):
    learner: str = "default"
    answers: dict[int, str]
    seconds: int | None = None


def create_app(use_llm: bool = True, tutor=None, practice=None) -> FastAPI:
    from ..agents.learner import load_learner, summary
    from ..agents.toolkit import load_session

    if tutor is None:
        from ..agents.orchestrator import WritingTutor
        from ..agents.toolkit import Toolkit

        tutor = WritingTutor(Toolkit.load(use_llm=use_llm))
        # The first search builds the keyword indexes (~25 s); pay that at startup, not on a student's essay.
        tutor.tk.rag.rubric.examiner_comments("warm up", 1)
        tutor.tk.rag.corpus.retrieve("warm up", 1)
    if practice is None:
        from ..grading.practice import PracticeService

        st = tutor.tk.settings
        practice = PracticeService(st.exam_dir, st.index_dir.parent)
    app = FastAPI(title="IELTS tutor")
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

    @app.post("/api/sessions/{session_id}/audio")
    async def submit_audio(session_id: str, request: Request, seconds: int | None = None):
        """Speaking: the raw recording as the request body (audio/webm, audio/mp4, audio/wav ...)."""
        body = await request.body()
        if len(body) < 1000:
            raise HTTPException(400, "recording too short")
        if len(body) > 50_000_000:
            raise HTTPException(413, "recording too large")
        kind = request.headers.get("content-type", "audio/webm").split(";")[0]
        suffix = {"audio/webm": ".webm", "audio/mp4": ".m4a", "audio/mpeg": ".mp3", "audio/wav": ".wav",
                  "audio/x-wav": ".wav", "audio/ogg": ".ogg", "audio/aiff": ".aiff"}.get(kind, ".webm")
        try:
            return tutor.submit_audio(session_id, body, suffix, seconds).model_dump()
        except FileNotFoundError as exc:
            raise HTTPException(404, "unknown session") from exc
        except ValueError as exc:
            raise HTTPException(409 if "already" in str(exc) else 400, str(exc)) from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        try:
            return load_session(session_id, dirs["sessions"]).model_dump()
        except FileNotFoundError as exc:
            raise HTTPException(404, "unknown session") from exc

    @app.get("/api/learners/{learner_id}")
    def learner(learner_id: str):
        return summary(load_learner(learner_id, dirs["learners"]))

    # ---- Listening and Reading practice (Cambridge books)
    @app.get("/api/practice/tests")
    def practice_tests(learner: str = "default"):
        return practice.catalogue(learner)

    @app.get("/api/practice/{test_id}/page/{index}")
    def practice_page(test_id: str, index: int):
        try:
            return FileResponse(practice.page_png(test_id, index), media_type="image/png")
        except (KeyError, IndexError) as exc:
            raise HTTPException(404, "no such page") from exc

    @app.get("/api/practice/{test_id}/audio/{part}")
    def practice_audio(test_id: str, part: int):
        try:
            path = practice.audio_file(test_id, part)
        except KeyError as exc:
            raise HTTPException(404, "no such audio") from exc
        return FileResponse(path, media_type="audio/mpeg")

    @app.post("/api/practice/{test_id}/submit")
    def practice_submit(test_id: str, req: AnswersRequest):
        try:
            return practice.mark(test_id, req.answers, req.learner, req.seconds)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app

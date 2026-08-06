"""FastAPI surface.

`Deps` bundles the three collaborators the routes need (store, llm, and the two
pipeline entry points) behind one injectable object, so tests swap the whole
pipeline for a fake without patching module globals.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import DB_PATH
from app.lesson.generate import generate_lesson_plan
from app.llm import LlmClient
from app.models import LessonPlan, ResearchBundle
from app.qa.answer import answer_question, grade_answer
from app.research.orchestrator import gather_research
from app.research.reflect import reflect_and_fill
from app.storage.db import SessionStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Interview Prep Agent")

# The frontend (Vite dev server) runs on a different origin than the API, so
# the browser needs an explicit CORS allow before it will let fetch() through.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Deps:
    def __init__(self) -> None:
        self.store = SessionStore(DB_PATH)
        self.store.init_schema()
        self.llm = LlmClient()

    async def research(
        self, company: str, role: str, job_url: str | None = None
    ) -> ResearchBundle:
        bundle = await gather_research(company, role, job_url=job_url)
        return await reflect_and_fill(bundle, self.llm)

    def plan(self, bundle: ResearchBundle, resume_text: str) -> LessonPlan:
        return generate_lesson_plan(bundle, resume_text, self.llm)


_deps: Deps | None = None


def get_deps() -> Deps:
    global _deps
    if _deps is None:
        _deps = Deps()
    return _deps


class CreateSessionRequest(BaseModel):
    company: str
    role: str
    resume_text: str
    job_url: str | None = None


class AskRequest(BaseModel):
    question: str


class GradeRequest(BaseModel):
    module_number: int
    question_index: int
    answer: str


def _load_or_404(deps: Deps, session_id: str) -> tuple[ResearchBundle, LessonPlan]:
    loaded = deps.store.load(session_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return loaded


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions")
async def create_session(
    req: CreateSessionRequest, deps: Deps = Depends(get_deps)
) -> dict:
    try:
        bundle = await deps.research(req.company, req.role, req.job_url)
        # deps.plan is a synchronous, minutes-long Anthropic HTTP call. Run it
        # off the event loop, exactly like deps.research's own LLM/network
        # calls, so /health and every other route stay responsive while a
        # session is being built.
        plan = await asyncio.to_thread(deps.plan, bundle, req.resume_text)
    except Exception:
        logger.exception(
            "Session creation pipeline failed for company=%r role=%r",
            req.company,
            req.role,
        )
        raise HTTPException(
            status_code=502, detail="Failed to generate lesson plan"
        ) from None
    session_id = str(uuid.uuid4())
    deps.store.save(session_id, bundle, plan)
    return {
        "session_id": session_id,
        "company": plan.company,
        "role": plan.role,
        "gap_analysis": plan.gap_analysis,
        "module_titles": [m.title for m in plan.modules],
        "sources_used": len(bundle.docs),
    }


@app.get("/sessions/{session_id}")
def get_session(session_id: str, deps: Deps = Depends(get_deps)) -> dict:
    bundle, plan = _load_or_404(deps, session_id)
    return {
        "company": plan.company,
        "role": plan.role,
        "gap_analysis": plan.gap_analysis,
        "module_titles": [m.title for m in plan.modules],
        "research_gaps": bundle.gaps,
        "sources_used": len(bundle.docs),
    }


@app.get("/sessions/{session_id}/modules/{number}")
def get_module(
    session_id: str, number: int, deps: Deps = Depends(get_deps)
) -> dict:
    _bundle, plan = _load_or_404(deps, session_id)
    if number < 1 or number > len(plan.modules):
        raise HTTPException(status_code=404, detail="Module not found")
    module = plan.modules[number - 1]
    return {
        "number": module.number,
        "title": module.title,
        "content": module.content,
        "quiz": [
            {"question": q.question, "expected_points": q.expected_points}
            for q in module.quiz
        ],
    }


@app.post("/sessions/{session_id}/ask")
def ask(
    session_id: str, req: AskRequest, deps: Deps = Depends(get_deps)
) -> dict:
    bundle, _plan = _load_or_404(deps, session_id)
    try:
        answer, sources = answer_question(bundle, req.question, deps.llm)
    except Exception:
        logger.exception(
            "answer_question failed for session=%s question=%r",
            session_id,
            req.question,
        )
        raise HTTPException(
            status_code=502, detail="Failed to answer question"
        ) from None
    return {"answer": answer, "sources": sources}


@app.post("/sessions/{session_id}/grade")
def grade(
    session_id: str, req: GradeRequest, deps: Deps = Depends(get_deps)
) -> dict:
    _bundle, plan = _load_or_404(deps, session_id)
    if req.module_number < 1 or req.module_number > len(plan.modules):
        raise HTTPException(status_code=404, detail="Module not found")
    quiz = plan.modules[req.module_number - 1].quiz
    if req.question_index < 0 or req.question_index >= len(quiz):
        raise HTTPException(status_code=404, detail="Question not found")
    try:
        return grade_answer(quiz[req.question_index], req.answer, deps.llm)
    except Exception:
        logger.exception(
            "grade_answer failed for session=%s module=%s question_index=%s",
            session_id,
            req.module_number,
            req.question_index,
        )
        raise HTTPException(status_code=502, detail="Failed to grade answer") from None

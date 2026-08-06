"""Phase 3 answering: live Q&A grounded in the bundle, and quiz grading.

Neither function ever searches the web. Q&A is answered strictly from retrieved
documents, which keeps responses fast and makes every claim traceable to a URL.
"""

from __future__ import annotations

from app.llm import LlmClient
from app.models import QuizQuestion, ResearchBundle
from app.qa.retriever import retrieve

_QA_SYSTEM = (
    "Answer the user's question about this company using ONLY the supplied "
    "excerpts. If the excerpts do not contain the answer, say so plainly rather "
    "than guessing. Be concise and concrete."
)

_NO_EVIDENCE = "I don't have any research for this company yet."

GRADE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "feedback": {
            "type": "string",
            "description": "1-3 sentences explaining the verdict to the candidate.",
        },
        "missed_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Expected points the answer did not cover.",
        },
    },
    "required": ["verdict", "feedback", "missed_points"],
}

_GRADE_SYSTEM = (
    "Grade a candidate's practice answer against the expected points. 'correct' "
    "means every expected point is covered, 'partial' means some, 'incorrect' "
    "means none. Judge substance, not wording — a paraphrase that conveys the "
    "point counts as covering it."
)


def answer_question(
    bundle: ResearchBundle, question: str, llm: LlmClient
) -> tuple[str, list[str]]:
    docs = retrieve(bundle, question)
    if not docs:
        return _NO_EVIDENCE, []
    excerpts = "\n\n".join(f"[{d.source_type}] {d.url}\n{d.content}" for d in docs)
    user = f"Question: {question}\n\n=== EXCERPTS ===\n{excerpts}"
    answer = llm.complete(system=_QA_SYSTEM, user=user)
    return answer, [d.url for d in docs]


def grade_answer(
    question: QuizQuestion, user_answer: str, llm: LlmClient
) -> dict:
    user = (
        f"Question: {question.question}\n"
        f"Expected points: {'; '.join(question.expected_points)}\n\n"
        f"Candidate answer: {user_answer}"
    )
    result = llm.complete_json(system=_GRADE_SYSTEM, user=user, schema=GRADE_SCHEMA)
    return {
        "verdict": result.get("verdict") or "incorrect",
        "feedback": result.get("feedback") or "",
        "missed_points": [str(p) for p in result.get("missed_points") or []],
    }

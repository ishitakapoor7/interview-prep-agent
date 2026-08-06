"""Phase 2: turn the finalized research bundle plus the candidate's resume into a
structured, personalized lesson plan.

One call, schema-enforced. Module numbers and titles are normalized server-side
rather than trusted from the model, so the frontend can index modules positionally.
"""

from __future__ import annotations

from app.lesson.schema import LESSON_SCHEMA
from app.llm import LlmClient
from app.models import MODULE_TITLES, LessonPlan, Module, QuizQuestion, ResearchBundle

_SYSTEM = (
    "You build interview-prep curricula. Ground every claim in the supplied "
    "evidence — never invent a fact that is not present. Personalize the content "
    "to the candidate's resume where relevant, especially the role and interview "
    "modules. Prefer specific, checkable statements over generic advice. Produce "
    f"exactly {len(MODULE_TITLES)} modules in the order given."
)


def generate_lesson_plan(
    bundle: ResearchBundle, resume_text: str, llm: LlmClient
) -> LessonPlan:
    user = (
        f"Company: {bundle.company}\n"
        f"Role: {bundle.role}\n\n"
        f"Module order: {', '.join(MODULE_TITLES)}\n\n"
        f"=== EVIDENCE ===\n{bundle.all_text()}\n\n"
        f"=== CANDIDATE RESUME ===\n{resume_text}"
    )
    result = llm.complete_json(system=_SYSTEM, user=user, schema=LESSON_SCHEMA)

    raw_modules = result.get("modules") or []
    if len(raw_modules) != len(MODULE_TITLES):
        raise ValueError(
            f"expected {len(MODULE_TITLES)} modules, got {len(raw_modules)}"
        )

    modules: list[Module] = []
    for i, raw in enumerate(raw_modules):
        quiz = [
            QuizQuestion(
                question=str(q.get("question", "")),
                expected_points=[str(p) for p in q.get("expected_points") or []],
            )
            for q in raw.get("quiz") or []
        ]
        # Number and title come from our constant, not the model — positional
        # indexing in the frontend depends on them being exact.
        modules.append(
            Module(
                number=i + 1,
                title=MODULE_TITLES[i],
                content=str(raw.get("content", "")),
                quiz=quiz,
            )
        )

    return LessonPlan(
        company=bundle.company,
        role=bundle.role,
        gap_analysis=str(result.get("gap_analysis", "")),
        modules=modules,
    )

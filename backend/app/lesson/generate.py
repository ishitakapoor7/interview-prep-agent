"""Phase 2: turn the finalized research bundle plus the candidate's resume into a
structured, personalized lesson plan.

One call, schema-enforced. Module numbers and titles are normalized server-side
rather than trusted from the model, so the frontend can index modules positionally.

Content is matched to its canonical slot by *title*, not by array position — the
model can (and does, occasionally) return the right seven titles out of order.
Trusting position alone would silently relabel a module's content with the wrong
canonical title: structurally valid, semantically wrong, and invisible to Task 8
and Task 12 downstream. Position is used only as a fallback when a returned
module's title is missing, non-canonical, or claimed by more than one module.
"""

from __future__ import annotations

import logging

from app.lesson.schema import LESSON_SCHEMA
from app.llm import LlmClient
from app.models import MODULE_TITLES, LessonPlan, Module, QuizQuestion, ResearchBundle

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You build interview-prep curricula. Ground every claim in the supplied "
    "evidence — never invent a fact that is not present. Personalize the content "
    "to the candidate's resume where relevant, especially the role and interview "
    "modules. Prefer specific, checkable statements over generic advice. Produce "
    f"exactly {len(MODULE_TITLES)} modules in the order given."
)


def _normalize_title(title: object) -> str:
    """Case/whitespace-insensitive key used only for matching, never for output."""
    return " ".join(str(title).split()).strip().lower()


def _match_modules_to_slots(raw_modules: list[dict]) -> list[dict]:
    """Assign each returned module dict to its canonical `MODULE_TITLES` slot.

    Matching is by title first, array position second. `raw_modules` and
    `MODULE_TITLES` are always the same length here — the caller has already
    enforced that.
    """
    # Which raw indices claim each normalized title.
    claims: dict[str, list[int]] = {}
    for idx, raw in enumerate(raw_modules):
        norm = _normalize_title(raw.get("title", ""))
        claims.setdefault(norm, []).append(idx)

    slots: list[dict | None] = [None] * len(MODULE_TITLES)
    used_raw_indices: set[int] = set()

    for slot_idx, canonical_title in enumerate(MODULE_TITLES):
        candidates = claims.get(_normalize_title(canonical_title), [])
        if not candidates:
            continue  # filled by positional fallback below
        raw_idx = candidates[0]
        slots[slot_idx] = raw_modules[raw_idx]
        used_raw_indices.add(raw_idx)
        if len(candidates) > 1:
            # Two or more modules claim the same canonical title. Take the
            # first deterministically; the rest are unclaimed and fall back to
            # position for whichever slot they land in.
            logger.warning(
                "lesson generation: %d modules claimed title %r; using raw "
                "index %d for slot %d, remaining fall back to position",
                len(candidates), canonical_title, raw_idx, slot_idx + 1,
            )

    unfilled_slots = [i for i, s in enumerate(slots) if s is None]
    unused_raw = [i for i in range(len(raw_modules)) if i not in used_raw_indices]
    for slot_idx, raw_idx in zip(unfilled_slots, unused_raw):
        slots[slot_idx] = raw_modules[raw_idx]

    for slot_idx, raw in enumerate(slots):
        raw_title = raw.get("title", "") if raw else ""
        if _normalize_title(raw_title) != _normalize_title(MODULE_TITLES[slot_idx]):
            logger.warning(
                "lesson generation: module titled %r placed in slot %d (%r) by "
                "positional fallback — model's title did not match a canonical "
                "slot, or was claimed twice",
                raw_title, slot_idx + 1, MODULE_TITLES[slot_idx],
            )

    return slots


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

    ordered = _match_modules_to_slots(raw_modules)

    modules: list[Module] = []
    for i, raw in enumerate(ordered):
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

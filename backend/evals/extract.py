"""Pull the scored fields out of the agent's lesson plan so they can be compared
against hand-annotated ground truth.

Extraction reads `LessonPlan` — the prose the user actually receives — not the
raw `ResearchBundle`. Scoring what the deliverable said, rather than what was
merely retrieved, is what lets Task 11 tell a synthesis failure (fact was in the
bundle but never made it into the plan) apart from an extraction failure (fact
is in the plan but this pass misread it). Scoring straight off the bundle would
collapse both into one bucket, because nothing downstream of research would ever
be measured.

Extraction only reports what the lesson plan said — it never judges whether that
content is correct. Task 10 does the comparison against `ground_truth.json` with
plain string/number/set operations; no model is involved in scoring. Keeping
that line intact is what lets this suite claim "no LLM-as-judge": the LLM here
is a reader, not a grader.
"""

from __future__ import annotations

import json

from app.llm import LlmClient
from app.models import CompanyFacts, LessonPlan, ResearchBundle

EXTRACT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "funding_usd": {
            "type": ["integer", "null"],
            "description": "Total disclosed funding in USD as a plain integer. Null if unknown.",
        },
        "founded_year": {"type": ["integer", "null"]},
        "founders": {"type": "array", "items": {"type": "string"}},
        "product_line": {
            "type": "string",
            "description": "One sentence describing what the company sells.",
        },
        "required_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top 5 required skills from the job posting evidence.",
        },
        "recent_events": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to 3 notable events from the last 6 months.",
        },
    },
    "required": [
        "funding_usd",
        "founded_year",
        "founders",
        "product_line",
        "required_skills",
        "recent_events",
    ],
}

_SYSTEM = (
    "Extract structured company facts from the supplied lesson plan text. Report "
    "only what the text states — use null or an empty list rather than guessing. "
    "Do not use prior knowledge about the company; the point is to measure what "
    "this lesson plan actually says."
)

# Fields that must load from a ground-truth row for it to be usable at all. Every
# other field is optional and defaults to the same "not claimed" value extraction
# uses, so a hand-annotated row and an agent-extracted row are comparable field by
# field even when one side is incomplete.
_REQUIRED_ROW_FIELDS = ("company", "tier")

# Mirrors CompanyFacts.tier's `Literal["large", "mid", "early"]`. Kept as a
# plain set (rather than introspected from the dataclass) because `from __future__
# import annotations` turns dataclass field types into strings at runtime, which
# would make introspection more fragile than just stating the three values here.
_VALID_TIERS = {"large", "mid", "early"}


def load_ground_truth(path: str) -> list[CompanyFacts]:
    """Load hand-annotated ground truth from a JSON array of rows shaped like
    `CompanyFacts` (plus a `job_posting_url` annotation field that is recorded for
    provenance but not part of `CompanyFacts` and is intentionally dropped here).

    Works for any number of rows — the seed file ships 3; the full dataset will
    grow to 30 with no code change required here.

    Raises `ValueError`, naming `path` and (where applicable) the offending row
    index, on any malformed input: missing file, invalid JSON, a top level that
    isn't an array, a row that isn't an object, a row missing `company`/`tier`,
    or a row whose `tier` isn't one of `large`/`mid`/`early`. A hand-annotated
    file should fail loudly at load time, not silently drop or misreport a row.
    """
    try:
        with open(path) as f:
            rows = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"{path}: ground-truth file not found") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e

    if not isinstance(rows, list):
        raise ValueError(
            f"{path}: expected a JSON array of ground-truth rows, got {type(rows).__name__}"
        )

    facts: list[CompanyFacts] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {i} is not a JSON object")
        missing = [k for k in _REQUIRED_ROW_FIELDS if k not in row]
        if missing:
            raise ValueError(f"{path}: row {i} is missing required field(s) {missing}")
        tier = row["tier"]
        if tier not in _VALID_TIERS:
            raise ValueError(
                f"{path}: row {i} has invalid tier {tier!r}; "
                f"expected one of {sorted(_VALID_TIERS)}"
            )
        facts.append(
            CompanyFacts(
                company=row["company"],
                tier=tier,
                funding_usd=row.get("funding_usd"),
                founded_year=row.get("founded_year"),
                founders=list(row.get("founders") or []),
                product_line=row.get("product_line") or "",
                required_skills=list(row.get("required_skills") or []),
                recent_events=list(row.get("recent_events") or []),
            )
        )
    return facts


def plan_text(plan: LessonPlan) -> str:
    """Flat text of the lesson plan, labeled the same way `ResearchBundle.all_text()`
    labels its sections, so the extraction prompt reads consistently regardless of
    which pipeline stage supplied the evidence.

    Public (not `_plan_text`) because `evals/attribute.py` needs the exact same
    plan text extraction saw, to tell a synthesis failure (fact never made it
    into the plan) apart from an extraction failure (fact is in the plan but
    was misread). Reusing this function instead of reimplementing it there is
    what keeps the two from silently diverging.
    """
    sections = [f"[gap_analysis]\n{plan.gap_analysis}"]
    for m in plan.modules:
        section = f"[module {m.number}] {m.title}\n{m.content}"
        points = "; ".join(p for q in m.quiz for p in q.expected_points)
        if points:
            section += f"\nExpected quiz points: {points}"
        sections.append(section)
    return "\n\n".join(sections)


def extract_facts(
    bundle: ResearchBundle, plan: LessonPlan, tier: str, llm: LlmClient
) -> CompanyFacts:
    """Ask the model to read `plan` — the lesson plan the user actually reads —
    and report what it says about the scored fields, then map that report onto
    `CompanyFacts`.

    `bundle` supplies only `company`; facts are deliberately never pulled from
    it. Extracting from the plan instead of the bundle is what makes the
    three-way attribution in Task 11 possible: a fact absent from the bundle is
    a research failure, a fact present in the bundle but absent from the plan is
    a synthesis failure, and a fact present in the plan but extracted wrong is
    an extraction failure. Scoring the bundle directly would erase the middle
    case.

    A field the plan never mentions comes back as `None` (scalars) or `[]`
    (lists) — distinct from a claimed value of `0` or `[""]`. That distinction
    must not be collapsed here; Task 11's attribution depends on telling "not
    claimed" apart from "claimed as empty/zero".
    """
    user = f"Company: {bundle.company}\n\n=== LESSON PLAN ===\n{plan_text(plan)}"
    result = llm.complete_json(system=_SYSTEM, user=user, schema=EXTRACT_SCHEMA)

    return CompanyFacts(
        company=bundle.company,
        tier=tier,  # type: ignore[arg-type]
        funding_usd=result.get("funding_usd"),
        founded_year=result.get("founded_year"),
        founders=[str(x) for x in result.get("founders") or []],
        product_line=str(result.get("product_line") or ""),
        required_skills=[str(x) for x in result.get("required_skills") or []],
        recent_events=[str(x) for x in result.get("recent_events") or []],
    )

"""Pull the scored fields out of a research bundle so they can be compared against
hand-annotated ground truth.

This is a separate extraction pass rather than a reuse of the lesson plan, because
the lesson plan is prose written for a human. Scoring needs typed fields, and a
prose-to-fields regex would introduce its own failure mode into the measurement.

Extraction only reports what the agent's evidence and output said — it never
judges whether that content is correct. Task 10 does the comparison against
`ground_truth.json` with plain string/number/set operations; no model is involved
in scoring. Keeping that line intact is what lets this suite claim "no
LLM-as-judge": the LLM here is a reader, not a grader.
"""

from __future__ import annotations

import json

from app.llm import LlmClient
from app.models import CompanyFacts, ResearchBundle

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
    "Extract structured company facts from the supplied evidence. Report only what "
    "the evidence supports — use null or an empty list rather than guessing. Do not "
    "use prior knowledge about the company; the point is to measure what this "
    "evidence contains."
)

# Fields that must load from a ground-truth row for it to be usable at all. Every
# other field is optional and defaults to the same "not claimed" value extraction
# uses, so a hand-annotated row and an agent-extracted row are comparable field by
# field even when one side is incomplete.
_REQUIRED_ROW_FIELDS = ("company", "tier")


def load_ground_truth(path: str) -> list[CompanyFacts]:
    """Load hand-annotated ground truth from a JSON array of rows shaped like
    `CompanyFacts` (plus a `job_posting_url` annotation field that is recorded for
    provenance but not part of `CompanyFacts` and is intentionally dropped here).

    Works for any number of rows — the seed file ships 3; the full dataset will
    grow to 30 with no code change required here.
    """
    with open(path) as f:
        rows = json.load(f)

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
        facts.append(
            CompanyFacts(
                company=row["company"],
                tier=row["tier"],
                funding_usd=row.get("funding_usd"),
                founded_year=row.get("founded_year"),
                founders=list(row.get("founders") or []),
                product_line=row.get("product_line") or "",
                required_skills=list(row.get("required_skills") or []),
                recent_events=list(row.get("recent_events") or []),
            )
        )
    return facts


def extract_facts(bundle: ResearchBundle, tier: str, llm: LlmClient) -> CompanyFacts:
    """Ask the model to read `bundle`'s evidence and report what it says about the
    scored fields, then map that report onto `CompanyFacts`.

    A field the evidence never mentions comes back as `None` (scalars) or `[]`
    (lists) — distinct from a claimed value of `0` or `[""]`. Task 11 attributes a
    wrong answer to research (fact never retrieved), extraction (fact retrieved but
    not pulled out), or synthesis (fact used incorrectly downstream) based on that
    distinction, so it must not be collapsed here.
    """
    user = f"Company: {bundle.company}\n\n=== EVIDENCE ===\n{bundle.all_text()}"
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

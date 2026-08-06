"""Turn a wrong answer into a diagnosis.

`extract_facts` (Task 9) reads the *lesson plan*, not the raw research bundle
— that's what the deliverable actually says. So a wrong field can fail at any
of three points, and attribution is keyed on where the truth value can still
be found:

| Truth value in bundle? | In lesson plan? | Field scored correct? | Attribution   |
|-------------------------|------------------|-------------------------|---------------|
| —                       | —                | yes                     | "none"        |
| no                      | —                | no                       | "research"    |
| yes                     | no               | no                       | "synthesis"   |
| yes                     | yes              | no                       | "extraction"  |

Absent from the bundle means the research step never retrieved it — no later
step could have used it. Present in the bundle but absent from the plan means
`generate_lesson_plan` had the evidence and dropped it. Present in the plan
(the text the user actually reads) but still scored wrong means this pass of
`extract_facts` misread text that was right there.

The distinction matters because the fixes are different for each: a research
failure calls for better queries or more sources; a synthesis failure calls
for a better generation prompt or a stronger model; an extraction failure
calls for a better extraction prompt, independent of research or generation.

Everything here is a mechanical substring/surface-form check over text — no
model is ever consulted to decide where a failure belongs.
"""

from __future__ import annotations

from app.models import Attribution, CompanyFacts, LessonPlan, ResearchBundle
from evals.extract import plan_text
from evals.score import FieldScore, normalize

# Earliest-in-pipeline step wins when a set field's members disagree on where
# they were lost (see _attribute_set). Lower number = earlier = preferred.
_ATTRIBUTION_PRIORITY: dict[Attribution, int] = {
    "research": 0,
    "synthesis": 1,
    "extraction": 2,
}


def _funding_surface_forms(amount: int) -> list[str]:
    """How a funding figure actually appears in prose. Press writes '$9.1M', not
    '9100000', so a raw-integer search would report false research failures."""
    forms = [str(amount), f"{amount:,}"]
    millions = amount / 1_000_000
    if millions >= 1:
        trimmed = f"{millions:.1f}".rstrip("0").rstrip(".")
        forms += [f"{trimmed}m", f"{trimmed} million"]
    billions = amount / 1_000_000_000
    if billions >= 1:
        trimmed = f"{billions:.1f}".rstrip("0").rstrip(".")
        forms += [f"{trimmed}b", f"{trimmed} billion"]
    return forms


def text_contains_value(text: str, value: str) -> bool:
    """Normalized substring check, used symmetrically against both the bundle
    text and the plan text -- one helper, so the two checks can never drift
    apart in how they normalize."""
    return normalize(value) in normalize(text)


def _attribute_scalar(
    bundle_text: str, plan_text_str: str, forms: list[str] | None
) -> Attribution:
    """Attribute a scalar field (funding_usd, founded_year) given the surface
    forms the truth value could appear as. `forms is None` means truth itself
    is null -- see the module-level note on that case below."""
    if forms is None:
        # Nothing was annotated as truth, so the prediction must have
        # hallucinated a value with nothing to search for in either text.
        # There's no way to presence-check a value that doesn't exist;
        # default to synthesis, the step responsible for grounding claims in
        # evidence, rather than guessing.
        return "synthesis"
    if not any(text_contains_value(bundle_text, f) for f in forms):
        return "research"
    if not any(text_contains_value(plan_text_str, f) for f in forms):
        return "synthesis"
    return "extraction"


def _attribute_set(
    values: list[str], bundle_text: str, plan_text_str: str
) -> Attribution:
    """Attribute a set field (founders, required_skills, recent_events).

    `attribute_failure` only receives the score, not the predicted list, so we
    can't tell precisely which truth members were missed versus which extra
    ones were hallucinated -- only that the field as a whole scored
    imperfectly. So we check each truth member's own fate (bundle? plan?) and,
    when members disagree on where they were lost, report the earliest
    failing pipeline step (research > synthesis > extraction). That keeps the
    diagnosis pointed at the root cause: if even one expected member was never
    retrieved, fixing extraction or synthesis wouldn't have saved this field.
    """
    if not values:
        # Same reasoning as the scalar null case: no truth members means
        # nothing to search for, so an incorrect score can only be a
        # hallucination with no ground-truth value to trace.
        return "synthesis"

    per_value: list[Attribution] = []
    for v in values:
        if not text_contains_value(bundle_text, v):
            per_value.append("research")
        elif not text_contains_value(plan_text_str, v):
            per_value.append("synthesis")
        else:
            per_value.append("extraction")

    return min(per_value, key=lambda a: _ATTRIBUTION_PRIORITY[a])


def attribute_failure(
    score: FieldScore, bundle: ResearchBundle, plan: LessonPlan, truth: CompanyFacts
) -> Attribution:
    if score.correct:
        return "none"

    bundle_text = bundle.all_text()
    plan_text_str = plan_text(plan)

    if score.field == "funding_usd":
        forms = (
            _funding_surface_forms(truth.funding_usd)
            if truth.funding_usd is not None
            else None
        )
        return _attribute_scalar(bundle_text, plan_text_str, forms)

    if score.field == "founded_year":
        forms = [str(truth.founded_year)] if truth.founded_year is not None else None
        return _attribute_scalar(bundle_text, plan_text_str, forms)

    set_values = {
        "founders": truth.founders,
        "required_skills": truth.required_skills,
        "recent_events": truth.recent_events,
    }.get(score.field)
    if set_values is None:
        raise ValueError(f"attribute_failure: unrecognized field {score.field!r}")
    return _attribute_set(set_values, bundle_text, plan_text_str)


def summarize(results: list[dict]) -> str:
    """Markdown results table: per-tier, per-field accuracy plus the failure mix."""
    tiers = ["large", "mid", "early"]
    fields = ["funding_usd", "founded_year", "founders", "required_skills", "recent_events"]

    lines = ["| Tier | " + " | ".join(fields) + " | n |", "|---" * (len(fields) + 2) + "|"]
    for tier in tiers:
        rows = [r for r in results if r["tier"] == tier]
        if not rows:
            continue
        cells = []
        for field in fields:
            scores = [s for r in rows for s in r["scores"] if s.field == field]
            if not scores:
                cells.append("—")
                continue
            pct = 100 * sum(1 for s in scores if s.correct) / len(scores)
            cells.append(f"{pct:.0f}%")
        lines.append(f"| {tier} | " + " | ".join(cells) + f" | {len(rows)} |")

    counts: dict[str, int] = {"research": 0, "extraction": 0, "synthesis": 0}
    for r in results:
        for a in r["attributions"]:
            if a in counts:
                counts[a] += 1
    total = sum(counts.values())
    lines.append("")
    lines.append("**Failure attribution**")
    lines.append("")
    lines.append("| Step | Failures | Share |")
    lines.append("|---|---|---|")
    for step, n in counts.items():
        share = f"{100 * n / total:.0f}%" if total else "—"
        lines.append(f"| {step} | {n} | {share} |")
    return "\n".join(lines)

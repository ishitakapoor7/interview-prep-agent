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

One deliberate gap: when the *truth* value itself is null (`funding_usd` or
`founded_year` is `None`) or a set field's truth list is empty, there is
nothing to presence-check in either text -- the only way a field can still
score wrong in that case is a hallucinated prediction. The correct diagnosis
there depends on the *predicted* value (was the hallucination grounded in the
plan's prose, or invented by extraction alone?), and `attribute_failure`'s
approved signature does not receive the prediction -- only the `FieldScore`.
`_attribute_scalar` and `_attribute_set` both default this case to
`"synthesis"` as a documented guess, not a derived answer. Don't try to fix
this inside either function alone; it needs a signature change this task
intentionally did not make.
"""

from __future__ import annotations

import re

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
    """How a funding figure actually appears in prose. Press writes '$9.1M',
    '$9.1 M' (a real space before the unit), or rounds off entirely to
    '~$9M' -- never '9100000' -- so a raw-integer search, or one that only
    checks the unspaced form, reports false research failures.

    Both a tight ('9.1m') and spaced ('9.1 m') variant are included because
    normalize() does not itself insert or remove the space between a number
    and its unit -- whichever the source text used is what has to match.
    A rounded whole-number form ('9m') is included too, since coverage
    routinely rounds off the decimal.
    """
    forms = [str(amount), f"{amount:,}"]
    millions = amount / 1_000_000
    if millions >= 1:
        trimmed = f"{millions:.1f}".rstrip("0").rstrip(".")
        forms += [f"{trimmed}m", f"{trimmed} m", f"{trimmed} million"]
        whole = str(round(millions))
        forms += [f"{whole}m", f"{whole} m", f"{whole} million"]
    billions = amount / 1_000_000_000
    if billions >= 1:
        trimmed = f"{billions:.1f}".rstrip("0").rstrip(".")
        forms += [f"{trimmed}b", f"{trimmed} b", f"{trimmed} billion"]
        whole = str(round(billions))
        forms += [f"{whole}b", f"{whole} b", f"{whole} billion"]
    return forms


def text_contains_value(text: str, value: str) -> bool:
    """Word-boundary-anchored, normalized substring check, used symmetrically
    against both the bundle text and the plan text -- one helper, so the two
    checks can never drift apart in how they normalize or match.

    Anchoring on `\\w` boundaries (rather than a bare substring test) matters
    for short values: an unanchored search for the skill "Go" matches inside
    "Google" and inside "go deeper on payments", turning a genuine research
    miss into a confidently wrong "extraction" or "synthesis" verdict.
    `normalize()` already reduces both sides to space-separated alphanumeric
    tokens, so the token boundaries are exactly the string-edge/space
    boundaries a `\\w` lookaround catches here.
    """
    normalized_value = normalize(value)
    if not normalized_value:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_value)}(?!\w)"
    return re.search(pattern, normalize(text)) is not None


def _attribute_scalar(
    bundle_text: str,
    plan_text_str: str,
    forms: list[str] | None,
    *,
    presence_check=text_contains_value,
) -> Attribution:
    """Attribute a scalar field (funding_usd, founded_year) given the surface
    forms the truth value could appear as. `forms is None` means truth itself
    is null -- see the module-level note on that case below.

    `presence_check` is swappable so founded_year can use the proximity-aware
    `text_contains_founding_year` (see below) instead of a bare substring
    test, without duplicating the null-handling and research/synthesis/
    extraction ladder below."""
    if forms is None:
        # Nothing was annotated as truth, so the prediction must have
        # hallucinated a value with nothing to search for in either text.
        # There's no way to presence-check a value that doesn't exist;
        # default to synthesis, the step responsible for grounding claims in
        # evidence, rather than guessing.
        return "synthesis"
    if not any(presence_check(bundle_text, f) for f in forms):
        return "research"
    if not any(presence_check(plan_text_str, f) for f in forms):
        return "synthesis"
    return "extraction"


# Founding language that must appear near a bare year for the year to count as
# "the founding year is present" rather than merely "this year appears
# somewhere in a 30-doc bundle for unrelated reasons." A bare token like
# "2025" collides constantly -- a funding headline, a conference name, a
# product version, a copyright year -- so an unqualified presence check turns
# routine noise into a confidently wrong "synthesis"/"extraction" verdict for
# what is actually a research miss. Same false-positive class as the "Go" /
# "Google" word-boundary bug above, just on a token that collides far more.
_FOUNDING_LANGUAGE = ("founded", "since", "started", "est", "launched", "incorporated", "began")

# Characters on each side of the year token to search for founding language.
# 60 is roughly one short clause -- enough to span "founded in 2025" or
# "2025, when the company started," without also reaching into an unrelated
# neighboring sentence.
_FOUNDING_YEAR_WINDOW_CHARS = 60


def text_contains_founding_year(text: str, year: str) -> bool:
    """Like `text_contains_value`, but for founded_year specifically: a bare
    year only counts as "present" when founding language appears within
    `_FOUNDING_YEAR_WINDOW_CHARS` characters of it.

    Deliberate accepted trade-off: this can miss a genuine founding mention
    phrased unusually (a date sitting alone in a table, with no nearby
    prose), producing a false "research" failure for a fact that actually was
    covered. That is the safer direction for an eval to err in -- under-
    crediting the pipeline is honest, over-crediting it (as the old bare
    presence check did) is not.
    """
    normalized_text = normalize(text)
    normalized_year = normalize(year)
    if not normalized_year:
        return False
    year_pattern = rf"(?<!\w){re.escape(normalized_year)}(?!\w)"
    for match in re.finditer(year_pattern, normalized_text):
        start = max(0, match.start() - _FOUNDING_YEAR_WINDOW_CHARS)
        end = min(len(normalized_text), match.end() + _FOUNDING_YEAR_WINDOW_CHARS)
        window = normalized_text[start:end]
        if any(
            re.search(rf"(?<!\w){kw}(?!\w)", window) for kw in _FOUNDING_LANGUAGE
        ):
            return True
    return False


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
        return _attribute_scalar(
            bundle_text, plan_text_str, forms, presence_check=text_contains_founding_year
        )

    set_values = {
        "founders": truth.founders,
        "required_skills": truth.required_skills,
        "recent_events": truth.recent_events,
    }.get(score.field)
    if set_values is None:
        raise ValueError(f"attribute_failure: unrecognized field {score.field!r}")
    return _attribute_set(set_values, bundle_text, plan_text_str)


def _errored(result: dict) -> bool:
    """A company whose run raised mid-eval (see run_eval.run_all) is recorded
    with an explicit `"error"` key and empty scores/attributions. We check
    both signals -- not just the `"error"` key -- so a future caller that
    forgets to set it but still leaves `scores` empty is still caught as
    errored rather than silently read as "100% correct on zero fields"."""
    return bool(result.get("error")) or not result.get("scores")


def summarize(results: list[dict]) -> str:
    """Markdown results table: per-tier, per-field accuracy plus the failure
    mix, plus how many companies in each tier never produced a score at all.

    `n` counts only companies that actually ran to completion; a separate
    `errors` column counts companies whose run raised (see `run_eval.run_all`)
    so a partially-crashed run cannot render identically to a clean one --
    the percentages above are silently computed from survivors only, so the
    error count has to be impossible to miss.
    """
    tiers = ["large", "mid", "early"]
    fields = ["funding_usd", "founded_year", "founders", "required_skills", "recent_events"]

    lines = [
        "| Tier | " + " | ".join(fields) + " | n | errors |",
        "|---" * (len(fields) + 3) + "|",
    ]
    total_errors = 0
    for tier in tiers:
        rows = [r for r in results if r["tier"] == tier]
        if not rows:
            continue
        errored_rows = [r for r in rows if _errored(r)]
        scored_rows = [r for r in rows if not _errored(r)]
        total_errors += len(errored_rows)
        cells = []
        for field in fields:
            scores = [s for r in scored_rows for s in r["scores"] if s.field == field]
            if not scores:
                cells.append("—")
                continue
            pct = 100 * sum(1 for s in scores if s.correct) / len(scores)
            cells.append(f"{pct:.0f}%")
        lines.append(
            f"| {tier} | " + " | ".join(cells) + f" | {len(scored_rows)} | {len(errored_rows)} |"
        )

    if total_errors:
        lines.append("")
        lines.append(
            f"**{total_errors} compan{'y' if total_errors == 1 else 'ies'} failed to run "
            "and are excluded from the field percentages above -- see the errors column "
            "and the run log for details.**"
        )

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

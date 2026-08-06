"""Tests for the three-way (four-value) failure attribution.

This suite implements the corrected design from the Task 11 dispatch, not the
brief's original two-way research/synthesis split: `extract_facts` reads the
*lesson plan*, not the bundle, so attribution must check both texts to tell a
research failure (never retrieved) apart from a synthesis failure (retrieved,
dropped by `generate_lesson_plan`) apart from an extraction failure (present
in the delivered plan, but misread).

Truth table pinned here (see attribute.py docstring for the same table):

| in bundle? | in plan? | correct? | Attribution   |
|------------|----------|----------|---------------|
| —          | —        | yes      | "none"        |
| no         | —        | no       | "research"    |
| yes        | no       | no       | "synthesis"   |
| yes        | yes      | no       | "extraction"  |
"""

import pytest

from app.models import CompanyFacts, LessonPlan, ResearchBundle, SourceDoc
from evals.attribute import attribute_failure, summarize, text_contains_value
from evals.score import FieldScore


def _bundle(text: str = "") -> ResearchBundle:
    docs = [SourceDoc("news", "u", "t", text, "ts")] if text else []
    return ResearchBundle(company="Acme", role="SWE", docs=docs)


def _plan(gap_analysis: str = "") -> LessonPlan:
    return LessonPlan(company="Acme", role="SWE", gap_analysis=gap_analysis, modules=[])


def _truth(**kw) -> CompanyFacts:
    base = dict(
        company="Acme",
        tier="early",
        funding_usd=9_100_000,
        founded_year=2025,
        founders=["Ada Lovelace"],
        product_line="things",
        required_skills=["Python"],
        recent_events=["raised money"],
    )
    base.update(kw)
    return CompanyFacts(**base)


# --- text_contains_value -----------------------------------------------------


def test_text_contains_value_matches_normalized_text():
    assert text_contains_value("Founded by Ada Lovelace.", "ada lovelace")
    assert not text_contains_value("Founded by someone else.", "ada lovelace")


def test_text_contains_value_rejects_substring_inside_a_longer_word():
    # Review finding: an unanchored substring search for "Go" matched inside
    # "Google" -- not evidence the skill "Go" was actually mentioned. Word
    # boundaries must reject that while still matching a genuine standalone
    # occurrence, including one that happens to be an ordinary English word
    # ("go deeper" really is the token "go", so it correctly matches -- the
    # false positive this fixes is specifically the *inside-a-word* case).
    assert not text_contains_value("Google Cloud", "Go")
    assert text_contains_value("You should go deeper on payments.", "Go")
    assert text_contains_value("We use Go for backend services.", "Go")


def test_text_contains_value_still_matches_multi_word_values():
    # Anchoring must not break matching a longer phrase embedded in a
    # sentence -- only bare-word-boundary edges matter, not whole-string
    # equality.
    assert text_contains_value("Ada Lovelace co-founded the company.", "ada lovelace")


# --- "none" --------------------------------------------------------------


def test_correct_score_attributes_to_none():
    score = FieldScore("founded_year", True, None, None, "")
    result = attribute_failure(score, _bundle("irrelevant"), _plan("irrelevant"), _truth())
    assert result == "none"


# --- "research": absent from the bundle entirely --------------------------


def test_missing_from_bundle_attributes_to_research():
    score = FieldScore("founded_year", False, None, None, "")
    bundle = _bundle("No dates here.")
    plan = _plan("No dates here either.")
    assert attribute_failure(score, bundle, plan, _truth()) == "research"


def test_research_attribution_is_independent_of_plan_content():
    # Even if the plan happens to mention the value (e.g. a stale or unrelated
    # match), absence from the bundle means research never retrieved it, and
    # that verdict must not be overridden by what's in the plan.
    score = FieldScore("founded_year", False, None, None, "")
    bundle = _bundle("No dates here.")
    plan = _plan("Founded in 2025.")
    assert attribute_failure(score, bundle, plan, _truth()) == "research"


# --- "synthesis": in the bundle, dropped before the plan -------------------


def test_present_in_bundle_absent_from_plan_attributes_to_synthesis():
    score = FieldScore("founded_year", False, None, None, "")
    bundle = _bundle("Founded in 2025.")
    plan = _plan("No dates mentioned.")
    assert attribute_failure(score, bundle, plan, _truth()) == "synthesis"


def test_funding_shorthand_present_in_bundle_absent_from_plan_is_synthesis():
    score = FieldScore("funding_usd", False, None, None, "")
    bundle = _bundle("Acme raised $9.1M in seed funding.")
    plan = _plan("Acme is an early-stage company.")
    assert attribute_failure(score, bundle, plan, _truth()) == "synthesis"


# --- "extraction": in the bundle AND the plan -- the value the old design
# could never produce, because it only ever checked the bundle. ------------


def test_present_in_bundle_and_plan_attributes_to_extraction():
    score = FieldScore("founded_year", False, None, None, "")
    bundle = _bundle("Founded in 2025.")
    plan = _plan("Acme was founded in 2025 by its team.")
    assert attribute_failure(score, bundle, plan, _truth()) == "extraction"


def test_funding_shorthand_present_in_bundle_and_plan_is_extraction():
    score = FieldScore("funding_usd", False, None, None, "")
    bundle = _bundle("raised $9.1M")
    plan = _plan("The company raised $9.1M to date.")
    assert attribute_failure(score, bundle, plan, _truth()) == "extraction"


# --- set fields: per-element check, earliest-failing-step priority --------


def test_founders_prefers_research_over_synthesis_when_elements_differ():
    # Ada is in the bundle but dropped from the plan (synthesis); Grace never
    # appears anywhere (research). The root cause is research, so that must
    # win even though a later element also failed for a different reason.
    score = FieldScore("founders", False, 0.5, 0.5, "")
    bundle = _bundle("Ada Lovelace co-founded Acme.")
    plan = _plan("Acme was founded by a small team.")
    truth = _truth(founders=["Ada Lovelace", "Grace Hopper"])
    assert attribute_failure(score, bundle, plan, truth) == "research"


def test_founders_prefers_synthesis_over_extraction_when_elements_differ():
    # Ada is in both bundle and plan (would be "extraction" alone); Grace is
    # in the bundle but dropped from the plan ("synthesis"). Synthesis is the
    # earlier-in-pipeline failure, so it wins.
    score = FieldScore("founders", False, 0.5, 0.5, "")
    bundle = _bundle("Ada Lovelace and Grace Hopper co-founded Acme.")
    plan = _plan("Ada Lovelace co-founded Acme.")
    truth = _truth(founders=["Ada Lovelace", "Grace Hopper"])
    assert attribute_failure(score, bundle, plan, truth) == "synthesis"


def test_founders_all_present_in_bundle_and_plan_attributes_to_extraction():
    score = FieldScore("founders", False, 1.0, 1.0, "")
    bundle = _bundle("Ada Lovelace co-founded Acme.")
    plan = _plan("Ada Lovelace co-founded Acme.")
    truth = _truth(founders=["Ada Lovelace"])
    assert attribute_failure(score, bundle, plan, truth) == "extraction"


def test_founders_all_absent_from_bundle_attributes_to_research():
    score = FieldScore("founders", False, 0.0, 0.0, "")
    bundle = _bundle("Acme is a company.")
    plan = _plan("Acme is a company.")
    truth = _truth(founders=["Ada Lovelace"])
    assert attribute_failure(score, bundle, plan, truth) == "research"


def test_short_skill_value_does_not_falsely_match_inside_a_longer_word():
    # Review finding, reproduced end to end: "Go" is a real required_skills
    # entry (Stripe's row in ground_truth.json). Before anchoring, "Go"
    # matched inside "Google" (bundle) and inside "go deeper" (plan), so this
    # scored a confidently wrong "extraction" instead of the true "research"
    # verdict -- the language was never retrieved at all.
    score = FieldScore("required_skills", False, 0.0, 0.0, "")
    bundle = _bundle("Stripe engineers use Google Cloud and Ruby.")
    plan = _plan("You should go deeper on payments.")
    truth = _truth(required_skills=["Go"])
    assert attribute_failure(score, bundle, plan, truth) == "research"


def test_funding_spaced_decimal_form_matches_in_bundle():
    # Review finding, reproduced: "$9.1 M" (a literal space before the unit)
    # is real coverage phrasing. The old surface forms only had the unspaced
    # "9.1m", so this bundle text registered as a false "research" failure.
    score = FieldScore("funding_usd", False, None, None, "")
    bundle = _bundle("Raised $9.1 M in seed.")
    plan = _plan("Nothing about funding in this plan.")
    assert attribute_failure(score, bundle, plan, _truth()) == "synthesis"


# --- edge cases: null truth values ----------------------------------------


def test_scalar_field_with_null_truth_defaults_to_synthesis_without_crashing():
    # truth.funding_usd is None (nothing annotated); the prediction must have
    # hallucinated a value for the score to be incorrect. There is no truth
    # value to search for in either text, so we can't presence-check; the
    # module documents this as a synthesis default (the step responsible for
    # grounding claims in evidence).
    score = FieldScore("funding_usd", False, None, None, "")
    truth = _truth(funding_usd=None)
    assert attribute_failure(score, _bundle("anything"), _plan("anything"), truth) == "synthesis"


def test_founded_year_with_null_truth_defaults_to_synthesis_without_crashing():
    score = FieldScore("founded_year", False, None, None, "")
    truth = _truth(founded_year=None)
    assert attribute_failure(score, _bundle("anything"), _plan("anything"), truth) == "synthesis"


def test_set_field_with_empty_truth_defaults_to_synthesis_without_crashing():
    # truth.founders == [] but the score is incorrect, meaning the prediction
    # hallucinated founders with nothing annotated to search for.
    score = FieldScore("founders", False, 0.0, None, "")
    truth = _truth(founders=[])
    assert attribute_failure(score, _bundle("anything"), _plan("anything"), truth) == "synthesis"


def test_attribute_failure_raises_for_a_field_score_company_never_produces():
    # score_company only ever emits the 5 scored fields; an unrecognized
    # field name reaching attribute_failure is a caller bug and must be
    # loud, not silently misattributed.
    score = FieldScore("not_a_real_field", False, None, None, "")
    with pytest.raises(ValueError, match="not_a_real_field"):
        attribute_failure(score, _bundle("x"), _plan("x"), _truth())


# --- summarize --------------------------------------------------------------


def test_summarize_emits_a_markdown_table_with_every_tier():
    results = [
        {
            "company": "A",
            "tier": "large",
            "scores": [FieldScore("founded_year", True, None, None, "")],
            "attributions": ["none"],
        },
        {
            "company": "B",
            "tier": "early",
            "scores": [FieldScore("founded_year", False, None, None, "")],
            "attributions": ["research"],
        },
    ]
    table = summarize(results)
    assert "| Tier |" in table
    assert "large" in table and "early" in table
    assert "founded_year" in table


def test_summarize_reports_extraction_in_the_failure_mix():
    results = [
        {
            "company": "A",
            "tier": "early",
            "scores": [FieldScore("founded_year", False, None, None, "")],
            "attributions": ["extraction"],
        },
    ]
    table = summarize(results)
    assert "| extraction | 1 |" in table


def test_summarize_handles_a_company_with_no_scores_without_crashing():
    # Mirrors what the harness records when a company's run raised mid-eval:
    # scores/attributions come back empty rather than aborting the report.
    # This company has no "error" key either -- summarize must still treat
    # empty scores as errored (see _errored's defensive fallback) rather than
    # crash or silently count it as a scored company.
    results = [
        {"company": "A", "tier": "early", "scores": [], "attributions": []},
    ]
    table = summarize(results)
    # n=0 scored, 1 error, every field cell is "—" -- not "early" merely
    # appearing somewhere in the table.
    assert "| early | — | — | — | — | — | 0 | 1 |" in table
    assert "1 company failed to run" in table


def test_summarize_computes_exact_percentages_and_failure_counts():
    # Pins concrete computed content (not a self-comparison, which cannot
    # fail within a single process regardless of whether summarize is
    # correct): the exact per-tier row and the exact failure-mix rows.
    results = [
        {
            "company": "A",
            "tier": "mid",
            "scores": [FieldScore("founded_year", False, None, None, "")],
            "attributions": ["research"],
        },
    ]
    table = summarize(results)
    # founded_year: 0/1 correct -> 0%; no scores at all for the other four
    # fields in this tier -> "—". n=1 scored, 0 errors.
    assert "| mid | — | 0% | — | — | — | 1 | 0 |" in table
    assert "failed to run" not in table
    assert "| research | 1 | 100% |" in table
    assert "| extraction | 0 | 0% |" in table
    assert "| synthesis | 0 | 0% |" in table


def test_summarize_reports_an_errors_column_separate_from_successful_n():
    # Review finding: an errored company used to inflate `n` while
    # contributing zero scores, making a partially-crashed run visually
    # indistinguishable from a clean one.
    results = [
        {
            "company": "A",
            "tier": "early",
            "scores": [FieldScore("founded_year", True, None, None, "")],
            "attributions": ["none"],
        },
        {
            "company": "B",
            "tier": "early",
            "scores": [],
            "attributions": [],
            "error": "simulated network failure",
        },
    ]
    table = summarize(results)
    assert "| n | errors |" in table
    # One scored company (100% on founded_year), one errored -- n counts only
    # the scored company; errors counts the other.
    assert "| early | — | 100% | — | — | — | 1 | 1 |" in table
    assert "1 company failed to run" in table

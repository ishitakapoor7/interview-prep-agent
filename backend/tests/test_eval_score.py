from app.models import CompanyFacts
from evals.score import (
    normalize,
    score_company,
    score_funding,
    score_set,
    score_year,
)


def test_normalize_lowercases_and_strips_punctuation_and_spacing():
    assert normalize("  Patrick   Collison. ") == "patrick collison"
    assert normalize("CI/CD") == "ci cd"


def test_score_funding_exact_match_is_correct():
    assert score_funding(9_100_000, 9_100_000).correct is True


def test_score_funding_within_five_percent_is_correct():
    assert score_funding(9_000_000, 9_100_000).correct is True


def test_score_funding_outside_five_percent_is_wrong():
    assert score_funding(12_000_000, 9_100_000).correct is False


def test_score_funding_missing_prediction_is_wrong_not_crash():
    s = score_funding(None, 9_100_000)
    assert s.correct is False
    assert "no prediction" in s.detail


def test_score_funding_is_correct_when_both_are_none():
    assert score_funding(None, None).correct is True


def test_score_year_requires_exact_match():
    assert score_year(2025, 2025).correct is True
    assert score_year(2024, 2025).correct is False


def test_score_set_computes_precision_and_recall():
    s = score_set("founders", ["A B", "C D"], ["A B", "C D", "E F"])
    assert s.precision == 1.0
    assert round(s.recall, 2) == 0.67
    assert s.correct is False  # recall incomplete


def test_score_set_is_correct_only_on_exact_set_match():
    s = score_set("founders", ["A B", "C D"], ["C D", "A B"])
    assert s.correct is True
    assert s.precision == 1.0
    assert s.recall == 1.0


def test_score_set_penalizes_hallucinated_members():
    s = score_set("founders", ["A B", "Z Z"], ["A B"])
    assert s.precision == 0.5
    assert s.recall == 1.0
    assert s.correct is False


def test_score_set_handles_empty_prediction():
    s = score_set("founders", [], ["A B"])
    assert s.precision == 0.0
    assert s.recall == 0.0


def test_score_company_returns_one_score_per_field():
    truth = CompanyFacts(
        company="Acme",
        tier="early",
        funding_usd=9_100_000,
        founded_year=2025,
        founders=["A B"],
        product_line="things",
        required_skills=["Python"],
        recent_events=["raised money"],
    )
    scores = score_company(truth, truth)
    fields = {s.field for s in scores}
    assert fields == {
        "funding_usd",
        "founded_year",
        "founders",
        "recent_events",
    }
    assert all(s.correct for s in scores)


# --- Additional edge cases called out by the task brief's self-review ---


def test_score_funding_at_exactly_five_percent_boundary_is_correct():
    # 5% of 9,100,000 = 455,000 -> truth - 455,000 = 8,645,000, exactly on the edge.
    s = score_funding(8_645_000, 9_100_000)
    assert s.correct is True


def test_score_funding_zero_truth_requires_exact_match_no_crash():
    assert score_funding(0, 0).correct is True
    s = score_funding(500, 0)
    assert s.correct is False


def test_score_funding_truth_present_prediction_none_is_wrong_not_conflated_with_zero():
    s = score_funding(None, 0)
    assert s.correct is False


def test_score_funding_negative_truth_is_not_flipped_to_correct_by_sign():
    # Regression test for the abs(truth) fix: under signed division
    # (delta = abs(pred - truth) / truth), truth=-1,000,000 and
    # pred=-5,000,000 gives delta = abs(-4,000,000) / -1,000,000 = -4.0,
    # which wrongly satisfies `delta <= FUNDING_TOLERANCE`. Dividing by
    # abs(truth) gives delta = 4.0, correctly failing. This test only
    # passes under the abs(truth) implementation.
    s = score_funding(-5_000_000, -1_000_000)
    assert s.correct is False


def test_score_set_duplicates_in_list_do_not_inflate_score():
    s = score_set("founders", ["A B", "A B"], ["A B"])
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.correct is True


def test_score_set_both_empty_is_correct_with_no_precision_recall():
    s = score_set("founders", [], [])
    assert s.correct is True
    assert s.precision is None
    assert s.recall is None


def test_score_set_normalizes_case_and_punctuation_on_both_sides():
    s = score_set("founders", ["patrick collison"], ["Patrick Collison."])
    assert s.correct is True
    assert s.precision == 1.0
    assert s.recall == 1.0


def test_score_company_excludes_product_line_company_tier_and_required_skills():
    # Locks the set of scored fields: `required_skills` was investigated and
    # found to have no annotatable ground truth (real postings routinely
    # don't enumerate skills, and deriving them from prose is interpretation,
    # not extraction -- see evals/score.py's module docstring). This test
    # must fail if scoring is ever reintroduced for it.
    truth = CompanyFacts(company="Acme", tier="early", required_skills=["Python"])
    scores = score_company(truth, truth)
    fields = {s.field for s in scores}
    assert "product_line" not in fields
    assert "company" not in fields
    assert "tier" not in fields
    assert "required_skills" not in fields

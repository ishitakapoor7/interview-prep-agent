import json
import re
from pathlib import Path

import pytest

from app.models import CompanyFacts, LessonPlan, Module, QuizQuestion, ResearchBundle, SourceDoc
from evals.extract import EXTRACT_SCHEMA, extract_facts, load_ground_truth

# The real shipped ground-truth file, not a synthetic tmp_path copy — this is
# what CI actually needs to catch when a hand-annotated row is malformed.
GROUND_TRUTH_PATH = Path(__file__).resolve().parents[1] / "evals" / "ground_truth.json"


class _FakeLlm:
    def __init__(self, payload):
        self._payload = payload
        self.last_user = None

    def complete_json(self, system, user, schema, max_tokens=8192):
        self.last_user = user
        return self._payload


def _empty_plan() -> LessonPlan:
    return LessonPlan(company="Acme", role="SWE", gap_analysis="", modules=[])


def test_extract_schema_lists_every_scored_field():
    props = set(EXTRACT_SCHEMA["properties"])
    assert {
        "funding_usd",
        "founded_year",
        "founders",
        "product_line",
        "required_skills",
        "recent_events",
    } <= props


def test_load_ground_truth_parses_into_company_facts(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(
        json.dumps(
            [
                {
                    "company": "Acme",
                    "tier": "early",
                    "funding_usd": 9100000,
                    "founded_year": 2025,
                    "founders": ["A B"],
                    "product_line": "things",
                    "required_skills": ["Python"],
                    "recent_events": ["raised money"],
                    "job_posting_url": "https://acme.com/jobs",
                }
            ]
        )
    )
    facts = load_ground_truth(str(path))
    assert len(facts) == 1
    assert facts[0].company == "Acme"
    assert facts[0].funding_usd == 9100000
    assert facts[0].founders == ["A B"]


def test_load_ground_truth_loads_the_real_seed_file():
    """Exercises the actual shipped `evals/ground_truth.json`, not a synthetic
    copy. Written to stay valid as the human partner appends the other 27
    hand-annotated rows: it asserts the seed companies are present rather than
    pinning the row count, but it does insist every row in the file parses into
    a valid CompanyFacts, so a malformed appended row fails this test instead of
    surfacing later as a silently wrong eval number."""
    facts = load_ground_truth(str(GROUND_TRUTH_PATH))
    assert len(facts) >= 3
    for f in facts:
        assert isinstance(f, CompanyFacts)
        assert f.tier in ("large", "mid", "early")
    companies = {f.company for f in facts}
    assert {"Stripe", "Mechanize", "Modal"} <= companies


def test_load_ground_truth_rejects_non_list_top_level(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"company": "Acme"}))
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_ground_truth(str(path))


def test_load_ground_truth_rejects_non_dict_row(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps([{"company": "Ok", "tier": "large"}, "not a row"]))
    with pytest.raises(ValueError) as exc_info:
        load_ground_truth(str(path))
    message = str(exc_info.value)
    assert str(path) in message
    # "row 1", not a bare "1" -- a pytest tmp_path routinely contains a "1"
    # (e.g. ".../pytest-38/..."), so a bare-digit assertion can pass even
    # when the row index is never actually named in the message.
    assert "row 1" in message


def test_load_ground_truth_rejects_row_missing_required_field(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(
        json.dumps([{"company": "Ok", "tier": "large"}, {"company": "NoTier"}])
    )
    with pytest.raises(ValueError) as exc_info:
        load_ground_truth(str(path))
    message = str(exc_info.value)
    assert str(path) in message
    assert "row 1" in message
    assert "tier" in message


def test_load_ground_truth_rejects_invalid_tier(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(
        json.dumps(
            [
                {"company": "Ok", "tier": "large"},
                {"company": "Typo", "tier": "midd"},
            ]
        )
    )
    with pytest.raises(ValueError) as exc_info:
        load_ground_truth(str(path))
    message = str(exc_info.value)
    assert str(path) in message
    assert "row 1" in message
    assert "midd" in message


def test_load_ground_truth_missing_file_raises_clear_error(tmp_path):
    path = tmp_path / "does_not_exist.json"
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_ground_truth(str(path))


def test_load_ground_truth_malformed_json_raises_clear_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{this is not valid json")
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_ground_truth(str(path))


def test_extract_facts_reads_the_lesson_plan_not_the_bundle():
    bundle = ResearchBundle(
        company="Acme",
        role="SWE",
        docs=[
            SourceDoc(
                "news", "u", "t", "Bundle-only detail that must not leak into extraction.", "ts"
            )
        ],
    )
    plan = LessonPlan(
        company="Acme",
        role="SWE",
        gap_analysis="No major gaps identified.",
        modules=[
            Module(
                number=1,
                title="Company Overview",
                content="Acme raised $9.1M in 2025.",
                quiz=[QuizQuestion(question="When was Acme founded?", expected_points=["2025"])],
            )
        ],
    )
    llm = _FakeLlm(
        {
            "funding_usd": 9100000,
            "founded_year": 2025,
            "founders": ["A B"],
            "product_line": "things",
            "required_skills": ["Python"],
            "recent_events": ["raised money"],
        }
    )
    facts = extract_facts(bundle, plan, "early", llm)
    assert facts.company == "Acme"
    assert facts.tier == "early"
    assert facts.funding_usd == 9100000
    assert "Acme raised $9.1M in 2025." in llm.last_user
    assert "No major gaps identified." in llm.last_user
    assert "Bundle-only detail" not in llm.last_user


def test_extract_facts_defaults_nulls_to_none_and_empty_lists():
    bundle = ResearchBundle(company="Acme", role="SWE", docs=[])
    facts = extract_facts(bundle, _empty_plan(), "early", _FakeLlm({}))
    assert facts.funding_usd is None
    assert facts.founded_year is None
    assert facts.founders == []
    assert facts.recent_events == []

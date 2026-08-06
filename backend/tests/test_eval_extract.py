import json

from app.models import ResearchBundle, SourceDoc
from evals.extract import EXTRACT_SCHEMA, extract_facts, load_ground_truth


class _FakeLlm:
    def __init__(self, payload):
        self._payload = payload
        self.last_user = None

    def complete_json(self, system, user, schema, max_tokens=8192):
        self.last_user = user
        return self._payload


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


def test_extract_facts_maps_model_output_into_company_facts():
    bundle = ResearchBundle(
        company="Acme",
        role="SWE",
        docs=[SourceDoc("news", "u", "t", "Acme raised $9.1M in 2025.", "ts")],
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
    facts = extract_facts(bundle, "early", llm)
    assert facts.company == "Acme"
    assert facts.tier == "early"
    assert facts.funding_usd == 9100000
    assert "Acme raised $9.1M in 2025." in llm.last_user


def test_extract_facts_defaults_nulls_to_none_and_empty_lists():
    bundle = ResearchBundle(company="Acme", role="SWE", docs=[])
    facts = extract_facts(bundle, "early", _FakeLlm({}))
    assert facts.funding_usd is None
    assert facts.founded_year is None
    assert facts.founders == []
    assert facts.recent_events == []

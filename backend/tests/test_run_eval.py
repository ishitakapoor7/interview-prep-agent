"""Offline tests for the eval harness. Every test injects a fake LLM and/or a
fake researcher -- no test in this file makes a network call or needs an API
key, per the Task 11 dispatch's environment constraints.
"""

import argparse
import json

import pytest

from app.models import MODULE_TITLES, CompanyFacts, ResearchBundle, SourceDoc
from evals import run_eval
from evals.run_eval import _positive_int, run_all, run_company, select_rows


class _FakeLlm:
    """Dispatches on schema shape: the lesson schema has a `modules` property,
    the extract schema has `funding_usd`. Mirrors how `LlmClient.complete_json`
    is actually called by generate_lesson_plan vs extract_facts."""

    def __init__(self, lesson_payload, extract_payload):
        self._lesson_payload = lesson_payload
        self._extract_payload = extract_payload
        self.calls = []

    def complete_json(self, system, user, schema, max_tokens=8192):
        self.calls.append(schema)
        if "modules" in schema["properties"]:
            return self._lesson_payload
        return self._extract_payload


def _lesson_payload():
    return {
        "gap_analysis": "You match 2 of 3 required skills.",
        "modules": [
            {
                "number": i + 1,
                "title": title,
                "content": f"Acme was founded in 2025. content for {title}",
                "quiz": [],
            }
            for i, title in enumerate(MODULE_TITLES)
        ],
    }


def _extract_payload(**overrides):
    base = {
        "funding_usd": 9_100_000,
        "founded_year": 2025,
        "founders": ["Ada Lovelace"],
        "product_line": "things",
        "required_skills": ["Python"],
        "recent_events": ["raised money"],
    }
    base.update(overrides)
    return base


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


async def _fake_researcher(company: str) -> ResearchBundle:
    return ResearchBundle(
        company=company,
        role="Software Engineer",
        docs=[SourceDoc("news", "u", "t", "Acme raised $9.1M in 2025.", "ts")],
    )


# --- run_company -------------------------------------------------------------


async def test_run_company_uses_injected_researcher_no_network(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    llm = _FakeLlm(_lesson_payload(), _extract_payload())

    result = await run_company(_truth(), llm, researcher=_fake_researcher)

    assert result["company"] == "Acme"
    assert result["tier"] == "early"
    assert {s.field for s in result["scores"]} == {
        "funding_usd",
        "founded_year",
        "founders",
        "required_skills",
        "recent_events",
    }
    assert all(s.correct for s in result["scores"])
    assert result["attributions"] == ["none"] * 5


async def test_run_company_writes_a_run_log_with_bundle_plan_and_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    llm = _FakeLlm(_lesson_payload(), _extract_payload())

    await run_company(_truth(), llm, researcher=_fake_researcher)

    run_file = tmp_path / "Acme.json"
    assert run_file.exists()
    logged = json.loads(run_file.read_text())
    assert set(logged) == {"bundle", "plan", "predicted", "truth", "scores", "attributions"}
    assert logged["truth"]["company"] == "Acme"
    assert logged["plan"]["modules"][0]["title"] == MODULE_TITLES[0]


async def test_run_company_wrong_field_is_attributed_not_just_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    # Extraction misreads founded_year even though "2025" is right there in the
    # plan text baked into _lesson_payload's module content -- an extraction
    # failure end to end through the real harness path.
    llm = _FakeLlm(_lesson_payload(), _extract_payload(founded_year=1999))

    result = await run_company(_truth(), llm, researcher=_fake_researcher)

    by_field = dict(zip((s.field for s in result["scores"]), result["attributions"]))
    assert by_field["founded_year"] == "extraction"


async def test_run_company_sanitizes_company_name_for_run_log_path(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    llm = _FakeLlm(_lesson_payload(), _extract_payload())

    await run_company(_truth(company="A/B"), llm, researcher=_fake_researcher)

    assert (tmp_path / "A_B.json").exists()


async def test_run_company_survives_a_run_log_write_failure(tmp_path, monkeypatch, caplog):
    # A read-only RUNS_DIR simulates a disk/permissions failure while writing
    # the run log. The eval score itself must still be returned -- losing the
    # on-disk trace for one company must not lose that company's result.
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))

    def _boom(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("builtins.open", _boom)
    llm = _FakeLlm(_lesson_payload(), _extract_payload())

    with caplog.at_level("ERROR"):
        result = await run_company(_truth(), llm, researcher=_fake_researcher)

    assert result["company"] == "Acme"
    assert all(s.correct for s in result["scores"])
    assert "Acme" in caplog.text


async def test_run_company_survives_a_makedirs_failure(tmp_path, monkeypatch, caplog):
    # Regression for the review finding: os.makedirs used to sit outside the
    # try/except that covers the write itself, so a directory-creation
    # failure crashed run_company entirely while the identical failure one
    # line later (open()) was tolerated. Both must now be handled the same
    # way.
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "unwritable"))

    def _boom(*args, **kwargs):
        raise OSError("simulated permissions failure")

    monkeypatch.setattr("os.makedirs", _boom)
    llm = _FakeLlm(_lesson_payload(), _extract_payload())

    with caplog.at_level("ERROR"):
        result = await run_company(_truth(), llm, researcher=_fake_researcher)

    assert result["company"] == "Acme"
    assert all(s.correct for s in result["scores"])
    assert "Acme" in caplog.text


# --- select_rows ---------------------------------------------------------


def _rows():
    return [
        _truth(company="A", tier="large"),
        _truth(company="B", tier="mid"),
        _truth(company="C", tier="early"),
        _truth(company="D", tier="early"),
    ]


def test_select_rows_filters_by_tier():
    rows = select_rows(_rows(), tier="early", limit=None)
    assert [r.company for r in rows] == ["C", "D"]


def test_select_rows_applies_limit_after_tier_filter():
    rows = select_rows(_rows(), tier="early", limit=1)
    assert [r.company for r in rows] == ["C"]


def test_select_rows_limit_alone_caps_the_full_set():
    rows = select_rows(_rows(), tier=None, limit=2)
    assert [r.company for r in rows] == ["A", "B"]


def test_select_rows_no_filters_returns_everything():
    rows = select_rows(_rows(), tier=None, limit=None)
    assert len(rows) == 4


def test_select_rows_limit_zero_returns_empty_list_not_the_full_set():
    # Review finding: `if limit:` treats 0 the same as None (both falsy), so
    # `--limit 0` used to silently run the entire set instead of nothing.
    # limit must be checked against `None`, not truthiness.
    rows = select_rows(_rows(), tier=None, limit=0)
    assert rows == []


# --- --limit CLI validation ------------------------------------------------


def test_positive_int_accepts_positive_values():
    assert _positive_int("3") == 3


def test_positive_int_rejects_zero():
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")


def test_positive_int_rejects_negative_values():
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("-1")


def test_cli_limit_flag_rejects_zero_and_negative_at_the_argparse_boundary():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=_positive_int, default=None)

    with pytest.raises(SystemExit):
        parser.parse_args(["--limit", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--limit", "-1"])

    args = parser.parse_args(["--limit", "5"])
    assert args.limit == 5


# --- run_all: one company failing must not abort the run --------------------


async def test_run_all_continues_past_a_failing_company(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    llm = _FakeLlm(_lesson_payload(), _extract_payload())

    async def flaky_researcher(company: str) -> ResearchBundle:
        if company == "Bad":
            raise RuntimeError("simulated network failure")
        return await _fake_researcher(company)

    rows = [_truth(company="Bad", tier="early"), _truth(company="Good", tier="early")]
    results = await run_all(rows, llm, researcher=flaky_researcher)

    assert [r["company"] for r in results] == ["Bad", "Good"]
    assert results[0]["scores"] == []
    assert results[0]["attributions"] == []
    # "error" is what evals.attribute.summarize keys off of to keep this
    # company visible in RESULTS.md instead of silently dropping out.
    assert "simulated network failure" in results[0]["error"]
    assert "error" not in results[1]
    assert results[1]["scores"] != []
    assert all(s.correct for s in results[1]["scores"])


async def test_run_all_logs_the_failure(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    llm = _FakeLlm(_lesson_payload(), _extract_payload())

    async def always_fails(company: str) -> ResearchBundle:
        raise RuntimeError("boom")

    with caplog.at_level("ERROR"):
        await run_all([_truth(company="Bad", tier="early")], llm, researcher=always_fails)

    assert "Bad" in caplog.text

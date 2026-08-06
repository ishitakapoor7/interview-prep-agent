import logging

import pytest

from app.lesson.generate import generate_lesson_plan
from app.lesson.schema import LESSON_SCHEMA
from app.models import MODULE_TITLES, ResearchBundle, SourceDoc


class _FakeLlm:
    def __init__(self, payload):
        self._payload = payload
        self.last_user = None

    def complete_json(self, system, user, schema, max_tokens=8192):
        self.last_user = user
        return self._payload


def _bundle():
    return ResearchBundle(
        company="Acme",
        role="SWE",
        docs=[SourceDoc("news", "u1", "t1", "Acme raised $9.1M.", "2026-08-04T00:00:00Z")],
    )


def _title_for(i: int) -> str:
    # Beyond len(MODULE_TITLES) (used by the over-count test) the extra entries
    # just need distinct titles — their content is never used.
    return MODULE_TITLES[i] if i < len(MODULE_TITLES) else f"Extra Module {i}"


def _payload(n_modules=7):
    return {
        "gap_analysis": "You match 3 of 5 required skills.",
        "modules": [
            {
                "number": i + 1,
                "title": _title_for(i),
                "content": f"content for {_title_for(i)}",
                "quiz": [
                    {"question": f"q{i}?", "expected_points": [f"point{i}"]},
                ],
            }
            for i in range(n_modules)
        ],
    }


def test_lesson_schema_requires_gap_analysis_and_modules():
    assert set(LESSON_SCHEMA["required"]) == {"gap_analysis", "modules"}


def test_lesson_schema_pins_modules_array_to_exactly_seven():
    assert LESSON_SCHEMA["properties"]["modules"]["minItems"] == 7
    assert LESSON_SCHEMA["properties"]["modules"]["maxItems"] == 7


def test_generate_returns_seven_typed_modules():
    plan = generate_lesson_plan(_bundle(), "my resume", _FakeLlm(_payload()))
    assert len(plan.modules) == 7
    assert plan.company == "Acme"
    assert plan.gap_analysis.startswith("You match")
    assert plan.modules[0].quiz[0].question == "q0?"
    assert plan.modules[0].quiz[0].expected_points == ["point0"]


def test_generate_sends_both_evidence_and_resume_to_the_model():
    llm = _FakeLlm(_payload())
    generate_lesson_plan(_bundle(), "RESUME_MARKER", llm)
    assert "Acme raised $9.1M." in llm.last_user
    assert "RESUME_MARKER" in llm.last_user


def test_generate_normalizes_module_numbers_and_titles():
    payload = _payload()
    payload["modules"][2]["number"] = 99
    payload["modules"][2]["title"] = "Wrong Title"
    plan = generate_lesson_plan(_bundle(), "r", _FakeLlm(payload))
    assert plan.modules[2].number == 3
    assert plan.modules[2].title == MODULE_TITLES[2]
    # Content is preserved even though the title/number were garbage — the
    # module falls back to its original array position.
    assert plan.modules[2].content == "content for Team & Culture"


def test_generate_raises_when_module_count_is_too_low():
    with pytest.raises(ValueError, match="expected 7 modules"):
        generate_lesson_plan(_bundle(), "r", _FakeLlm(_payload(n_modules=5)))


def test_generate_raises_when_module_count_is_too_high():
    with pytest.raises(ValueError, match="expected 7 modules"):
        generate_lesson_plan(_bundle(), "r", _FakeLlm(_payload(n_modules=8)))


def test_generate_tolerates_a_module_with_no_quiz():
    payload = _payload()
    payload["modules"][1]["quiz"] = []
    plan = generate_lesson_plan(_bundle(), "r", _FakeLlm(payload))
    assert plan.modules[1].quiz == []


def test_generate_matches_modules_by_title_when_model_returns_them_shuffled():
    # This is the regression test for the order-mismatch bug: a model that
    # returns all seven correct titles, just not in MODULE_TITLES order, must
    # still land each module's content in its correct canonical slot.
    payload = _payload()
    shuffle = [6, 0, 5, 1, 4, 2, 3]
    payload["modules"] = [payload["modules"][i] for i in shuffle]

    plan = generate_lesson_plan(_bundle(), "r", _FakeLlm(payload))

    for i, title in enumerate(MODULE_TITLES):
        assert plan.modules[i].number == i + 1
        assert plan.modules[i].title == title
        assert plan.modules[i].content == f"content for {title}"
        assert plan.modules[i].quiz[0].question == f"q{i}?"


def test_generate_duplicate_title_claim_loser_fills_remaining_slot(caplog):
    # Two modules both claim "Product & Technology"; nothing claims "The
    # Role". By trace: the earlier (index 1) claimant wins the slot it
    # claimed; the later (index 3, the original "The Role" module, now
    # retitled) loses the claim, re-enters the unused-raw pool, and fills the
    # one remaining unfilled slot via positional fallback. Both the
    # duplicate-claim warning and the positional-fallback warning must fire.
    payload = _payload()
    payload["modules"][3]["title"] = "Product & Technology"

    with caplog.at_level(logging.WARNING, logger="app.lesson.generate"):
        plan = generate_lesson_plan(_bundle(), "r", _FakeLlm(payload))

    # Slot 1 ("Product & Technology") is won by raw index 1, the first claimant.
    assert plan.modules[1].title == "Product & Technology"
    assert plan.modules[1].content == "content for Product & Technology"

    # Slot 3 ("The Role") has no claimant left, so it's filled positionally by
    # the loser of the duplicate claim -- raw index 3, whose content is
    # unchanged even though its title was overwritten.
    assert plan.modules[3].title == "The Role"
    assert plan.modules[3].content == "content for The Role"

    messages = [r.message for r in caplog.records]
    assert any("claimed title" in m for m in messages)
    assert any("positional fallback" in m for m in messages)


def test_generate_falls_back_to_position_for_a_noncanonical_title(caplog):
    payload = _payload()
    payload["modules"][0]["title"] = "Totally Made Up Title"

    with caplog.at_level(logging.WARNING, logger="app.lesson.generate"):
        plan = generate_lesson_plan(_bundle(), "r", _FakeLlm(payload))

    assert plan.modules[0].title == MODULE_TITLES[0]
    assert plan.modules[0].content == "content for Company Overview"
    assert any(
        "positional fallback" in record.message for record in caplog.records
    )

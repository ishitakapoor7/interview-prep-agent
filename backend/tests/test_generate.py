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


def _payload(n_modules=7):
    return {
        "gap_analysis": "You match 3 of 5 required skills.",
        "modules": [
            {
                "number": i + 1,
                "title": MODULE_TITLES[i],
                "content": f"content for {MODULE_TITLES[i]}",
                "quiz": [
                    {"question": f"q{i}?", "expected_points": [f"point{i}"]},
                ],
            }
            for i in range(n_modules)
        ],
    }


def test_lesson_schema_requires_gap_analysis_and_modules():
    assert set(LESSON_SCHEMA["required"]) == {"gap_analysis", "modules"}


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


def test_generate_raises_when_module_count_is_wrong():
    with pytest.raises(ValueError, match="expected 7 modules"):
        generate_lesson_plan(_bundle(), "r", _FakeLlm(_payload(n_modules=5)))


def test_generate_tolerates_a_module_with_no_quiz():
    payload = _payload()
    payload["modules"][1]["quiz"] = []
    plan = generate_lesson_plan(_bundle(), "r", _FakeLlm(payload))
    assert plan.modules[1].quiz == []

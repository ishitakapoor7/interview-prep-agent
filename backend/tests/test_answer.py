from app.models import QuizQuestion, ResearchBundle, SourceDoc
from app.qa.answer import GRADE_SCHEMA, answer_question, grade_answer


class _FakeLlm:
    def __init__(self, text="", payload=None):
        self._text = text
        self._payload = payload or {}
        self.last_user = None

    def complete(self, system, user, max_tokens=4096):
        self.last_user = user
        return self._text

    def complete_json(self, system, user, schema, max_tokens=8192):
        self.last_user = user
        return self._payload


def _bundle():
    return ResearchBundle(
        company="Acme",
        role="SWE",
        docs=[
            SourceDoc("news", "https://n1", "Funding", "Acme raised $9.1M.", "t"),
            SourceDoc("blog", "https://b1", "Stack", "We run Kubernetes.", "t"),
        ],
    )


def test_answer_question_returns_text_and_cited_urls():
    llm = _FakeLlm(text="They raised $9.1M.")
    answer, urls = answer_question(_bundle(), "How much did they raise?", llm)
    assert answer == "They raised $9.1M."
    assert "https://n1" in urls


def test_answer_question_grounds_the_prompt_in_retrieved_docs():
    llm = _FakeLlm(text="ok")
    answer_question(_bundle(), "funding raised", llm)
    assert "Acme raised $9.1M." in llm.last_user


def test_answer_question_handles_empty_bundle():
    llm = _FakeLlm(text="unused")
    empty = ResearchBundle(company="Acme", role="SWE", docs=[])
    answer, urls = answer_question(empty, "anything", llm)
    assert urls == []
    assert "don't have" in answer.lower() or "no research" in answer.lower()


def test_grade_schema_requires_verdict_and_feedback():
    assert set(GRADE_SCHEMA["required"]) == {"verdict", "feedback", "missed_points"}


def test_grade_answer_passes_expected_points_to_the_model():
    q = QuizQuestion(question="How much?", expected_points=["$9.1M", "April 2026"])
    llm = _FakeLlm(payload={"verdict": "partial", "feedback": "close", "missed_points": ["April 2026"]})
    result = grade_answer(q, "nine million", llm)
    assert result["verdict"] == "partial"
    assert result["missed_points"] == ["April 2026"]
    assert "$9.1M" in llm.last_user
    assert "nine million" in llm.last_user


def test_grade_answer_defaults_missing_fields():
    q = QuizQuestion(question="q", expected_points=["p"])
    result = grade_answer(q, "a", _FakeLlm(payload={}))
    assert result["verdict"] == "incorrect"
    assert result["missed_points"] == []

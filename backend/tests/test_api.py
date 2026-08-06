import pytest
from fastapi.testclient import TestClient

from app.main import app, get_deps
from app.models import LessonPlan, Module, QuizQuestion, ResearchBundle, SourceDoc
from app.storage.db import SessionStore


class _Deps:
    def __init__(self, store):
        self.store = store
        self.llm = _FakeLlm()

    async def research(self, company, role, job_url=None):
        return ResearchBundle(
            company=company,
            role=role,
            docs=[SourceDoc("news", "https://n1", "t", "Acme raised $9.1M.", "ts")],
            gaps=["g"],
            reflection_queries=["q"],
        )

    def plan(self, bundle, resume_text):
        return LessonPlan(
            company=bundle.company,
            role=bundle.role,
            gap_analysis="ga",
            modules=[
                Module(i + 1, f"M{i+1}", f"content{i+1}", [QuizQuestion("q?", ["p"])])
                for i in range(7)
            ],
        )


class _FakeLlm:
    def complete(self, system, user, max_tokens=4096):
        return "grounded answer"

    def complete_json(self, system, user, schema, max_tokens=8192):
        return {"verdict": "correct", "feedback": "nice", "missed_points": []}


@pytest.fixture
def client(tmp_path):
    store = SessionStore(str(tmp_path / "api.db"))
    store.init_schema()
    deps = _Deps(store)
    app.dependency_overrides[get_deps] = lambda: deps
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_session_returns_id_and_gap_analysis(client):
    r = client.post(
        "/sessions",
        json={"company": "Acme", "role": "SWE", "resume_text": "my resume"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert body["gap_analysis"] == "ga"
    assert len(body["module_titles"]) == 7


def test_get_session_returns_plan_and_research_gaps(client):
    sid = client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    r = client.get(f"/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["research_gaps"] == ["g"]


def test_get_module_returns_content_and_quiz(client):
    sid = client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    r = client.get(f"/sessions/{sid}/modules/1")
    assert r.status_code == 200
    assert r.json()["content"] == "content1"
    assert r.json()["quiz"][0]["question"] == "q?"


def test_get_module_404s_out_of_range(client):
    sid = client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    assert client.get(f"/sessions/{sid}/modules/99").status_code == 404


def test_ask_returns_answer_and_sources(client):
    sid = client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    r = client.post(f"/sessions/{sid}/ask", json={"question": "funding?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "grounded answer"
    assert r.json()["sources"] == ["https://n1"]


def test_grade_returns_verdict(client):
    sid = client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    r = client.post(
        f"/sessions/{sid}/grade",
        json={"module_number": 1, "question_index": 0, "answer": "nine million"},
    )
    assert r.status_code == 200
    assert r.json()["verdict"] == "correct"


def test_unknown_session_404s(client):
    assert client.get("/sessions/nope").status_code == 404


def test_get_module_non_integer_number_is_422_not_500(client):
    sid = client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    assert client.get(f"/sessions/{sid}/modules/abc").status_code == 422


def test_get_module_zero_and_negative_404(client):
    sid = client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    assert client.get(f"/sessions/{sid}/modules/0").status_code == 404
    assert client.get(f"/sessions/{sid}/modules/-1").status_code == 404


def test_grade_unknown_session_404s(client):
    r = client.post(
        "/sessions/nope/grade",
        json={"module_number": 1, "question_index": 0, "answer": "x"},
    )
    assert r.status_code == 404


def test_grade_out_of_range_module_404s(client):
    sid = client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    r = client.post(
        f"/sessions/{sid}/grade",
        json={"module_number": 99, "question_index": 0, "answer": "x"},
    )
    assert r.status_code == 404


def test_grade_out_of_range_question_index_404s(client):
    sid = client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    r = client.post(
        f"/sessions/{sid}/grade",
        json={"module_number": 1, "question_index": 5, "answer": "x"},
    )
    assert r.status_code == 404


def test_ask_unknown_session_404s(client):
    r = client.post("/sessions/nope/ask", json={"question": "funding?"})
    assert r.status_code == 404


def test_create_session_malformed_body_422s(client):
    r = client.post("/sessions", json={"company": "Acme"})
    assert r.status_code == 422


class _RaisingResearchDeps(_Deps):
    async def research(self, company, role, job_url=None):
        raise RuntimeError("network exploded")


@pytest.fixture
def failing_pipeline_client(tmp_path):
    store = SessionStore(str(tmp_path / "fail.db"))
    store.init_schema()
    deps = _RaisingResearchDeps(store)
    app.dependency_overrides[get_deps] = lambda: deps
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_session_pipeline_failure_is_502_not_500(failing_pipeline_client):
    r = failing_pipeline_client.post(
        "/sessions",
        json={"company": "Acme", "role": "SWE", "resume_text": "r"},
    )
    assert r.status_code == 502


class _RaisingLlm:
    def complete(self, system, user, max_tokens=4096):
        raise RuntimeError("llm api error")

    def complete_json(self, system, user, schema, max_tokens=8192):
        raise RuntimeError("llm api error")


class _RaisingLlmDeps(_Deps):
    def __init__(self, store):
        super().__init__(store)
        self.llm = _RaisingLlm()


@pytest.fixture
def failing_llm_client(tmp_path):
    store = SessionStore(str(tmp_path / "fail_llm.db"))
    store.init_schema()
    deps = _RaisingLlmDeps(store)
    app.dependency_overrides[get_deps] = lambda: deps
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ask_llm_failure_is_502_not_500(failing_llm_client):
    sid = failing_llm_client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    r = failing_llm_client.post(f"/sessions/{sid}/ask", json={"question": "funding?"})
    assert r.status_code == 502


def test_grade_llm_failure_is_502_not_500(failing_llm_client):
    sid = failing_llm_client.post(
        "/sessions", json={"company": "Acme", "role": "SWE", "resume_text": "r"}
    ).json()["session_id"]
    r = failing_llm_client.post(
        f"/sessions/{sid}/grade",
        json={"module_number": 1, "question_index": 0, "answer": "x"},
    )
    assert r.status_code == 502

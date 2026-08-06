from app.models import LessonPlan, Module, QuizQuestion, ResearchBundle, SourceDoc
from app.storage.db import SessionStore


def _fixtures():
    bundle = ResearchBundle(
        company="Acme",
        role="SWE",
        docs=[SourceDoc("news", "u1", "t1", "body", "2026-08-04T00:00:00Z")],
        gaps=["missing year"],
        reflection_queries=["Acme founded"],
    )
    plan = LessonPlan(
        company="Acme",
        role="SWE",
        gap_analysis="ga",
        modules=[
            Module(1, "Company Overview", "c", [QuizQuestion("q?", ["p"])]),
        ],
    )
    return bundle, plan


def test_save_and_load_roundtrips_bundle_and_plan(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    store.init_schema()
    bundle, plan = _fixtures()
    store.save("s1", bundle, plan)

    loaded = store.load("s1")
    assert loaded is not None
    got_bundle, got_plan = loaded
    assert got_bundle.company == "Acme"
    assert got_bundle.docs[0].url == "u1"
    assert got_bundle.gaps == ["missing year"]
    assert got_plan.modules[0].quiz[0].expected_points == ["p"]


def test_load_returns_none_for_unknown_session(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    store.init_schema()
    assert store.load("nope") is None


def test_save_is_idempotent_on_same_id(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    store.init_schema()
    bundle, plan = _fixtures()
    store.save("s1", bundle, plan)
    bundle.company = "Acme2"
    store.save("s1", bundle, plan)
    loaded = store.load("s1")
    assert loaded is not None
    assert loaded[0].company == "Acme2"

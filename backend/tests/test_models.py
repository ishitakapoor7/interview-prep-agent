from app.models import (
    MODULE_TITLES,
    CompanyFacts,
    LessonPlan,
    Module,
    QuizQuestion,
    ResearchBundle,
    SourceDoc,
)


def test_module_titles_has_seven_entries_in_fixed_order():
    assert len(MODULE_TITLES) == 7
    assert MODULE_TITLES[0] == "Company Overview"
    assert MODULE_TITLES[6] == "Interview Prep"


def test_source_doc_is_hashable_and_carries_provenance():
    doc = SourceDoc(
        source_type="news",
        url="https://example.com/a",
        title="Example raises money",
        content="Example raised $9.1M.",
        retrieved_at="2026-08-04T00:00:00Z",
    )
    assert doc.url == "https://example.com/a"
    assert {doc}  # frozen dataclass is hashable


def test_bundle_can_find_docs_by_type():
    news = SourceDoc("news", "u1", "t1", "c1", "2026-08-04T00:00:00Z")
    site = SourceDoc("website", "u2", "t2", "c2", "2026-08-04T00:00:00Z")
    bundle = ResearchBundle(company="Acme", role="SWE", docs=[news, site])
    assert bundle.by_type("news") == [news]
    assert bundle.by_type("blog") == []


def test_bundle_all_text_concatenates_every_doc():
    bundle = ResearchBundle(
        company="Acme",
        role="SWE",
        docs=[
            SourceDoc("news", "u1", "t1", "raised $9.1M", "2026-08-04T00:00:00Z"),
            SourceDoc("website", "u2", "t2", "we build things", "2026-08-04T00:00:00Z"),
        ],
    )
    text = bundle.all_text()
    assert "raised $9.1M" in text
    assert "we build things" in text


def test_lesson_plan_holds_seven_modules_with_quizzes():
    modules = [
        Module(
            number=i + 1,
            title=title,
            content=f"content {i}",
            quiz=[QuizQuestion(question="q?", expected_points=["p1"])],
        )
        for i, title in enumerate(MODULE_TITLES)
    ]
    plan = LessonPlan(
        company="Acme", role="SWE", gap_analysis="you match 3 of 5", modules=modules
    )
    assert len(plan.modules) == 7
    assert plan.modules[3].title == "The Role"


def test_company_facts_defaults_are_empty_not_none():
    facts = CompanyFacts(company="Acme", tier="early")
    assert facts.founders == []
    assert facts.required_skills == []
    assert facts.recent_events == []
    assert facts.funding_usd is None

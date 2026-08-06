from app.models import ResearchBundle, SourceDoc
from app.research.reflect import GAP_SCHEMA, identify_gaps, reflect_and_fill


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
        docs=[SourceDoc("news", "u1", "t1", "Acme raised money.", "2026-08-04T00:00:00Z")],
    )


def test_gap_schema_requires_both_lists():
    assert GAP_SCHEMA["type"] == "object"
    assert set(GAP_SCHEMA["required"]) == {"gaps", "followup_queries"}


def test_identify_gaps_returns_both_lists():
    llm = _FakeLlm({"gaps": ["no founding year"], "followup_queries": ["Acme founded"]})
    gaps, queries = identify_gaps(_bundle(), llm)
    assert gaps == ["no founding year"]
    assert queries == ["Acme founded"]


def test_identify_gaps_sends_bundle_text_to_the_model():
    llm = _FakeLlm({"gaps": [], "followup_queries": []})
    identify_gaps(_bundle(), llm)
    assert "Acme raised money." in llm.last_user


def test_identify_gaps_tolerates_missing_keys():
    llm = _FakeLlm({})
    assert identify_gaps(_bundle(), llm) == ([], [])


async def test_reflect_and_fill_adds_new_docs_and_records_gaps():
    llm = _FakeLlm(
        {"gaps": ["founding year unknown"], "followup_queries": ["Acme founded year"]}
    )

    def searcher(query, *, source_type, max_results=3, client=None):
        return [SourceDoc("news", "u2", "t2", "Founded in 2021.", "2026-08-04T00:00:00Z")]

    out = await reflect_and_fill(_bundle(), llm, searcher=searcher)
    assert out.gaps == ["founding year unknown"]
    assert out.reflection_queries == ["Acme founded year"]
    assert any("Founded in 2021." in d.content for d in out.docs)


async def test_reflect_and_fill_is_a_noop_when_no_gaps():
    llm = _FakeLlm({"gaps": [], "followup_queries": []})
    calls = []

    def searcher(query, *, source_type, max_results=3, client=None):
        calls.append(query)
        return []

    out = await reflect_and_fill(_bundle(), llm, searcher=searcher)
    assert calls == []
    assert len(out.docs) == 1


async def test_reflect_and_fill_does_not_duplicate_existing_urls():
    llm = _FakeLlm({"gaps": ["g"], "followup_queries": ["q"]})

    def searcher(query, *, source_type, max_results=3, client=None):
        return [SourceDoc("news", "u1", "dup", "dup body", "2026-08-04T00:00:00Z")]

    out = await reflect_and_fill(_bundle(), llm, searcher=searcher)
    assert len([d for d in out.docs if d.url == "u1"]) == 1

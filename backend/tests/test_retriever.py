from app.models import ResearchBundle, SourceDoc
from app.qa.retriever import retrieve, tokenize


def _bundle():
    return ResearchBundle(
        company="Acme",
        role="SWE",
        docs=[
            SourceDoc("news", "u1", "Funding", "Acme raised nine million dollars.", "t"),
            SourceDoc("blog", "u2", "Stack", "We run Kubernetes and Postgres.", "t"),
            SourceDoc("glassdoor", "u3", "Interviews", "Two rounds of system design.", "t"),
        ],
    )


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Acme's Kubernetes, Postgres!") == {"acme", "s", "kubernetes", "postgres"}


def test_retrieve_ranks_the_lexically_closest_doc_first():
    # "run" (not a stopword) is the token that actually overlaps doc u2's
    # content ("We run Kubernetes and Postgres."); every other query word here
    # is filtered as a stopword or matches nothing in the bundle.
    docs = retrieve(_bundle(), "what does their infrastructure run", top_k=3)
    assert docs[0].url == "u2"


def test_retrieve_respects_top_k():
    assert len(retrieve(_bundle(), "Acme", top_k=1)) == 1


def test_retrieve_returns_empty_for_empty_bundle():
    empty = ResearchBundle(company="Acme", role="SWE", docs=[])
    assert retrieve(empty, "anything") == []


def test_retrieve_falls_back_to_first_docs_when_nothing_overlaps():
    docs = retrieve(_bundle(), "zzzzz qqqqq", top_k=2)
    assert len(docs) == 2  # never returns nothing when the bundle is non-empty

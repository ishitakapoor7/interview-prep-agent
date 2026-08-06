import time

import pytest

from app.models import SourceDoc
from app.research.orchestrator import build_queries, gather_research


def _doc(source_type, url="u"):
    return SourceDoc(source_type, url, "t", "c", "2026-08-04T00:00:00Z")


def test_build_queries_covers_every_expected_source_type():
    pairs = build_queries("Acme", "Backend Engineer")
    types = {t for _q, t in pairs}
    assert {"website", "blog", "linkedin", "glassdoor", "news", "crunchbase", "video"} <= types


def test_build_queries_mentions_company_in_every_query():
    for query, _t in build_queries("Acme", "Backend Engineer"):
        assert "Acme" in query


async def test_gather_research_collects_docs_from_all_queries():
    def searcher(query, *, source_type, max_results=3, client=None):
        return [_doc(source_type, url=f"https://{source_type}.com")]

    bundle = await gather_research("Acme", "SWE", searcher=searcher)
    assert bundle.company == "Acme"
    assert bundle.role == "SWE"
    types = {d.source_type for d in bundle.docs}
    assert "news" in types and "website" in types
    assert bundle.gaps == []


async def test_gather_research_survives_a_failing_source():
    def searcher(query, *, source_type, max_results=3, client=None):
        if source_type == "glassdoor":
            raise RuntimeError("blocked")
        return [_doc(source_type)]

    bundle = await gather_research("Acme", "SWE", searcher=searcher)
    assert bundle.docs  # other sources still made it
    assert not bundle.by_type("glassdoor")


async def test_gather_research_scrapes_job_url_when_given():
    def searcher(query, *, source_type, max_results=3, client=None):
        return []

    def scraper(url, *, source_type, fetcher=None):
        return _doc("job_posting", url=url)

    bundle = await gather_research(
        "Acme", "SWE", job_url="https://acme.com/jobs/1", searcher=searcher, scraper=scraper
    )
    posting = bundle.by_type("job_posting")
    assert len(posting) == 1
    assert posting[0].url == "https://acme.com/jobs/1"


async def test_gather_research_deduplicates_by_url():
    def searcher(query, *, source_type, max_results=3, client=None):
        return [_doc(source_type, url="https://same.com")]

    bundle = await gather_research("Acme", "SWE", searcher=searcher)
    urls = [d.url for d in bundle.docs]
    assert len(urls) == len(set(urls))


async def test_gather_research_runs_sources_concurrently():
    """Each searcher call blocks for a fixed slice of time. If gather_research
    fanned the queries out sequentially, total wall-clock time would scale with
    the number of queries (8 * SLEEP). Run concurrently, it should stay close to
    one SLEEP regardless of how many source types exist."""

    SLEEP = 0.2

    def searcher(query, *, source_type, max_results=3, client=None):
        time.sleep(SLEEP)
        return [_doc(source_type, url=f"https://{source_type}.com")]

    num_queries = len(build_queries("Acme", "SWE"))
    assert num_queries >= 6  # sanity: this test is only meaningful with several sources

    start = time.monotonic()
    bundle = await gather_research("Acme", "SWE", searcher=searcher)
    elapsed = time.monotonic() - start

    assert len(bundle.docs) == num_queries
    # Sequential execution would take num_queries * SLEEP (>= 1.2s here).
    # Concurrent execution should take roughly one SLEEP plus scheduling slack.
    assert elapsed < SLEEP * (num_queries / 2)


# --- video transcript wiring (Fix 4) -----------------------------------------


def _video_searcher(url: str):
    """A searcher that returns one video-source hit at `url` and a plain doc
    for every other source type."""

    def searcher(query, *, source_type, max_results=3, client=None):
        if source_type == "video":
            return [_doc("video", url=url)]
        return [_doc(source_type)]

    return searcher


async def test_gather_research_fetches_transcripts_for_video_results():
    def transcript_fetcher(video_id):
        return SourceDoc(
            "video",
            f"https://www.youtube.com/watch?v={video_id}",
            "Transcript",
            "we talked about scaling",
            "2026-08-04T00:00:00Z",
        )

    bundle = await gather_research(
        "Acme",
        "SWE",
        searcher=_video_searcher("https://www.youtube.com/watch?v=abc123"),
        transcript_fetcher=transcript_fetcher,
    )

    transcripts = [d for d in bundle.docs if d.title == "Transcript"]
    assert len(transcripts) == 1
    assert transcripts[0].content == "we talked about scaling"
    # The transcript supersedes the shallow video search hit at the same URL
    # rather than both surviving as duplicates.
    same_url = [d for d in bundle.docs if d.url == "https://www.youtube.com/watch?v=abc123"]
    assert len(same_url) == 1


async def test_gather_research_skips_video_urls_with_no_extractable_id():
    calls = []

    def transcript_fetcher(video_id):
        calls.append(video_id)
        return None

    bundle = await gather_research(
        "Acme",
        "SWE",
        searcher=_video_searcher("https://example.com/blog/some-talk"),
        transcript_fetcher=transcript_fetcher,
    )

    assert calls == []  # no YouTube ID to fetch, so the fetcher is never called
    assert bundle.docs  # other sources still populated the bundle


async def test_gather_research_survives_a_failing_transcript_fetch():
    def transcript_fetcher(video_id):
        raise RuntimeError("transcript API down")

    bundle = await gather_research(
        "Acme",
        "SWE",
        searcher=_video_searcher("https://www.youtube.com/watch?v=abc123"),
        transcript_fetcher=transcript_fetcher,
    )

    assert bundle.docs  # other sources still made it through
    assert not any(d.title == "Transcript" for d in bundle.docs)

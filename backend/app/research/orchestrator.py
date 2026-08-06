"""Phase 1a: fan out across every public source in parallel.

Each (query, source_type) pair becomes one concurrent task. Failures are absorbed
per-task so one blocked source never takes down the run. Results are deduplicated
by URL, since several queries commonly surface the same page.

The "video" query is a plain web search, so its hits are pages *about* a talk,
not the talk's words. Whichever of those hits are actual YouTube videos get
their transcripts fetched too, concurrently with the rest of the fan-out --
still fail-soft, still never abort the phase.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from app.models import ResearchBundle, SourceDoc
from app.research.sources import (
    extract_youtube_video_id,
    fetch_video_transcript,
    scrape_page,
    search_web,
)

logger = logging.getLogger(__name__)


def build_queries(company: str, role: str) -> list[tuple[str, str]]:
    """One query per source type. Phrased the way a person would search, because
    Tavily ranks natural queries better than keyword soup."""
    return [
        (f"{company} official website product", "website"),
        (f"{company} engineering blog", "blog"),
        (f"{company} company LinkedIn employees leadership", "linkedin"),
        (f"{company} Glassdoor interview questions experience", "glassdoor"),
        (f"{company} news funding announcement 2026", "news"),
        (f"{company} Crunchbase funding founders", "crunchbase"),
        (f"{company} founder interview talk podcast youtube", "video"),
        (f"{company} {role} job description requirements", "job_posting"),
    ]


async def gather_research(
    company: str,
    role: str,
    *,
    job_url: str | None = None,
    searcher: Callable | None = None,
    scraper: Callable | None = None,
    transcript_fetcher: Callable | None = None,
) -> ResearchBundle:
    searcher = searcher or search_web
    scraper = scraper or scrape_page
    transcript_fetcher = transcript_fetcher or fetch_video_transcript

    async def run_search(query: str, source_type: str) -> list[SourceDoc]:
        try:
            return await asyncio.to_thread(searcher, query, source_type=source_type)
        except Exception:
            # one dead source must not abort the phase
            logger.warning(
                "search failed for query=%r source_type=%s", query, source_type, exc_info=True
            )
            return []

    async def run_scrape(url: str) -> list[SourceDoc]:
        try:
            doc = await asyncio.to_thread(scraper, url, source_type="job_posting")
        except Exception:
            logger.warning("scrape failed for url=%s", url, exc_info=True)
            return []
        return [doc] if doc else []

    async def run_transcript(video_id: str) -> list[SourceDoc]:
        try:
            doc = await asyncio.to_thread(transcript_fetcher, video_id)
        except Exception:
            # fetch_video_transcript is itself fail-soft, but this belt-and-
            # suspenders guard matches run_search/run_scrape above and covers
            # a broken injected fetcher in tests.
            logger.warning("transcript fetch failed for video_id=%s", video_id, exc_info=True)
            return []
        return [doc] if doc else []

    queries = build_queries(company, role)
    tasks = [run_search(q, t) for q, t in queries]
    if job_url:
        tasks.append(run_scrape(job_url))

    results = await asyncio.gather(*tasks)

    # The "video" search only ever returns a page *about* a talk -- never its
    # words. Pull a YouTube video ID out of each such result and fetch its
    # transcript in the same fan-out, concurrently with everything else,
    # rather than leaving "the talk was found but never transcribed" as a gap
    # for the one-round reflection pass to notice after the fact.
    video_index = next((i for i, (_q, t) in enumerate(queries) if t == "video"), None)
    video_ids: list[str] = []
    if video_index is not None:
        for doc in results[video_index]:
            video_id = extract_youtube_video_id(doc.url)
            if video_id:
                video_ids.append(video_id)
    transcript_results = await asyncio.gather(*(run_transcript(v) for v in video_ids))

    seen: set[str] = set()
    docs: list[SourceDoc] = []
    # Transcripts are checked first so that when a transcript and its source
    # video-search hit share the same URL, the richer transcript content wins
    # the dedup instead of the shallow search snippet that found it.
    for group in [*transcript_results, *results]:
        for doc in group:
            if doc.url in seen:
                continue
            seen.add(doc.url)
            docs.append(doc)

    return ResearchBundle(company=company, role=role, docs=docs)

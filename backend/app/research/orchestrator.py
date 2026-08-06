"""Phase 1a: fan out across every public source in parallel.

Each (query, source_type) pair becomes one concurrent task. Failures are absorbed
per-task so one blocked source never takes down the run. Results are deduplicated
by URL, since several queries commonly surface the same page.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from app.models import ResearchBundle, SourceDoc
from app.research.sources import scrape_page, search_web


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
) -> ResearchBundle:
    searcher = searcher or search_web
    scraper = scraper or scrape_page

    async def run_search(query: str, source_type: str) -> list[SourceDoc]:
        try:
            return await asyncio.to_thread(searcher, query, source_type=source_type)
        except Exception:
            return []  # one dead source must not abort the phase

    async def run_scrape(url: str) -> list[SourceDoc]:
        try:
            doc = await asyncio.to_thread(scraper, url, source_type="job_posting")
        except Exception:
            return []
        return [doc] if doc else []

    tasks = [run_search(q, t) for q, t in build_queries(company, role)]
    if job_url:
        tasks.append(run_scrape(job_url))

    results = await asyncio.gather(*tasks)

    seen: set[str] = set()
    docs: list[SourceDoc] = []
    for group in results:
        for doc in group:
            if doc.url in seen:
                continue
            seen.add(doc.url)
            docs.append(doc)

    return ResearchBundle(company=company, role=role, docs=docs)

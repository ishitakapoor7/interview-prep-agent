"""Phase 1b: the reflection half of the plan-reflect-act loop.

Phase 1a gathers whatever the standard query set surfaces. This step asks the
model to audit that evidence — what is missing, what contradicts, what is only
shallowly covered — and to propose targeted follow-up searches. Those searches
run, and their results are merged back in.

Exactly one round. The bound is structural (no loop), not a max-iteration guard,
so the cost of a run is predictable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from app.llm import LlmClient
from app.models import ResearchBundle, SourceDoc
from app.research.sources import search_web

logger = logging.getLogger(__name__)

GAP_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific facts that are missing, contradictory, or too shallow.",
        },
        "followup_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Web search queries that would resolve those gaps. Max 4.",
        },
    },
    "required": ["gaps", "followup_queries"],
}

_SYSTEM = (
    "You audit research evidence for an interview-prep briefing. Identify only "
    "concrete, checkable gaps: a missing founding year, two sources disagreeing on "
    "a funding amount, a founder talk whose transcript came back empty or disabled. "
    "Do not invent gaps for facts that are already well covered. Propose at most 4 "
    "follow-up queries."
)


def identify_gaps(bundle: ResearchBundle, llm: LlmClient) -> tuple[list[str], list[str]]:
    user = (
        f"Company: {bundle.company}\nRole: {bundle.role}\n\n"
        f"Evidence gathered so far:\n\n{bundle.all_text()}"
    )
    result = llm.complete_json(system=_SYSTEM, user=user, schema=GAP_SCHEMA)
    gaps = [str(g) for g in result.get("gaps") or []]
    queries = [str(q) for q in result.get("followup_queries") or []][:4]
    return gaps, queries


async def reflect_and_fill(
    bundle: ResearchBundle,
    llm: LlmClient,
    *,
    searcher: Callable | None = None,
) -> ResearchBundle:
    searcher = searcher or search_web
    # identify_gaps makes a synchronous Anthropic HTTP call. Off-load it like
    # every other blocking call in the pipeline (see orchestrator.py) so it
    # doesn't freeze the event loop for the duration of the call.
    gaps, queries = await asyncio.to_thread(identify_gaps, bundle, llm)
    bundle.gaps = gaps
    bundle.reflection_queries = queries
    if not queries:
        return bundle

    async def run(query: str) -> list[SourceDoc]:
        try:
            return await asyncio.to_thread(searcher, query, source_type="news")
        except Exception:
            logger.warning("reflection searcher failed for query=%r", query, exc_info=True)
            return []

    results = await asyncio.gather(*(run(q) for q in queries))
    seen = {d.url for d in bundle.docs}
    for group in results:
        for doc in group:
            if doc.url in seen:
                continue
            seen.add(doc.url)
            bundle.docs.append(doc)
    return bundle

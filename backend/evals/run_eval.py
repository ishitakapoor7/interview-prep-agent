"""Offline eval harness.

Replays Phases 1a+1b+2 for every annotated company (research, reflection,
lesson generation), extracts the scored fields from the delivered lesson
plan, compares them against ground truth, attributes each failure to a
pipeline step, and writes a markdown report. Bundles, plans, and predictions
are saved per company under `evals/runs/` so a failure can be inspected by
hand after the fact.

Usage:
    python -m evals.run_eval                    # every row in ground_truth.json
    python -m evals.run_eval --limit 3          # smoke test
    python -m evals.run_eval --tier early       # one tier
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict
from typing import Awaitable, Callable

from app.lesson.generate import generate_lesson_plan
from app.llm import LlmClient
from app.models import CompanyFacts, ResearchBundle
from app.research.orchestrator import gather_research
from app.research.reflect import reflect_and_fill
from evals.attribute import attribute_failure, summarize
from evals.extract import extract_facts, load_ground_truth
from evals.score import score_company

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

# The eval grades factual accuracy, not personalization, so every company gets
# the same generic resume rather than a per-company fixture that would give
# generate_lesson_plan something real to tailor Module 4 against.
_GENERIC_RESUME = (
    "Software engineer with experience across backend systems, distributed "
    "infrastructure, and API design."
)


async def run_company(
    row: CompanyFacts,
    llm: LlmClient,
    *,
    researcher: Callable[[str], Awaitable[ResearchBundle]] | None = None,
) -> dict:
    """Run one company through research (or the injected `researcher` seam),
    lesson generation, extraction, scoring, and attribution. Writes the full
    trace to `evals/runs/<company>.json` and returns the summary dict that
    `summarize()` consumes.
    """
    if researcher is not None:
        bundle = await researcher(row.company)
    else:
        bundle = await gather_research(row.company, "Software Engineer")
        bundle = await reflect_and_fill(bundle, llm)

    plan = generate_lesson_plan(bundle, _GENERIC_RESUME, llm)
    predicted = extract_facts(bundle, plan, row.tier, llm)
    scores = score_company(predicted, row)
    attributions = [attribute_failure(s, bundle, plan, row) for s in scores]

    run_path = os.path.join(RUNS_DIR, f"{row.company.replace('/', '_')}.json")
    try:
        # Directory creation shares the try/except with the write itself: a
        # permissions failure creating RUNS_DIR is exactly as tolerable as one
        # failing the write one line later, and must be handled the same way.
        os.makedirs(RUNS_DIR, exist_ok=True)
        with open(run_path, "w") as f:
            json.dump(
                {
                    "bundle": asdict(bundle),
                    "plan": asdict(plan),
                    "predicted": asdict(predicted),
                    "truth": asdict(row),
                    "scores": [asdict(s) for s in scores],
                    "attributions": attributions,
                },
                f,
                indent=2,
            )
    except OSError:
        # A failure to log the trace must not fail the eval run itself, but it
        # must not vanish silently either -- an un-inspectable failure would
        # be worse than a slow one.
        logger.exception("failed to write run log for %s to %s", row.company, run_path)

    return {
        "company": row.company,
        "tier": row.tier,
        "scores": scores,
        "attributions": attributions,
    }


def select_rows(
    rows: list[CompanyFacts], *, tier: str | None, limit: int | None
) -> list[CompanyFacts]:
    """Apply --tier then --limit, in that order, so --limit always caps the
    already-filtered set rather than sampling from the full set and possibly
    returning zero rows for the requested tier.

    `limit` is checked against `None`, not truthiness: `--limit 0` must mean
    "run zero companies," not "no limit was given" (`0` and `None` are both
    falsy, so a bare `if limit:` silently treats them the same and runs the
    entire set). The CLI additionally rejects non-positive `--limit` values
    outright (see `_positive_int`), so `0` only reaches here via a direct
    call to `select_rows`, where it correctly produces an empty list.
    """
    if tier:
        rows = [r for r in rows if r.tier == tier]
    if limit is not None:
        rows = rows[:limit]
    return rows


async def run_all(
    rows: list[CompanyFacts],
    llm: LlmClient,
    *,
    researcher: Callable[[str], Awaitable[ResearchBundle]] | None = None,
) -> list[dict]:
    """Run every row, one company at a time. A single company's failure is
    logged and recorded as a scoreless, `"error"`-tagged result rather than
    aborting the whole run -- a bad network day for one company shouldn't
    cost the rest of the set. `evals.attribute.summarize` reads the `"error"`
    key (and, defensively, empty `scores`) to keep failed companies visible
    in the report instead of silently dropping out of the percentages."""
    results = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row.company} ({row.tier})...")
        try:
            results.append(await run_company(row, llm, researcher=researcher))
        except Exception as exc:
            logger.exception(
                "eval run failed for %s; recording as a failure and continuing",
                row.company,
            )
            results.append(
                {
                    "company": row.company,
                    "tier": row.tier,
                    "scores": [],
                    "attributions": [],
                    "error": str(exc),
                }
            )
    return results


def _positive_int(value: str) -> int:
    """argparse `type=` for --limit: `--limit 0` must mean "run nothing," not
    "no limit was given" (see select_rows), and a negative limit would slice
    from the end of the list, which is never what's meant here. Both are
    rejected at the CLI boundary rather than silently reinterpreted."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"--limit must be a positive integer, got {value!r}")
    return n


async def main() -> None:
    # Configured here, not at import time, so importing this module (e.g. from
    # tests) never mutates global logging state as a side effect.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--tier", choices=["large", "mid", "early"], default=None)
    args = parser.parse_args()

    rows = select_rows(load_ground_truth(GROUND_TRUTH_PATH), tier=args.tier, limit=args.limit)

    llm = LlmClient()
    results = await run_all(rows, llm)

    report = summarize(results)
    print("\n" + report)
    with open(os.path.join(os.path.dirname(__file__), "RESULTS.md"), "w") as f:
        f.write("# Eval Results\n\n" + report + "\n")


if __name__ == "__main__":
    asyncio.run(main())

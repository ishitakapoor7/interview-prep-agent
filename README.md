# Interview Prep Agent

Given a company, a role, and a candidate's resume, this tool researches the
company from public sources, figures out where the resume is thin against
the role, and builds a seven-module interview prep plan with a quiz at the
end of each module. Once the plan exists, the candidate can ask free-form
questions about the company and get answers grounded in the sources that
were actually collected — no re-searching the web on every question.

It is a backend (FastAPI + SQLite, calling the Anthropic API and the Tavily
search API) and a small React frontend that consumes it.

## Running it

Requirements: Python 3.11+, Node 18+, an `ANTHROPIC_API_KEY`, and a
`TAVILY_API_KEY` (used for the web research phase). See `.env.example` for
the two variables the backend expects.

Backend:

```bash
cd backend
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export TAVILY_API_KEY=tvly-...
.venv/bin/uvicorn app.main:app --reload
```

Frontend (separate terminal):

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the backend at `http://localhost:8000` and the backend
allows CORS from `http://localhost:5173` (Vite's default dev port) — both
are hardcoded for local development, not configurable via env var yet.

Open the Vite dev URL, enter a company, a role, and paste a resume, and wait.
Building a plan runs the full research and generation pipeline synchronously
and can take a minute or two; the UI's loading state says so rather than
looking stuck.

Backend tests: `cd backend && .venv/bin/python -m pytest -v`.

## Architecture

The pipeline is four phases, each with a narrow job:

**1. Parallel research fan-out.** For a given company and role, the backend
issues a fixed set of searches — official site, engineering blog, LinkedIn,
Glassdoor, news, Crunchbase, founder interviews/talks, and the job posting
itself (or a scraped URL if one was supplied) — all concurrently. Each
search is independently fault-tolerant: one dead source returns an empty
result rather than aborting the run, and results are deduplicated by URL.

**2. One bounded reflection round.** The model reads everything phase 1
gathered and is asked to name concrete, checkable gaps — a missing founding
year, two sources disagreeing on a funding figure, a talk that was found but
never actually scraped — and to propose up to four follow-up searches. Those
searches run once and get merged back into the evidence bundle. The bound is
structural (there is no loop, no retry-until-satisfied), so a run's cost and
latency are predictable regardless of how much the model would like to keep
digging.

**3. Schema-enforced lesson generation.** The finalized research bundle plus
the candidate's resume go into a single model call that must return exactly
seven modules, in a fixed order (Company Overview, Product & Technology,
Team & Culture, The Role, Competitive Landscape, Recent Developments,
Interview Prep), each with prose content and a short quiz. Structured output
is enforced by defining a tool and forcing a tool call, rather than asking
for JSON in the prompt and hoping it parses. Content is matched to its
canonical module slot by title, not array position, because the model
occasionally returns the right seven titles out of order.

**4. RAG-grounded interactive delivery that never re-searches.** Once a plan
exists, free-form questions and quiz grading are both answered strictly from
the research bundle already on disk. Retrieval is lexical (token-overlap
ranking over the collected documents), not embeddings — a bundle holds on
the order of tens of documents, so exact keyword overlap is accurate enough
at that scale and avoids shipping a model, a vector index, and a cold start
to solve a problem this size doesn't have. No phase after generation ever
calls the search API again; a session's cost is fixed at creation time.

## The eval

The eval answers one question: when the pipeline claims a fact about a
company, is it right, and if not, which phase broke?

### Ground truth

`backend/evals/ground_truth.json` holds hand-annotated facts per company —
funding, founding year, founders, required skills, recent events, and a
free-text product description — across three tiers meant to stress different
parts of the research phase: **large** (well-covered, established
companies), **mid** (funded but with thinner public coverage), and
**early-stage** (minimal public footprint, the case most likely to expose
research gaps rather than generation gaps).

**The dataset currently ships 3 seed rows** — Stripe (large), Modal (mid),
and Mechanize (early) — one per tier, enough to exercise the harness
end-to-end. The target is 30 companies, 10 per tier; annotating the
remaining 27 is still outstanding work, not something this repo claims is
done.

### Scoring is deterministic, not model-graded

Every scored field is compared against ground truth with plain code — exact
match, set precision/recall, or numeric tolerance — never with a model
judging similarity:

- `founded_year`: exact match.
- `funding_usd`: correct within 5% relative tolerance, since press rounds
  figures ("$9.1M" vs "$9,100,000" vs "~$9M") and an exact-integer
  requirement would score correct answers as wrong. `None` (never claimed)
  and `0` (claimed as zero) are treated as distinct states, not coerced
  together.
- `founders`, `required_skills`, `recent_events`: set precision/recall after
  normalizing case and punctuation.

No LLM-as-judge appears anywhere in the scoring path. An eval graded by a
model can't be trusted to measure a model — the same failure mode (a
plausible-sounding but wrong claim) that the eval exists to catch is exactly
the kind of thing a model grader can also be fooled by, and a judge call
would make every score only as trustworthy as that one extra model call. The
one place a model *is* involved — pulling the scored fields back out of the
generated lesson plan's prose (`backend/evals/extract.py`) — is a reader,
not a grader: it converts prose into structured values, and the actual
comparison against `ground_truth.json` afterward is plain string/number/set
code with no model in the loop.

`product_line` is deliberately collected in ground truth but never scored.
It's a free-text description of what the company sells, and there's no
non-judgmental way to compare two descriptions of "what a company does" —
any reasonable comparison is really asking "are these similar enough," which
is a judgment call, and the only tool that reliably makes that call is
another model. Scoring it would mean quietly reintroducing an LLM judge
through the back door, which breaks the guarantee the rest of the eval
rests on. It's kept in the dataset because it's useful for a human skimming
a company's row, just not as a number that feeds the results table.

### Three-way failure attribution

When a field scores wrong, the harness doesn't just report "wrong" — it
diagnoses which pipeline phase lost the fact, by mechanically checking
whether the ground-truth value's surface form appears in the research bundle
text and in the generated lesson plan text:

| In research bundle? | In lesson plan? | Attribution |
|---|---|---|
| no | — | **research** — never retrieved; no later phase could have used it |
| yes | no | **synthesis** — generation had the evidence and dropped it |
| yes | yes | **extraction** — the eval's own reading of the plan misread text that was right there |

This distinction is what makes the eval actionable rather than just a
pass/fail count. Each failure mode has a different fix: a research failure
means better search queries or more sources; a synthesis failure means a
better generation prompt or a stronger model for that step; an extraction
failure means fixing the eval's own extraction prompt, and says nothing
about the pipeline being evaluated. Lumping all three into one "wrong"
bucket would point every fix at the wrong place at least some of the time.

Attribution is itself mechanical — a normalized, word-boundary-anchored
substring search, including funding-specific surface-form handling (`$9.1M`,
`9.1 million`, `~$9M`, etc.) — not a model call.

### Running it

```bash
cd backend
python -m evals.run_eval                # every row in ground_truth.json
python -m evals.run_eval --limit 3       # smoke test
python -m evals.run_eval --tier early    # one tier
```

Output is written to `backend/evals/RESULTS.md` (a per-tier, per-field
accuracy table plus the failure-attribution breakdown) and a full per-company
trace — the bundle, the generated plan, the extracted prediction, the
ground-truth row, and every field score and attribution — is saved to
`backend/evals/runs/<company>.json` so any wrong score can be inspected by
hand.

### Results

> **Not yet populated.** The eval harness, scoring, and attribution logic
> are implemented and tested, but a full run requires live
> `ANTHROPIC_API_KEY` / `TAVILY_API_KEY` access and the completed 30-company
> ground truth set — neither is available in this environment. Running
> `python -m evals.run_eval` against the current 3-row seed set and pasting
> the resulting `backend/evals/RESULTS.md` here (plus a short note on the
> dominant failure mode per tier) is the next step, not something reflected
> below.

## Design decisions

**Lexical retrieval over embeddings.** A research bundle holds on the order
of tens of documents per session. At that scale, token-overlap ranking finds
the relevant documents just as well as an embedding index would, without the
cost of shipping a model, building an index, or eating a cold start on every
new session — solving a scale problem this project doesn't have.

**No LLM-as-judge anywhere in scoring.** Covered above in detail; the short
version is that an eval whose grader is itself a model can't credibly
measure whether a model is right, so every scored comparison in this project
is plain deterministic code.

**`product_line` unscored.** Also covered above: free-text similarity has no
non-judgmental comparison, and adding one would undermine the "no
LLM-as-judge" guarantee for every other field too.

**One reflection round, not a loop.** The gap-finding step could in
principle keep asking for more searches until it's satisfied. It's capped at
exactly one round instead, so the cost and latency of building a plan are
predictable rather than open-ended.

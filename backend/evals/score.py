"""Deterministic scoring. No model is consulted anywhere in this file.

Every score is a comparison against a hand-annotated value, which is what makes
the resulting numbers trustworthy: rerunning the suite on the same bundle always
produces the same score.

`product_line` is intentionally not scored — free-text similarity has no
non-judgmental comparison, and adding an LLM judge to score it would undermine
the guarantee above. It is collected for reading, not grading. `company` and
`tier` are not scored either: they identify which row is being compared, not a
fact the pipeline is being graded on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import CompanyFacts

# Funding figures are reported rounded ("$9.1M" vs "$9,100,000" vs "~$9M"), so an
# exact-integer requirement would score correct answers as wrong.
FUNDING_TOLERANCE = 0.05

_PUNCT = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class FieldScore:
    field: str
    correct: bool
    precision: float | None
    recall: float | None
    detail: str


def normalize(s: str) -> str:
    """Lowercase, collapse punctuation to spaces, and collapse whitespace.

    Applied identically to both sides of every set/string comparison in this
    module — a normalizer applied asymmetrically produces false failures.
    """
    return " ".join(_PUNCT.sub(" ", s.lower()).split())


def score_funding(pred: int | None, truth: int | None) -> FieldScore:
    """Correct if within FUNDING_TOLERANCE (relative) of the annotated amount.

    Absence is not the same as zero: `None` means "never claimed", `0` means
    "claimed as zero". Those are compared as distinct states below, not
    coerced into each other.
    """
    if truth is None:
        # Nothing was annotated as claimed; correct only if the prediction
        # also claims nothing.
        correct = pred is None
        return FieldScore(
            "funding_usd", correct, None, None, f"truth is null, predicted {pred}"
        )
    if pred is None:
        return FieldScore("funding_usd", False, None, None, "no prediction")
    if truth == 0:
        # A relative (percentage) tolerance is undefined at zero — dividing by
        # truth here would raise ZeroDivisionError. Zero truth requires an
        # exact match instead of a tolerance band.
        correct = pred == 0
        return FieldScore(
            "funding_usd",
            correct,
            None,
            None,
            f"predicted {pred} vs truth 0 (zero truth requires exact match)",
        )
    # abs(truth) (not truth) as the denominator: a negative truth would
    # otherwise flip the sign of delta and make an out-of-tolerance
    # prediction compare as "correct" (e.g. truth=-1_000_000, pred=-5_000_000
    # gives delta=-4.0 under signed division, which wrongly passes <= 0.05;
    # abs(truth) gives 4.0 and correctly fails). Do not "simplify" this back
    # to `/ truth` — see
    # test_score_funding_negative_truth_is_not_flipped_to_correct_by_sign.
    delta = abs(pred - truth) / abs(truth)
    return FieldScore(
        "funding_usd",
        delta <= FUNDING_TOLERANCE,
        None,
        None,
        f"predicted {pred} vs truth {truth} ({delta:.1%} off)",
    )


def score_year(pred: int | None, truth: int | None) -> FieldScore:
    """Exact match only — a founding year has no meaningful tolerance band.
    `None == None` compares equal, so "neither side claimed a year" scores
    correct rather than being penalized as a miss.
    """
    correct = pred == truth
    return FieldScore(
        "founded_year", correct, None, None, f"predicted {pred} vs truth {truth}"
    )


def score_set(field: str, pred: list[str], truth: list[str]) -> FieldScore:
    """Set-based precision/recall over normalized strings. Both `pred` and
    `truth` are normalized the same way, so casing/punctuation differences
    never register as mismatches. Building a `set` also means duplicate
    entries in either list (e.g. the same founder listed twice) do not
    inflate the count of hits.

    Precision when the prediction is empty is defined as 0.0, not 1.0 or
    undefined — "predicted nothing" must not score as if the agent got
    everything right by omission.

    When the truth set itself is empty there is nothing to compute precision
    or recall against (0/0), so both are reported as `None` rather than a
    fabricated number; correctness falls back to "did the prediction also
    claim nothing".
    """
    p = {normalize(x) for x in pred if x.strip()}
    t = {normalize(x) for x in truth if x.strip()}
    if not t:
        return FieldScore(field, not p, None, None, "truth set is empty")
    hits = p & t
    precision = len(hits) / len(p) if p else 0.0
    recall = len(hits) / len(t)
    missed = sorted(t - p)
    spurious = sorted(p - t)
    return FieldScore(
        field,
        precision == 1.0 and recall == 1.0,
        precision,
        recall,
        f"missed={missed} hallucinated={spurious}",
    )


def score_company(pred: CompanyFacts, truth: CompanyFacts) -> list[FieldScore]:
    """One FieldScore per scored field. `company`, `tier`, and `product_line`
    are deliberately excluded — see module docstring."""
    return [
        score_funding(pred.funding_usd, truth.funding_usd),
        score_year(pred.founded_year, truth.founded_year),
        score_set("founders", pred.founders, truth.founders),
        score_set("required_skills", pred.required_skills, truth.required_skills),
        score_set("recent_events", pred.recent_events, truth.recent_events),
    ]

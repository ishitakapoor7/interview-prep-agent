"""Every dataclass that crosses a module boundary lives here.

Keeping them in one file means the research, lesson, qa, and eval packages all
agree on the same shapes without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# The lesson plan always has these seven modules, in this order. Generation fills
# in the content; the structure is fixed so the frontend and the eval harness can
# both rely on it.
MODULE_TITLES: list[str] = [
    "Company Overview",
    "Product & Technology",
    "Team & Culture",
    "The Role",
    "Competitive Landscape",
    "Recent Developments",
    "Interview Prep",
]

SourceType = Literal[
    "website",
    "blog",
    "job_posting",
    "linkedin",
    "glassdoor",
    "news",
    "video",
    "crunchbase",
]

# Which pipeline step is responsible for a wrong answer. "none" means correct.
Attribution = Literal["research", "extraction", "synthesis", "none"]


@dataclass(frozen=True)
class SourceDoc:
    """One retrieved document. Provenance (url + retrieved_at) is mandatory so the
    eval harness can always answer 'was this fact ever in the bundle?'"""

    source_type: SourceType
    url: str
    title: str
    content: str
    retrieved_at: str  # ISO 8601


@dataclass
class ResearchBundle:
    """Everything the agent gathered about one company for one role.

    `gaps` and `reflection_queries` are empty after Phase 1a and populated by the
    Phase 1b reflection pass, so the bundle records not just what was found but
    what the agent noticed was missing.
    """

    company: str
    role: str
    docs: list[SourceDoc] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    reflection_queries: list[str] = field(default_factory=list)

    def by_type(self, source_type: str) -> list[SourceDoc]:
        return [d for d in self.docs if d.source_type == source_type]

    def all_text(self) -> str:
        """Flat text of every doc, labeled with its source. This is what gets fed
        to synthesis and what the eval harness greps for attribution."""
        return "\n\n".join(
            f"[{d.source_type}] {d.title} ({d.url})\n{d.content}" for d in self.docs
        )


@dataclass(frozen=True)
class QuizQuestion:
    """`expected_points` are the key facts a good answer covers. Grading checks
    coverage against these, so grading stays explainable."""

    question: str
    expected_points: list[str]


@dataclass(frozen=True)
class Module:
    number: int
    title: str
    content: str
    quiz: list[QuizQuestion]


@dataclass(frozen=True)
class LessonPlan:
    company: str
    role: str
    gap_analysis: str
    modules: list[Module]


@dataclass(frozen=True)
class CompanyFacts:
    """Ground-truth (or agent-extracted) facts for one company. Used on both sides
    of the eval comparison — annotated by hand for truth, extracted by the agent
    for the prediction."""

    company: str
    tier: Literal["large", "mid", "early"]
    funding_usd: int | None = None
    founded_year: int | None = None
    founders: list[str] = field(default_factory=list)
    product_line: str = ""
    required_skills: list[str] = field(default_factory=list)
    recent_events: list[str] = field(default_factory=list)

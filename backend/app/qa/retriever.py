"""Lexical retrieval over the research bundle.

Deliberately not embeddings: a bundle holds on the order of tens of documents,
so token-overlap ranking is accurate enough and avoids shipping a model, an index,
and a cold start to solve a problem this size does not have.
"""

from __future__ import annotations

import re

from app.config import RETRIEVAL_TOP_K
from app.models import ResearchBundle, SourceDoc

_WORD = re.compile(r"[a-z0-9]+")

# Words too common to carry signal when scoring overlap.
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "on", "at", "by", "with", "what", "how", "does", "do", "did", "their",
    "they", "it", "this", "that", "be", "as", "from", "you", "your",
}


def tokenize(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _score(doc: SourceDoc, query_tokens: set[str]) -> int:
    doc_tokens = tokenize(f"{doc.title} {doc.content}")
    return len(query_tokens & doc_tokens)


def retrieve(
    bundle: ResearchBundle, query: str, top_k: int = RETRIEVAL_TOP_K
) -> list[SourceDoc]:
    if not bundle.docs:
        return []
    query_tokens = tokenize(query) - _STOP
    ranked = sorted(bundle.docs, key=lambda d: _score(d, query_tokens), reverse=True)
    # Even a zero-overlap query returns docs: an ungrounded answer is worse than
    # an answer grounded in loosely-related evidence the model can decline to use.
    return ranked[:top_k]

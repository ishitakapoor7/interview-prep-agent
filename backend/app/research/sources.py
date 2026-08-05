"""Individual source fetchers.

Every function here is fail-soft: a dead URL, a blocked scrape, or a video with
transcripts disabled returns None/[] instead of raising. The research phase runs
many of these in parallel and must survive any single one failing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from bs4 import BeautifulSoup

from app.config import MAX_DOC_CHARS, TAVILY_API_KEY
from app.models import SourceDoc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def truncate(text: str, limit: int) -> str:
    return text[:limit]


def clean_html(html: str) -> str:
    """Visible text only — scripts and styles removed, whitespace collapsed."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _default_tavily() -> Any:  # pragma: no cover - needs a key
    from tavily import TavilyClient

    return TavilyClient(api_key=TAVILY_API_KEY)


def search_web(
    query: str,
    *,
    source_type: str,
    max_results: int = 3,
    client: Any = None,
) -> list[SourceDoc]:
    """Tavily search. `raw_content` is the full page when available; `content` is
    the snippet fallback."""
    client = client or _default_tavily()
    try:
        raw = client.search(query, max_results=max_results, include_raw_content=True)
    except Exception:
        return []
    docs: list[SourceDoc] = []
    for r in raw.get("results", []):
        body = r.get("raw_content") or r.get("content") or ""
        if not body:
            continue
        docs.append(
            SourceDoc(
                source_type=source_type,
                url=r.get("url", ""),
                title=r.get("title", ""),
                content=truncate(body, MAX_DOC_CHARS),
                retrieved_at=_now(),
            )
        )
    return docs


def _default_fetcher(url: str) -> str:  # pragma: no cover - network
    import httpx

    resp = httpx.get(
        url,
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; InterviewPrepAgent/1.0)"},
    )
    resp.raise_for_status()
    return resp.text


def scrape_page(
    url: str, *, source_type: str, fetcher: Callable[[str], str] | None = None
) -> SourceDoc | None:
    fetcher = fetcher or _default_fetcher
    try:
        html = fetcher(url)
    except Exception:
        return None
    text = clean_html(html)
    if not text:
        return None
    return SourceDoc(
        source_type=source_type,
        url=url,
        title=url,
        content=truncate(text, MAX_DOC_CHARS),
        retrieved_at=_now(),
    )


def _default_transcript_api() -> Any:  # pragma: no cover - network
    from youtube_transcript_api import YouTubeTranscriptApi

    return YouTubeTranscriptApi


def fetch_video_transcript(video_id: str, *, api: Any = None) -> SourceDoc | None:
    api = api or _default_transcript_api()
    try:
        segments = api.get_transcript(video_id)
    except Exception:
        return None
    text = " ".join(s["text"] for s in segments)
    if not text.strip():
        return None
    return SourceDoc(
        source_type="video",
        url=f"https://www.youtube.com/watch?v={video_id}",
        title=f"YouTube transcript {video_id}",
        content=truncate(text, MAX_DOC_CHARS),
        retrieved_at=_now(),
    )

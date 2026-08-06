from app.config import MAX_DOC_CHARS
from app.models import SourceDoc
from app.research.sources import (
    clean_html,
    extract_youtube_video_id,
    fetch_video_transcript,
    scrape_page,
    search_web,
    truncate,
)


class _FakeTavily:
    def __init__(self, results):
        self._results = results

    def search(self, query, max_results=3, include_raw_content=True):
        return {"results": self._results}


class _BoomTavily:
    def search(self, *a, **k):
        raise RuntimeError("network down")


class _NonDictTavily:
    """Response shape isn't the expected dict at all — e.g. a transport-level bug
    returning a bare list instead of {"results": [...]}."""

    def search(self, *a, **k):
        return ["not", "a", "dict"]


class _MalformedResultTavily:
    """One result item isn't a dict (a bare string slipped into the list). A
    well-behaved fetcher skips it and keeps the valid ones."""

    def search(self, *a, **k):
        return {
            "results": [
                "not-a-dict",
                {"url": "https://x.com/2", "title": "ok", "raw_content": "good content"},
            ]
        }


class _Snippet:
    """Mirrors the real youtube_transcript_api FetchedTranscriptSnippet: an
    object with a `.text` attribute, not a dict with a "text" key."""

    def __init__(self, text):
        self.text = text


def test_clean_html_strips_tags_scripts_and_styles():
    html = "<html><script>bad()</script><style>x{}</style><p>Hello</p><p>World</p></html>"
    assert clean_html(html) == "Hello World"


def test_truncate_caps_length_and_leaves_short_text_alone():
    assert truncate("abcdef", 3) == "abc"
    assert truncate("ab", 10) == "ab"


def test_search_web_maps_results_to_source_docs():
    fake = _FakeTavily(
        [
            {
                "url": "https://x.com/1",
                "title": "Acme raises",
                "raw_content": "Acme raised $9.1M in April 2026.",
            }
        ]
    )
    docs = search_web("Acme funding", source_type="news", client=fake)
    assert len(docs) == 1
    assert docs[0].source_type == "news"
    assert docs[0].url == "https://x.com/1"
    assert "9.1M" in docs[0].content
    assert docs[0].retrieved_at.endswith("Z")


def test_search_web_falls_back_to_content_when_raw_content_missing():
    fake = _FakeTavily([{"url": "u", "title": "t", "content": "snippet only"}])
    docs = search_web("q", source_type="news", client=fake)
    assert docs[0].content == "snippet only"


def test_search_web_returns_empty_list_on_failure():
    assert search_web("q", source_type="news", client=_BoomTavily()) == []


def test_search_web_returns_empty_list_when_response_is_not_a_dict():
    assert search_web("q", source_type="news", client=_NonDictTavily()) == []


def test_search_web_skips_malformed_result_items_but_keeps_valid_ones():
    docs = search_web("q", source_type="news", client=_MalformedResultTavily())
    assert len(docs) == 1
    assert docs[0].url == "https://x.com/2"


def test_scrape_page_returns_doc_with_clean_text():
    def fetcher(url):
        return "<html><h1>Title</h1><p>We build robots.</p></html>"

    doc = scrape_page("https://acme.com", source_type="website", fetcher=fetcher)
    assert isinstance(doc, SourceDoc)
    assert doc.source_type == "website"
    assert "We build robots." in doc.content


def test_scrape_page_returns_none_on_failure():
    def fetcher(url):
        raise RuntimeError("404")

    assert scrape_page("https://acme.com", source_type="website", fetcher=fetcher) is None


def test_scrape_page_truncates_content_to_max_doc_chars():
    def fetcher(url):
        return "<p>" + ("word " * (MAX_DOC_CHARS // 4)) + "</p>"

    doc = scrape_page("https://acme.com", source_type="website", fetcher=fetcher)
    assert doc is not None
    assert len(doc.content) == MAX_DOC_CHARS


def test_fetch_video_transcript_joins_segments():
    class _Api:
        def fetch(self, video_id):
            return [_Snippet("we started"), _Snippet("in 2021")]

    doc = fetch_video_transcript("abc123", api=_Api())
    assert doc is not None
    assert doc.source_type == "video"
    assert doc.content == "we started in 2021"
    assert "abc123" in doc.url


def test_fetch_video_transcript_returns_none_when_disabled():
    class _Api:
        def fetch(self, video_id):
            raise RuntimeError("transcripts disabled")

    assert fetch_video_transcript("abc123", api=_Api()) is None


def test_fetch_video_transcript_returns_none_for_malformed_segments():
    class _Api:
        def fetch(self, video_id):
            # Plain dicts have no `.text` attribute — this is the exact shape
            # bug that made the fetcher a silent no-op before the `.fetch()`/
            # `.text` fix: parsing must not raise past the fail-soft guard.
            return [{"text": "we started"}]

    assert fetch_video_transcript("abc123", api=_Api()) is None


def test_extract_youtube_video_id_from_watch_url():
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=abc123XYZ") == "abc123XYZ"


def test_extract_youtube_video_id_from_watch_url_with_extra_query_params():
    assert (
        extract_youtube_video_id("https://www.youtube.com/watch?v=abc123&t=42s") == "abc123"
    )


def test_extract_youtube_video_id_from_short_url():
    assert extract_youtube_video_id("https://youtu.be/abc123") == "abc123"


def test_extract_youtube_video_id_from_embed_url():
    assert extract_youtube_video_id("https://www.youtube.com/embed/abc123") == "abc123"


def test_extract_youtube_video_id_returns_none_for_non_video_url():
    assert extract_youtube_video_id("https://example.com/blog/some-talk") is None


def test_extract_youtube_video_id_returns_none_for_youtube_channel_url():
    assert extract_youtube_video_id("https://www.youtube.com/@SomeChannel") is None


def test_installed_youtube_transcript_api_still_exposes_fetch():
    """Regression guard for the library's 1.x rename (get_transcript -> fetch)
    that this fetcher's fail-soft `except` silently swallowed in production.
    Checks the shape only; makes no network call."""
    from youtube_transcript_api import YouTubeTranscriptApi

    assert hasattr(YouTubeTranscriptApi, "fetch")
    assert callable(YouTubeTranscriptApi.fetch)

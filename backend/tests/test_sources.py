from app.models import SourceDoc
from app.research.sources import (
    clean_html,
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


def test_fetch_video_transcript_joins_segments():
    class _Api:
        @staticmethod
        def get_transcript(video_id):
            return [{"text": "we started"}, {"text": "in 2021"}]

    doc = fetch_video_transcript("abc123", api=_Api)
    assert doc is not None
    assert doc.source_type == "video"
    assert doc.content == "we started in 2021"
    assert "abc123" in doc.url


def test_fetch_video_transcript_returns_none_when_disabled():
    class _Api:
        @staticmethod
        def get_transcript(video_id):
            raise RuntimeError("transcripts disabled")

    assert fetch_video_transcript("abc123", api=_Api) is None

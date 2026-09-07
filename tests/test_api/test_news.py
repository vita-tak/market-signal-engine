"""Behaviour of fetching and parsing Alpha Vantage NEWS_SENTIMENT payloads."""

from datetime import UTC, datetime

import pytest

from src.api.errors import MalformedResponseError, RateLimitError
from src.api.models import NewsArticle
from src.api.news import fetch_news
from tests.doubles import FakeSource

SINCE = datetime(2026, 9, 6, 8, 0, tzinfo=UTC)


def _feed_entry(
    *,
    title: str = "CrowdStrike beats Q2 estimates and raises guidance",
    url: str = "https://www.example.com/crwd-q2-beat",
    time_published: str = "20260907T060000",
    summary: str = "CrowdStrike reported Q2 revenue above consensus.",
    source: str = "Example Newswire",
) -> dict[str, object]:
    return {
        "title": title,
        "url": url,
        "time_published": time_published,
        "summary": summary,
        "source": source,
        "authors": ["Jane Reporter"],
        "overall_sentiment_label": "Somewhat-Bullish",
    }


def test_fetch_news_parses_feed_into_articles() -> None:
    source = FakeSource({"items": "1", "feed": [_feed_entry()]})

    articles = fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert articles == [
        NewsArticle(
            title="CrowdStrike beats Q2 estimates and raises guidance",
            url="https://www.example.com/crwd-q2-beat",
            published_at=datetime(2026, 9, 7, 6, 0, tzinfo=UTC),
            summary="CrowdStrike reported Q2 revenue above consensus.",
            source="Example Newswire",
        )
    ]


def test_fetch_news_returns_empty_list_for_an_empty_feed() -> None:
    """A quiet day is not an error."""
    source = FakeSource({"items": "0", "feed": []})

    assert fetch_news("IRDM", source=source, since=SINCE, limit=10) == []


def test_fetch_news_truncates_the_feed_to_the_limit() -> None:
    source = FakeSource(
        {
            "items": "3",
            "feed": [
                _feed_entry(url="https://www.example.com/one"),
                _feed_entry(url="https://www.example.com/two"),
                _feed_entry(url="https://www.example.com/three"),
            ],
        }
    )

    articles = fetch_news("CRWD", source=source, since=SINCE, limit=2)

    assert [article.url for article in articles] == [
        "https://www.example.com/one",
        "https://www.example.com/two",
    ]


def test_fetch_news_requests_the_news_window_from_alpha_vantage() -> None:
    source = FakeSource({"items": "0", "feed": []})

    fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert source.calls == [
        (
            "NEWS_SENTIMENT",
            {
                "tickers": "CRWD",
                "time_from": "20260906T0800",
                "limit": "10",
                "sort": "LATEST",
            },
        )
    ]


def test_fetch_news_raises_when_the_feed_key_is_missing() -> None:
    source = FakeSource({"items": "0"})

    with pytest.raises(MalformedResponseError) as excinfo:
        fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert "feed" in str(excinfo.value)
    assert "CRWD" in str(excinfo.value)


def test_fetch_news_raises_when_an_entry_is_missing_a_required_field() -> None:
    entry = _feed_entry()
    del entry["url"]
    source = FakeSource({"items": "1", "feed": [entry]})

    with pytest.raises(MalformedResponseError) as excinfo:
        fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert "url" in str(excinfo.value)


def test_fetch_news_raises_when_a_timestamp_is_unparseable() -> None:
    source = FakeSource(
        {"items": "1", "feed": [_feed_entry(time_published="not-a-timestamp")]}
    )

    with pytest.raises(MalformedResponseError) as excinfo:
        fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert "not-a-timestamp" in str(excinfo.value)


def test_fetch_news_propagates_a_rate_limit_payload() -> None:
    source = FakeSource({"Information": "Our standard API rate limit is 25 per day."})

    with pytest.raises(RateLimitError):
        fetch_news("CRWD", source=source, since=SINCE, limit=10)

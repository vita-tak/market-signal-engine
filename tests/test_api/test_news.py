"""Behaviour of fetching and parsing Alpha Vantage NEWS_SENTIMENT payloads."""

from datetime import UTC, datetime

import pytest

from src.api.errors import MalformedResponseError, RateLimitError
from src.api.models import NewsArticle
from src.api.news import fetch_news
from tests.doubles import FakeSource

SINCE = datetime(2026, 9, 6, 8, 0, tzinfo=UTC)

# Distinguishes "the key is absent" from "the key holds None", which are
# different payload shapes the parser has to survive.
_ABSENT = object()


def _feed_entry(
    *,
    title: str = "CrowdStrike beats Q2 estimates and raises guidance",
    url: str = "https://www.example.com/crwd-q2-beat",
    time_published: str = "20260907T060000",
    summary: str = "CrowdStrike reported Q2 revenue above consensus.",
    source: str = "Example Newswire",
    ticker_sentiment: object = _ABSENT,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "title": title,
        "url": url,
        "time_published": time_published,
        "summary": summary,
        "source": source,
        "authors": ["Jane Reporter"],
        "overall_sentiment_label": "Somewhat-Bullish",
    }
    if ticker_sentiment is not _ABSENT:
        entry["ticker_sentiment"] = ticker_sentiment
    return entry


def _sentiment(ticker: str, relevance_score: object) -> dict[str, object]:
    """One element of a ticker_sentiment array."""
    return {
        "ticker": ticker,
        "relevance_score": relevance_score,
        "ticker_sentiment_score": "0.221",
        "ticker_sentiment_label": "Somewhat-Bullish",
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


def test_fetch_news_reads_the_relevance_score_for_the_requested_ticker() -> None:
    """One feed entry scores several tickers. Only this ticker's score counts."""
    source = FakeSource(
        {
            "items": "1",
            "feed": [
                _feed_entry(
                    ticker_sentiment=[
                        _sentiment("AAPL", "0.120"),
                        _sentiment("CRWD", "0.951"),
                        _sentiment("MSFT", "0.400"),
                    ]
                )
            ],
        }
    )

    articles = fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert articles[0].relevance_score == 0.951


@pytest.mark.parametrize(
    ("case", "ticker_sentiment"),
    [
        ("key absent", _ABSENT),
        ("empty array", []),
        ("other tickers only", [_sentiment("AAPL", "0.120")]),
    ],
)
def test_fetch_news_scores_an_article_zero_when_the_ticker_is_absent(
    case: str, ticker_sentiment: object
) -> None:
    """No score for this ticker is not an error. It ranks last."""
    source = FakeSource(
        {
            "items": "1",
            "feed": [_feed_entry(ticker_sentiment=ticker_sentiment)],
        }
    )

    articles = fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert articles[0].relevance_score == 0.0, case


@pytest.mark.parametrize(
    ("case", "ticker_sentiment"),
    [
        ("not a number", [_sentiment("CRWD", "n/a")]),
        ("null score", [_sentiment("CRWD", None)]),
        ("score key absent", [{"ticker": "CRWD"}]),
        ("nan", [_sentiment("CRWD", "NaN")]),
        ("infinity", [_sentiment("CRWD", "inf")]),
        ("array is not a list", {"ticker": "CRWD", "relevance_score": "0.9"}),
        ("array holds junk", ["nonsense", 7]),
    ],
)
def test_fetch_news_scores_an_article_zero_when_the_score_is_unreadable(
    case: str, ticker_sentiment: object
) -> None:
    """A ranking hint must never cost a ticker its signal for the day."""
    source = FakeSource(
        {
            "items": "1",
            "feed": [_feed_entry(ticker_sentiment=ticker_sentiment)],
        }
    )

    articles = fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert articles[0].relevance_score == 0.0, case


def test_fetch_news_keeps_looking_past_a_malformed_sentiment_entry() -> None:
    """A junk element must not hide a good score later in the array."""
    source = FakeSource(
        {
            "items": "1",
            "feed": [
                _feed_entry(ticker_sentiment=["nonsense", _sentiment("CRWD", "0.951")])
            ],
        }
    )

    articles = fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert articles[0].relevance_score == 0.951


def test_fetch_news_returns_articles_most_relevant_first() -> None:
    """Alpha Vantage sends them newest first. Relevance decides the order."""
    source = FakeSource(
        {
            "items": "3",
            "feed": [
                _feed_entry(
                    url="https://www.example.com/roundup",
                    time_published="20260907T090000",
                    ticker_sentiment=[_sentiment("CRWD", "0.104")],
                ),
                _feed_entry(
                    url="https://www.example.com/targets",
                    time_published="20260907T080000",
                    ticker_sentiment=[_sentiment("CRWD", "0.844")],
                ),
                _feed_entry(
                    url="https://www.example.com/earnings",
                    time_published="20260907T070000",
                    ticker_sentiment=[_sentiment("CRWD", "0.951")],
                ),
            ],
        }
    )

    articles = fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert [article.url for article in articles] == [
        "https://www.example.com/earnings",
        "https://www.example.com/targets",
        "https://www.example.com/roundup",
    ]


def test_fetch_news_keeps_the_alpha_vantage_order_for_equal_relevance() -> None:
    """Ties fall back to the order the feed arrived in, which is newest first."""
    source = FakeSource(
        {
            "items": "3",
            "feed": [
                _feed_entry(
                    url="https://www.example.com/newest",
                    time_published="20260907T090000",
                    ticker_sentiment=[_sentiment("CRWD", "0.500")],
                ),
                _feed_entry(
                    url="https://www.example.com/middle",
                    time_published="20260907T080000",
                    ticker_sentiment=[_sentiment("CRWD", "0.500")],
                ),
                _feed_entry(
                    url="https://www.example.com/oldest",
                    time_published="20260907T070000",
                    ticker_sentiment=[_sentiment("CRWD", "0.500")],
                ),
            ],
        }
    )

    articles = fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert [article.url for article in articles] == [
        "https://www.example.com/newest",
        "https://www.example.com/middle",
        "https://www.example.com/oldest",
    ]


def test_fetch_news_applies_the_limit_before_sorting_by_relevance() -> None:
    """The window is a recency decision. Relevance only orders what survives."""
    source = FakeSource(
        {
            "items": "3",
            "feed": [
                _feed_entry(
                    url="https://www.example.com/newest",
                    time_published="20260907T090000",
                    ticker_sentiment=[_sentiment("CRWD", "0.104")],
                ),
                _feed_entry(
                    url="https://www.example.com/middle",
                    time_published="20260907T080000",
                    ticker_sentiment=[_sentiment("CRWD", "0.500")],
                ),
                _feed_entry(
                    url="https://www.example.com/cut",
                    time_published="20260907T070000",
                    ticker_sentiment=[_sentiment("CRWD", "0.999")],
                ),
            ],
        }
    )

    articles = fetch_news("CRWD", source=source, since=SINCE, limit=2)

    # The most relevant article was outside the limit, so it is not here at all.
    assert [article.url for article in articles] == [
        "https://www.example.com/middle",
        "https://www.example.com/newest",
    ]

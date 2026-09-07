"""Fetching and parsing Alpha Vantage NEWS_SENTIMENT payloads."""

from datetime import UTC, datetime
from math import isfinite
from typing import Any

import config

from src.api.client import ResponseSource, ensure_ok
from src.api.errors import MalformedResponseError
from src.api.models import NewsArticle

_REQUIRED_FIELDS = ("title", "url", "time_published", "summary", "source")

# Keys of the per ticker relevance block Alpha Vantage attaches to each entry.
_TICKER_SENTIMENT_KEY = "ticker_sentiment"
_TICKER_KEY = "ticker"
_RELEVANCE_KEY = "relevance_score"
# Used when the feed carries no relevance score for the ticker being fetched.
_DEFAULT_RELEVANCE = 0.0


def fetch_news(
    ticker: str,
    *,
    source: ResponseSource,
    since: datetime,
    limit: int,
) -> list[NewsArticle]:
    """Return the articles published for a ticker since a given moment.

    Args:
        ticker: Ticker symbol to fetch news for.
        source: Where to read the Alpha Vantage payload from.
        since: Start of the news window, sent as the time_from parameter.
        limit: Maximum number of articles to return.

    Returns:
        The articles, most relevant first, with ties newest first. Empty when
        the ticker had no news.

    Raises:
        AlphaVantageError: If the payload reports a failure or is malformed.
    """
    payload = source.get(
        config.NEWS_FUNCTION,
        {
            "tickers": ticker,
            "time_from": since.strftime(config.AV_TIME_FROM_FORMAT),
            "limit": str(limit),
            "sort": config.NEWS_SORT,
        },
    )
    ensure_ok(payload, context=ticker)

    feed = payload.get("feed")
    if not isinstance(feed, list):
        raise MalformedResponseError(
            f"Alpha Vantage response for {ticker} has no 'feed' list"
        )

    # Alpha Vantage honours the limit parameter server side, but a fixture
    # may hold more entries than the caller asked for. The cut happens before
    # the sort, so which articles survive stays a recency decision and
    # relevance only orders what was kept.
    articles = [_to_article(entry, ticker=ticker) for entry in feed[:limit]]
    return _by_relevance(articles)


def _by_relevance(articles: list[NewsArticle]) -> list[NewsArticle]:
    """Order articles most relevant first.

    Python's sort is stable, and stays stable under reverse, so articles
    sharing a score keep the order Alpha Vantage sent them in. With sort=LATEST
    that means ties fall back to newest first.
    """
    return sorted(articles, key=lambda article: article.relevance_score, reverse=True)


def _to_article(entry: dict[str, Any], *, ticker: str) -> NewsArticle:
    """Convert one raw feed entry into a NewsArticle.

    Raises:
        MalformedResponseError: If a field is absent or the timestamp is not
            in the documented format.
    """
    missing = [field for field in _REQUIRED_FIELDS if field not in entry]
    if missing:
        raise MalformedResponseError(
            f"Article in the {ticker} feed is missing: {', '.join(missing)}"
        )

    raw_published = entry["time_published"]
    try:
        published = datetime.strptime(raw_published, config.AV_TIME_PUBLISHED_FORMAT)
    except (TypeError, ValueError) as error:
        raise MalformedResponseError(
            f"Article in the {ticker} feed has an unreadable time_published "
            f"'{raw_published}'"
        ) from error
    return NewsArticle(
        title=entry["title"],
        url=entry["url"],
        # Alpha Vantage timestamps carry no timezone marker. They are read as
        # UTC here, and written back as UTC everywhere downstream.
        published_at=published.replace(tzinfo=UTC),
        summary=entry["summary"],
        source=entry["source"],
        relevance_score=_relevance_score(entry, ticker=ticker),
    )


def _relevance_score(entry: dict[str, Any], *, ticker: str) -> float:
    """Return how relevant this article is to the ticker it was fetched for.

    One entry carries a score for every ticker it mentions, so the array is
    searched for the one that was asked about. The match is exact, which is
    also what stops a CRYPTO: or FOREX: entry in the same array from ever
    matching an equity ticker.

    This never raises. A relevance score is a ranking hint rather than
    content, so an absent or unreadable one must not cost a ticker its signal
    for the day. Every such case scores 0.0, which ranks the article last.

    Args:
        entry: One raw feed entry.
        ticker: The ticker the feed was fetched for.

    Returns:
        The score, or 0.0 if the feed carries no readable one for this ticker.
    """
    sentiments = entry.get(_TICKER_SENTIMENT_KEY)
    if not isinstance(sentiments, list):
        return _DEFAULT_RELEVANCE

    for sentiment in sentiments:
        if not isinstance(sentiment, dict):
            continue
        if sentiment.get(_TICKER_KEY) != ticker:
            continue
        try:
            score = float(sentiment[_RELEVANCE_KEY])
        except (KeyError, TypeError, ValueError):
            return _DEFAULT_RELEVANCE
        # A NaN sort key orders arbitrarily rather than loudly, so a
        # non-finite score is rejected here instead of reaching the sort.
        return score if isfinite(score) else _DEFAULT_RELEVANCE
    return _DEFAULT_RELEVANCE

"""Fetching and parsing Alpha Vantage NEWS_SENTIMENT payloads."""

from datetime import UTC, datetime
from typing import Any

import config

from src.api.client import ResponseSource, ensure_ok
from src.api.errors import MalformedResponseError
from src.api.models import NewsArticle

_REQUIRED_FIELDS = ("title", "url", "time_published", "summary", "source")


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
        The articles, newest first. Empty when the ticker had no news.

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
    # may hold more entries than the caller asked for.
    return [_to_article(entry, ticker=ticker) for entry in feed[:limit]]


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
    )

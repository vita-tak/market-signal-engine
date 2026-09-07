"""The saved fixtures must survive the real parsers.

These tests are the safety net for offline runs: if a fixture drifts out of
shape, it fails here rather than halfway through a run.
"""

import json
from datetime import UTC, datetime

import pytest

import config

from src.api.client import FixtureSource, ensure_ok
from src.api.errors import (
    InvalidRequestError,
    MalformedResponseError,
    RateLimitError,
)
from src.api.news import fetch_news
from src.api.prices import fetch_quote

SINCE = datetime(2026, 9, 6, 8, 0, tzinfo=UTC)


@pytest.fixture(name="source")
def fixture_source() -> FixtureSource:
    return FixtureSource(config.FIXTURES_DIR)


def _load(name: str) -> dict[str, object]:
    path = config.FIXTURES_DIR / name
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return payload


@pytest.mark.parametrize("ticker", config.WATCHLIST)
def test_every_watchlist_ticker_parses_end_to_end(
    source: FixtureSource, ticker: str
) -> None:
    articles = fetch_news(
        ticker, source=source, since=SINCE, limit=config.NEWS_LIMIT_PER_DAY
    )
    quote = fetch_quote(ticker, source=source)

    assert quote.symbol == ticker
    assert quote.price > 0
    assert all(article.published_at.tzinfo is UTC for article in articles)
    assert all(article.url.startswith("https://") for article in articles)


def test_market_proxy_quote_parses(source: FixtureSource) -> None:
    quote = fetch_quote(config.MARKET_PROXY, source=source)

    assert quote.symbol == "SPY"
    assert quote.change_percent == 0.8283


def test_crwd_fixture_holds_a_corroborated_event(source: FixtureSource) -> None:
    """Three independent sources, which is HIGH confidence territory."""
    articles = fetch_news("CRWD", source=source, since=SINCE, limit=10)

    assert len(articles) == 3
    assert len({article.source for article in articles}) == 3
    assert articles[0].published_at == datetime(2026, 9, 7, 6, 0, tzinfo=UTC)


def test_irdm_fixture_is_a_quiet_day(source: FixtureSource) -> None:
    assert fetch_news("IRDM", source=source, since=SINCE, limit=10) == []


def test_rate_limit_fixtures_are_recognised() -> None:
    for name in ("error_rate_limit.json", "error_rate_limit_note.json"):
        with pytest.raises(RateLimitError):
            ensure_ok(_load(name), context="CRWD")


def test_invalid_call_fixture_is_recognised() -> None:
    with pytest.raises(InvalidRequestError):
        ensure_ok(_load("error_invalid_call.json"), context="CRWD")


def test_empty_quote_fixture_is_rejected(source: FixtureSource) -> None:
    with pytest.raises(MalformedResponseError):
        fetch_quote("EMPTY", source=source)


def test_malformed_news_fixture_is_rejected(source: FixtureSource) -> None:
    with pytest.raises(MalformedResponseError) as excinfo:
        fetch_news("MALFORMED", source=source, since=SINCE, limit=10)

    assert "url" in str(excinfo.value)
    assert "time_published" in str(excinfo.value)

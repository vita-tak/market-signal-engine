"""A record built from the real fixtures, through the real chain.

Guards the grounding property: a record's sources are the articles that were
actually fetched, never anything the model produced.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import config

from src.analysis.analyzer import analyze
from src.analysis.llm import StubGenerator
from src.api.client import FixtureSource
from src.api.news import fetch_news
from src.api.prices import fetch_quote
from src.output.signal_log import append_record, build_record

RUN_TIMESTAMP = datetime(2026, 9, 7, 8, 0, tzinfo=UTC)
SINCE = datetime(2026, 9, 6, 8, 0, tzinfo=UTC)
SOURCE = FixtureSource(config.FIXTURES_DIR)


def _build(ticker: str) -> dict[str, object]:
    articles = fetch_news(ticker, source=SOURCE, since=SINCE, limit=config.NEWS_LIMIT)
    quote = fetch_quote(ticker, source=SOURCE)
    market = fetch_quote(config.MARKET_PROXY, source=SOURCE)
    analysis = analyze(
        ticker,
        articles=articles,
        quote=quote,
        market=market,
        generator=StubGenerator(),
    )
    return build_record(
        run_timestamp=RUN_TIMESTAMP,
        ticker=ticker,
        analysis=analysis,
        articles=articles,
        quote=quote,
        market=market,
    )


def test_sources_are_exactly_the_fetched_articles() -> None:
    articles = fetch_news("CRWD", source=SOURCE, since=SINCE, limit=config.NEWS_LIMIT)

    record = _build("CRWD")

    sources = record["news_sources"]
    assert isinstance(sources, list)
    assert [source["url"] for source in sources] == [a.url for a in articles]
    assert [source["title"] for source in sources] == [a.title for a in articles]


def test_a_quiet_ticker_records_no_sources() -> None:
    record = _build("IRDM")

    assert record["news_sources"] == []
    assert record["signal"] == "HOLD"


def test_a_full_watchlist_pass_writes_one_valid_line_per_ticker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signals.jsonl"

    for ticker in config.WATCHLIST:
        append_record(_build(ticker), path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert [record["ticker"] for record in records] == config.WATCHLIST
    assert {record["run_timestamp"] for record in records} == {"2026-09-07T08:00:00Z"}
    assert all(record["market_price_at_signal"] == 645.2 for record in records)
    assert all(record["evaluation"]["outcome"] is None for record in records)

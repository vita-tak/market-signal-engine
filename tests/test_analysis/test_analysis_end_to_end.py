"""The analysis module against the real saved fixtures.

Runs the same path an offline run takes, so a mismatch between a fixture and
the prompt builder fails here rather than mid-run.
"""

from datetime import UTC, datetime

import config

from src.analysis.analyzer import analyze, build_market_context
from src.analysis.llm import StubGenerator
from src.analysis.models import Confidence, SignalAnalysis, SignalType
from src.api.client import FixtureSource
from src.api.news import fetch_news
from src.api.prices import fetch_quote
from tests.doubles import NeverCalledGenerator, RecordingGenerator

SINCE = datetime(2026, 9, 6, 8, 0, tzinfo=UTC)
SOURCE = FixtureSource(config.FIXTURES_DIR)
ANALYSIS = SignalAnalysis(
    signal=SignalType.BUY,
    confidence=Confidence.HIGH,
    event="Q2 beat and guidance raise",
    reasoning="Corroborated by three sources.",
    news_summary="Revenue came in above consensus.",
)


def test_a_corroborated_event_reaches_the_model_with_every_source() -> None:
    articles = fetch_news(
        "CRWD", source=SOURCE, since=SINCE, limit=config.NEWS_LIMIT_PER_DAY
    )
    quote = fetch_quote("CRWD", source=SOURCE)
    market = fetch_quote(config.MARKET_PROXY, source=SOURCE)
    generator = RecordingGenerator(ANALYSIS)

    result = analyze(
        "CRWD",
        articles=articles,
        quote=quote,
        market=market,
        generator=generator,
    )

    _, user = generator.prompts[0]
    assert "3 article(s) retrieved" in user
    for article in articles:
        assert article.url in user
    assert build_market_context(market) in user
    assert result.analysis is ANALYSIS


def test_a_quiet_ticker_never_reaches_the_model() -> None:
    articles = fetch_news(
        "IRDM", source=SOURCE, since=SINCE, limit=config.NEWS_LIMIT_PER_DAY
    )
    quote = fetch_quote("IRDM", source=SOURCE)
    market = fetch_quote(config.MARKET_PROXY, source=SOURCE)

    result = analyze(
        "IRDM",
        articles=articles,
        quote=quote,
        market=market,
        generator=NeverCalledGenerator(),
    )

    assert result.analysis.signal is SignalType.HOLD
    assert result.analysis.confidence is Confidence.LOW
    assert result.market_context == "SPY 645.20, +0.83% on 2026-09-05"


def test_the_offline_stub_produces_a_valid_analysis_for_every_ticker() -> None:
    """The default configuration must carry a full run without a model."""
    market = fetch_quote(config.MARKET_PROXY, source=SOURCE)

    for ticker in config.WATCHLIST:
        articles = fetch_news(
            ticker, source=SOURCE, since=SINCE, limit=config.NEWS_LIMIT_PER_DAY
        )
        quote = fetch_quote(ticker, source=SOURCE)

        result = analyze(
            ticker,
            articles=articles,
            quote=quote,
            market=market,
            generator=StubGenerator(),
        )

        assert isinstance(result.analysis, SignalAnalysis)
        assert result.market_context == "SPY 645.20, +0.83% on 2026-09-05"

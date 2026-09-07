"""Behaviour of turning news and prices into a signal."""

from datetime import UTC, datetime

import config

from src.analysis.analyzer import analyze, build_market_context
from src.analysis.models import Confidence, SignalAnalysis, SignalType
from src.analysis.prompts import SYSTEM_PROMPT
from src.api.models import NewsArticle, Quote
from tests.doubles import NeverCalledGenerator, RecordingGenerator

MARKET = Quote(
    symbol="SPY",
    price=645.2,
    change_percent=0.8283,
    latest_trading_day="2026-09-05",
)
CRWD = Quote(
    symbol="CRWD",
    price=342.5,
    change_percent=1.0324,
    latest_trading_day="2026-09-05",
)
ARTICLE = NewsArticle(
    title="CrowdStrike beats Q2 estimates",
    url="https://www.example.com/crwd-q2-beat",
    published_at=datetime(2026, 9, 7, 6, 0, tzinfo=UTC),
    summary="Revenue came in above consensus.",
    source="Example Newswire",
)


def test_build_market_context_formats_a_rising_market() -> None:
    assert build_market_context(MARKET) == "SPY 645.20, +0.83% on 2026-09-05"


def test_build_market_context_formats_a_falling_market() -> None:
    falling = Quote(
        symbol="SPY",
        price=630.05,
        change_percent=-1.2461,
        latest_trading_day="2026-09-05",
    )

    assert build_market_context(falling) == "SPY 630.05, -1.25% on 2026-09-05"


def test_analyze_holds_on_a_quiet_day_without_asking_the_model() -> None:
    """Criterion 4 already fixes the answer, so no request is worth making."""
    result = analyze(
        "IRDM",
        articles=[],
        quote=CRWD,
        market=MARKET,
        generator=NeverCalledGenerator(),
    )

    assert result.analysis == SignalAnalysis(
        signal=SignalType.HOLD,
        confidence=Confidence.LOW,
        event=config.QUIET_DAY_EVENT,
        reasoning=config.QUIET_DAY_REASONING,
        news_summary=config.QUIET_DAY_NEWS_SUMMARY,
    )
    assert result.market_context == "SPY 645.20, +0.83% on 2026-09-05"


def _analysis() -> SignalAnalysis:
    return SignalAnalysis(
        signal=SignalType.BUY,
        confidence=Confidence.HIGH,
        event="Q2 beat and guidance raise",
        reasoning="Three sources report the same beat.",
        news_summary="CrowdStrike reported Q2 revenue above consensus.",
    )


def test_analyze_delegates_to_the_generator_when_there_is_news() -> None:
    expected = _analysis()
    generator = RecordingGenerator(expected)

    result = analyze(
        "CRWD",
        articles=[ARTICLE],
        quote=CRWD,
        market=MARKET,
        generator=generator,
    )

    assert result.analysis is expected
    assert result.market_context == "SPY 645.20, +0.83% on 2026-09-05"
    assert len(generator.prompts) == 1


def test_analyze_sends_the_system_prompt_and_a_grounded_user_prompt() -> None:
    generator = RecordingGenerator(_analysis())

    analyze(
        "CRWD",
        articles=[ARTICLE],
        quote=CRWD,
        market=MARKET,
        generator=generator,
    )

    system, user = generator.prompts[0]
    assert system == SYSTEM_PROMPT
    assert "CRWD" in user
    assert "342.50" in user
    assert "645.20" in user
    assert ARTICLE.url in user

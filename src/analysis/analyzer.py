"""Turning a ticker's news and prices into a signal."""

from collections.abc import Sequence

import config

from src.analysis.llm import SignalGenerator
from src.analysis.models import (
    Confidence,
    SignalAnalysis,
    SignalType,
    TickerAnalysis,
)
from src.analysis.prompts import SYSTEM_PROMPT, build_user_prompt
from src.api.models import NewsArticle, Quote


def build_market_context(market: Quote) -> str:
    """Describe the broader market move in one factual line.

    Built from the quote rather than generated, so the market backdrop on a
    record is always the real number.
    """
    return config.MARKET_CONTEXT_TEMPLATE.format(
        symbol=market.symbol,
        price=market.price,
        change_percent=market.change_percent,
        latest_trading_day=market.latest_trading_day,
    )


def analyze(
    ticker: str,
    *,
    articles: Sequence[NewsArticle],
    quote: Quote,
    market: Quote,
    generator: SignalGenerator,
) -> TickerAnalysis:
    """Produce a signal for one ticker.

    A ticker with no news in the lookback window returns HOLD with LOW
    confidence without reaching the model at all. Signal criterion 4 already
    fixes the answer for a quiet day, so asking would only add cost and the
    chance of an invented event.

    Args:
        ticker: The ticker being analysed.
        articles: News in the lookback window, most relevant first. May be
            empty.
        quote: The ticker's own price snapshot.
        market: The market proxy's price snapshot.
        generator: What produces the analysis when there is news to judge.

    Returns:
        The analysis and the market context it was judged against.

    Raises:
        AnalysisError: If the generator fails or returns an unusable analysis.
    """
    market_context = build_market_context(market)

    if not articles:
        return TickerAnalysis(
            analysis=_quiet_day_analysis(), market_context=market_context
        )

    user_prompt = build_user_prompt(
        ticker=ticker,
        articles=articles,
        quote=quote,
        market_context=market_context,
    )
    analysis = generator.generate(system=SYSTEM_PROMPT, user=user_prompt)
    return TickerAnalysis(analysis=analysis, market_context=market_context)


def _quiet_day_analysis() -> SignalAnalysis:
    """The fixed answer for a ticker with no news in the lookback window."""
    return SignalAnalysis(
        signal=SignalType.HOLD,
        confidence=Confidence.LOW,
        event=config.QUIET_DAY_EVENT,
        reasoning=config.QUIET_DAY_REASONING,
        news_summary=config.QUIET_DAY_NEWS_SUMMARY,
    )

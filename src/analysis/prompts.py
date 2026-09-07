"""The prompts sent to the model.

The system prompt encodes the signal criteria and confidence definitions from
CLAUDE.md. The user prompt carries only facts: prices, the market backdrop,
and the articles themselves. Nothing here asks the model to restate market
data, so a record's market context is always a real number.
"""

from collections.abc import Sequence

import config

from src.api.models import NewsArticle, Quote

SYSTEM_PROMPT = """You are an equity research analyst producing event-driven \
signals for a watchlist. You do not trade and you do not give financial \
advice. You read one day of news for a single ticker and judge whether it \
contains a concrete event worth acting on.

A valid signal must:
1. Be driven by a concrete, identifiable event, not general sentiment.
2. Be actionable in advance. The news must predate the price movement it
   would explain.
3. Account for market context, not just the stock in isolation. A move that
   merely tracks the broader market is not a stock specific event.
4. Return HOLD with LOW confidence on a quiet day. Never invent an event.

Confidence definitions:
HIGH: multiple corroborating sources, clear causal link to price impact.
MEDIUM: one strong source or an indirect causal link.
LOW: weak signal, noisy news, or no clear event identified.

Field rules:
signal: BUY, HOLD or SELL.
confidence: HIGH, MEDIUM or LOW.
event: one short sentence naming the concrete event. If no event qualifies,
say so plainly instead of describing the mood.
reasoning: why this event supports this signal. Cite the numbered articles you
relied on, and say how the market backdrop affected your read.
news_summary: a factual summary of the articles, with no interpretation.

Ground every claim in the articles you are given. Do not draw on outside
knowledge about the company, and do not speculate about news that is not in
the list. Several weak articles do not add up to a strong signal."""

_HEADER_TEMPLATE = """Ticker: {ticker}
Price: {price:.2f} ({change_percent:+.2f}% on {latest_trading_day})
Market: {market_context}

{count} article(s) retrieved for this ticker, newest first:"""

_ARTICLE_TEMPLATE = """
[{index}] {title}
    Source: {source}
    Published: {published_at}
    URL: {url}
    Summary: {summary}"""


def build_user_prompt(
    *,
    ticker: str,
    articles: Sequence[NewsArticle],
    quote: Quote,
    market_context: str,
) -> str:
    """Lay out one ticker's facts for the model.

    Args:
        ticker: The ticker being analysed.
        articles: News in the lookback window, newest first.
        quote: The ticker's own price snapshot.
        market_context: The market backdrop line, already formatted.

    Returns:
        The user prompt.
    """
    header = _HEADER_TEMPLATE.format(
        ticker=ticker,
        price=quote.price,
        change_percent=quote.change_percent,
        latest_trading_day=quote.latest_trading_day,
        market_context=market_context,
        count=len(articles),
    )
    body = "".join(
        _ARTICLE_TEMPLATE.format(
            index=index,
            title=article.title,
            source=article.source,
            published_at=article.published_at.strftime(config.TIMESTAMP_FORMAT),
            url=article.url,
            summary=article.summary,
        )
        for index, article in enumerate(articles, start=1)
    )
    return f"{header}\n{body}"

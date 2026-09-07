"""The prompts carry the rules and the facts, and nothing else."""

from datetime import UTC, datetime

import pytest

from src.analysis.prompts import SYSTEM_PROMPT, build_user_prompt
from src.api.models import NewsArticle, Quote

QUOTE = Quote(
    symbol="CRWD",
    price=342.5,
    change_percent=1.0324,
    latest_trading_day="2026-09-05",
)
MARKET_CONTEXT = "SPY 645.20, +0.83% on 2026-09-05"


def _article(index: int) -> NewsArticle:
    return NewsArticle(
        title=f"Headline {index}",
        url=f"https://www.example.com/article-{index}",
        published_at=datetime(2026, 9, 7, 6, index, tzinfo=UTC),
        summary=f"Summary {index}.",
        source=f"Source {index}",
    )


@pytest.mark.parametrize(
    "requirement",
    [
        "concrete, identifiable event",
        "must predate the price movement",
        "not just the stock in isolation",
        "Never invent an event",
    ],
)
def test_system_prompt_states_every_signal_criterion(requirement: str) -> None:
    assert requirement in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "definition",
    [
        "HIGH: multiple corroborating sources",
        "MEDIUM: one strong source",
        "LOW: weak signal, noisy news",
    ],
)
def test_system_prompt_states_every_confidence_definition(definition: str) -> None:
    assert definition in SYSTEM_PROMPT


def test_system_prompt_forbids_outside_knowledge() -> None:
    """The record has to be judgeable from the sources it lists."""
    assert "Do not draw on outside" in SYSTEM_PROMPT
    assert "Ground every claim in the articles" in SYSTEM_PROMPT


def test_user_prompt_leads_with_the_ticker_and_both_price_lines() -> None:
    prompt = build_user_prompt(
        ticker="CRWD",
        articles=[_article(1)],
        quote=QUOTE,
        market_context=MARKET_CONTEXT,
    )

    assert prompt.startswith("Ticker: CRWD\n")
    assert "Price: 342.50 (+1.03% on 2026-09-05)" in prompt
    assert f"Market: {MARKET_CONTEXT}" in prompt


def test_user_prompt_numbers_every_article_with_all_its_fields() -> None:
    articles = [_article(1), _article(2), _article(3)]

    prompt = build_user_prompt(
        ticker="CRWD",
        articles=articles,
        quote=QUOTE,
        market_context=MARKET_CONTEXT,
    )

    assert "3 article(s) retrieved" in prompt
    for index, article in enumerate(articles, start=1):
        assert f"[{index}] {article.title}" in prompt
        assert f"Source: {article.source}" in prompt
        assert f"URL: {article.url}" in prompt
        assert f"Summary: {article.summary}" in prompt


def test_user_prompt_renders_timestamps_as_utc() -> None:
    prompt = build_user_prompt(
        ticker="CRWD",
        articles=[_article(1)],
        quote=QUOTE,
        market_context=MARKET_CONTEXT,
    )

    assert "Published: 2026-09-07T06:01:00Z" in prompt


def test_user_prompt_tells_the_model_the_articles_are_ordered_by_relevance() -> None:
    """The order is a fact about the list, so the model is told what it means."""
    prompt = build_user_prompt(
        ticker="CRWD",
        articles=[_article(1), _article(2)],
        quote=QUOTE,
        market_context=MARKET_CONTEXT,
    )

    assert "most relevant first" in prompt
    assert "newest first" not in prompt

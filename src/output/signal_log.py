"""Building and appending JSONL signal records."""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import config

from src.analysis.models import TickerAnalysis
from src.api.models import NewsArticle, Quote


def build_record(
    *,
    run_timestamp: datetime,
    ticker: str,
    analysis: TickerAnalysis,
    articles: Sequence[NewsArticle],
    quote: Quote,
    market: Quote,
) -> dict[str, Any]:
    """Assemble one log entry.

    Args:
        run_timestamp: When the run started. Shared by every record in a run.
        ticker: The ticker this record is about.
        analysis: The signal and the market backdrop it was judged against.
        articles: The articles that were actually analysed. These become the
            record's sources, so no URL can come from the model.
        quote: The ticker's price snapshot at signal time.
        market: The market proxy's snapshot, kept as the evaluation baseline.

    Returns:
        A dict of plain JSON values, in the documented key order.

    Raises:
        ValueError: If a timestamp carries no timezone.
    """
    signal = analysis.analysis
    return {
        "run_timestamp": _utc_string(run_timestamp, field="run_timestamp"),
        "ticker": ticker,
        "signal": signal.signal.value,
        "confidence": signal.confidence.value,
        "event": signal.event,
        "reasoning": signal.reasoning,
        "news_sources": [_to_source(article) for article in articles],
        "news_summary": signal.news_summary,
        "market_context": analysis.market_context,
        "price_at_signal": quote.price,
        "market_price_at_signal": market.price,
        # Filled in by hand after 7 and 30 days, not by this run.
        "evaluation": {
            "price_7d": None,
            "price_30d": None,
            "market_7d": None,
            "market_30d": None,
            "outcome": None,
        },
    }


def _to_source(article: NewsArticle) -> dict[str, str]:
    """Reduce an article to the grounding a later reader needs."""
    return {
        "url": article.url,
        "published_at": _utc_string(article.published_at, field="published_at"),
        "title": article.title,
    }


def _utc_string(value: datetime, *, field: str) -> str:
    """Format a timestamp as UTC.

    The output format ends in Z, so a timestamp with no timezone is rejected
    rather than silently labelled as UTC.

    Raises:
        ValueError: If the timestamp carries no timezone.
    """
    if value.tzinfo is None:
        raise ValueError(
            f"'{field}' must be timezone aware so it can be written as UTC, "
            f"got the naive value {value.isoformat()}"
        )
    return value.astimezone(UTC).strftime(config.TIMESTAMP_FORMAT)


def append_record(record: Mapping[str, Any], *, path: Path) -> None:
    """Append one record to the signal log as a single JSONL line.

    The parent directory is created if it does not exist. The whole line is
    written in one call, so a crash cannot leave half a record behind.

    Args:
        record: The record to write, as returned by build_record.
        path: The log file to append to.

    Raises:
        OSError: If the directory or the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False keeps non-ASCII headlines readable in the log.
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as log:
        log.write(line)

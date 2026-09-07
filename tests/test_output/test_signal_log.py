"""Behaviour of building and appending JSONL signal records."""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.analysis.models import (
    Confidence,
    SignalAnalysis,
    SignalType,
    TickerAnalysis,
)
from src.api.models import NewsArticle, Quote
from src.output.signal_log import append_record, build_record

RUN_TIMESTAMP = datetime(2026, 9, 7, 8, 0, tzinfo=UTC)
QUOTE = Quote(
    symbol="CRWD",
    price=342.5,
    change_percent=1.0324,
    latest_trading_day="2026-09-05",
)
MARKET = Quote(
    symbol="SPY",
    price=645.2,
    change_percent=0.8283,
    latest_trading_day="2026-09-05",
)
ARTICLE = NewsArticle(
    title="CrowdStrike beats Q2 estimates",
    url="https://www.example.com/crwd-q2-beat",
    published_at=datetime(2026, 9, 7, 6, 0, tzinfo=UTC),
    summary="Revenue came in above consensus.",
    source="Example Newswire",
)
ANALYSIS = TickerAnalysis(
    analysis=SignalAnalysis(
        signal=SignalType.BUY,
        confidence=Confidence.HIGH,
        event="Q2 beat and guidance raise",
        reasoning="Corroborated by three sources.",
        news_summary="Revenue came in above consensus.",
    ),
    market_context="SPY 645.20, +0.83% on 2026-09-05",
)


def _record(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "run_timestamp": RUN_TIMESTAMP,
        "ticker": "CRWD",
        "analysis": ANALYSIS,
        "articles": [ARTICLE],
        "quote": QUOTE,
        "market": MARKET,
        "lookback_days": 3,
    }
    kwargs.update(overrides)
    return build_record(**kwargs)  # type: ignore[arg-type]


def test_record_has_the_documented_keys_in_order() -> None:
    assert list(_record()) == [
        "run_timestamp",
        "ticker",
        "signal",
        "confidence",
        "event",
        "reasoning",
        "news_sources",
        "news_summary",
        "market_context",
        "price_at_signal",
        "market_price_at_signal",
        "lookback_days",
        "evaluation",
    ]


def test_record_carries_the_analysis_as_plain_json_values() -> None:
    record = _record()

    assert record["run_timestamp"] == "2026-09-07T08:00:00Z"
    assert record["ticker"] == "CRWD"
    assert record["signal"] == "BUY"
    assert type(record["signal"]) is str
    assert record["confidence"] == "HIGH"
    assert type(record["confidence"]) is str
    assert record["event"] == "Q2 beat and guidance raise"
    assert record["reasoning"] == "Corroborated by three sources."
    assert record["news_summary"] == "Revenue came in above consensus."
    assert record["market_context"] == "SPY 645.20, +0.83% on 2026-09-05"


def test_record_carries_the_lookback_window_it_was_built_from() -> None:
    """The evaluation reads this log 30 days later, when the window a signal
    was judged on is not otherwise recoverable from the file."""
    record = _record(lookback_days=4)

    assert record["lookback_days"] == 4
    assert type(record["lookback_days"]) is int


def test_record_carries_both_price_baselines() -> None:
    """The market baseline is what makes market_7d and market_30d computable."""
    record = _record()

    assert record["price_at_signal"] == 342.5
    assert record["market_price_at_signal"] == 645.2


def test_news_sources_mirror_the_articles_that_were_analysed() -> None:
    """Sources come from the fetched articles, never from the model."""
    second = NewsArticle(
        title="Analysts lift price targets",
        url="https://www.example.com/crwd-targets",
        published_at=datetime(2026, 9, 7, 7, 15, tzinfo=UTC),
        summary="Four brokerages raised targets.",
        source="Finance Example",
    )

    record = _record(articles=[ARTICLE, second])

    assert record["news_sources"] == [
        {
            "url": "https://www.example.com/crwd-q2-beat",
            "published_at": "2026-09-07T06:00:00Z",
            "title": "CrowdStrike beats Q2 estimates",
        },
        {
            "url": "https://www.example.com/crwd-targets",
            "published_at": "2026-09-07T07:15:00Z",
            "title": "Analysts lift price targets",
        },
    ]


def test_news_sources_is_empty_on_a_quiet_day() -> None:
    assert _record(articles=[])["news_sources"] == []


def test_evaluation_block_is_written_empty() -> None:
    """Filled in by hand after 7 and 30 days, not by this run."""
    assert _record()["evaluation"] == {
        "price_7d": None,
        "price_30d": None,
        "market_7d": None,
        "market_30d": None,
        "outcome": None,
    }


def test_timestamps_are_normalised_to_utc() -> None:
    """A record must never claim Z for a time that is not UTC."""
    stockholm = timezone(timedelta(hours=2))
    record = _record(run_timestamp=datetime(2026, 9, 7, 10, 0, tzinfo=stockholm))

    assert record["run_timestamp"] == "2026-09-07T08:00:00Z"


def test_a_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        _record(run_timestamp=datetime(2026, 9, 7, 8, 0))

    assert "run_timestamp" in str(excinfo.value)


def test_append_creates_the_log_and_its_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "output" / "signals.jsonl"

    append_record(_record(), path=path)

    assert path.exists()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_append_adds_a_line_without_touching_the_earlier_ones(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signals.jsonl"

    append_record(_record(ticker="CRWD"), path=path)
    append_record(_record(ticker="IRDM"), path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["ticker"] for line in lines] == ["CRWD", "IRDM"]


def test_every_line_round_trips_through_json(tmp_path: Path) -> None:
    path = tmp_path / "signals.jsonl"
    record = _record()

    append_record(record, path=path)

    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_numbers_stay_numbers_in_the_log(tmp_path: Path) -> None:
    """The evaluation later does arithmetic on these."""
    path = tmp_path / "signals.jsonl"

    append_record(_record(), path=path)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["price_at_signal"] == 342.5
    assert isinstance(written["price_at_signal"], float)
    assert isinstance(written["market_price_at_signal"], float)
    assert written["evaluation"]["price_7d"] is None


def test_non_ascii_headlines_stay_readable(tmp_path: Path) -> None:
    path = tmp_path / "signals.jsonl"
    article = NewsArticle(
        title="Nordea höjer riktkursen för bolaget i Malmö",
        url="https://www.example.com/nordea-malmo",
        published_at=datetime(2026, 9, 7, 6, 0, tzinfo=UTC),
        summary="Höjd riktkurs.",
        source="Example",
    )

    append_record(_record(articles=[article]), path=path)

    raw = path.read_text(encoding="utf-8")
    assert "Malmö" in raw
    assert "\\u" not in raw


def test_the_log_ends_with_a_newline(tmp_path: Path) -> None:
    """So the next append starts a new line rather than extending this one."""
    path = tmp_path / "signals.jsonl"

    append_record(_record(), path=path)

    assert path.read_text(encoding="utf-8").endswith("\n")

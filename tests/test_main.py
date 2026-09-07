"""Behaviour of a full run: what gets logged, what gets reported, exit code."""

import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

import config
import main

from src.analysis.errors import LLMRequestError
from src.analysis.llm import SignalGenerator, StubGenerator
from src.api.client import FixtureSource, ResponseSource
from src.api.errors import RateLimitError
from tests.doubles import FailingGenerator, SpySleep, SpySource

RUN_TIMESTAMP = datetime(2026, 9, 7, 8, 0, tzinfo=UTC)


def _no_delay(seconds: float) -> None:
    """A no-op sleep so the suite is not slowed down by the real delay."""


def _run(
    path: Path,
    *,
    source: ResponseSource | None = None,
    generator: SignalGenerator | None = None,
    sleep: Callable[[float], None] = _no_delay,
) -> tuple[int, str, str]:
    """Run the pipeline, returning the exit code and both streams."""
    stdout, stderr = io.StringIO(), io.StringIO()
    code = main.run(
        source=source if source is not None else FixtureSource(config.FIXTURES_DIR),
        generator=generator if generator is not None else StubGenerator(),
        run_timestamp=RUN_TIMESTAMP,
        output_path=path,
        stdout=stdout,
        stderr=stderr,
        sleep=sleep,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def _records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_clean_run_logs_every_ticker_and_exits_zero(tmp_path: Path) -> None:
    path = tmp_path / "output" / "signals.jsonl"

    code, stdout, stderr = _run(path)

    assert code == 0
    assert stderr == ""
    assert [record["ticker"] for record in _records(path)] == config.WATCHLIST
    for ticker in config.WATCHLIST:
        assert ticker in stdout


def test_every_line_of_a_run_shares_one_timestamp(tmp_path: Path) -> None:
    """The evaluation groups records by run, so the stamp must not drift."""
    path = tmp_path / "signals.jsonl"

    _run(path)

    assert {record["run_timestamp"] for record in _records(path)} == {
        "2026-09-07T08:00:00Z"
    }


def test_the_market_proxy_is_quoted_once_per_run(tmp_path: Path) -> None:
    """The free tier allows 25 calls a day. One SPY quote covers the run."""
    source = SpySource(FixtureSource(config.FIXTURES_DIR))

    _run(tmp_path / "signals.jsonl", source=source)

    market_calls = [call for call in source.calls if call == ("GLOBAL_QUOTE", "SPY")]
    assert len(market_calls) == 1


def test_a_run_stays_within_the_daily_call_budget(tmp_path: Path) -> None:
    source = SpySource(FixtureSource(config.FIXTURES_DIR))

    _run(tmp_path / "signals.jsonl", source=source)

    assert len(source.calls) == 2 * len(config.WATCHLIST) + 1


def test_a_delay_separates_every_alpha_vantage_call(tmp_path: Path) -> None:
    """The free tier allows 1 request per second; calls must be paced."""
    source = SpySource(FixtureSource(config.FIXTURES_DIR))
    sleep = SpySleep()

    _run(tmp_path / "signals.jsonl", source=source, sleep=sleep)

    assert sleep.delays == [config.ALPHAVANTAGE_DELAY_SECONDS] * (len(source.calls) - 1)


def test_a_failing_ticker_does_not_stop_the_others(tmp_path: Path) -> None:
    path = tmp_path / "signals.jsonl"
    source = SpySource(
        FixtureSource(config.FIXTURES_DIR),
        failures={("NEWS_SENTIMENT", "CRSP"): RateLimitError("daily limit reached")},
    )

    code, stdout, stderr = _run(path, source=source)

    assert code == 1
    logged = [record["ticker"] for record in _records(path)]
    assert logged == [t for t in config.WATCHLIST if t != "CRSP"]
    assert "CRSP" in stderr
    assert "daily limit reached" in stderr
    assert "1 failed" in stdout


def test_a_failing_ticker_writes_no_partial_record(tmp_path: Path) -> None:
    path = tmp_path / "signals.jsonl"
    source = SpySource(
        FixtureSource(config.FIXTURES_DIR),
        failures={("GLOBAL_QUOTE", "AROC"): RateLimitError("daily limit reached")},
    )

    _run(path, source=source)

    assert all(record["ticker"] != "AROC" for record in _records(path))


def test_an_analysis_failure_fails_only_the_tickers_that_needed_the_model(
    tmp_path: Path,
) -> None:
    """IRDM is a quiet day, so it never reaches the model and still logs."""
    path = tmp_path / "signals.jsonl"

    code, stdout, stderr = _run(
        path, generator=FailingGenerator(LLMRequestError("Anthropic returned 529"))
    )

    assert code == 1
    assert [record["ticker"] for record in _records(path)] == ["IRDM"]
    assert stderr.count("Anthropic returned 529") == 3


def test_a_failing_market_quote_aborts_before_any_ticker(tmp_path: Path) -> None:
    """Every record needs the market baseline, so a partial run is worthless."""
    path = tmp_path / "signals.jsonl"
    source = SpySource(
        FixtureSource(config.FIXTURES_DIR),
        failures={("GLOBAL_QUOTE", "SPY"): RateLimitError("daily limit reached")},
    )

    code, _, stderr = _run(path, source=source)

    assert code == 1
    assert _records(path) == []
    assert "SPY" in stderr
    assert source.calls == [("GLOBAL_QUOTE", "SPY")]


def test_an_unwritable_log_fails_the_tickers_rather_than_crashing(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    code, stdout, stderr = _run(blocker / "signals.jsonl")

    assert code == 1
    for ticker in config.WATCHLIST:
        assert ticker in stderr
    assert "0 of 4" in stdout


def test_the_summary_reports_what_was_logged(tmp_path: Path) -> None:
    code, stdout, _ = _run(tmp_path / "signals.jsonl")

    assert code == 0
    assert f"{len(config.WATCHLIST)} of {len(config.WATCHLIST)}" in stdout


def test_main_reports_a_missing_key_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "USE_FIXTURES", False)
    monkeypatch.setattr(config, "ALPHAVANTAGE_API_KEY", None)
    monkeypatch.setattr(config, "OUTPUT_PATH", tmp_path / "signals.jsonl")

    code = main.main()

    assert code == 1
    assert "ALPHAVANTAGE_API_KEY" in capsys.readouterr().err

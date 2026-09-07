"""Entry point. Orchestrates one run of the Market Signal Engine.

This module wires the three packages together and owns the exit code. All
business logic lives in src/api/, src/analysis/ and src/output/.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

import config

from src.analysis.analyzer import analyze
from src.analysis.errors import AnalysisError, MissingApiKeyError
from src.analysis.llm import SignalGenerator, build_generator
from src.api.client import ResponseSource, build_source
from src.api.errors import AlphaVantageError
from src.api.errors import MissingApiKeyError as MissingDataApiKeyError
from src.api.news import fetch_news
from src.api.prices import fetch_quote
from src.output.signal_log import append_record, build_record

_EXIT_OK = 0
_EXIT_FAILED = 1


def run(
    *,
    source: ResponseSource,
    generator: SignalGenerator,
    run_timestamp: datetime,
    output_path: Path,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Analyse every ticker on the watchlist and log one record each.

    A ticker that fails is skipped and reported, and the run continues, so one
    bad symbol does not cost the whole day's signals. Nothing is written for a
    ticker that failed.

    Args:
        source: Where Alpha Vantage payloads come from.
        generator: What produces a signal analysis.
        run_timestamp: Shared by every record written by this run.
        output_path: The JSONL log to append to.
        stdout: Stream for the per-ticker results and the summary.
        stderr: Stream for failures.

    Returns:
        0 if every ticker was logged, 1 otherwise.
    """
    since = run_timestamp - timedelta(days=config.NEWS_DAYS_BACK)

    # Fetched once, before any ticker. Every record needs the market baseline,
    # so there is nothing worth writing if this fails.
    try:
        market = fetch_quote(config.MARKET_PROXY, source=source)
    except AlphaVantageError as error:
        print(
            f"Aborted: could not quote the market proxy "
            f"{config.MARKET_PROXY}. {error}",
            file=stderr,
        )
        return _EXIT_FAILED

    failed = 0
    for ticker in config.WATCHLIST:
        try:
            articles = fetch_news(
                ticker, source=source, since=since, limit=config.NEWS_LIMIT
            )
            quote = fetch_quote(ticker, source=source)
            analysis = analyze(
                ticker,
                articles=articles,
                quote=quote,
                market=market,
                generator=generator,
            )
            record = build_record(
                run_timestamp=run_timestamp,
                ticker=ticker,
                analysis=analysis,
                articles=articles,
                quote=quote,
                market=market,
            )
            append_record(record, path=output_path)
        except (AlphaVantageError, AnalysisError, OSError) as error:
            failed += 1
            print(f"{ticker}: {type(error).__name__}: {error}", file=stderr)
            continue

        print(
            f"{ticker:<6} {record['signal']:<4} {record['confidence']}",
            file=stdout,
        )

    logged = len(config.WATCHLIST) - failed
    summary = f"{logged} of {len(config.WATCHLIST)} tickers logged"
    if failed:
        summary = f"{summary}, {failed} failed"
    print(summary, file=stdout)

    return _EXIT_FAILED if failed else _EXIT_OK


def main() -> int:
    """Build the run's collaborators from configuration and run it."""
    try:
        source = build_source()
        generator = build_generator()
    except (MissingDataApiKeyError, MissingApiKeyError) as error:
        print(f"Aborted: {error}", file=sys.stderr)
        return _EXIT_FAILED

    return run(
        source=source,
        generator=generator,
        run_timestamp=datetime.now(UTC),
        output_path=config.OUTPUT_PATH,
    )


if __name__ == "__main__":
    sys.exit(main())

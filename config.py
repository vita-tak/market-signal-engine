"""Central configuration for the Market Signal Engine.

This is the only module that reads environment variables or loads .env.
Every other module imports the constants below or receives them as arguments.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, *, default: bool) -> bool:
    """Read a boolean flag from the environment.

    Args:
        name: Name of the environment variable to read.
        default: Value to use when the variable is unset or blank.

    Returns:
        The parsed boolean value.

    Raises:
        ValueError: If the variable is set to a value that is neither a
            recognised true value nor a recognised false value. Failing here
            is deliberate: a typo must not silently decide whether the run
            reaches a real API.
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if not normalized:
        return default
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    valid = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(
        f"Environment variable '{name}' must be one of: {valid}. Got '{raw}'."
    )


# Secrets, loaded from .env
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID")

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# Tickers analysed on every run
WATCHLIST = [
    "ALNY",
    "CRWD",
    "CRSP",
    "IRDM",
]

# Base lookback for an ordinary trading day. src/api/market_calendar.py
# adds one day for every market closure immediately before a run.
NEWS_DAYS_BACK = 1
# Max articles per day of the news window, so a longer window asks for
# proportionally more news rather than only the newest headlines.
NEWS_LIMIT_PER_DAY = 10
# Bounds the backward scan for the last trading day. The rule never
# produces more than 4, so reaching this means the holiday table is wrong.
MAX_LOOKBACK_DAYS = 10
# The market keeps its own calendar, so the run timestamp is read in this
# zone before its calendar day decides the lookback.
MARKET_TIMEZONE = "America/New_York"

# Alpha Vantage
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
NEWS_FUNCTION = "NEWS_SENTIMENT"
QUOTE_FUNCTION = "GLOBAL_QUOTE"
NEWS_SORT = "LATEST"
REQUEST_TIMEOUT_SECONDS = 30.0

# Free tier allows 1 request per second. Callers must pace their calls.
ALPHAVANTAGE_DELAY_SECONDS = 1.2

# Timestamp format of the time_published field in a NEWS_SENTIMENT response
AV_TIME_PUBLISHED_FORMAT = "%Y%m%dT%H%M%S"
# Timestamp format of the time_from request parameter
AV_TIME_FROM_FORMAT = "%Y%m%dT%H%M"

# Broader market proxy. Provides the market context on each signal and the
# baseline price needed to evaluate a signal against the market later on.
MARKET_PROXY = "SPY"
MARKET_CONTEXT_TEMPLATE = (
    "{symbol} {price:.2f}, {change_percent:+.2f}% on {latest_trading_day}"
)

# Anthropic
ANTHROPIC_MODEL = "claude-haiku-4-5"
ANTHROPIC_MAX_TOKENS = 2000

# Offline mode. Both flags default to True so a development run never reaches
# a real API. Set USE_FIXTURES=false and USE_LLM_STUB=false for a live run.
USE_FIXTURES = env_flag("USE_FIXTURES", default=True)
USE_LLM_STUB = env_flag("USE_LLM_STUB", default=True)

FIXTURES_DIR = PROJECT_ROOT / "src" / "api" / "fixtures"

# Output
OUTPUT_PATH = PROJECT_ROOT / "output" / "signals.jsonl"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Used when a ticker has no news in the lookback window. Signal criterion 4
# fixes the answer for a quiet day, so no analysis request is made.
QUIET_DAY_EVENT = "No qualifying event. No articles in the lookback window."
QUIET_DAY_REASONING = (
    "No news was returned for this ticker in the lookback window, so there is "
    "no concrete event to act on. Holding rather than reading a signal into "
    "the absence of news."
)
QUIET_DAY_NEWS_SUMMARY = (
    "No articles were returned for this ticker in the lookback window."
)

# Used when USE_LLM_STUB is enabled. The text marks the record as a stub so an
# offline run can never be mistaken for a real one.
STUB_EVENT = "STUB: no analysis performed, USE_LLM_STUB is enabled."
STUB_REASONING = (
    "STUB: this record came from the offline signal generator. "
    "Set USE_LLM_STUB=false to run a real analysis."
)
STUB_NEWS_SUMMARY = "STUB: news was not analysed."

"""Fetching and parsing Alpha Vantage GLOBAL_QUOTE payloads."""

from typing import Any

import config

from src.api.client import ResponseSource, ensure_ok
from src.api.errors import MalformedResponseError
from src.api.models import Quote

_QUOTE_KEY = "Global Quote"
_SYMBOL_FIELD = "01. symbol"
_PRICE_FIELD = "05. price"
_TRADING_DAY_FIELD = "07. latest trading day"
_CHANGE_PERCENT_FIELD = "10. change percent"


def fetch_quote(symbol: str, *, source: ResponseSource) -> Quote:
    """Return the latest price snapshot for a symbol.

    Args:
        symbol: Ticker or index proxy to quote.
        source: Where to read the Alpha Vantage payload from.

    Returns:
        The parsed quote.

    Raises:
        AlphaVantageError: If the payload reports a failure or is malformed.
    """
    payload = source.get(config.QUOTE_FUNCTION, {"symbol": symbol})
    ensure_ok(payload, context=symbol)

    quote = payload.get(_QUOTE_KEY)
    if not isinstance(quote, dict) or not quote:
        # An unknown symbol comes back as an empty object rather than an error.
        raise MalformedResponseError(
            f"Alpha Vantage returned no '{_QUOTE_KEY}' data for {symbol}"
        )

    return Quote(
        symbol=_read_str(quote, _SYMBOL_FIELD, symbol=symbol),
        price=_read_float(quote, _PRICE_FIELD, symbol=symbol),
        change_percent=_read_float(
            quote, _CHANGE_PERCENT_FIELD, symbol=symbol, strip="%"
        ),
        latest_trading_day=_read_str(quote, _TRADING_DAY_FIELD, symbol=symbol),
    )


def _read_str(quote: dict[str, Any], field: str, *, symbol: str) -> str:
    """Read a required string field out of a quote."""
    value = quote.get(field)
    if not isinstance(value, str) or not value:
        raise MalformedResponseError(
            f"Quote for {symbol} has no usable '{field}', got '{value}'"
        )
    return value


def _read_float(
    quote: dict[str, Any], field: str, *, symbol: str, strip: str = ""
) -> float:
    """Read a required numeric field, which arrives as a string.

    Args:
        quote: The Global Quote object.
        field: Field name to read.
        symbol: Symbol the quote belongs to, used in the error message.
        strip: Trailing characters to remove, such as the percent sign.
    """
    raw = quote.get(field)
    try:
        return float(str(raw).rstrip(strip))
    except (TypeError, ValueError) as error:
        raise MalformedResponseError(
            f"Quote for {symbol} has an unreadable '{field}', got '{raw}'"
        ) from error

"""Behaviour of fetching and parsing Alpha Vantage GLOBAL_QUOTE payloads."""

import pytest

from src.api.errors import MalformedResponseError, RateLimitError
from src.api.models import Quote
from src.api.prices import fetch_quote
from tests.doubles import FakeSource


def _quote_payload(
    *,
    symbol: str = "CRWD",
    price: str = "342.5000",
    change_percent: str = "1.0324%",
    latest_trading_day: str = "2026-09-05",
) -> dict[str, object]:
    return {
        "Global Quote": {
            "01. symbol": symbol,
            "02. open": "339.1000",
            "03. high": "345.8000",
            "04. low": "337.5000",
            "05. price": price,
            "06. volume": "2841903",
            "07. latest trading day": latest_trading_day,
            "08. previous close": "339.0000",
            "09. change": "3.5000",
            "10. change percent": change_percent,
        }
    }


def test_fetch_quote_parses_a_quote() -> None:
    """Every number arrives as a string, and the change carries a percent sign."""
    source = FakeSource(_quote_payload())

    quote = fetch_quote("CRWD", source=source)

    assert quote == Quote(
        symbol="CRWD",
        price=342.5,
        change_percent=1.0324,
        latest_trading_day="2026-09-05",
    )


def test_fetch_quote_parses_a_negative_change() -> None:
    source = FakeSource(_quote_payload(change_percent="-0.8412%"))

    assert fetch_quote("CRWD", source=source).change_percent == -0.8412


def test_fetch_quote_requests_the_symbol_from_alpha_vantage() -> None:
    source = FakeSource(_quote_payload(symbol="SPY"))

    fetch_quote("SPY", source=source)

    assert source.calls == [("GLOBAL_QUOTE", {"symbol": "SPY"})]


def test_fetch_quote_raises_for_an_empty_quote() -> None:
    """An unknown symbol yields an empty object, never a zero price."""
    source = FakeSource({"Global Quote": {}})

    with pytest.raises(MalformedResponseError) as excinfo:
        fetch_quote("NOPE", source=source)

    assert "NOPE" in str(excinfo.value)


def test_fetch_quote_raises_when_the_quote_key_is_missing() -> None:
    source = FakeSource({"items": "0"})

    with pytest.raises(MalformedResponseError) as excinfo:
        fetch_quote("CRWD", source=source)

    assert "Global Quote" in str(excinfo.value)


def test_fetch_quote_raises_for_an_unparseable_price() -> None:
    source = FakeSource(_quote_payload(price="n/a"))

    with pytest.raises(MalformedResponseError) as excinfo:
        fetch_quote("CRWD", source=source)

    assert "05. price" in str(excinfo.value)
    assert "n/a" in str(excinfo.value)


def test_fetch_quote_propagates_a_rate_limit_payload() -> None:
    source = FakeSource({"Note": "Thank you for using Alpha Vantage!"})

    with pytest.raises(RateLimitError):
        fetch_quote("CRWD", source=source)

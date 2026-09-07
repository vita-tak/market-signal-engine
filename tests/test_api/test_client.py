"""Behaviour of the Alpha Vantage transport layer."""

from collections.abc import Mapping

import pytest
import requests

import config

from src.api.client import (
    FixtureSource,
    LiveSource,
    build_source,
    ensure_ok,
)
from src.api.errors import (
    FixtureNotFoundError,
    InvalidRequestError,
    MissingApiKeyError,
    RateLimitError,
    TransportError,
)


def test_ensure_ok_accepts_a_normal_payload() -> None:
    ensure_ok({"feed": []}, context="CRWD")


def test_ensure_ok_raises_rate_limit_for_an_information_payload() -> None:
    """The free tier reports exhaustion with HTTP 200 and an Information key."""
    payload = {
        "Information": (
            "Thank you for using Alpha Vantage! Our standard API rate limit is "
            "25 requests per day."
        )
    }

    with pytest.raises(RateLimitError) as excinfo:
        ensure_ok(payload, context="CRWD")

    assert "CRWD" in str(excinfo.value)
    assert "25 requests per day" in str(excinfo.value)


def test_ensure_ok_raises_rate_limit_for_a_note_payload() -> None:
    payload = {"Note": "Thank you for using Alpha Vantage! Please consider..."}

    with pytest.raises(RateLimitError):
        ensure_ok(payload, context="SPY")


def test_ensure_ok_raises_invalid_request_for_an_error_message_payload() -> None:
    payload = {"Error Message": "Invalid API call. Please retry."}

    with pytest.raises(InvalidRequestError) as excinfo:
        ensure_ok(payload, context="NOPE")

    assert "NOPE" in str(excinfo.value)
    assert "Invalid API call" in str(excinfo.value)


def test_fixture_source_resolves_a_news_payload_by_ticker() -> None:
    source = FixtureSource(config.FIXTURES_DIR)

    payload = source.get("NEWS_SENTIMENT", {"tickers": "CRWD", "limit": "10"})

    assert len(payload["feed"]) == 3


def test_fixture_source_resolves_a_quote_payload_by_symbol() -> None:
    source = FixtureSource(config.FIXTURES_DIR)

    payload = source.get("GLOBAL_QUOTE", {"symbol": "SPY"})

    assert payload["Global Quote"]["01. symbol"] == "SPY"


def test_fixture_source_raises_when_no_fixture_matches() -> None:
    source = FixtureSource(config.FIXTURES_DIR)

    with pytest.raises(FixtureNotFoundError) as excinfo:
        source.get("GLOBAL_QUOTE", {"symbol": "TSLA"})

    assert "global_quote_TSLA.json" in str(excinfo.value)


def test_fixture_source_raises_when_the_request_names_no_symbol() -> None:
    source = FixtureSource(config.FIXTURES_DIR)

    with pytest.raises(FixtureNotFoundError):
        source.get("GLOBAL_QUOTE", {})


def test_every_watchlist_ticker_has_both_fixtures() -> None:
    """An offline run must not fail on a missing fixture."""
    source = FixtureSource(config.FIXTURES_DIR)

    for ticker in [*config.WATCHLIST, config.MARKET_PROXY]:
        source.get("GLOBAL_QUOTE", {"symbol": ticker})

    for ticker in config.WATCHLIST:
        source.get("NEWS_SENTIMENT", {"tickers": ticker})


class FakeResponse:
    """Stands in for a requests Response."""

    def __init__(self, payload: object, *, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        if self._error is not None:
            raise self._error
        return self._payload


class FakeGetter:
    """Records the HTTP call a LiveSource would have made."""

    def __init__(self, response: FakeResponse | Exception) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(
        self, url: str, *, params: Mapping[str, str], timeout: float
    ) -> FakeResponse:
        self.calls.append((url, dict(params), timeout))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_live_source_sends_the_function_and_api_key() -> None:
    getter = FakeGetter(FakeResponse({"feed": []}))
    source = LiveSource(
        api_key="test-key",
        base_url="https://example.invalid/query",
        timeout=30.0,
        getter=getter,
    )

    payload = source.get("NEWS_SENTIMENT", {"tickers": "CRWD"})

    assert payload == {"feed": []}
    assert getter.calls == [
        (
            "https://example.invalid/query",
            {
                "tickers": "CRWD",
                "function": "NEWS_SENTIMENT",
                "apikey": "test-key",
            },
            30.0,
        )
    ]


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_live_source_rejects_a_missing_api_key(api_key: str | None) -> None:
    """The failure must land before any request leaves the machine."""
    with pytest.raises(MissingApiKeyError) as excinfo:
        LiveSource(
            api_key=api_key,
            base_url="https://example.invalid/query",
            timeout=30.0,
            getter=FakeGetter(FakeResponse({})),
        )

    assert "ALPHAVANTAGE_API_KEY" in str(excinfo.value)


def test_live_source_wraps_a_request_failure() -> None:
    source = LiveSource(
        api_key="test-key",
        base_url="https://example.invalid/query",
        timeout=30.0,
        getter=FakeGetter(requests.ConnectionError("name resolution failed")),
    )

    with pytest.raises(TransportError) as excinfo:
        source.get("GLOBAL_QUOTE", {"symbol": "CRWD"})

    assert "CRWD" in str(excinfo.value)


def test_live_source_wraps_a_non_json_body() -> None:
    source = LiveSource(
        api_key="test-key",
        base_url="https://example.invalid/query",
        timeout=30.0,
        getter=FakeGetter(FakeResponse(None, error=ValueError("not json"))),
    )

    with pytest.raises(TransportError):
        source.get("GLOBAL_QUOTE", {"symbol": "CRWD"})


def test_build_source_returns_a_fixture_source_when_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "USE_FIXTURES", True)

    assert isinstance(build_source(), FixtureSource)


def test_build_source_returns_a_live_source_when_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "USE_FIXTURES", False)
    monkeypatch.setattr(config, "ALPHAVANTAGE_API_KEY", "test-key")

    assert isinstance(build_source(), LiveSource)


def test_build_source_fails_fast_when_going_live_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "USE_FIXTURES", False)
    monkeypatch.setattr(config, "ALPHAVANTAGE_API_KEY", None)

    with pytest.raises(MissingApiKeyError):
        build_source()

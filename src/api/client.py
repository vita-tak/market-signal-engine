"""Transport layer for Alpha Vantage requests."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import requests

import config

from src.api.errors import (
    FixtureNotFoundError,
    InvalidRequestError,
    MissingApiKeyError,
    RateLimitError,
    TransportError,
)

# Keys Alpha Vantage uses to report a failure inside an HTTP 200 response.
_RATE_LIMIT_KEYS = ("Note", "Information")
_ERROR_KEY = "Error Message"

# Request parameters that name the symbol, depending on the endpoint.
_SYMBOL_PARAMS = ("tickers", "symbol")


class ResponseSource(Protocol):
    """Something that can return an Alpha Vantage payload for a request.

    Implemented by the live HTTP source and by the fixture source, so the rest
    of src/api/ is unaware of where a payload came from.
    """

    def get(self, function: str, params: Mapping[str, str]) -> dict[str, Any]:
        """Return the decoded JSON payload for one Alpha Vantage call."""
        ...


def ensure_ok(payload: Mapping[str, Any], *, context: str) -> None:
    """Raise if the payload reports a failure instead of carrying data.

    Args:
        payload: Decoded Alpha Vantage response.
        context: Ticker or symbol the payload belongs to, used in the message.

    Raises:
        RateLimitError: If the call limit is exhausted or calls are throttled.
        InvalidRequestError: If Alpha Vantage rejected the request.
    """
    for key in _RATE_LIMIT_KEYS:
        message = payload.get(key)
        if message is not None:
            raise RateLimitError(f"Alpha Vantage rate limit for {context}: {message}")

    error = payload.get(_ERROR_KEY)
    if error is not None:
        raise InvalidRequestError(
            f"Alpha Vantage rejected the request for {context}: {error}"
        )


class FixtureSource:
    """Reads saved Alpha Vantage payloads from disk instead of calling the API.

    A fixture is named after the request that produced it, for example
    news_sentiment_CRWD.json or global_quote_SPY.json. The news window is not
    re-applied here: offline, the fixture is the window.
    """

    def __init__(self, fixtures_dir: Path) -> None:
        self._fixtures_dir = fixtures_dir

    def get(self, function: str, params: Mapping[str, str]) -> dict[str, Any]:
        """Return the saved payload for a request.

        Raises:
            FixtureNotFoundError: If the request names no symbol, or no
                matching fixture file exists.
        """
        symbol = _symbol_of(params)
        if symbol is None:
            raise FixtureNotFoundError(
                f"Cannot resolve a {function} fixture: the request names no "
                f"symbol (looked for {' or '.join(_SYMBOL_PARAMS)})"
            )

        path = self._fixtures_dir / f"{function.lower()}_{symbol}.json"
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as error:
            raise FixtureNotFoundError(f"No fixture at {path}") from error

        payload: dict[str, Any] = json.loads(raw)
        return payload


def _symbol_of(params: Mapping[str, str]) -> str | None:
    """Return the symbol a request is about, whichever parameter names it."""
    for key in _SYMBOL_PARAMS:
        value = params.get(key)
        if value:
            return value
    return None


class HttpResponse(Protocol):
    """The part of a requests Response that this module uses."""

    def raise_for_status(self) -> Any: ...

    def json(self) -> Any: ...


class HttpGetter(Protocol):
    """Performs one HTTP GET. Substituted in tests so no request is made."""

    def __call__(
        self, url: str, *, params: Mapping[str, str], timeout: float
    ) -> HttpResponse: ...


def _requests_get(
    url: str, *, params: Mapping[str, str], timeout: float
) -> HttpResponse:
    """Adapter that gives requests.get the shape of an HttpGetter."""
    return requests.get(url, params=params, timeout=timeout)


class LiveSource:
    """Calls the real Alpha Vantage API over HTTP."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout: float,
        getter: HttpGetter = _requests_get,
    ) -> None:
        """Validate the credentials before any request can be made.

        Raises:
            MissingApiKeyError: If no usable API key was supplied.
        """
        if api_key is None or not api_key.strip():
            raise MissingApiKeyError(
                "ALPHAVANTAGE_API_KEY is not set. Set it in .env, or run with "
                "USE_FIXTURES=true to work from saved responses."
            )
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._getter = getter

    def get(self, function: str, params: Mapping[str, str]) -> dict[str, Any]:
        """Return the decoded payload for one Alpha Vantage call.

        Raises:
            TransportError: If the request failed or the body was not JSON.
        """
        context = _symbol_of(params) or function
        query = {**params, "function": function, "apikey": self._api_key}
        try:
            response = self._getter(
                self._base_url, params=query, timeout=self._timeout
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except requests.RequestException as error:
            raise TransportError(
                f"Alpha Vantage request for {context} failed: {error}"
            ) from error
        except ValueError as error:
            raise TransportError(
                f"Alpha Vantage returned a non-JSON body for {context}: {error}"
            ) from error
        return payload


def build_source() -> ResponseSource:
    """Return the response source the current configuration asks for.

    Raises:
        MissingApiKeyError: If live mode is on but no API key is configured.
    """
    if config.USE_FIXTURES:
        return FixtureSource(config.FIXTURES_DIR)
    return LiveSource(
        api_key=config.ALPHAVANTAGE_API_KEY,
        base_url=config.ALPHAVANTAGE_BASE_URL,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )

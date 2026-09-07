"""Exceptions raised by the Alpha Vantage client.

Alpha Vantage answers with HTTP 200 for every one of these conditions, so the
payload itself has to be inspected. Each error names the ticker or symbol that
caused it, because a run touches several symbols in a row.
"""


class AlphaVantageError(Exception):
    """Base class for every failure reaching Alpha Vantage."""


class MissingApiKeyError(AlphaVantageError):
    """A live request was attempted without ALPHAVANTAGE_API_KEY set."""


class RateLimitError(AlphaVantageError):
    """The daily call limit is exhausted, or calls are being throttled."""


class InvalidRequestError(AlphaVantageError):
    """Alpha Vantage rejected the request, usually a bad symbol or key."""


class MalformedResponseError(AlphaVantageError):
    """The payload arrived but does not hold the fields it should."""


class TransportError(AlphaVantageError):
    """The request never completed, or the body was not JSON."""


class FixtureNotFoundError(AlphaVantageError):
    """Offline mode was asked for a payload with no matching fixture file."""

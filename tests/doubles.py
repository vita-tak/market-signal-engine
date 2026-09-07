"""Test doubles shared across the test suite."""

from collections.abc import Mapping
from typing import Any


class FakeSource:
    """A ResponseSource that returns a canned payload and records its calls."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, function: str, params: Mapping[str, str]) -> dict[str, Any]:
        self.calls.append((function, dict(params)))
        return self.payload


class NeverCalledGenerator:
    """A SignalGenerator that fails the test if it is ever used."""

    def generate(self, *, system: str, user: str) -> Any:
        raise AssertionError(
            "The signal generator was called when it should not have been"
        )


class RecordingGenerator:
    """A SignalGenerator that returns a canned analysis and records its prompts."""

    def __init__(self, analysis: Any) -> None:
        self._analysis = analysis
        self.prompts: list[tuple[str, str]] = []

    def generate(self, *, system: str, user: str) -> Any:
        self.prompts.append((system, user))
        return self._analysis


class SpySource:
    """Wraps a ResponseSource, recording every call and failing on demand."""

    def __init__(
        self,
        inner: Any,
        failures: Mapping[tuple[str, str], Exception] | None = None,
    ) -> None:
        self._inner = inner
        self._failures = dict(failures or {})
        self.calls: list[tuple[str, str]] = []

    def get(self, function: str, params: Mapping[str, str]) -> dict[str, Any]:
        symbol = params.get("tickers") or params.get("symbol") or ""
        self.calls.append((function, symbol))
        error = self._failures.get((function, symbol))
        if error is not None:
            raise error
        result: dict[str, Any] = self._inner.get(function, params)
        return result


class FailingGenerator:
    """A SignalGenerator that always fails."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def generate(self, *, system: str, user: str) -> Any:
        raise self._error

"""Behaviour at the boundary between the analyser and the Anthropic API.

No test here reaches the network. The Anthropic call is replaced by a fake
parser with the same shape.
"""

import anthropic
import httpx2
import pytest

import config

from src.analysis.errors import (
    InvalidSignalError,
    LLMRequestError,
    MissingApiKeyError,
)
from src.analysis.llm import (
    AnthropicGenerator,
    ParsedSignal,
    StubGenerator,
    build_generator,
)
from src.analysis.models import Confidence, SignalAnalysis, SignalType

ANALYSIS = SignalAnalysis(
    signal=SignalType.BUY,
    confidence=Confidence.HIGH,
    event="Q2 beat and guidance raise",
    reasoning="Three sources report the same beat.",
    news_summary="Revenue came in above consensus.",
)


class FakeParser:
    """Stands in for one structured-output request."""

    def __init__(self, result: ParsedSignal | Exception) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def __call__(
        self, *, model: str, max_tokens: int, system: str, user: str
    ) -> ParsedSignal:
        self.calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "user": user,
            }
        )
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _generator(result: ParsedSignal | Exception) -> tuple[AnthropicGenerator, FakeParser]:
    parser = FakeParser(result)
    generator = AnthropicGenerator(
        api_key="test-key",
        model="claude-haiku-4-5",
        max_tokens=2000,
        parser=parser,
    )
    return generator, parser


def test_stub_generator_returns_an_analysis_marked_as_a_stub() -> None:
    """An offline run must never be mistaken for a real one."""
    analysis = StubGenerator().generate(system="ignored", user="ignored")

    assert analysis.signal is SignalType.HOLD
    assert analysis.confidence is Confidence.LOW
    assert analysis.event == config.STUB_EVENT
    assert "STUB" in analysis.reasoning


def test_anthropic_generator_returns_the_parsed_analysis() -> None:
    generator, parser = _generator(ParsedSignal(analysis=ANALYSIS, stop_reason="end_turn"))

    result = generator.generate(system="rules", user="facts")

    assert result is ANALYSIS
    assert parser.calls == [
        {
            "model": "claude-haiku-4-5",
            "max_tokens": 2000,
            "system": "rules",
            "user": "facts",
        }
    ]


def test_anthropic_generator_raises_when_no_analysis_came_back() -> None:
    """A refusal or a truncated response must not become a silent HOLD."""
    generator, _ = _generator(ParsedSignal(analysis=None, stop_reason="max_tokens"))

    with pytest.raises(InvalidSignalError) as excinfo:
        generator.generate(system="rules", user="facts")

    assert "max_tokens" in str(excinfo.value)


def test_anthropic_generator_wraps_an_api_status_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=request), body=None
    )
    generator, _ = _generator(error)

    with pytest.raises(LLMRequestError) as excinfo:
        generator.generate(system="rules", user="facts")

    assert "429" in str(excinfo.value)


def test_anthropic_generator_wraps_a_connection_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    generator, _ = _generator(
        anthropic.APIConnectionError(message="connection refused", request=request)
    )

    with pytest.raises(LLMRequestError):
        generator.generate(system="rules", user="facts")


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_anthropic_generator_rejects_a_missing_api_key(api_key: str | None) -> None:
    with pytest.raises(MissingApiKeyError) as excinfo:
        AnthropicGenerator(
            api_key=api_key,
            model="claude-haiku-4-5",
            max_tokens=2000,
            parser=FakeParser(ParsedSignal(analysis=ANALYSIS, stop_reason=None)),
        )

    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_build_generator_returns_the_stub_when_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "USE_LLM_STUB", True)

    assert isinstance(build_generator(), StubGenerator)


def test_build_generator_returns_the_anthropic_generator_when_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "USE_LLM_STUB", False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")

    assert isinstance(build_generator(), AnthropicGenerator)


def test_build_generator_fails_fast_when_going_live_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "USE_LLM_STUB", False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)

    with pytest.raises(MissingApiKeyError):
        build_generator()

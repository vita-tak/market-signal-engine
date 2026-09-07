"""Producing a signal analysis from a pair of prompts."""

from dataclasses import dataclass
from typing import Protocol

import anthropic
from pydantic import ValidationError

import config

from src.analysis.errors import (
    InvalidSignalError,
    LLMRequestError,
    MissingApiKeyError,
)
from src.analysis.models import Confidence, SignalAnalysis, SignalType


class SignalGenerator(Protocol):
    """Something that turns a pair of prompts into a signal analysis.

    Implemented by the Anthropic generator and by the offline stub, so the
    analyzer is unaware of which one is in play.
    """

    def generate(self, *, system: str, user: str) -> SignalAnalysis:
        """Return the analysis for one ticker."""
        ...


@dataclass(frozen=True, slots=True)
class ParsedSignal:
    """The outcome of one structured-output request.

    `analysis` is None when the model refused or ran out of output tokens, in
    which case `stop_reason` says which.
    """

    analysis: SignalAnalysis | None
    stop_reason: str | None


class SignalParser(Protocol):
    """Performs one structured-output request against a model.

    This is the seam that keeps the network out of the tests.
    """

    def __call__(
        self, *, model: str, max_tokens: int, system: str, user: str
    ) -> ParsedSignal: ...


def _anthropic_parser(api_key: str) -> SignalParser:
    """Adapter giving the Anthropic SDK the shape of a SignalParser.

    Structured outputs are used rather than free text, so the signal and
    confidence values can only ever be members of their enums.
    """
    client = anthropic.Anthropic(api_key=api_key)

    def parse(
        *, model: str, max_tokens: int, system: str, user: str
    ) -> ParsedSignal:
        response = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=SignalAnalysis,
        )
        return ParsedSignal(
            analysis=response.parsed_output, stop_reason=response.stop_reason
        )

    return parse


class StubGenerator:
    """Returns a fixed analysis without contacting any model.

    Used when USE_LLM_STUB is on. The text marks the record as a stub so an
    offline run can never be mistaken for a real one.
    """

    def generate(self, *, system: str, user: str) -> SignalAnalysis:
        """Return the fixed stub analysis, ignoring both prompts."""
        return SignalAnalysis(
            signal=SignalType.HOLD,
            confidence=Confidence.LOW,
            event=config.STUB_EVENT,
            reasoning=config.STUB_REASONING,
            news_summary=config.STUB_NEWS_SUMMARY,
        )


class AnthropicGenerator:
    """Asks Claude for a signal analysis."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        max_tokens: int,
        parser: SignalParser | None = None,
    ) -> None:
        """Validate the credentials before any request can be made.

        Raises:
            MissingApiKeyError: If no usable API key was supplied.
        """
        if api_key is None or not api_key.strip():
            raise MissingApiKeyError(
                "ANTHROPIC_API_KEY is not set. Set it in .env, or run with "
                "USE_LLM_STUB=true to work without a model."
            )
        self._model = model
        self._max_tokens = max_tokens
        self._parser = parser if parser is not None else _anthropic_parser(api_key)

    def generate(self, *, system: str, user: str) -> SignalAnalysis:
        """Return the analysis for one ticker.

        Raises:
            LLMRequestError: If the request did not complete. The SDK already
                retries rate limits and server errors on its own, so anything
                surfacing here has exhausted those attempts.
            InvalidSignalError: If the model refused, ran out of output
                tokens, or answered outside the schema.
        """
        try:
            result = self._parser(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                user=user,
            )
        except anthropic.APIStatusError as error:
            raise LLMRequestError(
                f"Anthropic returned {error.status_code} for the {self._model} "
                f"request: {error}"
            ) from error
        except anthropic.APIConnectionError as error:
            raise LLMRequestError(f"Could not reach the Anthropic API: {error}") from error
        except anthropic.APIError as error:
            raise LLMRequestError(f"Anthropic request failed: {error}") from error
        except ValidationError as error:
            raise InvalidSignalError(
                f"The {self._model} response did not match the signal schema: {error}"
            ) from error

        if result.analysis is None:
            # A refusal or a truncated response must not become a silent HOLD.
            raise InvalidSignalError(
                f"The {self._model} response carried no analysis "
                f"(stop_reason: {result.stop_reason})"
            )
        return result.analysis


def build_generator() -> SignalGenerator:
    """Return the generator the current configuration asks for.

    Raises:
        MissingApiKeyError: If live mode is on but no API key is configured.
    """
    if config.USE_LLM_STUB:
        return StubGenerator()
    return AnthropicGenerator(
        api_key=config.ANTHROPIC_API_KEY,
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.ANTHROPIC_MAX_TOKENS,
    )

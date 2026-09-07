"""Domain types produced by the analysis module."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class SignalType(StrEnum):
    """The action a signal recommends."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class Confidence(StrEnum):
    """How much weight the signal carries.

    HIGH: multiple corroborating sources, clear causal link to price impact.
    MEDIUM: one strong source or an indirect causal link.
    LOW: weak signal, noisy news, or no clear event identified.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SignalAnalysis(BaseModel):
    """The contract the model must fill in.

    Market data is deliberately absent: the model is given the numbers but
    never asked to restate them, so it cannot invent a market move.
    """

    model_config = ConfigDict(extra="forbid")

    signal: SignalType
    confidence: Confidence
    event: str
    reasoning: str
    news_summary: str


@dataclass(frozen=True, slots=True)
class TickerAnalysis:
    """One ticker's analysis, plus the market backdrop it was judged against."""

    analysis: SignalAnalysis
    market_context: str

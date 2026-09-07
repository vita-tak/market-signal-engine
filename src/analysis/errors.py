"""Exceptions raised while producing a signal.

Kept separate from the Alpha Vantage hierarchy rather than sharing a base
class, so no shared error module has to live outside the three packages the
architecture defines. main.py catches both roots.
"""


class AnalysisError(Exception):
    """Base class for every failure while analysing a ticker."""


class MissingApiKeyError(AnalysisError):
    """A live analysis was attempted without ANTHROPIC_API_KEY set."""


class LLMRequestError(AnalysisError):
    """The request to the model did not complete."""


class InvalidSignalError(AnalysisError):
    """The model answered, but not with a usable analysis."""

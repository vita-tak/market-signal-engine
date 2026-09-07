"""Domain types returned by the Alpha Vantage client."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """A single news article about a ticker."""

    title: str
    url: str
    published_at: datetime
    summary: str
    source: str


@dataclass(frozen=True, slots=True)
class Quote:
    """A price snapshot for one symbol."""

    symbol: str
    price: float
    change_percent: float
    latest_trading_day: str

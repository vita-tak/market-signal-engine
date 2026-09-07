"""When the US stock market was last open, and what that means for the news
window.

Pure date arithmetic. Nothing here makes a request or reads a clock.
"""

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import config

from src.api.models import NewsWindow

_MONDAY = 0
_THURSDAY = 3
_SATURDAY = 5
_SUNDAY = 6
_NEW_YEARS_DAY = (1, 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth given weekday of a month, for example the 3rd Monday."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7, weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last given weekday of a month, for example the last Monday."""
    next_month = date(year + month // 12, month % 12 + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Return Easter Sunday by the anonymous Gregorian computus.

    Good Friday is the market closure; Easter is only how it is located.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lunar = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lunar) // 451
    month, day = divmod(h + lunar - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(day: date) -> date:
    """Shift a fixed-date holiday to the weekday the market observes it on.

    NYSE Rule 7.2: a holiday on a Saturday closes the preceding Friday, and one
    on a Sunday closes the following Monday. New Year's Day is the exception,
    because 31 December is the last trading day of the year and the market
    stays open on it. Leaving that holiday on its Saturday is harmless, since
    a Saturday is already closed by the weekend rule.

    That exception is also what keeps every observed date inside its own
    calendar year, which _market_holidays relies on to key its cache by year.
    """
    if day.weekday() == _SATURDAY:
        if (day.month, day.day) == _NEW_YEARS_DAY:
            return day
        return day - timedelta(days=1)
    if day.weekday() == _SUNDAY:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=8)
def _market_holidays(year: int) -> frozenset[date]:
    """Return the days the US stock market is closed in a given year.

    This is the NYSE calendar, not the federal one. Columbus Day and Veterans
    Day are federal holidays on which the market trades, and Good Friday is a
    market closure that is not federal.

    A scan near a year boundary touches two years, so the result is cached.
    """
    fixed = (
        date(year, 1, 1),  # New Year's Day
        date(year, 6, 19),  # Juneteenth
        date(year, 7, 4),  # Independence Day
        date(year, 12, 25),  # Christmas
    )
    floating = (
        _nth_weekday(year, 1, _MONDAY, 3),  # MLK Day
        _nth_weekday(year, 2, _MONDAY, 3),  # Presidents Day
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, _MONDAY),  # Memorial Day
        _nth_weekday(year, 9, _MONDAY, 1),  # Labor Day
        _nth_weekday(year, 11, _THURSDAY, 4),  # Thanksgiving
    )
    # Only the fixed dates can land on a weekend and need shifting. The
    # floating six are defined as a weekday of a month already.
    return frozenset([_observed(day) for day in fixed] + list(floating))


def is_trading_day(day: date) -> bool:
    """Return whether the US stock market is open on a given calendar day.

    Args:
        day: The calendar day to judge.

    Returns:
        True if the market trades that day.
    """
    return day.weekday() < _SATURDAY and day not in _market_holidays(day.year)


def lookback_days(moment: datetime) -> int:
    """Return how many days of news a run starting at this moment must cover.

    The base lookback, plus one day for every consecutive non-trading day
    immediately before the run. A Monday therefore reaches back three days to
    Friday.

    The count of days comes from the market's own calendar in New York; the
    arithmetic stays in UTC. A timedelta of n days is exactly n * 24h, so no
    daylight saving transition can shift the window.

    Args:
        moment: When the run started. Must be timezone aware.

    Returns:
        The number of days to look back.

    Raises:
        ValueError: If the moment is naive, or if no trading day is found
            within MAX_LOOKBACK_DAYS.
    """
    if moment.tzinfo is None:
        raise ValueError(
            f"'moment' must be timezone aware so it can be read in the "
            f"market's own zone, got the naive value {moment.isoformat()}"
        )

    # ZoneInfo caches its instances, so this is built per call rather than
    # at import, where a missing tz database would break the whole package.
    today = moment.astimezone(ZoneInfo(config.MARKET_TIMEZONE)).date()
    for closed in range(config.MAX_LOOKBACK_DAYS):
        if is_trading_day(today - timedelta(days=closed + 1)):
            return config.NEWS_DAYS_BACK + closed

    raise ValueError(
        f"No trading day found in the {config.MAX_LOOKBACK_DAYS} days before "
        f"{today.isoformat()}. The market holiday table is wrong."
    )


def news_window(moment: datetime) -> NewsWindow:
    """Return the news window for a run starting at a given moment.

    Args:
        moment: When the run started. Must be timezone aware.

    Returns:
        The window start, its length in days, and the article limit. The limit
        scales with the window: Alpha Vantage sorts by LATEST, so a flat limit
        over a longer window would let the newest day crowd out the older ones
        the longer window exists to reach.

    Raises:
        ValueError: If the moment is naive, or if no trading day is found
            within MAX_LOOKBACK_DAYS.
    """
    days = lookback_days(moment)
    return NewsWindow(
        start=moment.astimezone(UTC) - timedelta(days=days),
        days=days,
        limit=config.NEWS_LIMIT_PER_DAY * days,
    )

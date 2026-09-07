"""Behaviour of the news window: how far back a run reaches for news.

Every test here is a pure function of a fixed datetime. Nothing opens a
socket, a file or the clock.
"""

from datetime import UTC, date, datetime

import pytest

import config

from src.api.market_calendar import is_trading_day, news_window


def test_an_ordinary_trading_day_looks_back_one_day() -> None:
    """Tuesday 2026-09-15. The market traded on Monday, so 24 hours is enough."""
    window = news_window(datetime(2026, 9, 15, 8, 0, tzinfo=UTC))

    assert window.days == 1
    assert window.start == datetime(2026, 9, 14, 8, 0, tzinfo=UTC)
    assert window.limit == config.NEWS_LIMIT_PER_DAY


def test_a_monday_reaches_back_to_friday() -> None:
    """The market was shut all weekend, so 24 hours would see only Sunday."""
    window = news_window(datetime(2026, 9, 14, 8, 0, tzinfo=UTC))

    assert window.days == 3
    assert window.start == datetime(2026, 9, 11, 8, 0, tzinfo=UTC)


def test_the_market_calendar_day_decides_the_window_not_the_utc_day() -> None:
    """01:00 UTC on a Tuesday is 21:00 the previous Monday in New York, and
    the market keeps its own calendar."""
    window = news_window(datetime(2026, 9, 8, 1, 0, tzinfo=UTC))

    assert window.days == 3


# The NYSE holiday calendar, which is not the federal one: Columbus Day and
# Veterans Day are federal holidays on which the market trades, and Good
# Friday is a market closure that is not federal. Listed across two years so
# every holiday appears at least once on a weekday of its own.
@pytest.mark.parametrize(
    ("day", "holiday"),
    [
        (date(2026, 1, 1), "New Year's Day"),
        (date(2026, 1, 19), "MLK Day"),
        (date(2026, 2, 16), "Presidents Day"),
        (date(2026, 4, 3), "Good Friday"),
        (date(2026, 5, 25), "Memorial Day"),
        (date(2026, 6, 19), "Juneteenth"),
        (date(2026, 9, 7), "Labor Day"),
        (date(2026, 11, 26), "Thanksgiving"),
        (date(2026, 12, 25), "Christmas"),
        (date(2027, 1, 1), "New Year's Day"),
        (date(2027, 1, 18), "MLK Day"),
        (date(2027, 2, 15), "Presidents Day"),
        (date(2027, 3, 26), "Good Friday"),
        (date(2027, 5, 31), "Memorial Day"),
        (date(2027, 9, 6), "Labor Day"),
        (date(2027, 11, 25), "Thanksgiving"),
        (date(2027, 7, 4), "Independence Day"),
    ],
)
def test_the_market_is_closed_on_a_holiday(day: date, holiday: str) -> None:
    assert is_trading_day(day) is False, f"{holiday} {day} should be closed"


@pytest.mark.parametrize(
    "day",
    [
        date(2026, 9, 15),  # an ordinary Tuesday
        date(2026, 11, 27),  # the half day after Thanksgiving, still a trading day
        date(2026, 12, 24),  # Christmas Eve, a Thursday, still a trading day
        date(2026, 1, 2),  # the day after New Year's Day
    ],
)
def test_the_market_trades_on_an_ordinary_weekday(day: date) -> None:
    assert is_trading_day(day) is True


@pytest.mark.parametrize(
    ("day", "why"),
    [
        (date(2026, 7, 3), "4 July 2026 is a Saturday, observed the Friday before"),
        (date(2027, 6, 18), "Juneteenth 2027 is a Saturday, observed the Friday"),
        (date(2027, 7, 5), "4 July 2027 is a Sunday, observed the Monday after"),
        (date(2027, 12, 24), "Christmas 2027 is a Saturday, observed the Friday"),
    ],
)
def test_a_fixed_date_holiday_on_a_weekend_closes_an_adjacent_weekday(
    day: date, why: str
) -> None:
    assert is_trading_day(day) is False, why


def test_new_years_day_on_a_saturday_leaves_new_years_eve_a_trading_day() -> None:
    """The NYSE exception: 31 December is the last trading day of the year and
    stays open, unlike every other holiday that falls on a Saturday."""
    assert date(2028, 1, 1).weekday() == 5
    assert is_trading_day(date(2027, 12, 31)) is True


@pytest.mark.parametrize(
    ("run_day", "expected", "why"),
    [
        (date(2026, 9, 8), 4, "Tuesday after Labor Day: Mon, Sun, Sat all closed"),
        (date(2026, 11, 27), 2, "Friday after Thanksgiving: Thursday closed"),
        (date(2026, 4, 6), 4, "Monday after Good Friday: Fri, Sat, Sun closed"),
        (date(2026, 7, 6), 4, "Monday after the observed 4 July"),
        (date(2026, 12, 28), 4, "Monday after Christmas on a Friday"),
        (date(2027, 12, 27), 4, "Monday after Christmas observed on the Friday"),
        (date(2028, 1, 3), 3, "Monday after a weekend, 31 December stayed open"),
        (date(2026, 1, 20), 4, "Tuesday after MLK Day"),
        (date(2026, 9, 12), 1, "a Saturday run, Friday traded"),
        (date(2026, 9, 13), 2, "a Sunday run, Saturday closed"),
    ],
)
def test_the_window_extends_by_every_consecutive_closed_day(
    run_day: date, expected: int, why: str
) -> None:
    moment = datetime(run_day.year, run_day.month, run_day.day, 12, 0, tzinfo=UTC)

    assert news_window(moment).days == expected, why


def test_the_article_limit_scales_with_the_window() -> None:
    """Alpha Vantage sorts by LATEST, so a flat limit over a three day window
    would let the weekend crowd out the Friday news the window exists for."""
    ordinary = news_window(datetime(2026, 9, 15, 8, 0, tzinfo=UTC))
    monday = news_window(datetime(2026, 9, 14, 8, 0, tzinfo=UTC))

    assert ordinary.limit == config.NEWS_LIMIT_PER_DAY
    assert monday.days == 3
    assert monday.limit == 3 * config.NEWS_LIMIT_PER_DAY


def test_a_naive_run_timestamp_is_rejected() -> None:
    """astimezone on a naive value silently assumes the machine's local zone,
    which would make the window depend on where the job happens to run."""
    with pytest.raises(ValueError) as excinfo:
        news_window(datetime(2026, 9, 15, 8, 0))

    message = str(excinfo.value)
    assert "moment" in message
    assert "2026-09-15T08:00:00" in message


def test_the_backward_scan_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreachable with a correct holiday table. Raising rather than clamping
    means a broken table cannot hide behind a plausible looking window."""
    monkeypatch.setattr(config, "MAX_LOOKBACK_DAYS", 0)

    with pytest.raises(ValueError) as excinfo:
        news_window(datetime(2026, 9, 15, 8, 0, tzinfo=UTC))

    message = str(excinfo.value)
    assert "2026-09-15" in message
    assert "holiday table" in message

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

EXCHANGE_TZ = "Asia/Kolkata"


@dataclass(frozen=True)
class TradingSession:
    open_time: time = time(9, 15)
    close_time: time = time(15, 30)
    freq: str = "1min"
    timezone: str = EXCHANGE_TZ


def load_holiday_dates(path: str | None) -> set[pd.Timestamp]:
    if path is None:
        return set()
    holidays = pd.read_csv(path)
    if holidays.empty:
        return set()
    date_col = "date" if "date" in holidays.columns else holidays.columns[0]
    return {pd.Timestamp(value).normalize() for value in holidays[date_col].dropna()}


def build_trading_session_grid(
    start: pd.Timestamp,
    end: pd.Timestamp,
    holidays: set[pd.Timestamp] | None = None,
    session: TradingSession | None = None,
) -> pd.DatetimeIndex:
    session = session or TradingSession()
    holidays = holidays or set()

    start = normalize_timestamp(start, session.timezone)
    end = normalize_timestamp(end, session.timezone)
    days = pd.date_range(start.normalize(), end.normalize(), freq="B", tz=session.timezone)

    grids: list[pd.DatetimeIndex] = []
    for day in days:
        if day.tz_localize(None).normalize() in holidays:
            continue
        open_ts = day.replace(
            hour=session.open_time.hour,
            minute=session.open_time.minute,
            second=0,
            microsecond=0,
        )
        close_ts = day.replace(
            hour=session.close_time.hour,
            minute=session.close_time.minute,
            second=0,
            microsecond=0,
        )
        grids.append(pd.date_range(open_ts, close_ts, freq=session.freq, tz=session.timezone))

    if not grids:
        return pd.DatetimeIndex([], tz=session.timezone, name="timestamp")
    return grids[0].append(grids[1:]).rename("timestamp")


def normalize_timestamp(value: pd.Timestamp | str, timezone: str = EXCHANGE_TZ) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(timezone)
    return ts.tz_convert(timezone)


def normalize_timestamp_series(series: pd.Series, timezone: str = EXCHANGE_TZ) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(timezone)
    return parsed.dt.tz_convert(timezone)

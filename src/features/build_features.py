from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.calendar import TradingSession
from src.data.schemas import FEATURE_COLUMNS


def _returns(close: pd.Series, periods: int) -> pd.Series:
    return close.pct_change(periods, fill_method=None)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(2, window // 3)).mean()
    std = series.rolling(window, min_periods=max(2, window // 3)).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def _time_of_day_zscore(df: pd.DataFrame, value_col: str) -> pd.Series:
    minute_key = df["timestamp"].dt.strftime("%H:%M")
    grouped = df.groupby(minute_key, sort=False)[value_col]
    mean = grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0))
    return (df[value_col] - mean) / std.replace(0, np.nan)


def compute_features(df: pd.DataFrame, session: TradingSession | None = None) -> pd.DataFrame:
    session = session or TradingSession()
    out = df.copy()

    out["spot_ret_1m"] = _returns(out["spot_close"], 1)
    out["spot_ret_5m"] = _returns(out["spot_close"], 5)
    out["spot_ret_15m"] = _returns(out["spot_close"], 15)
    out["fut_ret_1m"] = _returns(out["fut_close"], 1)
    out["fut_ret_5m"] = _returns(out["fut_close"], 5)
    out["fut_ret_15m"] = _returns(out["fut_close"], 15)

    out["spot_range"] = (out["spot_high"] - out["spot_low"]) / out["spot_close"]
    out["fut_range"] = (out["fut_high"] - out["fut_low"]) / out["fut_close"]
    out["basis"] = out["fut_close"] - out["spot_close"]
    out["basis_pct"] = out["basis"] / out["spot_close"]
    out["basis_change"] = out["basis"].diff()
    out["basis_zscore_60m"] = _rolling_zscore(out["basis"], 60)
    out["fut_volume_zscore_by_time_of_day"] = _time_of_day_zscore(out, "fut_volume")
    out["oi_change_1m"] = out["fut_open_interest"].diff()
    out["oi_change_15m"] = out["fut_open_interest"].diff(15)
    out["realized_vol_15m"] = out["fut_ret_1m"].rolling(15, min_periods=5).std(ddof=0)
    out["realized_vol_30m"] = out["fut_ret_1m"].rolling(30, min_periods=10).std(ddof=0)

    open_minutes = session.open_time.hour * 60 + session.open_time.minute
    close_minutes = session.close_time.hour * 60 + session.close_time.minute
    minute_of_day = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    out["minutes_from_open"] = minute_of_day - open_minutes
    out["minutes_to_close"] = close_minutes - minute_of_day
    out["day_of_week"] = out["timestamp"].dt.dayofweek

    expiry_ts = pd.to_datetime(out["fut_expiry"])
    ts_date = out["timestamp"].dt.tz_localize(None).dt.normalize()
    out["days_to_expiry"] = (expiry_ts - ts_date).dt.days
    out["is_expiry_day"] = (out["days_to_expiry"] == 0).astype(float)
    out["is_expiry_week"] = (out["days_to_expiry"].between(0, 6)).astype(float)

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out

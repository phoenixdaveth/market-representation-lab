from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schemas import LABEL_COLUMNS


def _future_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def _future_realized_vol(close: pd.Series, horizon: int) -> pd.Series:
    returns = close.pct_change(fill_method=None).shift(-1)
    return returns.rolling(horizon, min_periods=horizon).std(ddof=0).shift(-(horizon - 1))


def _future_window_stat(series: pd.Series, horizon: int, func: str) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.rolling(horizon, min_periods=horizon).agg(func).shift(-(horizon - 1))


def compute_future_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["fut_close"]
    out["future_ret_15m"] = _future_return(close, 15)
    out["future_ret_30m"] = _future_return(close, 30)
    out["future_realized_vol_30m"] = _future_realized_vol(close, 30)

    future_high = _future_window_stat(out["fut_high"], 30, "max")
    future_low = _future_window_stat(out["fut_low"], 30, "min")
    out["future_range_30m"] = (future_high - future_low) / close
    out["future_basis_change_30m"] = out["basis"].shift(-30) - out["basis"]
    out["max_adverse_excursion_30m"] = (future_low / close) - 1.0
    out["max_favorable_excursion_30m"] = (future_high / close) - 1.0

    out[LABEL_COLUMNS] = out[LABEL_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out

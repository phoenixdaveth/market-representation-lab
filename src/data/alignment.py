from __future__ import annotations

import pandas as pd

from src.data.calendar import TradingSession, build_trading_session_grid


def align_spot_and_futures(
    spot: pd.DataFrame,
    futures_continuous: pd.DataFrame,
    holidays: set[pd.Timestamp] | None = None,
    session: TradingSession | None = None,
) -> pd.DataFrame:
    start = max(spot["timestamp"].min(), futures_continuous["timestamp"].min())
    end = min(spot["timestamp"].max(), futures_continuous["timestamp"].max())
    grid = build_trading_session_grid(start, end, holidays=holidays, session=session)
    base = pd.DataFrame({"timestamp": grid})

    spot_cols = {
        "open": "spot_open",
        "high": "spot_high",
        "low": "spot_low",
        "close": "spot_close",
    }
    fut_cols = {
        "contract": "fut_contract",
        "expiry": "fut_expiry",
        "open": "fut_open",
        "high": "fut_high",
        "low": "fut_low",
        "close": "fut_close",
        "volume": "fut_volume",
        "open_interest": "fut_open_interest",
    }

    spot_aligned = spot.rename(columns=spot_cols)[["timestamp", *spot_cols.values()]]
    fut_aligned = futures_continuous.rename(columns=fut_cols)[["timestamp", *fut_cols.values()]]
    merged = base.merge(spot_aligned, on="timestamp", how="left").merge(
        fut_aligned, on="timestamp", how="left"
    )
    price_cols = [
        "spot_open",
        "spot_high",
        "spot_low",
        "spot_close",
        "fut_open",
        "fut_high",
        "fut_low",
        "fut_close",
        "fut_volume",
        "fut_open_interest",
    ]
    return merged.dropna(subset=price_cols).reset_index(drop=True)

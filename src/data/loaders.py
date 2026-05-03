from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.calendar import EXCHANGE_TZ, normalize_timestamp_series
from src.data.schemas import FUTURES_COLUMNS, SPOT_COLUMNS


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_spot_data(path: str | Path, timezone: str = EXCHANGE_TZ) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, SPOT_COLUMNS, "spot data")
    df = df.copy()
    df["timestamp"] = normalize_timestamp_series(df["timestamp"], timezone)
    numeric_cols = ["open", "high", "low", "close"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("timestamp").drop_duplicates("timestamp")


def load_futures_data(path: str | Path, timezone: str = EXCHANGE_TZ) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, FUTURES_COLUMNS, "futures data")
    df = df.copy()
    df["timestamp"] = normalize_timestamp_series(df["timestamp"], timezone)
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    numeric_cols = ["open", "high", "low", "close", "volume", "open_interest"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values(["timestamp", "expiry", "contract"])


def load_expiry_calendar(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "expiry" if "expiry" in df.columns else df.columns[0]
    out = df.rename(columns={date_col: "expiry"}).copy()
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.date
    return out.dropna(subset=["expiry"]).drop_duplicates("expiry").sort_values("expiry")

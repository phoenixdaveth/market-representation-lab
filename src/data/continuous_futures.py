from __future__ import annotations

import pandas as pd


def build_near_month_continuous(futures: pd.DataFrame) -> pd.DataFrame:
    """Select the nearest non-expired futures row for each timestamp.

    The rule is deterministic and deliberately simple for research repeatability:
    at each timestamp, choose the contract with the smallest expiry date that is
    on or after the timestamp's calendar date. If none are available, use the
    closest expiry still present in the data for that timestamp.
    """

    required = {"timestamp", "contract", "expiry", "open", "high", "low", "close"}
    missing = required - set(futures.columns)
    if missing:
        raise ValueError(f"futures data missing columns: {sorted(missing)}")

    df = futures.copy()
    df["timestamp_date"] = df["timestamp"].dt.date
    df["is_unexpired"] = df["expiry"] >= df["timestamp_date"]
    df["expiry_distance"] = (
        pd.to_datetime(df["expiry"]) - pd.to_datetime(df["timestamp_date"])
    ).dt.days.abs()
    df = df.sort_values(
        ["timestamp", "is_unexpired", "expiry_distance", "expiry", "contract"],
        ascending=[True, False, True, True, True],
    )
    selected = df.groupby("timestamp", as_index=False).head(1)
    selected = selected.drop(columns=["timestamp_date", "is_unexpired", "expiry_distance"])
    return selected.sort_values("timestamp").reset_index(drop=True)

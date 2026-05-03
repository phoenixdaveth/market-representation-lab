from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.alignment import align_spot_and_futures
from src.data.calendar import EXCHANGE_TZ, TradingSession, load_holiday_dates
from src.data.continuous_futures import build_near_month_continuous
from src.data.loaders import load_futures_data, load_spot_data
from src.data.schemas import FEATURE_COLUMNS, LABEL_COLUMNS
from src.data.windowing import build_rolling_windows
from src.features.build_features import compute_features
from src.features.labels import compute_future_labels


@dataclass(frozen=True)
class DatasetPaths:
    spot: Path
    futures: Path
    expiry_calendar: Path | None = None
    holiday_calendar: Path | None = None


def prepare_frame(
    paths: DatasetPaths,
    timezone: str = EXCHANGE_TZ,
    session: TradingSession | None = None,
) -> pd.DataFrame:
    spot = load_spot_data(paths.spot, timezone=timezone)
    futures = load_futures_data(paths.futures, timezone=timezone)
    holidays = load_holiday_dates(str(paths.holiday_calendar)) if paths.holiday_calendar else set()
    continuous = build_near_month_continuous(futures)
    aligned = align_spot_and_futures(spot, continuous, holidays=holidays, session=session)
    featured = compute_features(aligned, session=session)
    return compute_future_labels(featured)


def prepare_windows(
    paths: DatasetPaths,
    seq_len: int,
    timezone: str = EXCHANGE_TZ,
    session: TradingSession | None = None,
    feature_columns: list[str] | None = None,
    label_columns: list[str] | None = None,
):
    frame = prepare_frame(paths, timezone=timezone, session=session)
    return build_rolling_windows(
        frame,
        feature_columns or FEATURE_COLUMNS,
        label_columns or LABEL_COLUMNS,
        seq_len=seq_len,
    )

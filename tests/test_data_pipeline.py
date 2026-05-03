import numpy as np
import pandas as pd

from src.data.calendar import TradingSession, build_trading_session_grid
from src.data.schemas import FEATURE_COLUMNS, LABEL_COLUMNS
from src.data.windowing import build_rolling_windows
from src.features.build_features import compute_features
from src.features.labels import compute_future_labels


def test_feature_label_window_pipeline_with_synthetic_data():
    session = TradingSession()
    grid = build_trading_session_grid(
        pd.Timestamp("2025-01-02 09:15", tz="Asia/Kolkata"),
        pd.Timestamp("2025-01-03 15:30", tz="Asia/Kolkata"),
        session=session,
    )
    n = len(grid)
    idx = np.arange(n, dtype=float)
    spot_close = 24_000 + idx * 0.1 + np.sin(idx / 20.0)
    fut_close = spot_close + 12.0 + np.cos(idx / 30.0)

    frame = pd.DataFrame(
        {
            "timestamp": grid,
            "spot_open": spot_close - 0.5,
            "spot_high": spot_close + 1.0,
            "spot_low": spot_close - 1.0,
            "spot_close": spot_close,
            "fut_contract": "NIFTY25JANFUT",
            "fut_expiry": pd.to_datetime("2025-01-30").date(),
            "fut_open": fut_close - 0.5,
            "fut_high": fut_close + 1.2,
            "fut_low": fut_close - 1.2,
            "fut_close": fut_close,
            "fut_volume": 1000 + (idx % 50) + (idx // 376) * 10,
            "fut_open_interest": 500_000 + idx * 2,
        }
    )

    featured = compute_features(frame, session=session)
    labelled = compute_future_labels(featured)
    x, y, metadata = build_rolling_windows(
        labelled,
        feature_columns=FEATURE_COLUMNS,
        label_columns=LABEL_COLUMNS,
        seq_len=32,
    )

    assert x.ndim == 3
    assert y.ndim == 2
    assert x.shape[1:] == (32, len(FEATURE_COLUMNS))
    assert y.shape[1] == len(LABEL_COLUMNS)
    assert len(metadata) == len(x) == len(y)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import trading_metrics


def pure_model_positions(pred_return_30m: pd.Series, threshold: float) -> pd.Series:
    values = pred_return_30m.to_numpy()
    positions = np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))
    return pd.Series(positions, index=pred_return_30m.index, name="position")


def backtest_positions(
    metadata: pd.DataFrame,
    positions: pd.Series,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = metadata.copy()
    frame["position"] = positions.to_numpy()
    gross = frame["position"] * frame["future_ret_30m"]
    turnover = frame["position"].diff().abs().fillna(frame["position"].abs())
    cost = turnover * (cost_bps / 10_000.0)
    frame["strategy_return"] = gross - cost
    metrics = trading_metrics(frame["strategy_return"], frame["position"])
    metrics["estimated_cost_impact"] = float(cost.sum())
    return frame, metrics

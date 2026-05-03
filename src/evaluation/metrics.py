from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prediction_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in y_true.columns:
        actual = y_true[column].to_numpy()
        pred = y_pred[column].to_numpy()
        rows.append(
            {
                "target": column,
                "mae": mean_absolute_error(actual, pred),
                "rmse": mean_squared_error(actual, pred, squared=False),
                "r2": r2_score(actual, pred),
                "corr": np.corrcoef(actual, pred)[0, 1] if len(actual) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max.replace(0, np.nan) - 1.0
    return float(drawdown.min())


def trading_metrics(
    returns: pd.Series,
    positions: pd.Series,
    periods_per_year: int = 252 * 375,
) -> dict[str, float]:
    returns = returns.fillna(0.0)
    positions = positions.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    std = returns.std(ddof=0)
    trades = returns[positions != 0]
    return {
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "sharpe": float((returns.mean() / std) * np.sqrt(periods_per_year)) if std > 0 else 0.0,
        "max_drawdown": max_drawdown(equity) if len(equity) else 0.0,
        "hit_rate": float((trades > 0).mean()) if len(trades) else 0.0,
        "average_trade": float(trades.mean()) if len(trades) else 0.0,
        "turnover": float(positions.diff().abs().fillna(positions.abs()).sum()),
    }

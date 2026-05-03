from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.schemas import PREDICTION_TARGETS
from src.evaluation.clustering import (
    apply_cluster_rules,
    cluster_summary,
    fit_kmeans,
    identify_directional_clusters,
)
from src.evaluation.hidden_states import extract_hidden_states, predict_outputs
from src.evaluation.metrics import prediction_metrics
from src.evaluation.trading import backtest_positions, pure_model_positions


def evaluate_predictions(
    model: torch.nn.Module,
    x: np.ndarray,
    metadata: pd.DataFrame,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_preds = predict_outputs(model, x, batch_size=batch_size)
    pred_frame = pd.DataFrame(
        {
            target: raw_preds[pred_key]
            for pred_key, target in PREDICTION_TARGETS.items()
            if pred_key in raw_preds
        }
    )
    metric_frame = prediction_metrics(metadata[pred_frame.columns], pred_frame)
    return pred_frame, metric_frame


def evaluate_hidden_state_clusters(
    model: torch.nn.Module,
    x: np.ndarray,
    metadata: pd.DataFrame,
    n_clusters: int = 8,
    seed: int = 7,
    batch_size: int = 256,
) -> tuple[np.ndarray, pd.DataFrame]:
    hidden = extract_hidden_states(model, x, batch_size=batch_size)
    kmeans = fit_kmeans(hidden, n_clusters=n_clusters, seed=seed)
    clusters = kmeans.predict(hidden)
    return clusters, cluster_summary(metadata, clusters)


def evaluate_pure_model_signal(
    metadata: pd.DataFrame,
    pred_frame: pd.DataFrame,
    threshold: float,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    positions = pure_model_positions(pred_frame["future_ret_30m"], threshold=threshold)
    return backtest_positions(metadata, positions, cost_bps=cost_bps)


def evaluate_cluster_signal(
    train_metadata: pd.DataFrame,
    train_clusters: np.ndarray,
    test_metadata: pd.DataFrame,
    test_clusters: np.ndarray,
    min_count: int = 50,
    min_abs_return: float = 0.0,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float], dict[int, int]]:
    rules = identify_directional_clusters(
        train_metadata,
        train_clusters,
        min_count=min_count,
        min_abs_return=min_abs_return,
    )
    positions = apply_cluster_rules(test_clusters, rules)
    backtest, metrics = backtest_positions(test_metadata, positions, cost_bps=cost_bps)
    return backtest, metrics, rules

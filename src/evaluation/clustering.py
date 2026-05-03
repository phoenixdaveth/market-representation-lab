from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def fit_kmeans(hidden_states: np.ndarray, n_clusters: int = 8, seed: int = 7) -> KMeans:
    return KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed).fit(hidden_states)


def cluster_summary(metadata: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    frame = metadata.copy()
    frame["cluster"] = clusters
    grouped = frame.groupby("cluster", observed=True)
    return grouped.agg(
        count=("cluster", "size"),
        average_future_15m_return=("future_ret_15m", "mean"),
        average_future_30m_return=("future_ret_30m", "mean"),
        average_future_30m_range=("future_range_30m", "mean"),
        average_future_30m_volatility=("future_realized_vol_30m", "mean"),
        win_rate=("future_ret_30m", lambda x: float((x > 0).mean())),
        max_adverse_excursion=("max_adverse_excursion_30m", "mean"),
        max_favorable_excursion=("max_favorable_excursion_30m", "mean"),
    ).reset_index()


def identify_directional_clusters(
    train_metadata: pd.DataFrame,
    train_clusters: np.ndarray,
    min_count: int = 50,
    min_abs_return: float = 0.0,
) -> dict[int, int]:
    summary = cluster_summary(train_metadata, train_clusters)
    rules: dict[int, int] = {}
    for row in summary.itertuples(index=False):
        avg_return = row.average_future_30m_return
        if row.count < min_count or abs(avg_return) < min_abs_return:
            rules[int(row.cluster)] = 0
        else:
            rules[int(row.cluster)] = 1 if avg_return > 0 else -1
    return rules


def apply_cluster_rules(clusters: np.ndarray, rules: dict[int, int]) -> pd.Series:
    return pd.Series([rules.get(int(cluster), 0) for cluster in clusters], name="position")

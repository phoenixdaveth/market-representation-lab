from __future__ import annotations

import numpy as np
import pandas as pd


def build_rolling_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if seq_len < 1:
        raise ValueError("seq_len must be positive")

    clean = df.dropna(subset=feature_columns + label_columns).reset_index(drop=True)
    features = clean[feature_columns].to_numpy(dtype=np.float32)
    labels = clean[label_columns].to_numpy(dtype=np.float32)
    if len(clean) < seq_len:
        empty_x = np.empty((0, seq_len, len(feature_columns)), dtype=np.float32)
        empty_y = np.empty((0, len(label_columns)), dtype=np.float32)
        return empty_x, empty_y, clean.iloc[0:0].copy()

    windows = []
    targets = []
    metadata_rows = []
    for end_idx in range(seq_len - 1, len(clean)):
        start_idx = end_idx - seq_len + 1
        windows.append(features[start_idx : end_idx + 1])
        targets.append(labels[end_idx])
        metadata_rows.append(clean.iloc[end_idx])

    x = np.stack(windows).astype(np.float32)
    y = np.stack(targets).astype(np.float32)
    metadata = pd.DataFrame(metadata_rows).reset_index(drop=True)
    return x, y, metadata

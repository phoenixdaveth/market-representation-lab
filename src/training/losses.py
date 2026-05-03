from __future__ import annotations

import torch
from torch import nn

from src.data.schemas import LABEL_COLUMNS


DEFAULT_LOSS_WEIGHTS = {
    "pred_return_15m": 1.0,
    "pred_return_30m": 1.0,
    "pred_vol_30m": 0.5,
    "pred_range_30m": 0.5,
    "pred_basis_change_30m": 0.25,
}

TARGET_INDEX = {
    "pred_return_15m": LABEL_COLUMNS.index("future_ret_15m"),
    "pred_return_30m": LABEL_COLUMNS.index("future_ret_30m"),
    "pred_vol_30m": LABEL_COLUMNS.index("future_realized_vol_30m"),
    "pred_range_30m": LABEL_COLUMNS.index("future_range_30m"),
    "pred_basis_change_30m": LABEL_COLUMNS.index("future_basis_change_30m"),
}


def multi_task_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    weights: dict[str, float] | None = None,
    criterion: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or DEFAULT_LOSS_WEIGHTS
    criterion = criterion or nn.SmoothL1Loss()

    total = targets.new_tensor(0.0)
    parts: dict[str, torch.Tensor] = {}
    for output_key, weight in weights.items():
        target_idx = TARGET_INDEX[output_key]
        loss = criterion(outputs[output_key], targets[:, target_idx])
        parts[output_key] = loss.detach()
        total = total + float(weight) * loss
    parts["total"] = total.detach()
    return total, parts

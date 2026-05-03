from __future__ import annotations

import torch
from torch import nn


class MultiTaskHeads(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.pred_return_15m = nn.Linear(d_model, 1)
        self.pred_return_30m = nn.Linear(d_model, 1)
        self.pred_vol_30m = nn.Linear(d_model, 1)
        self.pred_range_30m = nn.Linear(d_model, 1)
        self.pred_basis_change_30m = nn.Linear(d_model, 1)

    def forward(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.shared(hidden_state)
        return {
            "hidden_state": hidden_state,
            "pred_return_15m": self.pred_return_15m(z).squeeze(-1),
            "pred_return_30m": self.pred_return_30m(z).squeeze(-1),
            "pred_vol_30m": self.pred_vol_30m(z).squeeze(-1),
            "pred_range_30m": self.pred_range_30m(z).squeeze(-1),
            "pred_basis_change_30m": self.pred_basis_change_30m(z).squeeze(-1),
        }

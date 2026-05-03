from __future__ import annotations

import torch
from torch import nn


class VariableSelectionNetwork(nn.Module):
    """Small feature-wise selector for numeric market tensors.

    Input shape is [batch, seq_len, num_features]. Each scalar feature is mapped
    into d_model, scored, and combined with a softmax over features.
    """

    def __init__(self, num_features: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.feature_embeddings = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(num_features)]
        )
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, seq_len, num_features], got {tuple(x.shape)}")
        if x.shape[-1] != self.num_features:
            raise ValueError(f"expected {self.num_features} features, got {x.shape[-1]}")

        embedded = []
        scores = []
        for idx, projection in enumerate(self.feature_embeddings):
            feature = x[..., idx : idx + 1]
            z = projection(feature)
            embedded.append(z)
            scores.append(self.score(z))

        feature_tensor = torch.stack(embedded, dim=-2)
        score_tensor = torch.cat(scores, dim=-1)
        weights = torch.softmax(score_tensor, dim=-1)
        selected = (feature_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return self.output_norm(selected), weights

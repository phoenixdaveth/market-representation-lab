from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class Mamba2Block(nn.Module):
    """Compact numeric sequence block inspired by Mamba-style mixing.

    This is intentionally implemented from scratch for market tensors. It uses
    local depthwise convolution, gated projections, and residual state mixing;
    it does not load or imitate pretrained language-model checkpoints.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = d_model * expansion
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, hidden * 2)
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden,
        )
        self.state_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.out_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = value.transpose(1, 2)
        mixed = self.depthwise(value)[..., : x.shape[1]].transpose(1, 2)
        mixed = torch.nn.functional.silu(mixed)
        gated = mixed * torch.sigmoid(gate)
        state = gated * self.state_gate(gated)
        return residual + self.dropout(self.out_proj(state))


class Mamba3Block(nn.Module):
    """Experimental third-generation numeric sequence block.

    The Mamba3 variant adds multi-scale depthwise filters and a learned residual
    blend. It is small, configurable, and suitable for representation-learning
    experiments, not a pretrained text model.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 2,
        kernel_sizes: tuple[int, ...] = (3, 7),
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = d_model * expansion
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, hidden * 3)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden,
                    hidden,
                    kernel_size=kernel,
                    padding=kernel - 1,
                    groups=hidden,
                )
                for kernel in kernel_sizes
            ]
        )
        self.scale_weights = nn.Parameter(torch.zeros(len(kernel_sizes)))
        self.state_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        seq_len = x.shape[1]
        x = self.norm(x)
        value, gate, skip = self.in_proj(x).chunk(3, dim=-1)
        conv_in = value.transpose(1, 2)
        weights = torch.softmax(self.scale_weights, dim=0)
        mixed = 0.0
        for weight, conv in zip(weights, self.convs):
            mixed = mixed + weight * conv(conv_in)[..., :seq_len].transpose(1, 2)
        mixed = torch.nn.functional.silu(mixed + skip)
        state = torch.tanh(self.state_proj(mixed)) * torch.sigmoid(gate)
        update = self.dropout(self.out_proj(state))
        return residual + torch.sigmoid(self.residual_scale) * update

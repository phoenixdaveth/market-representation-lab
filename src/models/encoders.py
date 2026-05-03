from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.heads import MultiTaskHeads
from src.models.mamba_blocks import Mamba2Block, Mamba3Block
from src.models.vsn import VariableSelectionNetwork


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        head_dim = d_model * (2 if bidirectional else 1)
        self.heads = MultiTaskHeads(head_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.input_projection(x)
        output, _ = self.lstm(z)
        hidden = output[:, -1]
        return self.heads(hidden)


class VSNLSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.vsn = VariableSelectionNetwork(num_features, d_model, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        head_dim = d_model * (2 if bidirectional else 1)
        self.heads = MultiTaskHeads(head_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z, weights = self.vsn(x)
        output, _ = self.lstm(z)
        outputs = self.heads(output[:, -1])
        outputs["vsn_weights"] = weights
        return outputs


class _MambaEncoderBase(nn.Module):
    block_cls: type[nn.Module]

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        num_layers: int = 2,
        expansion: int = 2,
        dropout: float = 0.0,
        use_vsn: bool = False,
        **block_kwargs: Any,
    ):
        super().__init__()
        self.use_vsn = use_vsn
        if use_vsn:
            self.input_adapter = VariableSelectionNetwork(num_features, d_model, dropout=dropout)
        else:
            self.input_adapter = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList(
            [
                self.block_cls(
                    d_model=d_model,
                    expansion=expansion,
                    dropout=dropout,
                    **block_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = MultiTaskHeads(d_model, dropout=dropout)

    def _adapt_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_vsn:
            return self.input_adapter(x)
        return self.input_adapter(x), None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z, weights = self._adapt_input(x)
        for block in self.blocks:
            z = block(z)
        hidden = self.final_norm(z[:, -1])
        outputs = self.heads(hidden)
        if weights is not None:
            outputs["vsn_weights"] = weights
        return outputs


class Mamba2Encoder(_MambaEncoderBase):
    block_cls = Mamba2Block


class VSNMamba2Encoder(Mamba2Encoder):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, use_vsn=True, **kwargs)


class Mamba3Encoder(_MambaEncoderBase):
    block_cls = Mamba3Block


class VSNMamba3Encoder(Mamba3Encoder):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, use_vsn=True, **kwargs)


MODEL_REGISTRY = {
    "LSTMEncoder": LSTMEncoder,
    "VSNLSTMEncoder": VSNLSTMEncoder,
    "Mamba2Encoder": Mamba2Encoder,
    "VSNMamba2Encoder": VSNMamba2Encoder,
    "Mamba3Encoder": Mamba3Encoder,
    "VSNMamba3Encoder": VSNMamba3Encoder,
}


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = config.get("model", config)
    name = model_config["name"]
    kwargs = {key: value for key, value in model_config.items() if key != "name"}
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; choose one of {sorted(MODEL_REGISTRY)}") from exc
    return model_cls(**kwargs)

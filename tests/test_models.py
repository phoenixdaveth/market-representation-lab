import pytest
import torch

from src.models import (
    LSTMEncoder,
    Mamba2Encoder,
    Mamba3Encoder,
    VSNLSTMEncoder,
    VSNMamba2Encoder,
    VSNMamba3Encoder,
)


@pytest.mark.parametrize(
    "model_cls,kwargs,has_vsn",
    [
        (LSTMEncoder, {}, False),
        (VSNLSTMEncoder, {}, True),
        (Mamba2Encoder, {"kernel_size": 5}, False),
        (VSNMamba2Encoder, {"kernel_size": 5}, True),
        (Mamba3Encoder, {"kernel_sizes": (3, 5)}, False),
        (VSNMamba3Encoder, {"kernel_sizes": (3, 5)}, True),
    ],
)
def test_encoder_forward_shapes(model_cls, kwargs, has_vsn):
    batch, seq_len, num_features, d_model = 4, 32, 23, 16
    x = torch.randn(batch, seq_len, num_features)
    model = model_cls(
        num_features=num_features,
        d_model=d_model,
        num_layers=1,
        dropout=0.0,
        **kwargs,
    )

    outputs = model(x)

    assert outputs["hidden_state"].shape == (batch, d_model)
    for key in [
        "pred_return_15m",
        "pred_return_30m",
        "pred_vol_30m",
        "pred_range_30m",
        "pred_basis_change_30m",
    ]:
        assert outputs[key].shape == (batch,)
        assert torch.isfinite(outputs[key]).all()

    if has_vsn:
        assert outputs["vsn_weights"].shape == (batch, seq_len, num_features)
        assert torch.allclose(outputs["vsn_weights"].sum(dim=-1), torch.ones(batch, seq_len))
    else:
        assert "vsn_weights" not in outputs

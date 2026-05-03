from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def extract_hidden_states(
    model: torch.nn.Module,
    x: np.ndarray | torch.Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    tensor = torch.as_tensor(x, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size)
    states = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            outputs = model(xb)
            states.append(outputs["hidden_state"].detach().cpu().numpy())
    return np.concatenate(states, axis=0) if states else np.empty((0, 0), dtype=np.float32)


def predict_outputs(
    model: torch.nn.Module,
    x: np.ndarray | torch.Tensor,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    tensor = torch.as_tensor(x, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size)
    collected: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            outputs = model(xb)
            for key, value in outputs.items():
                if key in {"hidden_state", "vsn_weights"}:
                    continue
                collected.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}

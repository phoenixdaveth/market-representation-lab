from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.data.pipeline import DatasetPaths, prepare_windows
from src.models import build_model
from src.training.dataset import SequenceDataset
from src.training.losses import multi_task_loss
from src.utils.config import load_yaml
from src.utils.seed import seed_everything


def train_from_config(config_path: str | Path) -> None:
    cfg = load_yaml(config_path)
    seed_everything(cfg.get("seed", 7))

    data_cfg = cfg["data"]
    paths = DatasetPaths(
        spot=Path(data_cfg["spot_path"]),
        futures=Path(data_cfg["futures_path"]),
        expiry_calendar=Path(data_cfg["expiry_calendar_path"])
        if data_cfg.get("expiry_calendar_path")
        else None,
        holiday_calendar=Path(data_cfg["holiday_calendar_path"])
        if data_cfg.get("holiday_calendar_path")
        else None,
    )
    x, y, _ = prepare_windows(paths, seq_len=data_cfg["seq_len"])
    dataset = SequenceDataset(x, y)
    val_size = max(1, int(len(dataset) * cfg["training"].get("val_fraction", 0.2)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 7)),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])

    model = build_model(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)
            loss, _ = multi_task_loss(outputs, yb, cfg["training"].get("loss_weights"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"].get("max_grad_norm", 1.0),
            )
            optimizer.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                outputs = model(xb)
                loss, _ = multi_task_loss(outputs, yb, cfg["training"].get("loss_weights"))
                val_loss += loss.item() * len(xb)

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss / max(1, train_size):.6f} "
            f"val_loss={val_loss / max(1, val_size):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()

"""Training loop for neural postprocessor.

Использование:
    python scripts/postproc/train_neural_postproc.py \\
        --train-parquet data/postproc/corpus_train.parquet \\
        --val-parquet   data/postproc/corpus_val.parquet \\
        --out-dir       experiments/neural_postproc_v1 \\
        --epochs 30 --batch-size 4096

См. docs/postprocessing_rfc.md.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import DEFAULT_FEATURES, StationCorpusDataset, build_balanced_sampler
from .losses import compute_total_loss
from .models import MultiTaskResidualMLP


@dataclass
class TrainConfig:
    train_parquet: str
    val_parquet: str
    out_dir: str
    feature_cols: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    hidden: List[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.1
    probabilistic: bool = False
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    wind_alpha: float = 0.5
    w_t2m: float = 1.0
    w_wind: float = 1.0
    balanced_sampling: bool = True
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 1.0
    seed: int = 42


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_pass(
    model: MultiTaskResidualMLP,
    loader: DataLoader,
    cfg: TrainConfig,
    optim: Optional[torch.optim.Optimizer],
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "loss_t2m": 0.0, "loss_wind": 0.0, "n": 0}
    # extra metrics
    sq_err = {"t2m": 0.0, "u10": 0.0, "v10": 0.0}

    for batch in loader:
        features = batch["features"].to(cfg.device, non_blocking=True)
        gnn = {
            "t2m": batch["gnn_t2m"].to(cfg.device),
            "u10": batch["gnn_u10"].to(cfg.device),
            "v10": batch["gnn_v10"].to(cfg.device),
        }
        targets = {
            "t2m": batch["t2m"].to(cfg.device),
            "u10": batch["u10"].to(cfg.device),
            "v10": batch["v10"].to(cfg.device),
        }

        if train:
            optim.zero_grad(set_to_none=True)

        out = model(features, gnn_targets=gnn)
        losses = compute_total_loss(
            out,
            targets,
            probabilistic=cfg.probabilistic,
            w_t2m=cfg.w_t2m,
            w_wind=cfg.w_wind,
            huber_delta=cfg.huber_delta,
            wind_alpha=cfg.wind_alpha,
        )

        if train:
            losses["loss"].backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

        bs = features.size(0)
        totals["loss"] += losses["loss"].item() * bs
        totals["loss_t2m"] += losses["loss_t2m"].item() * bs
        totals["loss_wind"] += losses["loss_wind"].item() * bs
        totals["n"] += bs

        with torch.no_grad():
            if cfg.probabilistic:
                t_pred = out["t2m_mu"]
                u_pred, v_pred = out["wind_mu"][:, 0], out["wind_mu"][:, 1]
            else:
                t_pred = out["t2m"]
                u_pred, v_pred = out["u10"], out["v10"]
            sq_err["t2m"] += ((t_pred - targets["t2m"]) ** 2).sum().item()
            sq_err["u10"] += ((u_pred - targets["u10"]) ** 2).sum().item()
            sq_err["v10"] += ((v_pred - targets["v10"]) ** 2).sum().item()

    n = max(totals["n"], 1)
    metrics = {
        "loss": totals["loss"] / n,
        "loss_t2m": totals["loss_t2m"] / n,
        "loss_wind": totals["loss_wind"] / n,
        "rmse_t2m": float(np.sqrt(sq_err["t2m"] / n)),
        "rmse_u10": float(np.sqrt(sq_err["u10"] / n)),
        "rmse_v10": float(np.sqrt(sq_err["v10"] / n)),
        "vec_rmse_wind": float(np.sqrt((sq_err["u10"] + sq_err["v10"]) / n)),
    }
    return metrics


def train(cfg: TrainConfig) -> Dict[str, float]:
    _seed_all(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = StationCorpusDataset(cfg.train_parquet, feature_cols=cfg.feature_cols)
    val_ds = StationCorpusDataset(
        cfg.val_parquet,
        feature_cols=cfg.feature_cols,
        scalers=train_ds.export_scalers(),
    )
    train_ds.save_scalers(out_dir / "scalers.json")

    sampler = build_balanced_sampler(train_ds) if cfg.balanced_sampling else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    model = MultiTaskResidualMLP(
        feature_dim=len(cfg.feature_cols),
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        probabilistic=cfg.probabilistic,
    ).to(cfg.device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    history: List[Dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_m = _epoch_pass(model, train_loader, cfg, optim, train=True)
        val_m = _epoch_pass(model, val_loader, cfg, optim=None, train=False)
        sched.step()

        row = {
            "epoch": epoch,
            "lr": optim.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        val_score = val_m["rmse_t2m"] + val_m["vec_rmse_wind"]
        print(
            f"[ep {epoch:03d}] "
            f"train loss={train_m['loss']:.4f} "
            f"val loss={val_m['loss']:.4f} "
            f"val rmse_t2m={val_m['rmse_t2m']:.3f}°C "
            f"val vec_rmse_wind={val_m['vec_rmse_wind']:.3f}m/s",
            flush=True,
        )

        if val_score < best_val:
            best_val = val_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "scalers": train_ds.export_scalers(),
                    "feature_cols": cfg.feature_cols,
                    "epoch": epoch,
                    "val_metrics": val_m,
                },
                out_dir / "best_model.pth",
            )

    return {"best_val_combined": best_val, "epochs_run": cfg.epochs}

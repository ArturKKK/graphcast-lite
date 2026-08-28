#!/usr/bin/env python
"""CLI для обучения v3 neural postprocessor (StationLeadBiasResidualMLP).

Главные отличия от v2:
  • per-station additive bias head (nn.Embedding(N,3)): чистый аддитивный сдвиг
    для (t2m,u,v) на каждую станцию — лечит остаточный bias после residual MLP;
  • wider station_emb (32 vs 16) — больше capacity на per-station специфику;
  • deeper trunk [192,192,128] vs [128,128];
  • рассчитан на корпус с ~689 станций РФ.

Пример:
    python scripts/postproc/train_neural_postproc_v3.py \\
        --train-parquet data/postproc/corpus_v3_train.parquet \\
        --val-parquet   data/postproc/corpus_v3_val.parquet \\
        --out-dir       experiments/neural_postproc_v3 \\
        --epochs 40 --batch-size 4096 --station-emb-dim 32 \\
        --hidden 192,192,128
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocessing.neural.dataset import StationCorpusDataset, build_balanced_sampler
from src.postprocessing.neural.losses import compute_total_loss
from src.postprocessing.neural.models import StationLeadBiasResidualMLP


# Same feature set as v2 (calendar + lapse + elevations already present)
DEFAULT_FEATURES_V3: List[str] = [
    "gnn_t2m", "gnn_u10", "gnn_v10", "gnn_msl", "gnn_sp",
    "gnn_t850", "gnn_t500", "gnn_q850", "gnn_z500",
    "gnn_u850", "gnn_v850", "gnn_u1000", "gnn_v1000",
    "lapse_t850_1000", "dewpoint_depression", "solar_zen",
    "lat", "lon", "elev",
    "z_surf", "lsm",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    "lead_norm",
]


@dataclass
class TrainConfigV3:
    train_parquet: str
    val_parquet: str
    out_dir: str
    feature_cols: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES_V3))
    hidden: List[int] = field(default_factory=lambda: [192, 192, 128])
    dropout: float = 0.1
    probabilistic: bool = False
    station_emb_dim: int = 32
    film_hidden: int = 64
    epochs: int = 40
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    wind_alpha: float = 0.5
    w_t2m: float = 1.0
    w_wind: float = 1.0
    balanced_sampling: bool = True
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 1.0
    seed: int = 42


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_station_idx(train_parquet: str, val_parquet: str) -> Dict[str, int]:
    seen: set[str] = set()
    for p in (train_parquet, val_parquet):
        df = pd.read_parquet(p, columns=["station_usaf"])
        seen.update(df["station_usaf"].astype(str).unique().tolist())
    return {s: i for i, s in enumerate(sorted(seen))}


def _epoch_pass(
    model: StationLeadBiasResidualMLP,
    loader: DataLoader,
    cfg: TrainConfigV3,
    optim: Optional[torch.optim.Optimizer],
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "loss_t2m": 0.0, "loss_wind": 0.0, "n": 0}
    sq_err = {"t2m": 0.0, "u10": 0.0, "v10": 0.0}
    abs_bias = {"t2m": 0.0}

    for step, batch in enumerate(loader):
        features = batch["features"].to(cfg.device, non_blocking=True)
        station_idx = batch["station_idx"].to(cfg.device, non_blocking=True)
        lead_norm = batch["lead_norm"].to(cfg.device, non_blocking=True)
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

        out = model(features, station_idx=station_idx, lead_norm=lead_norm, gnn_targets=gnn)
        losses = compute_total_loss(
            out,
            targets,
            probabilistic=cfg.probabilistic,
            w_t2m=cfg.w_t2m,
            w_wind=cfg.w_wind,
            huber_delta=cfg.huber_delta,
            wind_alpha=cfg.wind_alpha,
        )

        if train and not torch.isfinite(losses["loss"]):
            print(f"[NaN-DETECT] step={step} loss={losses['loss'].item()}", flush=True)
            sys.exit(2)

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
            abs_bias["t2m"] += (t_pred - targets["t2m"]).sum().item()

    n = max(totals["n"], 1)
    return {
        "loss": totals["loss"] / n,
        "loss_t2m": totals["loss_t2m"] / n,
        "loss_wind": totals["loss_wind"] / n,
        "rmse_t2m": float(np.sqrt(sq_err["t2m"] / n)),
        "rmse_u10": float(np.sqrt(sq_err["u10"] / n)),
        "rmse_v10": float(np.sqrt(sq_err["v10"] / n)),
        "vec_rmse_wind": float(np.sqrt((sq_err["u10"] + sq_err["v10"]) / n)),
        "bias_t2m": abs_bias["t2m"] / n,
    }


def train_v3(cfg: TrainConfigV3) -> Dict[str, float]:
    _seed_all(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_to_idx = _build_station_idx(cfg.train_parquet, cfg.val_parquet)
    num_stations = len(station_to_idx)
    print(f"[cfg] num_stations={num_stations}  features={len(cfg.feature_cols)}", flush=True)
    with open(out_dir / "station_to_idx.json", "w") as f:
        json.dump(station_to_idx, f, indent=2)

    train_ds = StationCorpusDataset(
        cfg.train_parquet,
        feature_cols=cfg.feature_cols,
        station_to_idx=station_to_idx,
    )
    # Датасет мог дополнить список признаками-наблюдениями, которых не было в
    # cfg. Модель, проверочная выборка и чекпойнт обязаны знать ТОТ ЖЕ список,
    # иначе размерность входа расходится: 28.08.2026 обучение падало сразу на
    # первом батче — «mat1 and mat2 shapes cannot be multiplied (4096x92 и
    # 58x192)». Приводим cfg к тому, что датасет собрал на самом деле.
    if list(train_ds.feature_cols) != list(cfg.feature_cols):
        was = len(cfg.feature_cols)
        cfg.feature_cols = list(train_ds.feature_cols)
        print(f"[cfg] признаков: {was} -> {len(cfg.feature_cols)} "
              f"(датасет дополнил список)", flush=True)

    val_ds = StationCorpusDataset(
        cfg.val_parquet,
        feature_cols=cfg.feature_cols,
        scalers=train_ds.export_scalers(),
        station_to_idx=station_to_idx,
    )
    train_ds.save_scalers(out_dir / "scalers.json")

    if train_ds.station_idx_arr is None or train_ds.lead_norm_arr is None:
        raise RuntimeError("v3 requires station_usaf + lead_norm in parquet")

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

    model = StationLeadBiasResidualMLP(
        feature_dim=len(cfg.feature_cols),
        num_stations=num_stations,
        station_emb_dim=cfg.station_emb_dim,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        probabilistic=cfg.probabilistic,
        film_hidden=cfg.film_hidden,
    ).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] StationLeadBiasResidualMLP params={n_params:,}  "
          f"(station_emb={cfg.station_emb_dim} hidden={cfg.hidden} bias_head=Embedding({num_stations},3))",
          flush=True)

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
            f"val bias_t2m={val_m['bias_t2m']:+.3f}°C "
            f"val vec_rmse_wind={val_m['vec_rmse_wind']:.3f}m/s",
            flush=True,
        )

        if val_score < best_val:
            best_val = val_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    # Класс модели пишем прямо в чекпойнт: оценка строила
                    # только версию v2, и веса v3 в неё не ложились —
                    # у v3 есть добавочная голова смещения по станции.
                    "model_class": type(model).__name__,
                    "cfg": asdict(cfg),
                    "scalers": train_ds.export_scalers(),
                    "feature_cols": cfg.feature_cols,
                    "station_to_idx": station_to_idx,
                    "epoch": epoch,
                    "val_metrics": val_m,
                },
                out_dir / "best_model.pth",
            )

    return {"best_val_combined": best_val, "epochs_run": cfg.epochs}


def _parse_args() -> TrainConfigV3:
    p = argparse.ArgumentParser(description="Train v3 neural postprocessor (station+lead+bias)")
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hidden", default="192,192,128")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--station-emb-dim", type=int, default=32)
    p.add_argument("--film-hidden", type=int, default=64)
    p.add_argument("--probabilistic", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--wind-alpha", type=float, default=0.5)
    p.add_argument("--w-t2m", type=float, default=1.0)
    p.add_argument("--w-wind", type=float, default=1.0)
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg_kwargs = dict(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        out_dir=args.out_dir,
        hidden=[int(x) for x in args.hidden.split(",")],
        dropout=args.dropout,
        station_emb_dim=args.station_emb_dim,
        film_hidden=args.film_hidden,
        probabilistic=args.probabilistic,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        huber_delta=args.huber_delta,
        wind_alpha=args.wind_alpha,
        w_t2m=args.w_t2m,
        w_wind=args.w_wind,
        balanced_sampling=(not args.no_balanced),
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if args.device:
        cfg_kwargs["device"] = args.device
    return TrainConfigV3(**cfg_kwargs)


def main() -> int:
    cfg = _parse_args()
    print("Config:")
    print(json.dumps({k: getattr(cfg, k) for k in vars(cfg)}, indent=2, default=str))
    result = train_v3(cfg)
    print("Done:", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""CLI для обучения neural postprocessor.

Пример:
    python scripts/postproc/train_neural_postproc.py \\
        --train-parquet data/postproc/corpus_train.parquet \\
        --val-parquet   data/postproc/corpus_val.parquet \\
        --out-dir       experiments/neural_postproc_v1 \\
        --epochs 30 --batch-size 4096

Probabilistic (CRPS):
    ... --probabilistic

См. docs/postprocessing_rfc.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# гарантируем, что репозиторий импортируем как пакет
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocessing.neural.train import TrainConfig, train


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train neural postprocessor (multi-task residual MLP)")
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", default=None, help="опц. JSON-файл с настройками, перекрывает CLI")
    p.add_argument("--hidden", default="128,128")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--probabilistic", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--wind-alpha", type=float, default=0.5)
    p.add_argument("--w-t2m", type=float, default=1.0)
    p.add_argument("--w-wind", type=float, default=1.0)
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg_kwargs = dict(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        out_dir=args.out_dir,
        hidden=[int(x) for x in args.hidden.split(",")],
        dropout=args.dropout,
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

    if args.config:
        with open(args.config) as f:
            extra = json.load(f)
        cfg_kwargs.update(extra)

    return TrainConfig(**cfg_kwargs)


def main() -> int:
    cfg = _parse_args()
    print("Config:")
    print(json.dumps({k: getattr(cfg, k) for k in vars(cfg)}, indent=2, default=str))
    result = train(cfg)
    print("Done:", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

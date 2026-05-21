"""Per-lead eval для v2 neural postprocessor на val parquet.

Usage:
    python scripts/postproc/eval_per_lead_v2.py \
        --val-parquet data/postproc/corpus_v2_val.parquet \
        --ckpt experiments/neural_postproc_v2/best_model.pth \
        --out-dir experiments/neural_postproc_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.postprocessing.neural.dataset import StationCorpusDataset
from src.postprocessing.neural.models import StationLeadAwareResidualMLP


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _bias(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(pred - obs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-parquet", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    feature_cols = ckpt["feature_cols"]
    station_to_idx = ckpt["station_to_idx"]
    scalers = ckpt["scalers"]

    ds = StationCorpusDataset(
        args.val_parquet,
        feature_cols=feature_cols,
        scalers=scalers,
        station_to_idx=station_to_idx,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = StationLeadAwareResidualMLP(
        feature_dim=len(feature_cols),
        num_stations=len(station_to_idx),
        station_emb_dim=cfg["station_emb_dim"],
        hidden=tuple(cfg["hidden"]),
        dropout=cfg["dropout"],
        probabilistic=cfg["probabilistic"],
        film_hidden=cfg["film_hidden"],
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n = len(ds)
    pp_t2m = np.zeros(n, dtype=np.float32)
    pp_u = np.zeros(n, dtype=np.float32)
    pp_v = np.zeros(n, dtype=np.float32)

    cur = 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["features"].size(0)
            features = batch["features"].to(args.device)
            station_idx = batch["station_idx"].to(args.device)
            lead_norm = batch["lead_norm"].to(args.device)
            gnn = {
                "t2m": batch["gnn_t2m"].to(args.device),
                "u10": batch["gnn_u10"].to(args.device),
                "v10": batch["gnn_v10"].to(args.device),
            }
            out = model(features, station_idx=station_idx, lead_norm=lead_norm, gnn_targets=gnn)
            if cfg["probabilistic"]:
                t = out["t2m_mu"].cpu().numpy()
                u = out["wind_mu"][:, 0].cpu().numpy()
                v = out["wind_mu"][:, 1].cpu().numpy()
            else:
                t = out["t2m"].cpu().numpy()
                u = out["u10"].cpu().numpy()
                v = out["v10"].cpu().numpy()
            pp_t2m[cur:cur + bs] = t
            pp_u[cur:cur + bs] = u
            pp_v[cur:cur + bs] = v
            cur += bs

    obs_t2m = ds.Y[:, 0]
    obs_u = ds.Y[:, 1]
    obs_v = ds.Y[:, 2]
    gnn_t2m = ds.G[:, 0]
    gnn_u = ds.G[:, 1]
    gnn_v = ds.G[:, 2]

    leads = pd.read_parquet(args.val_parquet, columns=["lead_h"])["lead_h"].to_numpy()
    # Align: dataset may have dropped rows with NaN targets — replicate
    df_full = pd.read_parquet(args.val_parquet, columns=["lead_h", "obs_t2m_K", "obs_u10", "obs_v10"])
    mask = df_full[["obs_t2m_K", "obs_u10", "obs_v10"]].notna().all(axis=1).to_numpy()
    leads = df_full["lead_h"].to_numpy()[mask]
    assert len(leads) == n, f"lead alignment mismatch: {len(leads)} vs {n}"

    unique_leads = sorted(np.unique(leads).tolist())

    def speed(u, v):
        return np.sqrt(u * u + v * v)

    def metrics_for(idx):
        return {
            "n": int(idx.sum()),
            "pp_rmse_t2m": _rmse(pp_t2m[idx], obs_t2m[idx]),
            "pp_mae_t2m": _mae(pp_t2m[idx], obs_t2m[idx]),
            "pp_bias_t2m": _bias(pp_t2m[idx], obs_t2m[idx]),
            "gnn_rmse_t2m": _rmse(gnn_t2m[idx], obs_t2m[idx]),
            "gnn_bias_t2m": _bias(gnn_t2m[idx], obs_t2m[idx]),
            "pp_vec_rmse_wind": float(np.sqrt(np.mean((pp_u[idx] - obs_u[idx]) ** 2 + (pp_v[idx] - obs_v[idx]) ** 2))),
            "gnn_vec_rmse_wind": float(np.sqrt(np.mean((gnn_u[idx] - obs_u[idx]) ** 2 + (gnn_v[idx] - obs_v[idx]) ** 2))),
            "pp_speed_rmse": _rmse(speed(pp_u[idx], pp_v[idx]), speed(obs_u[idx], obs_v[idx])),
            "gnn_speed_rmse": _rmse(speed(gnn_u[idx], gnn_v[idx]), speed(obs_u[idx], obs_v[idx])),
        }

    all_idx = np.ones(n, dtype=bool)
    overall = metrics_for(all_idx)
    per_lead = {}
    for lh in unique_leads:
        per_lead[int(lh)] = metrics_for(leads == lh)

    out_json = {"overall": overall, "per_lead": per_lead, "ckpt_epoch": ckpt.get("epoch")}
    with open(out_dir / "eval_per_lead_v2.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # Markdown
    lines = []
    lines.append("# Neural postproc v2 — per-lead evaluation\n")
    lines.append(f"Checkpoint epoch: {ckpt.get('epoch')}. Val n={n}.\n")
    lines.append("## Overall (val)\n")
    lines.append("|  | T2m RMSE °C | T2m MAE °C | T2m bias °C | Wind vec-RMSE m/s | Wind speed-RMSE m/s |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| **postproc v2** | **{overall['pp_rmse_t2m']:.3f}** | {overall['pp_mae_t2m']:.3f} | "
        f"{overall['pp_bias_t2m']:+.3f} | **{overall['pp_vec_rmse_wind']:.3f}** | "
        f"**{overall['pp_speed_rmse']:.3f}** |"
    )
    lines.append(
        f"| GNN raw | {overall['gnn_rmse_t2m']:.3f} | — | {overall['gnn_bias_t2m']:+.3f} | "
        f"{overall['gnn_vec_rmse_wind']:.3f} | {overall['gnn_speed_rmse']:.3f} |"
    )
    lines.append(
        f"| Δ | {overall['pp_rmse_t2m']-overall['gnn_rmse_t2m']:+.3f} | — | "
        f"{overall['pp_bias_t2m']-overall['gnn_bias_t2m']:+.3f} | "
        f"{overall['pp_vec_rmse_wind']-overall['gnn_vec_rmse_wind']:+.3f} | "
        f"{overall['pp_speed_rmse']-overall['gnn_speed_rmse']:+.3f} |\n"
    )

    lines.append("## Per lead_h\n")
    lines.append("| lead h | n | T2m RMSE pp | T2m RMSE GNN | Δ T2m | T2m bias pp | T2m bias GNN | Wind vec pp | Wind vec GNN | Δ wind | Wind speed pp | Wind speed GNN |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lh in unique_leads:
        m = per_lead[int(lh)]
        lines.append(
            f"| {int(lh)} | {m['n']:,} | {m['pp_rmse_t2m']:.3f} | {m['gnn_rmse_t2m']:.3f} | "
            f"{m['pp_rmse_t2m']-m['gnn_rmse_t2m']:+.3f} | {m['pp_bias_t2m']:+.3f} | "
            f"{m['gnn_bias_t2m']:+.3f} | {m['pp_vec_rmse_wind']:.3f} | {m['gnn_vec_rmse_wind']:.3f} | "
            f"{m['pp_vec_rmse_wind']-m['gnn_vec_rmse_wind']:+.3f} | {m['pp_speed_rmse']:.3f} | "
            f"{m['gnn_speed_rmse']:.3f} |"
        )

    with open(out_dir / "eval_per_lead_v2.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_dir / 'eval_per_lead_v2.json'}")
    print(f"Saved: {out_dir / 'eval_per_lead_v2.md'}")
    print(f"\nOverall: t2m RMSE={overall['pp_rmse_t2m']:.3f}°C, wind vec={overall['pp_vec_rmse_wind']:.3f}m/s")


if __name__ == "__main__":
    main()

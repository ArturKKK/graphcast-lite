"""Per-station eval for v2 neural postprocessor with optional bbox filter.

Designed to answer: "errors at Novosibirsk + Akademgorodok".
Default bbox covers Novosibirsk Oblast neighbourhood (54..56N, 81..85E).

Usage:
    python scripts/postproc/eval_per_station_v2.py \
        --val-parquet data/postproc/corpus_v2_val.parquet \
        --ckpt experiments/neural_postproc_v2/best_model.pth \
        --out-dir experiments/neural_postproc_v2 \
        --bbox 54 56 81 85 \
        --label nsk
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
from src.postprocessing.neural.models import (StationLeadAwareResidualMLP,
                                              StationLeadBiasResidualMLP)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def _mae(a, b):
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def _bias(p, o):
    return float(np.mean(p - o)) if len(p) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-parquet", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bbox", type=float, nargs=4, default=[54.0, 56.0, 81.0, 85.0],
                    help="lat_min lat_max lon_min lon_max")
    ap.add_argument("--label", default="nsk")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    lat_min, lat_max, lon_min, lon_max = args.bbox
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
        auto_obs_features=False,
        station_to_idx=station_to_idx,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Какой класс обучали, такой и строим: у v3 есть добавочная голова
    # смещения по станции, и веса v3 в модель v2 не лягут.
    cls_name = ckpt.get("model_class")
    if cls_name is None:
        cls_name = ("StationLeadBiasResidualMLP"
                    if any(k.startswith("bias_head.") for k in ckpt["model_state"])
                    else "StationLeadAwareResidualMLP")
    print(f"[eval] модель: {cls_name}", flush=True)

    model = {"StationLeadAwareResidualMLP": StationLeadAwareResidualMLP,
             "StationLeadBiasResidualMLP": StationLeadBiasResidualMLP}[cls_name](
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

    # Recover station/lat/lon aligned with ds (which drops obs-NaN rows)
    df_full = pd.read_parquet(
        args.val_parquet,
        columns=["station_usaf", "station_lat", "station_lon", "station_elev",
                 "lead_h", "obs_t2m_K", "obs_u10", "obs_v10"],
    )
    mask = df_full[["obs_t2m_K", "obs_u10", "obs_v10"]].notna().all(axis=1).to_numpy()
    df = df_full.loc[mask].reset_index(drop=True)
    assert len(df) == n, f"alignment mismatch {len(df)} vs {n}"

    # Per-station catalog
    stations = (
        df[["station_usaf", "station_lat", "station_lon", "station_elev"]]
        .drop_duplicates("station_usaf")
        .reset_index(drop=True)
    )
    in_box = stations[
        (stations.station_lat >= lat_min) & (stations.station_lat <= lat_max)
        & (stations.station_lon >= lon_min) & (stations.station_lon <= lon_max)
    ].sort_values(["station_lat", "station_lon"]).reset_index(drop=True)

    def speed(u, v):
        return np.sqrt(u * u + v * v)

    def metrics(idx: np.ndarray) -> dict:
        return {
            "n": int(idx.sum()),
            "pp_rmse_t2m": _rmse(pp_t2m[idx], obs_t2m[idx]),
            "pp_mae_t2m": _mae(pp_t2m[idx], obs_t2m[idx]),
            "pp_bias_t2m": _bias(pp_t2m[idx], obs_t2m[idx]),
            "gnn_rmse_t2m": _rmse(gnn_t2m[idx], obs_t2m[idx]),
            "gnn_mae_t2m": _mae(gnn_t2m[idx], obs_t2m[idx]),
            "gnn_bias_t2m": _bias(gnn_t2m[idx], obs_t2m[idx]),
            "pp_vec_rmse_wind": _rmse(
                np.stack([pp_u[idx], pp_v[idx]], axis=1),
                np.stack([obs_u[idx], obs_v[idx]], axis=1),
            ) * np.sqrt(2) if idx.sum() else float("nan"),
            "gnn_vec_rmse_wind": _rmse(
                np.stack([gnn_u[idx], gnn_v[idx]], axis=1),
                np.stack([obs_u[idx], obs_v[idx]], axis=1),
            ) * np.sqrt(2) if idx.sum() else float("nan"),
            "pp_speed_rmse": _rmse(speed(pp_u[idx], pp_v[idx]), speed(obs_u[idx], obs_v[idx])),
            "gnn_speed_rmse": _rmse(speed(gnn_u[idx], gnn_v[idx]), speed(obs_u[idx], obs_v[idx])),
        }

    # NOTE: the vec_rmse above multiplied by sqrt(2) compensates for stacking 2 channels
    # but to keep parity with eval_per_lead_v2.py (which uses sqrt(mean(du^2+dv^2))),
    # recompute directly:
    def vec_rmse(pu, pv, ou, ov):
        if len(pu) == 0:
            return float("nan")
        return float(np.sqrt(np.mean((pu - ou) ** 2 + (pv - ov) ** 2)))

    def metrics2(idx):
        return {
            "n": int(idx.sum()),
            "pp_rmse_t2m": _rmse(pp_t2m[idx], obs_t2m[idx]),
            "pp_mae_t2m": _mae(pp_t2m[idx], obs_t2m[idx]),
            "pp_bias_t2m": _bias(pp_t2m[idx], obs_t2m[idx]),
            "gnn_rmse_t2m": _rmse(gnn_t2m[idx], obs_t2m[idx]),
            "gnn_mae_t2m": _mae(gnn_t2m[idx], obs_t2m[idx]),
            "gnn_bias_t2m": _bias(gnn_t2m[idx], obs_t2m[idx]),
            "pp_vec_rmse_wind": vec_rmse(pp_u[idx], pp_v[idx], obs_u[idx], obs_v[idx]),
            "gnn_vec_rmse_wind": vec_rmse(gnn_u[idx], gnn_v[idx], obs_u[idx], obs_v[idx]),
            "pp_speed_rmse": _rmse(speed(pp_u[idx], pp_v[idx]), speed(obs_u[idx], obs_v[idx])),
            "gnn_speed_rmse": _rmse(speed(gnn_u[idx], gnn_v[idx]), speed(obs_u[idx], obs_v[idx])),
        }

    usaf_arr = df["station_usaf"].to_numpy()

    per_station = {}
    for _, srow in in_box.iterrows():
        sid = srow["station_usaf"]
        idx = (usaf_arr == sid)
        per_station[str(sid)] = {
            "lat": float(srow["station_lat"]),
            "lon": float(srow["station_lon"]),
            "elev": float(srow["station_elev"]),
            **metrics2(idx),
        }

    # Region aggregate
    region_idx = np.isin(usaf_arr, in_box["station_usaf"].to_numpy())
    region = metrics2(region_idx)
    overall = metrics2(np.ones(n, dtype=bool))

    out_json = {
        "label": args.label,
        "bbox": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        "n_stations_in_box": len(in_box),
        "region_aggregate": region,
        "per_station": per_station,
        "overall_all_stations": overall,
        "ckpt_epoch": ckpt.get("epoch"),
    }
    json_path = out_dir / f"eval_per_station_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    # Markdown
    lines = []
    lines.append(f"# Neural postproc v2 — per-station eval ({args.label})\n")
    lines.append(
        f"BBox: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]. "
        f"Stations in box: **{len(in_box)}** / 50. Checkpoint epoch: {ckpt.get('epoch')}.\n"
    )
    if len(in_box) == 0:
        lines.append("**Нет станций в bbox.** Расширь диапазон.\n")
    else:
        lines.append("## Stations in box\n")
        lines.append("| USAF | lat | lon | elev m | n samples |")
        lines.append("|---|---:|---:|---:|---:|")
        for sid, m in per_station.items():
            lines.append(f"| {sid} | {m['lat']:.3f} | {m['lon']:.3f} | {m['elev']:.0f} | {m['n']:,} |")
        lines.append("")

        lines.append("## Per-station metrics (T2m)\n")
        lines.append("| USAF | n | RMSE pp °C | RMSE GNN °C | Δ °C | bias pp °C | bias GNN °C | MAE pp °C | MAE GNN °C |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for sid, m in per_station.items():
            d = m["pp_rmse_t2m"] - m["gnn_rmse_t2m"]
            lines.append(
                f"| {sid} | {m['n']:,} | {m['pp_rmse_t2m']:.3f} | {m['gnn_rmse_t2m']:.3f} | "
                f"{d:+.3f} | {m['pp_bias_t2m']:+.3f} | {m['gnn_bias_t2m']:+.3f} | "
                f"{m['pp_mae_t2m']:.3f} | {m['gnn_mae_t2m']:.3f} |"
            )
        lines.append("")

        lines.append("## Per-station metrics (wind)\n")
        lines.append("| USAF | n | vec-RMSE pp m/s | vec-RMSE GNN m/s | Δ | speed-RMSE pp | speed-RMSE GNN |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for sid, m in per_station.items():
            d = m["pp_vec_rmse_wind"] - m["gnn_vec_rmse_wind"]
            lines.append(
                f"| {sid} | {m['n']:,} | {m['pp_vec_rmse_wind']:.3f} | {m['gnn_vec_rmse_wind']:.3f} | "
                f"{d:+.3f} | {m['pp_speed_rmse']:.3f} | {m['gnn_speed_rmse']:.3f} |"
            )
        lines.append("")

        lines.append("## Region aggregate (all stations in box)\n")
        lines.append("|  | n | T2m RMSE °C | T2m bias °C | wind vec-RMSE | wind speed-RMSE |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(
            f"| postproc v2 | {region['n']:,} | **{region['pp_rmse_t2m']:.3f}** | "
            f"{region['pp_bias_t2m']:+.3f} | **{region['pp_vec_rmse_wind']:.3f}** | "
            f"**{region['pp_speed_rmse']:.3f}** |"
        )
        lines.append(
            f"| GNN raw | {region['n']:,} | {region['gnn_rmse_t2m']:.3f} | "
            f"{region['gnn_bias_t2m']:+.3f} | {region['gnn_vec_rmse_wind']:.3f} | "
            f"{region['gnn_speed_rmse']:.3f} |"
        )
        lines.append(
            f"| Δ | — | {region['pp_rmse_t2m']-region['gnn_rmse_t2m']:+.3f} | "
            f"{region['pp_bias_t2m']-region['gnn_bias_t2m']:+.3f} | "
            f"{region['pp_vec_rmse_wind']-region['gnn_vec_rmse_wind']:+.3f} | "
            f"{region['pp_speed_rmse']-region['gnn_speed_rmse']:+.3f} |"
        )
        lines.append("")

    lines.append("## Overall (all 50 stations, reference)\n")
    lines.append(
        f"- n = {overall['n']:,}\n"
        f"- T2m RMSE: pp **{overall['pp_rmse_t2m']:.3f}** °C  vs GNN {overall['gnn_rmse_t2m']:.3f} °C\n"
        f"- T2m bias: pp {overall['pp_bias_t2m']:+.3f} °C  vs GNN {overall['gnn_bias_t2m']:+.3f} °C\n"
        f"- Wind vec-RMSE: pp **{overall['pp_vec_rmse_wind']:.3f}** m/s  vs GNN {overall['gnn_vec_rmse_wind']:.3f} m/s\n"
    )

    md_path = out_dir / f"eval_per_station_{args.label}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Stations in box: {len(in_box)} / 50; region n={region['n']:,}")
    if len(in_box):
        print(f"Region pp T2m RMSE = {region['pp_rmse_t2m']:.3f}°C (GNN {region['gnn_rmse_t2m']:.3f}°C)")
        print(f"Region pp wind vec-RMSE = {region['pp_vec_rmse_wind']:.3f}m/s (GNN {region['gnn_vec_rmse_wind']:.3f}m/s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract GNN t2m predictions at 19 station locations.

Runs the real_freeze6 model on ALL timesteps of the multires dataset,
extracts denormalized t2m predictions at the nearest grid nodes to each
station, and saves as CSV for MOS retraining.

Usage (on VM):
    python scripts/extract_gnn_at_stations.py experiments/multires_real_freeze6 \
        --split all --max-samples 0
"""

import argparse, json, os, sys, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig
from src.data.dataloader_chunked import load_chunked_datasets
from src.data.dataloader import load_train_and_test_datasets
from src.main import load_model_from_experiment_config
from src.utils import load_from_json_file

# ── Station coordinates ─────────────────────────────────────────────
STATIONS = {
    "284935": {"name": "Yemelyanovo", "lat": 56.173, "lon": 92.493, "elev": 287},
    "294670": {"name": "Achinsk", "lat": 56.283, "lon": 90.517, "elev": 257},
    "295810": {"name": "Kansk", "lat": 56.200, "lon": 95.633, "elev": 207},
    "287854": {"name": "Abakan", "lat": 53.740, "lon": 91.385, "elev": 253},
    "293740": {"name": "Kazachinskoe", "lat": 57.683, "lon": 93.267, "elev": 93},
    "294710": {"name": "Bolshaya Murta", "lat": 56.900, "lon": 93.133, "elev": 180},
    "293670": {"name": "Novobirilyussy", "lat": 56.967, "lon": 90.683, "elev": 181},
    "294770": {"name": "Sukhobuzimskoe", "lat": 56.500, "lon": 93.283, "elev": 164},
    "295530": {"name": "Bogotol", "lat": 56.217, "lon": 89.550, "elev": 290},
    "295710": {"name": "Minino", "lat": 56.067, "lon": 92.733, "elev": 235},
    "295630": {"name": "Kaca", "lat": 56.117, "lon": 92.200, "elev": 479},
    "295660": {"name": "Shumiha", "lat": 55.933, "lon": 92.283, "elev": 275},
    "293630": {"name": "Pirovskoe", "lat": 57.633, "lon": 92.267, "elev": 179},
    "293790": {"name": "Taseevo", "lat": 57.200, "lon": 94.550, "elev": 168},
    "294640": {"name": "Bolshoj Uluj", "lat": 56.650, "lon": 90.550, "elev": 231},
    "294810": {"name": "Dzerzhinskoe", "lat": 56.850, "lon": 95.217, "elev": 188},
    "295610": {"name": "Nazarovo", "lat": 56.033, "lon": 90.317, "elev": 256},
    "295620": {"name": "Kemchug", "lat": 56.100, "lon": 91.667, "elev": 332},
    "295800": {"name": "Solyanka", "lat": 56.167, "lon": 95.267, "elev": 357},
}


def find_nearest_nodes(grid_lat, grid_lon, stations):
    """Find nearest grid node for each station. Returns {usaf: grid_index}."""
    mapping = {}
    for usaf, info in stations.items():
        dist = (grid_lat - info["lat"])**2 + (grid_lon - info["lon"])**2
        gidx = int(np.argmin(dist))
        actual_lat, actual_lon = grid_lat[gidx], grid_lon[gidx]
        km = np.sqrt((actual_lat - info["lat"])**2 + (actual_lon - info["lon"])**2) * 111
        mapping[usaf] = {
            "grid_idx": gidx,
            "actual_lat": float(actual_lat),
            "actual_lon": float(actual_lon),
            "dist_km": float(km),
        }
        print(f"  {usaf} ({info['name']:20s}) → node {gidx:6d} "
              f"({actual_lat:.3f}°N, {actual_lon:.3f}°E) dist={km:.1f}km")
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="all", help="all|test|train")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--start-date", default="2010-01-01T00:00",
                     help="Start datetime of the dataset (ISO format)")
    ap.add_argument("--output", default="data/temp_train/gnn_predictions_at_stations.csv")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--ar-steps", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load config ─────────────────────────────────────────────────
    exp_dir = Path(args.experiment_dir)
    cfg_path = str(exp_dir / "config.json")
    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))

    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None:
        if hasattr(exp_cfg, "data_dir") and exp_cfg.data_dir:
            data_dir = Path(exp_cfg.data_dir)
        else:
            data_dir = Path("data/datasets") / exp_cfg.data.dataset_name
    print(f"Data dir: {data_dir}")

    # ── Load dataset ────────────────────────────────────────────────
    data_npy = data_dir / "data.npy"
    C = exp_cfg.data.num_features_used
    OBS = exp_cfg.data.obs_window_used
    P = exp_cfg.data.pred_window_used

    if data_npy.exists():
        from src.data.dataloader_chunked import load_chunked_datasets
        train_ds, val_ds, test_ds, meta = load_chunked_datasets(
            str(data_dir),
            obs_window=OBS,
            pred_steps=P,
            n_features=C,
            test_split=args.split,
        )
        ds = test_ds  # when split="all", test_ds has all samples
    else:
        train_ds, val_ds, test_ds, meta = load_train_and_test_datasets(
            str(data_dir), exp_cfg.data
        )
        ds = test_ds
    G = len(ds[0][0]) if ds[0][0].dim() >= 1 else meta.num_longitudes * meta.num_latitudes
    print(f"Dataset: {len(ds)} samples, G={G}, C={C}, OBS={OBS}")

    # ── Load coords ─────────────────────────────────────────────────
    npz_path = data_dir / "coords.npz"
    z = np.load(npz_path)
    grid_lat = z["latitude"].astype(np.float32)
    grid_lon = z["longitude"].astype(np.float32)
    print(f"Grid coords: lat({grid_lat.shape}), lon({grid_lon.shape})")

    # ── Load scalers ────────────────────────────────────────────────
    sc = np.load(data_dir / "scalers.npz")
    mean_all = sc["mean"][:C].astype(np.float64)
    std_all = sc["std"][:C].astype(np.float64)
    t2m_mean = mean_all[0]
    t2m_std = std_all[0]
    print(f"Scalers: t2m mean={t2m_mean:.2f}K, std={t2m_std:.2f}K")

    # ── Map stations → grid nodes ──────────────────────────────────
    print("\nStation → grid node mapping:")
    station_map = find_nearest_nodes(grid_lat, grid_lon, STATIONS)

    # ── Load model ──────────────────────────────────────────────────
    coords = (grid_lat, grid_lon) if npz_path.exists() else None
    flat_grid = getattr(meta, 'flat_grid', False)
    # Detect regional dataset → compute region_bounds for mesh trimming
    region_bounds = None
    if coords is not None:
        lat_span = float(grid_lat.max() - grid_lat.min())
        lon_span = float(grid_lon.max() - grid_lon.min())
        if lat_span < 90 and lon_span < 90:
            region_bounds = (
                float(grid_lat.min()), float(grid_lat.max()),
                float(grid_lon.min()), float(grid_lon.max()),
            )
            print(f"[region] lat=[{region_bounds[0]:.1f},{region_bounds[1]:.1f}] "
                  f"lon=[{region_bounds[2]:.1f},{region_bounds[3]:.1f}]")

    model = load_model_from_experiment_config(
        experiment_config=exp_cfg, device=device, dataset_metadata=meta,
        coordinates=coords, region_bounds=region_bounds, flat_grid=flat_grid,
    )
    ckpt = exp_dir / "best_model.pth"
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, loaded from {ckpt}")

    # ── Compute timestamps ──────────────────────────────────────────
    start_dt = datetime.fromisoformat(args.start_date)
    step_hours = 6

    # ── Run inference ───────────────────────────────────────────────
    max_samples = len(ds) if args.max_samples == 0 else min(args.max_samples, len(ds))
    AR = args.ar_steps

    print(f"\nRunning inference: {max_samples} samples, AR={AR}...")
    results = []
    t0 = time.time()

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            if i >= max_samples:
                break

            X = X.to(device)
            y = y.squeeze(0)

            # Single-step inference
            pred = model(X, attention_threshold=0.0).cpu()
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)

            if args.no_residual:
                out = pred.squeeze(0)  # [G, C]
            else:
                X_reshaped = X.view(1, G, OBS, C)
                out = (X_reshaped[:, :, -1, :].cpu() + pred).squeeze(0)  # [G, C]

            # Compute the timestamp for this prediction:
            # Sample i uses timesteps [offset, offset+1] as obs, predicts offset+2
            if hasattr(ds, '_sample_indices') and i < len(ds._sample_indices):
                chunk_idx, local_t = ds._sample_indices[i]
                # Global offset: sum of previous chunk lengths + local_t
                cum = np.cumsum([0] + ds.chunk_lengths)
                global_t = int(cum[chunk_idx]) + local_t
            else:
                global_t = i

            pred_time_offset = global_t + OBS  # the predicted timestep
            pred_dt = start_dt + timedelta(hours=pred_time_offset * step_hours)

            # Extract t2m (channel 0) at each station, denormalize
            for usaf, sinfo in station_map.items():
                gidx = sinfo["grid_idx"]

                # GNN prediction (normalized → denormalized → Celsius)
                gnn_t2m_norm = out[gidx, 0].item()
                gnn_t2m_K = gnn_t2m_norm * t2m_std + t2m_mean
                gnn_t2m_C = gnn_t2m_K - 273.15

                # ERA5 input (last obs frame, channel 0) — denormalized
                era5_t2m_norm = y[gidx, 0].item()  # ground truth (next step)
                era5_t2m_K = era5_t2m_norm * t2m_std + t2m_mean
                era5_t2m_C = era5_t2m_K - 273.15

                # Also get the last input t2m (for reference)
                X_flat = X.squeeze(0)  # [G, OBS*C]
                input_last_t2m_norm = X_flat[gidx, (OBS-1)*C].item()
                input_t2m_K = input_last_t2m_norm * t2m_std + t2m_mean
                input_t2m_C = input_t2m_K - 273.15

                results.append({
                    "time": pred_dt.isoformat(),
                    "station_usaf": usaf,
                    "gnn_t2m_C": round(gnn_t2m_C, 3),
                    "era5_t2m_C": round(era5_t2m_C, 3),
                    "input_t2m_C": round(input_t2m_C, 3),
                    "grid_lat": sinfo["actual_lat"],
                    "grid_lon": sinfo["actual_lon"],
                })

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (max_samples - i - 1) / rate
                print(f"  [{i+1}/{max_samples}] {rate:.1f} samples/s, ETA {eta:.0f}s")

    # ── Save CSV ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} rows in {elapsed:.1f}s")

    import pandas as pd
    df = pd.DataFrame(results)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["time", "station_usaf"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows, {out_path.stat().st_size / 1024:.0f} KB)")

    # Quick stats
    print(f"\nPer-station GNN t2m stats:")
    for usaf in sorted(STATIONS.keys()):
        sub = df[df.station_usaf == usaf]
        if len(sub) == 0:
            continue
        bias = sub.gnn_t2m_C - sub.era5_t2m_C
        print(f"  {usaf} ({STATIONS[usaf]['name']:20s}): "
              f"mean_bias={bias.mean():.3f}°C  mae={bias.abs().mean():.3f}°C  "
              f"n={len(sub)}")


if __name__ == "__main__":
    main()

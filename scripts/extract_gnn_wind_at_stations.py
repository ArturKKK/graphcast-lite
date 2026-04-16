#!/usr/bin/env python3
"""
Extract GNN wind (u10, v10) + t2m predictions at 19 station locations.

Unlike extract_gnn_at_stations.py, this builds multires input on-the-fly
from global + regional datasets (no need to pre-build the full multires dataset).

Outputs CSV with: time, station_usaf, gnn_t2m_C, gnn_u10, gnn_v10, gnn_ws_ms,
                   era5_t2m_C, era5_u10, era5_v10, era5_ws_ms

Usage (CPU, slow but works):
    python scripts/extract_gnn_wind_at_stations.py \
        experiments/multires_real_freeze6 \
        --max-samples 2000
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig
from src.main import load_model_from_experiment_config
from src.utils import load_from_json_file

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


def build_multires_coords(global_dir, regional_dir, roi=(50, 60, 83, 98)):
    """Build multires flat coordinates from global + regional grids."""
    g = np.load(Path(global_dir) / "coords.npz")
    r = np.load(Path(regional_dir) / "coords.npz")
    g_lats, g_lons = g["latitude"].astype(np.float64), g["longitude"].astype(np.float64)
    r_lats, r_lons = r["latitude"].astype(np.float64), r["longitude"].astype(np.float64)

    lat_min, lat_max, lon_min, lon_max = roi
    g_lon_mesh, g_lat_mesh = np.meshgrid(g_lons, g_lats)
    in_roi = ((g_lat_mesh >= lat_min) & (g_lat_mesh <= lat_max) &
              (g_lon_mesh >= lon_min) & (g_lon_mesh <= lon_max))
    keep_global = ~in_roi

    g_flat_lats = g_lat_mesh[keep_global]
    g_flat_lons = g_lon_mesh[keep_global]

    r_lon_mesh, r_lat_mesh = np.meshgrid(r_lons, r_lats)

    flat_lats = np.concatenate([g_flat_lats, r_lat_mesh.ravel()]).astype(np.float32)
    flat_lons = np.concatenate([g_flat_lons, r_lon_mesh.ravel()]).astype(np.float32)

    return flat_lats, flat_lons, keep_global, g_lats, g_lons, r_lats, r_lons


def load_timestep_multires(t, global_data, regional_data, keep_global,
                           g_nlat, g_nlon, r_nlat, r_nlon, C,
                           scalers_mean, scalers_std):
    """Load one timestep, build multires flat vector, normalize."""
    # Global: shape (T, nlon, nlat, C) → (nlat, nlon, C) → flat
    g_frame = global_data[t]  # (nlon, nlat, C) float16
    g_frame = np.transpose(g_frame, (1, 0, 2))  # (nlat, nlon, C)
    g_flat = g_frame[keep_global].astype(np.float32)  # (N_kept, C)

    # Regional: shape (T, nlon, nlat, C) → flat
    r_frame = regional_data[t]  # (nlon, nlat, C) float16
    r_frame = np.transpose(r_frame, (1, 0, 2))  # (nlat, nlon, C)
    r_flat = r_frame.reshape(-1, C).astype(np.float32)  # (N_reg, C)

    # Concatenate
    frame = np.concatenate([g_flat, r_flat], axis=0)  # (N_total, C)

    # Normalize
    frame_norm = (frame - scalers_mean) / scalers_std

    return frame, frame_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", default="experiments/multires_real_freeze6")
    ap.add_argument("--global-dir",
                    default="data/datasets/global_512x256_19f_2010-2021_07deg")
    ap.add_argument("--regional-dir",
                    default="data/datasets/region_krsk_61x41_19f_2010-2020_025deg")
    ap.add_argument("--start-date", default="2010-01-01T00:00")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Start from this timestep index (for resuming)")
    ap.add_argument("--step", type=int, default=1,
                    help="Process every N-th timestep (for faster extraction)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--output",
                    default="data/temp_train/gnn_wind_predictions_at_stations.csv")
    args = ap.parse_args()

    device = torch.device("cpu")
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
    elif args.device != "cpu":
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load experiment config
    exp_dir = Path(args.experiment_dir)
    exp_cfg = ExperimentConfig(**load_from_json_file(str(exp_dir / "config.json")))
    C = exp_cfg.data.num_features_used
    OBS = exp_cfg.data.obs_window_used

    # Build multires coordinates
    print("Building multires coordinates...")
    flat_lats, flat_lons, keep_global, g_lats, g_lons, r_lats, r_lons = \
        build_multires_coords(args.global_dir, args.regional_dir)
    N = len(flat_lats)
    print(f"Multires grid: {N} nodes (global kept={keep_global.sum()}, "
          f"regional={N - keep_global.sum()})")

    # Load datasets as memmap
    with open(Path(args.global_dir) / "dataset_info.json") as f:
        g_info = json.load(f)
    with open(Path(args.regional_dir) / "dataset_info.json") as f:
        r_info = json.load(f)

    g_shape = (g_info["n_time"], g_info["n_lon"], g_info["n_lat"], g_info["n_feat"])
    r_shape = (r_info["n_time"], r_info["n_lon"], r_info["n_lat"], r_info["n_feat"])

    global_data = np.memmap(Path(args.global_dir) / "data.npy",
                            dtype=np.float16, mode="r", shape=g_shape)
    regional_data = np.memmap(Path(args.regional_dir) / "data.npy",
                              dtype=np.float16, mode="r", shape=r_shape)

    print(f"Global data: {g_shape}, Regional data: {r_shape}")

    # Scalers (use global)
    sc = np.load(Path(args.global_dir) / "scalers.npz")
    s_mean = sc["mean"][:C].astype(np.float32)
    s_std = sc["std"][:C].astype(np.float32)

    # Variable indices
    variables = json.load(open(Path(args.global_dir) / "variables.json"))
    i_t2m = variables.index("t2m")
    i_u10 = variables.index("10u")
    i_v10 = variables.index("10v")
    print(f"Variable indices: t2m={i_t2m}, u10={i_u10}, v10={i_v10}")

    # Map stations to grid nodes
    print("\nStation → grid node mapping:")
    station_map = {}
    for usaf, info in STATIONS.items():
        dist = (flat_lats - info["lat"])**2 + (flat_lons - info["lon"])**2
        gidx = int(np.argmin(dist))
        km = np.sqrt(dist[gidx]) * 111
        station_map[usaf] = {"grid_idx": gidx, "dist_km": km}
        print(f"  {usaf} ({info['name']:20s}) → node {gidx:6d} "
              f"({flat_lats[gidx]:.3f}°N, {flat_lons[gidx]:.3f}°E) dist={km:.1f}km")

    # Load model
    coords = (flat_lats, flat_lons)
    lat_span = float(flat_lats.max() - flat_lats.min())
    lon_span = float(flat_lons.max() - flat_lons.min())
    region_bounds = None
    if lat_span < 90 and lon_span < 90:
        region_bounds = (float(flat_lats.min()), float(flat_lats.max()),
                         float(flat_lons.min()), float(flat_lons.max()))

    class FakeMeta:
        num_longitudes = 0
        num_latitudes = 0
        flat_grid = True

    model = load_model_from_experiment_config(
        experiment_config=exp_cfg, device=device, dataset_metadata=FakeMeta(),
        coordinates=coords, region_bounds=region_bounds, flat_grid=True,
    )
    ckpt = exp_dir / "best_model.pth"
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Timestamps
    start_dt = datetime.fromisoformat(args.start_date)
    step_hours = 6

    # Use min of both datasets
    T_max = min(g_info["n_time"], r_info["n_time"])
    T_usable = T_max - OBS  # need OBS frames for input + 1 for target
    start_idx = max(args.start_idx, 0)
    step = args.step
    n_available = (T_usable - start_idx + step - 1) // step
    max_samples = n_available if args.max_samples == 0 \
        else min(args.max_samples, n_available)

    print(f"\nTimesteps total: {T_max}, usable: {T_usable}")
    print(f"Processing: {max_samples} samples, start_idx={start_idx}, step={step}")

    # Run inference
    results = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(max_samples):
            t_idx = start_idx + i * step

            # Build input: OBS frames stacked (N, OBS*C)
            frames_phys = []
            frames_norm = []
            for obs_step in range(OBS):
                t = t_idx + obs_step
                phys, norm = load_timestep_multires(
                    t, global_data, regional_data, keep_global,
                    g_info["n_lat"], g_info["n_lon"],
                    r_info["n_lat"], r_info["n_lon"], C,
                    s_mean, s_std
                )
                frames_phys.append(phys)
                frames_norm.append(norm)

            # Stack: (N, OBS*C) flattened
            X_norm = np.concatenate(frames_norm, axis=1)  # (N, OBS*C)
            X = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, OBS*C)

            # Ground truth: the next timestep after obs window
            target_t = t_idx + OBS
            target_phys, target_norm = load_timestep_multires(
                target_t, global_data, regional_data, keep_global,
                g_info["n_lat"], g_info["n_lon"],
                r_info["n_lat"], r_info["n_lon"], C,
                s_mean, s_std
            )

            # Forward pass
            pred_norm = model(X, attention_threshold=0.0).cpu()
            if pred_norm.dim() == 2:
                pred_norm = pred_norm.unsqueeze(0)

            # use_residual=False → pred is direct normalized output
            out_norm = pred_norm.squeeze(0).numpy()  # (N, C)

            # Denormalize
            out_phys = out_norm * s_std + s_mean  # (N, C) in physical units

            # Prediction timestamp
            pred_dt = start_dt + timedelta(hours=(target_t) * step_hours)

            # Extract at each station
            for usaf, sinfo in station_map.items():
                gidx = sinfo["grid_idx"]

                gnn_t2m_K = out_phys[gidx, i_t2m]
                gnn_u10 = out_phys[gidx, i_u10]
                gnn_v10 = out_phys[gidx, i_v10]
                gnn_ws = np.sqrt(gnn_u10**2 + gnn_v10**2)

                era5_t2m_K = target_phys[gidx, i_t2m]
                era5_u10 = target_phys[gidx, i_u10]
                era5_v10 = target_phys[gidx, i_v10]
                era5_ws = np.sqrt(era5_u10**2 + era5_v10**2)

                results.append({
                    "time": pred_dt.isoformat(),
                    "station_usaf": usaf,
                    "gnn_t2m_C": round(float(gnn_t2m_K) - 273.15, 3),
                    "gnn_u10": round(float(gnn_u10), 4),
                    "gnn_v10": round(float(gnn_v10), 4),
                    "gnn_ws_ms": round(float(gnn_ws), 4),
                    "era5_t2m_C": round(float(era5_t2m_K) - 273.15, 3),
                    "era5_u10": round(float(era5_u10), 4),
                    "era5_v10": round(float(era5_v10), 4),
                    "era5_ws_ms": round(float(era5_ws), 4),
                })

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (max_samples - i - 1) / rate
                print(f"  [{i+1}/{max_samples}] {rate:.2f} samples/s, "
                      f"ETA {eta/60:.0f}m")

            # Save checkpoint every 500 samples
            if (i + 1) % 500 == 0:
                import pandas as pd
                df_tmp = pd.DataFrame(results)
                df_tmp.to_csv(args.output + ".tmp", index=False)
                print(f"  Checkpoint saved ({len(results)} rows)")

    # Save final CSV
    import pandas as pd
    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} rows in {elapsed:.1f}s "
          f"({len(results)/19:.0f} samples, {max_samples/elapsed:.2f} samples/s)")

    df = pd.DataFrame(results)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["time", "station_usaf"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")

    # Quick stats
    print(f"\nPer-station GNN wind stats (vs ERA5):")
    for usaf in sorted(STATIONS.keys()):
        sub = df[df.station_usaf == usaf]
        if len(sub) == 0:
            continue
        ws_bias = sub.gnn_ws_ms - sub.era5_ws_ms
        t2m_bias = sub.gnn_t2m_C - sub.era5_t2m_C
        print(f"  {usaf} ({STATIONS[usaf]['name']:20s}): "
              f"wind_bias={ws_bias.mean():+.3f} m/s  "
              f"t2m_bias={t2m_bias.mean():+.3f}°C  n={len(sub)}")


if __name__ == "__main__":
    main()

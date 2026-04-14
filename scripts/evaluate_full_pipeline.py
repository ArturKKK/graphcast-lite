#!/usr/bin/env python3
"""
Comprehensive evaluation: GNN vs GNN+UNet cascade vs post-processing.

Builds multires test data on-the-fly from global + regional datasets,
runs GNN inference, optional UNet cascade, and applies post-processing:
  - Lapse-rate elevation correction
  - Learned MOS (HistGBR)
  - Spatial IDW (MOS bias interpolated to full grid)
  - Optimal Interpolation (simulated live station obs)

Evaluates on the fine 0.25° regional grid against real ERA5 ground truth.

Usage:
  python scripts/evaluate_full_pipeline.py \
    --max-samples 50 --ar-steps 4
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig
from src.constants import FileNames
from src.data.data_configs import DatasetMetadata
from src.main import load_model_from_experiment_config, set_random_seeds
from src.postprocessing.mos_correction import (
    apply_learned_mos_t2m,
    load_learned_mos,
)
from src.unet.model import WeatherUNet
from src.utils import load_from_json_file


# ─── Constants ───────────────────────────────────────────
ROI = (50.0, 60.0, 83.0, 98.0)
LAPSE_RATE = 6.5e-3  # K/m

MOS_STATIONS = [
    {"lat": 56.173, "lon": 92.493, "elev": 287, "name": "Yemelyanovo"},
    {"lat": 56.283, "lon": 90.517, "elev": 257, "name": "Achinsk"},
    {"lat": 56.200, "lon": 95.633, "elev": 207, "name": "Kansk"},
    {"lat": 53.740, "lon": 91.385, "elev": 253, "name": "Abakan"},
    {"lat": 57.683, "lon": 93.267, "elev": 93,  "name": "Kazachinskoe"},
    {"lat": 56.900, "lon": 93.133, "elev": 180, "name": "Bolshaya Murta"},
    {"lat": 56.967, "lon": 90.683, "elev": 181, "name": "Novobirilyussy"},
    {"lat": 56.500, "lon": 93.283, "elev": 164, "name": "Sukhobuzimskoe"},
    {"lat": 56.217, "lon": 89.550, "elev": 290, "name": "Bogotol"},
    {"lat": 56.067, "lon": 92.733, "elev": 235, "name": "Minino"},
    {"lat": 56.117, "lon": 92.200, "elev": 479, "name": "Kaca"},
    {"lat": 55.933, "lon": 92.283, "elev": 275, "name": "Shumiha"},
    {"lat": 57.633, "lon": 92.267, "elev": 179, "name": "Pirovskoe"},
    {"lat": 57.200, "lon": 94.550, "elev": 168, "name": "Taseevo"},
    {"lat": 56.650, "lon": 90.550, "elev": 231, "name": "Bolshoj Uluj"},
    {"lat": 56.850, "lon": 95.217, "elev": 188, "name": "Dzerzhinskoe"},
    {"lat": 56.033, "lon": 90.317, "elev": 256, "name": "Nazarovo"},
    {"lat": 56.100, "lon": 91.667, "elev": 332, "name": "Kemchug"},
    {"lat": 56.167, "lon": 95.267, "elev": 357, "name": "Solyanka"},
]

# Mean station elevation (for regional average lapse)
MEAN_STATION_ELEV = np.mean([s["elev"] for s in MOS_STATIONS])

VAR_ORDER = [
    "t2m", "10u", "10v", "msl", "tp", "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
]

# ─── Data loading helpers ────────────────────────────────


def build_multires_coords(g_lats, g_lons, r_lats, r_lons, roi):
    """Build multires node mapping: global outside ROI + regional inside."""
    lat_min, lat_max, lon_min, lon_max = roi
    g_lat_mesh, g_lon_mesh = np.meshgrid(g_lats, g_lons, indexing="ij")  # (nlat, nlon)
    in_roi = (
        (g_lat_mesh >= lat_min) & (g_lat_mesh <= lat_max)
        & (g_lon_mesh >= lon_min) & (g_lon_mesh <= lon_max)
    )
    keep_global = ~in_roi

    g_flat_lats = g_lat_mesh[keep_global].astype(np.float32)
    g_flat_lons = g_lon_mesh[keep_global].astype(np.float32)

    r_lon_mesh, r_lat_mesh = np.meshgrid(r_lons, r_lats)
    r_flat_lats = r_lat_mesh.reshape(-1).astype(np.float32)
    r_flat_lons = r_lon_mesh.reshape(-1).astype(np.float32)

    flat_lats = np.concatenate([g_flat_lats, r_flat_lats])
    flat_lons = np.concatenate([g_flat_lons, r_flat_lons])
    n_global_kept = len(g_flat_lats)
    region_mask = np.zeros(len(flat_lats), dtype=bool)
    region_mask[n_global_kept:] = True

    return flat_lats, flat_lons, region_mask, keep_global, n_global_kept


def build_multires_frame(
    global_data, g_lats, g_lons, r_lats, r_lons, keep_global,
    n_global_kept, n_regional, C, t_global, t_regional,
    regional_data=None,
):
    """Build one multires frame from global + regional data on the fly.

    If regional_data is provided, uses real 0.25° data for ROI nodes.
    Otherwise interpolates from global grid.
    """
    # Global frame: (n_lon, n_lat, C) → (n_lat, n_lon, C)
    g_frame = np.array(global_data[t_global], dtype=np.float32).transpose(1, 0, 2)
    global_values = g_frame[keep_global]  # (n_kept, C)

    if regional_data is not None:
        # Use real regional data
        r_frame = np.array(regional_data[t_regional], dtype=np.float32)
        # r_frame: (n_lon, n_lat, C) → flatten lat-major: (n_lat, n_lon, C) → (N, C)
        regional_values = r_frame.transpose(1, 0, 2).reshape(-1, C)
    else:
        # Interpolate from global
        r_lon_mesh, r_lat_mesh = np.meshgrid(r_lons, r_lats)
        r_points = np.stack([r_lat_mesh.ravel(), r_lon_mesh.ravel()], axis=-1)
        regional_values = np.zeros((n_regional, C), dtype=np.float32)
        for c in range(C):
            interp = RegularGridInterpolator(
                (g_lats, g_lons), g_frame[:, :, c],
                method="linear", bounds_error=False, fill_value=None,
            )
            regional_values[:, c] = interp(r_points)

    return np.concatenate([global_values, regional_values], axis=0)  # (N_total, C)


def denormalize(data_norm, mean, std):
    """Denormalize: physical = normalized * std + mean."""
    return data_norm * std + mean


def crop_upsample_roi(pred_flat, flat_lats, flat_lons, region_mask,
                      target_lats, target_lons, C, mean, std):
    """Extract ROI from multires prediction (normalized) → upsample to 0.25° grid (normalized).

    Returns (n_lat_fine, n_lon_fine, C) normalized array.
    """
    # Region nodes are the last n_regional nodes
    roi_vals = pred_flat[region_mask]  # (N_reg, C)
    roi_lats = flat_lats[region_mask]
    roi_lons = flat_lons[region_mask]

    # Build unique lat/lon for the regional subgrid
    u_lats = np.unique(roi_lats)
    u_lons = np.unique(roi_lons)
    n_lat_r = len(u_lats)
    n_lon_r = len(u_lons)

    # Map flat → 2D
    lat_idx_map = {float(v): i for i, v in enumerate(u_lats)}
    lon_idx_map = {float(v): i for i, v in enumerate(u_lons)}
    grid_2d = np.zeros((n_lat_r, n_lon_r, C), dtype=np.float32)
    for k in range(len(roi_lats)):
        li = lat_idx_map.get(float(roi_lats[k]))
        lo = lon_idx_map.get(float(roi_lons[k]))
        if li is not None and lo is not None:
            grid_2d[li, lo] = roi_vals[k]

    # The regional grid IS the target grid → no interpolation needed
    # (since region nodes are exactly the 0.25° grid)
    return grid_2d  # (41, 61, C)


def apply_lapse(pred_phys, var_order, target_elevation):
    """Apply lapse-rate correction: adjusts t2m for elevation difference."""
    if "t2m" not in var_order or "z_surf" not in var_order:
        return pred_phys
    corrected = pred_phys.copy()
    t_idx = var_order.index("t2m")
    z_idx = var_order.index("z_surf")
    # z_surf is geopotential height in gpm ≈ metres
    z_grid = corrected[:, 0, z_idx] if corrected.ndim == 3 else corrected[:, z_idx]
    delta_z = z_grid - target_elevation
    delta_t = delta_z * LAPSE_RATE
    if corrected.ndim == 3:
        for step in range(corrected.shape[1]):
            corrected[:, step, t_idx] += delta_t
    else:
        corrected[:, t_idx] += delta_t
    return corrected


def simulate_station_obs(ground_truth_phys, r_lats, r_lons, stations, var_order):
    """Create synthetic station observations from ground truth for OI evaluation.

    Returns observations array (G, C) with NaN everywhere except station points.
    """
    G = len(r_lats)
    C = len(var_order)
    obs = np.full((G, C), np.nan, dtype=np.float32)

    # Create flat lat/lon arrays matching regional grid
    r_lon_mesh, r_lat_mesh = np.meshgrid(r_lons, r_lats)
    flat_lats = r_lat_mesh.ravel()
    flat_lons = r_lon_mesh.ravel()

    for st in stations:
        dist = (flat_lats - st["lat"]) ** 2 + (flat_lons - st["lon"]) ** 2
        gidx = int(np.argmin(dist))
        obs[gidx, :] = ground_truth_phys[gidx, :]

    return obs


# ─── Main evaluation ─────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-exp", default="experiments/multires_nores_freeze6")
    ap.add_argument("--unet-exp", default="experiments/downscaler_krsk")
    ap.add_argument("--global-dir", default="data/datasets/global_512x256_19f_2010-2021_07deg")
    ap.add_argument("--region-dir", default="data/datasets/region_krsk_61x41_19f_2010-2020_025deg")
    ap.add_argument("--learned-mos", default="live_runtime_bundle/learned_mos_t2m_19stations.joblib")
    ap.add_argument("--ar-steps", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--lapse-elev", type=float, default=MEAN_STATION_ELEV,
                    help="Target elevation for lapse correction (default: mean station elev)")
    ap.add_argument("--skip-unet", action="store_true", help="Skip UNet cascade")
    ap.add_argument("--skip-oi", action="store_true", help="Skip OI (slow on CPU)")
    args = ap.parse_args()

    set_random_seeds(42)
    device = torch.device("cpu")  # local evaluation
    AR = args.ar_steps
    OBS = 2  # obs_window for GNN

    print("=" * 70)
    print("COMPREHENSIVE PIPELINE EVALUATION")
    print("=" * 70)

    # ── 1. Load datasets ──
    print("\n[1/6] Loading datasets...")
    g_dir = Path(args.global_dir)
    r_dir = Path(args.region_dir)

    with open(g_dir / "dataset_info.json") as f:
        g_info = json.load(f)
    with open(r_dir / "dataset_info.json") as f:
        r_info = json.load(f)

    g_coords = np.load(g_dir / "coords.npz")
    g_lats = g_coords["latitude"].astype(np.float64)
    g_lons = g_coords["longitude"].astype(np.float64)
    r_coords = np.load(r_dir / "coords.npz")
    r_lats = r_coords["latitude"].astype(np.float64)
    r_lons = r_coords["longitude"].astype(np.float64)

    C = g_info["n_feat"]
    T_global = g_info["n_time"]
    T_regional = r_info["n_time"]
    T_overlap = min(T_global, T_regional)

    g_shape = (T_global, g_info["n_lon"], g_info["n_lat"], C)
    r_shape = (T_regional, r_info["n_lon"], r_info["n_lat"], C)
    global_data = np.memmap(g_dir / "data.npy", dtype=np.float16, mode="r", shape=g_shape)
    regional_data = np.memmap(r_dir / "data.npy", dtype=np.float16, mode="r", shape=r_shape)

    scalers = np.load(g_dir / "scalers.npz")
    y_mean = scalers["mean"][:C].astype(np.float32)
    y_std = scalers["std"][:C].astype(np.float32)

    print(f"  Global: {g_shape}, Regional: {r_shape}")
    print(f"  Overlap: {T_overlap} timesteps, using test portion")

    # ── 2. Build multires coordinates ──
    print("\n[2/6] Building multires coordinates...")
    flat_lats, flat_lons, region_mask, keep_global, n_global_kept = \
        build_multires_coords(g_lats, g_lons, r_lats, r_lons, ROI)
    N_total = len(flat_lats)
    n_regional = region_mask.sum()
    print(f"  Multires grid: {N_total} nodes (global={n_global_kept}, regional={n_regional})")

    # ── 3. Load GNN model ──
    print("\n[3/6] Loading GNN model...")
    gnn_cfg = ExperimentConfig(**load_from_json_file(
        os.path.join(args.gnn_exp, FileNames.EXPERIMENT_CONFIG)
    ))

    # Create metadata for model init
    metadata = DatasetMetadata(
        flattened=True, num_latitudes=0, num_longitudes=0,
        num_features=C, obs_window=OBS, pred_window=1,
    )
    metadata.flat_grid = True
    metadata.num_grid_nodes = N_total
    metadata.cordinates = (flat_lats, flat_lons)
    metadata.is_regional = region_mask

    gnn_model = load_model_from_experiment_config(
        gnn_cfg, device, metadata,
        coordinates=(flat_lats, flat_lons),
        region_bounds=ROI, mesh_buffer=15.0, flat_grid=True,
    )

    ckpt_path = os.path.join(args.gnn_exp, "best_model.pth")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    gnn_model.load_state_dict(state, strict=False)
    gnn_model.eval()
    n_params = sum(p.numel() for p in gnn_model.parameters())
    print(f"  GNN loaded: {n_params:,} params from {ckpt_path}")

    # ── 4. Load UNet model ──
    unet_model = None
    unet_static_t = None
    if not args.skip_unet:
        print("\n[4/6] Loading UNet downscaler...")
        unet_cfg_path = os.path.join(args.unet_exp, "config.json")
        with open(unet_cfg_path) as f:
            unet_cfg = json.load(f)
        unet_obs = unet_cfg.get("obs_window", 2)
        unet_filters = unet_cfg.get("base_filters", 64)
        unet_residual = unet_cfg.get("residual", True)
        static_channels = unet_cfg.get("static_channels", [7, 8])
        # Build static fine fields from the first regional frame (z_surf, lsm are static)
        r_frame0 = np.array(regional_data[0], dtype=np.float32)  # (n_lon, n_lat, C)
        r_frame0_hw = r_frame0.transpose(1, 0, 2)  # (H, W, C)
        static_fine = r_frame0_hw[:, :, static_channels]  # (H, W, n_static)
        # Normalize static channels with global scalers
        for si, ch in enumerate(static_channels):
            static_fine[:, :, si] = (static_fine[:, :, si] - y_mean[ch]) / y_std[ch]
        unet_static_t = torch.from_numpy(static_fine).permute(2, 0, 1).unsqueeze(0).float()
        n_static = len(static_channels)
        # Input: obs_window * C channels + static channels
        unet_in_ch = unet_obs * C + n_static
        unet_model = WeatherUNet(
            in_channels=unet_in_ch, out_channels=C, base_filters=unet_filters,
        )
        unet_ckpt = os.path.join(args.unet_exp, "best_model.pth")
        unet_model.load_state_dict(
            torch.load(unet_ckpt, map_location=device, weights_only=True)
        )
        unet_model.eval()
        print(f"  UNet loaded: {unet_model.num_params:,} params, residual={unet_residual}, "
              f"static_ch={static_channels}")
    else:
        unet_residual = False
        unet_obs = 2
        print("\n[4/6] UNet skipped")

    # ── 5. Load learned MOS ──
    mos_bundle = None
    if os.path.exists(args.learned_mos):
        mos_bundle = load_learned_mos(args.learned_mos)
        print(f"  MOS loaded: test_mae={mos_bundle.get('test_mae', '?')}°C")
    else:
        print(f"  WARNING: MOS not found at {args.learned_mos}")

    # ── 6. Prepare test samples ──
    print("\n[5/6] Preparing test samples...")
    # Test set: last 20% of overlap, skip first half (= val), use second half (= test_only)
    n_test_total = int(T_overlap * 0.2)
    n_val = n_test_total // 2
    test_start = T_overlap - n_test_total + n_val  # start of test_only
    test_end = T_overlap

    # Sample indices: each needs OBS + AR consecutive timesteps
    window = OBS + AR
    valid_indices = list(range(test_start, test_end - window + 1))
    max_samples = min(args.max_samples, len(valid_indices))

    # Evenly space samples across test period
    if max_samples < len(valid_indices):
        step = len(valid_indices) // max_samples
        sample_starts = [valid_indices[i * step] for i in range(max_samples)]
    else:
        sample_starts = valid_indices[:max_samples]

    print(f"  Test range: t={test_start}..{test_end-1} ({test_end - test_start} steps)")
    print(f"  Evaluating {max_samples} samples, AR={AR}")

    # ── 7. Run evaluation ──
    print("\n[6/6] Running inference...")
    print("  This may take a while on CPU...")

    # Fine grid for regional evaluation
    n_lat_fine = len(r_lats)
    n_lon_fine = len(r_lons)
    G_fine = n_lat_fine * n_lon_fine  # 2501

    # Regional flat coordinates
    r_lon_mesh, r_lat_mesh = np.meshgrid(r_lons, r_lats)
    r_flat_lats = r_lat_mesh.ravel().astype(np.float32)
    r_flat_lons = r_lon_mesh.ravel().astype(np.float32)

    # Map stations to regional grid indices
    station_grid_indices = []
    for st in MOS_STATIONS:
        dist = (r_flat_lats - st["lat"]) ** 2 + (r_flat_lons - st["lon"]) ** 2
        station_grid_indices.append(int(np.argmin(dist)))

    # Simulated forecast start time (doesn't matter much, use 2020-06-01)
    base_time = datetime(2020, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Accumulators for different configurations
    configs = ["GNN", "GNN+lapse", "GNN+MOS", "GNN+lapse+MOS", "GNN+lapse+MOS+IDW"]
    if unet_model is not None:
        configs += ["Cascade", "Cascade+lapse", "Cascade+lapse+MOS+IDW"]
    if not args.skip_oi:
        configs += ["GNN+lapse+MOS+IDW+OI"]
        if unet_model is not None:
            configs += ["Cascade+lapse+MOS+IDW+OI"]

    # Per-config, per-horizon, per-channel MSE (on regional grid)
    mse_grid = {cfg: [np.zeros(C, dtype=np.float64) for _ in range(AR)] for cfg in configs}
    mse_grid["Persistence"] = [np.zeros(C, dtype=np.float64) for _ in range(AR)]
    # Same at station points only
    mse_stn = {cfg: [np.zeros(C, dtype=np.float64) for _ in range(AR)] for cfg in configs}
    mse_stn["Persistence"] = [np.zeros(C, dtype=np.float64) for _ in range(AR)]
    count_grid = [0] * AR
    count_stn = [0] * AR

    # OI setup (on regional grid)
    oi = None
    if not args.skip_oi:
        try:
            from src.assimilation.optimal_interpolation import OptimalInterpolation
            oi = OptimalInterpolation(
                grid_lats=r_lats, grid_lons=r_lons,
                sigma_b=1.5, sigma_o=0.5, L=100_000.0,  # 100km decorrelation
                device=device,
            )
            print("  OI initialized on regional grid")
        except Exception as e:
            print(f"  WARNING: OI init failed: {e}")
            oi = None

    t0_total = time.time()
    for si, t_start in enumerate(sample_starts):
        t0_sample = time.time()

        # Build input: OBS multires frames (normalized)
        input_frames = []
        for t_off in range(OBS):
            t = t_start + t_off
            frame = build_multires_frame(
                global_data, g_lats, g_lons, r_lats, r_lons,
                keep_global, n_global_kept, n_regional, C,
                t_global=t, t_regional=t,
                regional_data=regional_data,
            )
            # Normalize with global scalers
            frame_norm = (frame - y_mean) / y_std
            input_frames.append(frame_norm)

        # Input tensor: (1, N_total, OBS * C)
        X_np = np.stack(input_frames, axis=1)  # (N, OBS, C)
        curr_state = torch.from_numpy(X_np).unsqueeze(0).float()  # (1, N, OBS, C)

        # Persistence baseline: last obs frame on REGIONAL grid (physical units)
        t_last_obs = t_start + OBS - 1
        persist_frame = np.array(regional_data[t_last_obs], dtype=np.float32)
        persist_phys = persist_frame.transpose(1, 0, 2).reshape(-1, C)  # (G_fine, C)
        # Regional data.npy stores RAW physical values (verified: t2m~238K, z_surf~428m).
        # Normalization is applied on-the-fly by the dataloader, not stored in files.

        use_residual = gnn_cfg.use_residual if hasattr(gnn_cfg, 'use_residual') else False

        # AR rollout
        for ar_step in range(AR):
            t_target = t_start + OBS + ar_step
            if t_target >= T_overlap:
                break

            # Ground truth on fine grid (physical)
            gt_frame = np.array(regional_data[t_target], dtype=np.float32)
            gt_phys = gt_frame.transpose(1, 0, 2).reshape(-1, C)  # already physical units

            # Forecast times for MOS
            valid_times = [
                base_time + timedelta(hours=6 * (step + 1))
                for step in range(AR)
            ]

            # ── GNN forward pass ──
            inp = curr_state.view(1, N_total, OBS * C)
            with torch.no_grad():
                pred = gnn_model(X=inp, attention_threshold=0.0)
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)

            if use_residual:
                gnn_out = curr_state[:, :, -1, :] + pred
            else:
                gnn_out = pred

            gnn_out_np = gnn_out[0].numpy()  # (N_total, C) normalized

            # Extract ROI (regional nodes)
            roi_pred_norm = gnn_out_np[region_mask]  # (G_fine, C) normalized
            roi_pred_phys = roi_pred_norm * y_std + y_mean  # physical units

            # ── UNet cascade ──
            cascade_phys = None
            if unet_model is not None:
                # Input to UNet: (1, unet_obs * C, H, W)
                # Use the same GNN output duplicated for obs_window frames
                roi_2d = roi_pred_norm.reshape(n_lat_fine, n_lon_fine, C)
                unet_input_frames = [roi_2d] * unet_obs
                unet_stack = np.concatenate(unet_input_frames, axis=-1)  # (H, W, obs*C)
                unet_x = torch.from_numpy(unet_stack).permute(2, 0, 1).unsqueeze(0).float()
                # Append static fine channels (z_surf, lsm normalized)
                if unet_static_t is not None:
                    unet_x = torch.cat([unet_x, unet_static_t], dim=1)

                with torch.no_grad():
                    unet_out = unet_model(unet_x)  # (1, C, H, W)

                if unet_residual:
                    x_last = unet_x[:, (unet_obs - 1) * C:unet_obs * C, :, :]
                    unet_out = x_last + unet_out

                cascade_norm = unet_out[0].numpy().transpose(1, 2, 0).reshape(-1, C)
                cascade_phys = cascade_norm * y_std + y_mean

            # ── Build prediction arrays for different configs (G_fine, steps=1, C) ──
            # We evaluate per-step, so shape (G_fine, 1, C) for MOS compatibility
            def make_pred_3d(flat_phys):
                return flat_phys[:, np.newaxis, :]  # (G, 1, C)

            # Config: GNN raw
            gnn_3d = make_pred_3d(roi_pred_phys)

            # Config: GNN + lapse
            gnn_lapse_3d = apply_lapse(gnn_3d.copy(), VAR_ORDER, args.lapse_elev)

            # Config: GNN + MOS (at station points)
            gnn_mos_3d = gnn_3d.copy()
            if mos_bundle is not None:
                valid_times_step = [valid_times[min(ar_step, len(valid_times) - 1)]]
                result = apply_learned_mos_t2m(
                    gnn_mos_3d, VAR_ORDER, mos_bundle,
                    r_flat_lats, r_flat_lons, valid_times_step,
                    stations=MOS_STATIONS, spatial_idw=False,
                )
                if isinstance(result, tuple):
                    gnn_mos_3d = result[0]

            # Config: GNN + lapse + MOS
            gnn_lm_3d = apply_lapse(gnn_3d.copy(), VAR_ORDER, args.lapse_elev)
            if mos_bundle is not None:
                result = apply_learned_mos_t2m(
                    gnn_lm_3d, VAR_ORDER, mos_bundle,
                    r_flat_lats, r_flat_lons, valid_times_step,
                    stations=MOS_STATIONS, spatial_idw=False,
                )
                if isinstance(result, tuple):
                    gnn_lm_3d = result[0]

            # Config: GNN + lapse + MOS + IDW
            gnn_lmi_3d = apply_lapse(gnn_3d.copy(), VAR_ORDER, args.lapse_elev)
            if mos_bundle is not None:
                result = apply_learned_mos_t2m(
                    gnn_lmi_3d, VAR_ORDER, mos_bundle,
                    r_flat_lats, r_flat_lons, valid_times_step,
                    stations=MOS_STATIONS, spatial_idw=True,
                    idw_power=2.0, idw_max_radius_km=300.0,
                )
                if isinstance(result, tuple):
                    gnn_lmi_3d = result[0]

            # Store predictions for comparison
            preds = {
                "GNN": gnn_3d[:, 0, :],
                "GNN+lapse": gnn_lapse_3d[:, 0, :],
                "GNN+MOS": gnn_mos_3d[:, 0, :],
                "GNN+lapse+MOS": gnn_lm_3d[:, 0, :],
                "GNN+lapse+MOS+IDW": gnn_lmi_3d[:, 0, :],
            }

            if cascade_phys is not None:
                casc_3d = make_pred_3d(cascade_phys)
                casc_lapse_3d = apply_lapse(casc_3d.copy(), VAR_ORDER, args.lapse_elev)
                casc_lmi_3d = apply_lapse(casc_3d.copy(), VAR_ORDER, args.lapse_elev)
                if mos_bundle is not None:
                    result = apply_learned_mos_t2m(
                        casc_lmi_3d, VAR_ORDER, mos_bundle,
                        r_flat_lats, r_flat_lons, valid_times_step,
                        stations=MOS_STATIONS, spatial_idw=True,
                        idw_power=2.0, idw_max_radius_km=300.0,
                    )
                    if isinstance(result, tuple):
                        casc_lmi_3d = result[0]

                preds["Cascade"] = cascade_phys
                preds["Cascade+lapse"] = casc_lapse_3d[:, 0, :]
                preds["Cascade+lapse+MOS+IDW"] = casc_lmi_3d[:, 0, :]

            # OI: use ground truth at station points as "observations"
            if oi is not None:
                for base_cfg, oi_cfg in [
                    ("GNN+lapse+MOS+IDW", "GNN+lapse+MOS+IDW+OI"),
                ]:
                    if base_cfg in preds and oi_cfg in configs:
                        pred_for_oi = preds[base_cfg].copy()
                        # Create obs: NaN everywhere except stations
                        obs_for_oi = simulate_station_obs(
                            gt_phys, r_lats, r_lons, MOS_STATIONS, VAR_ORDER,
                        )
                        pred_t = torch.from_numpy(pred_for_oi).float()
                        obs_t = torch.from_numpy(obs_for_oi).float()
                        try:
                            analys = oi.apply(pred_t, obs_t)
                            preds[oi_cfg] = analys.numpy()
                        except Exception:
                            preds[oi_cfg] = pred_for_oi

                if cascade_phys is not None and "Cascade+lapse+MOS+IDW+OI" in configs:
                    base = preds.get("Cascade+lapse+MOS+IDW", cascade_phys)
                    obs_for_oi = simulate_station_obs(
                        gt_phys, r_lats, r_lons, MOS_STATIONS, VAR_ORDER,
                    )
                    pred_t = torch.from_numpy(base).float()
                    obs_t = torch.from_numpy(obs_for_oi).float()
                    try:
                        analys = oi.apply(pred_t, obs_t)
                        preds["Cascade+lapse+MOS+IDW+OI"] = analys.numpy()
                    except Exception:
                        preds["Cascade+lapse+MOS+IDW+OI"] = base

            # ── Compute metrics ──
            gt = gt_phys  # (G_fine, C) — raw physical units

            for cfg_name, pred_flat in preds.items():
                if cfg_name not in mse_grid:
                    continue
                diff = (pred_flat - gt) ** 2
                mse_grid[cfg_name][ar_step] += diff.sum(axis=0)

                # Station points only
                for idx in station_grid_indices:
                    mse_stn[cfg_name][ar_step] += (pred_flat[idx] - gt[idx]) ** 2

            # Persistence
            diff_p = (persist_phys - gt) ** 2
            mse_grid["Persistence"][ar_step] += diff_p.sum(axis=0)
            for idx in station_grid_indices:
                mse_stn["Persistence"][ar_step] += (persist_phys[idx] - gt[idx]) ** 2

            count_grid[ar_step] += G_fine
            count_stn[ar_step] += len(station_grid_indices)

            # Update GNN state for next AR step
            curr_state = torch.cat(
                [curr_state[:, :, 1:, :], gnn_out.unsqueeze(2)], dim=2,
            )

        elapsed = time.time() - t0_sample
        if (si + 1) % 5 == 0 or si == 0:
            t2m_rmse = np.sqrt(mse_grid["GNN"][0][0] / max(count_grid[0], 1))
            eta = (time.time() - t0_total) / (si + 1) * (max_samples - si - 1)
            print(f"  [{si+1}/{max_samples}] {elapsed:.1f}s/sample, "
                  f"t2m GNN +6h RMSE={t2m_rmse:.2f}K, ETA={eta/60:.0f}min")

    total_time = time.time() - t0_total
    print(f"\n  Done in {total_time/60:.1f} minutes")

    # ── 8. Print results ──
    hours = [6 * (h + 1) for h in range(AR)]

    # --- t2m table (the main one) ---
    print("\n" + "=" * 90)
    print("═══ t2m RMSE (°C) at all regional grid points ═══")
    print("=" * 90)
    t_idx = VAR_ORDER.index("t2m")
    all_configs = ["Persistence"] + configs
    header = f"{'Config':>30}" + "".join(f" {f'+{h}h':>8}" for h in hours) + "   avg"
    print(header)
    print("-" * len(header))

    for cfg_name in all_configs:
        if cfg_name not in mse_grid:
            continue
        vals = []
        for h in range(AR):
            rmse = np.sqrt(mse_grid[cfg_name][h][t_idx] / max(count_grid[h], 1))
            vals.append(rmse)
        avg = np.mean(vals)
        line = f"{cfg_name:>30}"
        for v in vals:
            line += f" {v:8.2f}"
        line += f" {avg:7.2f}"
        print(line)

    # --- t2m at stations only ---
    print("\n" + "=" * 90)
    print("═══ t2m RMSE (°C) at 19 MOS STATION points ═══")
    print("=" * 90)
    print(header)
    print("-" * len(header))

    for cfg_name in all_configs:
        if cfg_name not in mse_stn:
            continue
        vals = []
        for h in range(AR):
            rmse = np.sqrt(mse_stn[cfg_name][h][t_idx] / max(count_stn[h], 1))
            vals.append(rmse)
        avg = np.mean(vals)
        line = f"{cfg_name:>30}"
        for v in vals:
            line += f" {v:8.2f}"
        line += f" {avg:7.2f}"
        print(line)

    # --- Skill vs persistence ---
    print("\n" + "=" * 90)
    print("═══ Skill vs Persistence (%, higher=better) ═══")
    print("=" * 90)

    # Overall skill (normalized, averaged over dynamic channels)
    static_ch = {7, 8}  # z_surf, lsm
    header2 = f"{'Config':>30}" + "".join(f" {f'+{h}h':>8}" for h in hours) + "   avg"
    print(header2)
    print("-" * len(header2))

    persist_rmse = {}
    for h in range(AR):
        skills = []
        for c in range(C):
            if c in static_ch:
                continue
            persist_rmse[(h, c)] = np.sqrt(
                mse_grid["Persistence"][h][c] / max(count_grid[h], 1)
            )

    for cfg_name in configs:
        if cfg_name not in mse_grid:
            continue
        vals = []
        for h in range(AR):
            skills_ch = []
            for c in range(C):
                if c in static_ch:
                    continue
                rmse_m = np.sqrt(mse_grid[cfg_name][h][c] / max(count_grid[h], 1))
                rmse_p = persist_rmse.get((h, c), 1e-8)
                sk = (1.0 - rmse_m / rmse_p) * 100 if rmse_p > 1e-8 else 0
                skills_ch.append(sk)
            vals.append(np.mean(skills_ch))
        avg = np.mean(vals)
        line = f"{cfg_name:>30}"
        for v in vals:
            line += f"  {v:6.1f}%"
        line += f" {avg:6.1f}%"
        print(line)

    # --- Per-channel RMSE for key variables ---
    print("\n" + "=" * 90)
    print("═══ Per-variable RMSE (physical units, best config highlighted) ═══")
    print("=" * 90)

    key_vars = ["t2m", "10u", "10v", "msl", "t@850", "z@500"]
    print(f"{'Variable':>10}", end="")
    for cfg in ["GNN", "GNN+lapse+MOS+IDW"]:
        if cfg in mse_grid:
            for h in hours:
                print(f" {cfg}+{h}h", end="")
    if unet_model:
        for h in hours:
            print(f" Cascade+{h}h", end="")
    print()

    for var in key_vars:
        if var not in VAR_ORDER:
            continue
        cidx = VAR_ORDER.index(var)
        print(f"{var:>10}", end="")
        for cfg_name in ["GNN", "GNN+lapse+MOS+IDW"]:
            if cfg_name not in mse_grid:
                continue
            for h in range(AR):
                rmse = np.sqrt(mse_grid[cfg_name][h][cidx] / max(count_grid[h], 1))
                if var == "t2m" or var.startswith("t@"):
                    print(f" {rmse:9.2f}", end="")
                elif "z@" in var:
                    print(f" {rmse/9.81:8.1f}m", end="")
                else:
                    print(f" {rmse:9.3f}", end="")
        if unet_model and "Cascade" in mse_grid:
            for h in range(AR):
                rmse = np.sqrt(mse_grid["Cascade"][h][cidx] / max(count_grid[h], 1))
                if var == "t2m" or var.startswith("t@"):
                    print(f" {rmse:9.2f}", end="")
                elif "z@" in var:
                    print(f" {rmse/9.81:8.1f}m", end="")
                else:
                    print(f" {rmse:9.3f}", end="")
        print()

    # --- Save results JSON ---
    results = {}
    for cfg_name in all_configs:
        if cfg_name not in mse_grid:
            continue
        results[cfg_name] = {}
        for h in range(AR):
            horizon = f"+{hours[h]}h"
            results[cfg_name][horizon] = {}
            for c in range(C):
                var = VAR_ORDER[c] if c < len(VAR_ORDER) else f"ch{c}"
                rmse_grid_v = float(np.sqrt(mse_grid[cfg_name][h][c] / max(count_grid[h], 1)))
                rmse_stn_v = float(np.sqrt(mse_stn[cfg_name][h][c] / max(count_stn[h], 1)))
                results[cfg_name][horizon][var] = {
                    "rmse_grid": rmse_grid_v,
                    "rmse_station": rmse_stn_v,
                }

    out_path = Path(args.gnn_exp) / "pipeline_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()

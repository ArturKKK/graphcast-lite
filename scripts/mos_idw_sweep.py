#!/usr/bin/env python3
"""
MOS/IDW parameter sweep: test different IDW power and max_radius_km values.

Runs evaluate_full_pipeline with varying IDW parameters to find optimal
post-processing settings. Uses the same GNN model (freeze6) as DA experiments.

Usage:
  python scripts/mos_idw_sweep.py --max-samples 50 --ar-steps 4
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
from src.utils import load_from_json_file

# ─── Import helpers from evaluate_full_pipeline ───
# Add scripts/ to path so we can import evaluate_full_pipeline as a module
_scripts_dir = str(REPO_ROOT / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from evaluate_full_pipeline import (
    ROI, LAPSE_RATE, MOS_STATIONS, VAR_ORDER, MEAN_STATION_ELEV,
    build_multires_coords, build_multires_frame, apply_lapse,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-exp", default="experiments/multires_nores_freeze6")
    ap.add_argument("--global-dir", default="data/datasets/global_512x256_19f_2010-2021_07deg")
    ap.add_argument("--region-dir", default="data/datasets/region_krsk_61x41_19f_2010-2020_025deg")
    ap.add_argument("--learned-mos", default="live_runtime_bundle/learned_mos_t2m_19stations.joblib")
    ap.add_argument("--ar-steps", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--lapse-elev", type=float, default=MEAN_STATION_ELEV)
    args = ap.parse_args()

    set_random_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AR = args.ar_steps
    OBS = 2

    print("=" * 70)
    print("MOS/IDW PARAMETER SWEEP")
    print("=" * 70)

    # ── Load datasets ──
    print("\n[1] Loading datasets...")
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

    # ── Build multires coordinates ──
    print("[2] Building multires coordinates...")
    flat_lats, flat_lons, region_mask, keep_global, n_global_kept = \
        build_multires_coords(g_lats, g_lons, r_lats, r_lons, ROI)
    N_total = len(flat_lats)
    n_regional = region_mask.sum()
    n_lat_fine = len(r_lats)
    n_lon_fine = len(r_lons)
    G_fine = n_lat_fine * n_lon_fine

    # Regional flat coordinates
    r_lon_mesh, r_lat_mesh = np.meshgrid(r_lons, r_lats)
    r_flat_lats = r_lat_mesh.ravel().astype(np.float32)
    r_flat_lons = r_lon_mesh.ravel().astype(np.float32)

    # ── Load GNN ──
    print("[3] Loading GNN model...")
    gnn_cfg = ExperimentConfig(**load_from_json_file(
        os.path.join(args.gnn_exp, FileNames.EXPERIMENT_CONFIG)
    ))
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
    print(f"  GNN loaded on {device}")

    # ── Load MOS ──
    mos_bundle = load_learned_mos(args.learned_mos)
    print(f"  MOS loaded: test_mae={mos_bundle.get('test_mae', '?')}°C")

    # ── Prepare test samples ──
    print("[4] Preparing test samples...")
    n_test_total = int(T_overlap * 0.2)
    n_val = n_test_total // 2
    test_start = T_overlap - n_test_total + n_val
    test_end = T_overlap
    window = OBS + AR
    valid_indices = list(range(test_start, test_end - window + 1))
    max_samples = min(args.max_samples, len(valid_indices))
    if max_samples < len(valid_indices):
        step_sz = len(valid_indices) // max_samples
        sample_starts = [valid_indices[i * step_sz] for i in range(max_samples)]
    else:
        sample_starts = valid_indices[:max_samples]
    print(f"  {max_samples} samples, AR={AR}")

    use_residual = gnn_cfg.use_residual if hasattr(gnn_cfg, 'use_residual') else False
    base_time = datetime(2020, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

    # ── IDW parameter grid ──
    idw_configs = [
        # (power, max_radius_km, label)
        (2.0, 300.0, "p2.0_r300"),   # current default
        (2.0, 150.0, "p2.0_r150"),
        (2.0, 100.0, "p2.0_r100"),
        (2.0, 200.0, "p2.0_r200"),
        (2.0, 50.0,  "p2.0_r50"),
        (3.0, 300.0, "p3.0_r300"),
        (3.0, 150.0, "p3.0_r150"),
        (3.0, 100.0, "p3.0_r100"),
        (1.5, 300.0, "p1.5_r300"),
        (1.5, 150.0, "p1.5_r150"),
    ]

    # Also test a Gaussian kernel (OI-style) replacement
    # We'll do this by comparing IDW vs no-IDW (station-only MOS)

    configs = ["GNN_raw", "GNN+lapse", "GNN+lapse+MOS_station"]
    for _, _, label in idw_configs:
        configs.append(f"GNN+lapse+MOS+IDW_{label}")

    t2m_idx = VAR_ORDER.index("t2m")

    # Accumulators: config → [per-horizon MSE sum for t2m]
    mse_t2m = {cfg: [0.0] * AR for cfg in configs}
    mse_t2m["Persistence"] = [0.0] * AR
    count = [0] * AR

    # ── Run evaluation ──
    print("[5] Running inference + MOS/IDW sweep...")
    t0 = time.time()

    for si, t_start in enumerate(sample_starts):
        # Build input
        input_frames = []
        for t_off in range(OBS):
            t = t_start + t_off
            frame = build_multires_frame(
                global_data, g_lats, g_lons, r_lats, r_lons,
                keep_global, n_global_kept, n_regional, C,
                t_global=t, t_regional=t,
                regional_data=regional_data,
            )
            frame_norm = (frame - y_mean) / y_std
            input_frames.append(frame_norm)

        X_np = np.stack(input_frames, axis=1)
        curr_state = torch.from_numpy(X_np).unsqueeze(0).float().to(device)

        # Persistence
        t_last_obs = t_start + OBS - 1
        persist_frame = np.array(regional_data[t_last_obs], dtype=np.float32)
        persist_phys = persist_frame.transpose(1, 0, 2).reshape(-1, C)

        for ar_step in range(AR):
            t_target = t_start + OBS + ar_step
            if t_target >= T_overlap:
                break

            gt_frame = np.array(regional_data[t_target], dtype=np.float32)
            gt_phys = gt_frame.transpose(1, 0, 2).reshape(-1, C)
            gt_t2m = gt_phys[:, t2m_idx]

            valid_times = [base_time + timedelta(hours=6 * (ar_step + 1))]

            # GNN forward
            inp = curr_state.view(1, N_total, OBS * C)
            with torch.no_grad():
                pred = gnn_model(X=inp, attention_threshold=0.0)
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)
            gnn_out = (curr_state[:, :, -1, :] + pred) if use_residual else pred
            gnn_out_np = gnn_out[0].cpu().numpy()

            roi_pred_norm = gnn_out_np[region_mask]
            roi_pred_phys = roi_pred_norm * y_std + y_mean

            # GNN raw
            mse_t2m["GNN_raw"][ar_step] += np.sum((roi_pred_phys[:, t2m_idx] - gt_t2m) ** 2)

            # GNN + lapse
            lapse_3d = apply_lapse(roi_pred_phys[:, np.newaxis, :].copy(), VAR_ORDER, args.lapse_elev)
            lapse_flat = lapse_3d[:, 0, :]
            mse_t2m["GNN+lapse"][ar_step] += np.sum((lapse_flat[:, t2m_idx] - gt_t2m) ** 2)

            # GNN + lapse + MOS (station-only, no IDW)
            mos_3d = lapse_3d.copy()
            result = apply_learned_mos_t2m(
                mos_3d, VAR_ORDER, mos_bundle,
                r_flat_lats, r_flat_lons, valid_times,
                stations=MOS_STATIONS, spatial_idw=False,
            )
            if isinstance(result, tuple):
                mos_3d = result[0]
            mse_t2m["GNN+lapse+MOS_station"][ar_step] += np.sum((mos_3d[:, 0, t2m_idx] - gt_t2m) ** 2)

            # GNN + lapse + MOS + IDW (sweep parameters)
            for power, max_r, label in idw_configs:
                idw_3d = lapse_3d.copy()
                result = apply_learned_mos_t2m(
                    idw_3d, VAR_ORDER, mos_bundle,
                    r_flat_lats, r_flat_lons, valid_times,
                    stations=MOS_STATIONS, spatial_idw=True,
                    idw_power=power, idw_max_radius_km=max_r,
                )
                if isinstance(result, tuple):
                    idw_3d = result[0]
                cfg_name = f"GNN+lapse+MOS+IDW_{label}"
                mse_t2m[cfg_name][ar_step] += np.sum((idw_3d[:, 0, t2m_idx] - gt_t2m) ** 2)

            # Persistence
            mse_t2m["Persistence"][ar_step] += np.sum((persist_phys[:, t2m_idx] - gt_t2m) ** 2)

            count[ar_step] += G_fine

            # Update state for next AR step
            curr_state = torch.cat(
                [curr_state[:, :, 1:, :], gnn_out.unsqueeze(2)], dim=2,
            )

        if (si + 1) % 10 == 0 or si == 0:
            elapsed = time.time() - t0
            eta = elapsed / (si + 1) * (max_samples - si - 1)
            print(f"  [{si+1}/{max_samples}] ETA: {eta/60:.0f}min")

    total_time = time.time() - t0
    print(f"\n  Done in {total_time/60:.1f} minutes")

    # ── Results ──
    hours = [6 * (h + 1) for h in range(AR)]

    print("\n" + "=" * 90)
    print("═══ t2m RMSE (°C) — MOS/IDW Parameter Sweep ═══")
    print("=" * 90)

    header = f"{'Config':>35}" + "".join(f" {f'+{h}h':>8}" for h in hours) + "   avg"
    print(header)
    print("-" * len(header))

    all_cfgs = ["Persistence", "GNN_raw", "GNN+lapse", "GNN+lapse+MOS_station"] + \
               [f"GNN+lapse+MOS+IDW_{label}" for _, _, label in idw_configs]

    for cfg_name in all_cfgs:
        vals = []
        for h in range(AR):
            rmse = np.sqrt(mse_t2m[cfg_name][h] / max(count[h], 1))
            vals.append(rmse)
        avg = np.mean(vals)
        line = f"{cfg_name:>35}"
        for v in vals:
            line += f" {v:8.3f}"
        line += f" {avg:7.3f}"
        print(line)

    # Skill table
    print("\n" + "=" * 90)
    print("═══ t2m Skill vs Persistence (%) ═══")
    print("=" * 90)
    header2 = f"{'Config':>35}" + "".join(f" {f'+{h}h':>8}" for h in hours) + "   avg"
    print(header2)
    print("-" * len(header2))

    persist_rmse = []
    for h in range(AR):
        persist_rmse.append(np.sqrt(mse_t2m["Persistence"][h] / max(count[h], 1)))

    for cfg_name in all_cfgs:
        if cfg_name == "Persistence":
            continue
        vals = []
        for h in range(AR):
            rmse = np.sqrt(mse_t2m[cfg_name][h] / max(count[h], 1))
            skill = (1.0 - rmse / persist_rmse[h]) * 100 if persist_rmse[h] > 1e-8 else 0
            vals.append(skill)
        avg = np.mean(vals)
        line = f"{cfg_name:>35}"
        for v in vals:
            line += f"  {v:6.2f}%"
        line += f" {avg:6.2f}%"
        print(line)

    print("\n" + "=" * 90)
    print("═══ Best IDW config per horizon ═══")
    print("=" * 90)
    for h in range(AR):
        best_cfg = None
        best_rmse = 9999.0
        for power, max_r, label in idw_configs:
            cfg_name = f"GNN+lapse+MOS+IDW_{label}"
            rmse = np.sqrt(mse_t2m[cfg_name][h] / max(count[h], 1))
            if rmse < best_rmse:
                best_rmse = rmse
                best_cfg = label
        print(f"  +{hours[h]}h: {best_cfg} → t2m RMSE = {best_rmse:.3f}°C")


if __name__ == "__main__":
    main()

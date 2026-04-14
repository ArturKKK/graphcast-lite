#!/usr/bin/env python3
"""
MOS/IDW parameter sweep using the merged multires dataset directly.

Uses the same data loading as predict.py (ChunkDataset from merged dataset)
to ensure model/graph compatibility. Only evaluates t2m on regional nodes.

Usage:
  python scripts/mos_idw_sweep_v2.py --max-samples 50 --ar-steps 4
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
from src.data.dataloader_chunked import load_chunked_datasets
from src.main import load_model_from_experiment_config
from src.postprocessing.mos_correction import (
    apply_learned_mos_t2m,
    load_learned_mos,
)
from src.utils import load_from_json_file

# ─── Constants ───
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

MEAN_STATION_ELEV = float(np.mean([s["elev"] for s in MOS_STATIONS]))

VAR_ORDER = [
    "t2m", "10u", "10v", "msl", "tp", "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
]


def apply_lapse_correction(pred_phys, var_order, target_elev, z_surf_field):
    """Apply lapse-rate correction: adjust t2m based on elevation difference."""
    if "t2m" not in var_order or "z_surf" not in var_order:
        return pred_phys
    t2m_idx = var_order.index("t2m")
    z_idx = var_order.index("z_surf")
    corrected = pred_phys.copy()
    # z_surf_field is geopotential (m²/s²), elevation = z_surf / 9.80665
    model_elev = z_surf_field / 9.80665  # (G,) in meters
    delta_h = model_elev - target_elev  # positive = model higher than station
    corrected[:, :, t2m_idx] += LAPSE_RATE * delta_h[:, None]  # warm up if model is higher
    return corrected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-exp", default="experiments/multires_nores_freeze6")
    ap.add_argument("--learned-mos", default="live_runtime_bundle/learned_mos_t2m_19stations.joblib")
    ap.add_argument("--ar-steps", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--lapse-elev", type=float, default=None)
    args = ap.parse_args()

    if args.lapse_elev is None:
        args.lapse_elev = MEAN_STATION_ELEV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AR = args.ar_steps

    print("=" * 70)
    print("MOS/IDW PARAMETER SWEEP (v2 — merged dataset)")
    print("=" * 70)

    # ── Load experiment config ──
    exp_dir = args.gnn_exp
    cfg_path = os.path.join(exp_dir, FileNames.EXPERIMENT_CONFIG)
    ckpt_path = os.path.join(exp_dir, "best_model.pth")
    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))

    data_dir = Path(exp_cfg.data_dir) if hasattr(exp_cfg, 'data_dir') and exp_cfg.data_dir else None
    assert data_dir and data_dir.exists(), f"data_dir not found: {data_dir}"

    # ── Load dataset (same as predict.py) ──
    print("\n[1] Loading dataset from", data_dir)
    ds_info = json.load(open(data_dir / "dataset_info.json"))
    coords_npz = np.load(data_dir / "coords.npz")
    flat_lats = coords_npz["latitude"].astype(np.float32)
    flat_lons = coords_npz["longitude"].astype(np.float32)
    is_regional = coords_npz.get("is_regional", None)
    if is_regional is not None:
        is_regional = is_regional.astype(bool)
    else:
        is_regional = np.ones(len(flat_lats), dtype=bool)

    region_idx = np.where(is_regional)[0]
    n_regional = len(region_idx)
    N_total = len(flat_lats)
    C = ds_info["n_feat"]
    OBS = 2  # observation window

    print(f"  N_total={N_total}, N_regional={n_regional}, C={C}")
    print(f"  mode={ds_info.get('mode', '?')}, n_time={ds_info['n_time']}")

    # Load scalers for physical unit conversion
    scalers_path = data_dir / "scalers.npz"
    if scalers_path.exists():
        sc = np.load(scalers_path)
        y_mean = sc["mean"][:C].astype(np.float32)
        y_std = sc["std"][:C].astype(np.float32)
    else:
        y_mean = np.zeros(C, dtype=np.float32)
        y_std = np.ones(C, dtype=np.float32)

    # ── Load ChunkDataset ──
    print("[2] Loading ChunkDataset...")
    train_ds, val_ds, test_ds, metadata = load_chunked_datasets(
        data_path=str(data_dir),
        obs_window=OBS,
        pred_steps=1,
        n_features=C,
        test_split="test_only",
    )

    # ── Load GNN ──
    print("[3] Loading GNN model...")
    ROI = (50.0, 60.0, 83.0, 98.0)
    gnn_model = load_model_from_experiment_config(
        exp_cfg, device, metadata,
        coordinates=(flat_lats, flat_lons),
        region_bounds=ROI, mesh_buffer=15.0, flat_grid=True,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Pruned mesh changes edge features — remove from state_dict (recomputed at init)
    state = {k: v for k, v in state.items()
             if not k.startswith("_processing_edge_features")}
    gnn_model.load_state_dict(state, strict=False)
    gnn_model = gnn_model.to(device)  # ensure all buffers (edge features) on GPU
    gnn_model.eval()
    print(f"  GNN loaded on {device}")

    # ── Load MOS ──
    mos_bundle = load_learned_mos(args.learned_mos)
    print(f"  MOS loaded: test_mae={mos_bundle.get('test_mae', '?')}°C")

    # ── Prepare samples ──
    print("[4] Preparing test samples...")
    max_samples = min(args.max_samples, len(test_ds))
    if max_samples < len(test_ds):
        step_sz = len(test_ds) // max_samples
        sample_indices = [i * step_sz for i in range(max_samples)]
    else:
        sample_indices = list(range(max_samples))
    print(f"  {max_samples} samples, AR={AR}")

    use_residual = not True  # freeze6 is --no-residual
    if hasattr(exp_cfg, 'use_residual'):
        use_residual = exp_cfg.use_residual

    # ── IDW parameter grid ──
    idw_configs = [
        (2.0, 300.0, "p2.0_r300"),   # current default
        (2.0, 200.0, "p2.0_r200"),
        (2.0, 150.0, "p2.0_r150"),
        (2.0, 100.0, "p2.0_r100"),
        (2.0, 50.0,  "p2.0_r50"),
        (3.0, 300.0, "p3.0_r300"),
        (3.0, 150.0, "p3.0_r150"),
        (3.0, 100.0, "p3.0_r100"),
        (1.5, 300.0, "p1.5_r300"),
        (1.5, 150.0, "p1.5_r150"),
    ]

    configs = ["GNN_raw", "GNN+lapse", "GNN+lapse+MOS_station"]
    for _, _, label in idw_configs:
        configs.append(f"GNN+lapse+MOS+IDW_{label}")

    t2m_idx = VAR_ORDER.index("t2m")

    # Accumulators: config → [per-horizon SE sum for t2m, regional nodes only]
    se_t2m = {cfg: [0.0] * AR for cfg in configs}
    se_t2m["Persistence"] = [0.0] * AR
    count = [0] * AR

    # Regional coordinates
    reg_lats = flat_lats[region_idx]
    reg_lons = flat_lons[region_idx]

    # z_surf for lapse rate
    z_surf_idx = VAR_ORDER.index("z_surf") if "z_surf" in VAR_ORDER else None

    base_time = datetime(2020, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

    # ── Run evaluation ──
    print("[5] Running inference + MOS/IDW sweep...")
    t0 = time.time()

    for si, ds_idx in enumerate(sample_indices):
        X_flat, Y_flat = test_ds[ds_idx]
        # X_flat: (N, OBS*C) normalized, Y_flat: (N, 1*C) normalized
        X_np = X_flat.numpy()  # (N, OBS*C)
        Y_np = Y_flat.numpy()  # (N, C)

        curr_state = X_flat.unsqueeze(0).float().to(device)  # (1, N, OBS*C)

        # Persistence: last obs step (last C features of X), regional nodes
        persist_norm = X_np[region_idx, -C:]  # (n_regional, C)
        persist_phys_t2m = persist_norm[:, t2m_idx] * y_std[t2m_idx] + y_mean[t2m_idx]

        # Ground truth: Y is (N, C) normalized — only +6h available
        gt_norm = Y_np[region_idx, :]  # (n_regional, C)
        gt_phys_t2m = gt_norm[:, t2m_idx] * y_std[t2m_idx] + y_mean[t2m_idx]

        valid_times = [base_time + timedelta(hours=6)]

        # GNN forward
        inp = curr_state  # already (1, N, OBS*C)
        with torch.no_grad():
            pred = gnn_model(X=inp, attention_threshold=0.0)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if use_residual:
            last_state = curr_state[:, :, -C:]
            gnn_out = last_state + pred
        else:
            gnn_out = pred
        gnn_out_np = gnn_out[0].cpu().numpy()  # (N, C) normalized

        # Physical units for regional nodes
        reg_pred_norm = gnn_out_np[region_idx]  # (n_regional, C)
        reg_pred_phys = reg_pred_norm * y_std + y_mean  # (n_regional, C)

        # GNN raw t2m
        se_t2m["GNN_raw"][0] += float(np.sum((reg_pred_phys[:, t2m_idx] - gt_phys_t2m) ** 2))

        # Lapse correction
        pred_3d = reg_pred_phys[:, np.newaxis, :]  # (n_regional, 1, C)
        if z_surf_idx is not None:
            z_surf_field = reg_pred_phys[:, z_surf_idx]
            lapse_3d = apply_lapse_correction(pred_3d, VAR_ORDER, args.lapse_elev, z_surf_field)
        else:
            lapse_3d = pred_3d.copy()
        se_t2m["GNN+lapse"][0] += float(np.sum((lapse_3d[:, 0, t2m_idx] - gt_phys_t2m) ** 2))

        # MOS station-only (no IDW)
        mos_3d = lapse_3d.copy()
        result = apply_learned_mos_t2m(
            mos_3d, VAR_ORDER, mos_bundle,
            reg_lats, reg_lons, valid_times,
            stations=MOS_STATIONS, spatial_idw=False,
        )
        if isinstance(result, tuple):
            mos_3d = result[0]
        se_t2m["GNN+lapse+MOS_station"][0] += float(np.sum((mos_3d[:, 0, t2m_idx] - gt_phys_t2m) ** 2))

        # IDW sweep
        for power, max_r, label in idw_configs:
            idw_3d = lapse_3d.copy()
            result = apply_learned_mos_t2m(
                idw_3d, VAR_ORDER, mos_bundle,
                reg_lats, reg_lons, valid_times,
                stations=MOS_STATIONS, spatial_idw=True,
                idw_power=power, idw_max_radius_km=max_r,
            )
            if isinstance(result, tuple):
                idw_3d = result[0]
            cfg_name = f"GNN+lapse+MOS+IDW_{label}"
            se_t2m[cfg_name][0] += float(np.sum((idw_3d[:, 0, t2m_idx] - gt_phys_t2m) ** 2))

        # Persistence
        se_t2m["Persistence"][0] += float(np.sum((persist_phys_t2m - gt_phys_t2m) ** 2))

        count[0] += n_regional

        if (si + 1) % 10 == 0 or si == 0:
            elapsed = time.time() - t0
            eta = elapsed / (si + 1) * (max_samples - si - 1)
            print(f"  [{si+1}/{max_samples}] ETA: {eta/60:.0f}min")

    total_time = time.time() - t0
    print(f"\n  Done in {total_time/60:.1f} minutes")

    # ── Results ──
    print("\n" + "=" * 90)
    print("═══ t2m RMSE (°C) on Regional Nodes — MOS/IDW Parameter Sweep ═══")
    print("=" * 90)

    AR_used = max(h for h in range(AR) if count[h] > 0) + 1
    hours = [6 * (h + 1) for h in range(AR_used)]

    header = f"{'Config':>35}" + "".join(f" {f'+{h}h':>8}" for h in hours)
    print(header)
    print("-" * len(header))

    all_cfgs = ["Persistence", "GNN_raw", "GNN+lapse", "GNN+lapse+MOS_station"] + \
               [f"GNN+lapse+MOS+IDW_{label}" for _, _, label in idw_configs]

    results = {}
    for cfg_name in all_cfgs:
        vals = []
        for h in range(AR_used):
            rmse = np.sqrt(se_t2m[cfg_name][h] / max(count[h], 1))
            vals.append(rmse)
        results[cfg_name] = vals
        line = f"{cfg_name:>35}"
        for v in vals:
            line += f" {v:8.3f}"
        print(line)

    # Skill relative to persistence
    print("\n" + "=" * 90)
    print("═══ t2m Skill vs Persistence (%) ═══")
    print("=" * 90)
    header2 = f"{'Config':>35}" + "".join(f" {f'+{h}h':>8}" for h in hours)
    print(header2)
    print("-" * len(header2))

    persist_rmse = results["Persistence"]
    for cfg_name in all_cfgs:
        if cfg_name == "Persistence":
            continue
        vals = []
        for h in range(AR_used):
            skill = (1.0 - results[cfg_name][h] / persist_rmse[h]) * 100 if persist_rmse[h] > 1e-8 else 0
            vals.append(skill)
        line = f"{cfg_name:>35}"
        for v in vals:
            line += f"  {v:6.2f}%"
        print(line)

    # Best config
    print("\n" + "=" * 90)
    print("═══ Best IDW config ═══")
    print("=" * 90)
    for h in range(AR_used):
        best_cfg = None
        best_rmse = 9999.0
        for power, max_r, label in idw_configs:
            cfg_name = f"GNN+lapse+MOS+IDW_{label}"
            rmse = results[cfg_name][h]
            if rmse < best_rmse:
                best_rmse = rmse
                best_cfg = label
        print(f"  +{hours[h]}h: {best_cfg} → t2m RMSE = {best_rmse:.3f}°C")


if __name__ == "__main__":
    main()

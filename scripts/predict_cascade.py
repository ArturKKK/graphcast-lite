#!/usr/bin/env python3
"""
scripts/predict_cascade.py

Каскадный инференс: GNN (глобальный) → UNet (региональный downscaler).

Шаги:
  1. GNN предсказывает на multires/глобальном гриде (авторегрессивно)
  2. Извлекаем ROI из предсказания
  3. Билинейно upsampl'им до 0.25°
  4. UNet downscaler уточняет

Для сравнения считает метрики:
  - GNN-only (coarse, upsampled)
  - GNN + UNet (cascade)
  - Оба против real 0.25° ERA5 (если доступно)

Usage:
  python scripts/predict_cascade.py \
    experiments/multires_nores_freeze6 \
    experiments/downscaler_krsk \
    --downscaler-data /data/datasets/downscaler_krsk_19f \
    --gnn-data /data/datasets/multires_krsk_19f \
    --roi 50 60 83 98 \
    --ar-steps 4 \
    --max-samples 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import FileNames
from src.config import ExperimentConfig
from src.utils import load_from_json_file
from src.data.dataloader_chunked import load_chunked_datasets
from src.main import load_model_from_experiment_config, set_random_seeds
from src.unet.model import WeatherUNet


def crop_and_upsample_roi(
    pred_flat: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    roi: tuple,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    flat_grid: bool = True,
):
    """
    Извлекает ROI из GNN-предсказания и билинейно upsampl'ит до target grid.

    pred_flat: (G, C) — предсказание GNN на flat/regular grid
    Returns: (n_lat_fine, n_lon_fine, C) — upsampled prediction
    """
    lat_min, lat_max, lon_min, lon_max = roi
    C = pred_flat.shape[-1]

    if flat_grid:
        # Для flat grid: извлекаем точки внутри ROI, интерполируем на fine grid
        roi_mask = (
            (grid_lats >= lat_min) & (grid_lats <= lat_max) &
            (grid_lons >= lon_min) & (grid_lons <= lon_max)
        )
        roi_lats = grid_lats[roi_mask]
        roi_lons = grid_lons[roi_mask]
        roi_vals = pred_flat[roi_mask]  # (N_roi, C)

        # Для регулярного субгрида в ROI: определяем уникальные lat/lon
        u_lats = np.unique(roi_lats)
        u_lons = np.unique(roi_lons)

        # Reshape в 2D grid (lat, lon, C) для RegularGridInterpolator
        grid_2d = np.full((len(u_lats), len(u_lons), C), np.nan, dtype=np.float32)
        lat_idx_map = {v: i for i, v in enumerate(u_lats)}
        lon_idx_map = {v: i for i, v in enumerate(u_lons)}
        for k in range(len(roi_lats)):
            li = lat_idx_map.get(roi_lats[k])
            lo = lon_idx_map.get(roi_lons[k])
            if li is not None and lo is not None:
                grid_2d[li, lo] = roi_vals[k]

        # Fill NaN with nearest (safety)
        from scipy.ndimage import generic_filter
        for c in range(C):
            mask = np.isnan(grid_2d[:, :, c])
            if mask.any():
                grid_2d[:, :, c][mask] = np.nanmean(grid_2d[:, :, c])

    else:
        # Regular grid: simple slice
        lat_idx = np.where((grid_lats >= lat_min - 1) & (grid_lats <= lat_max + 1))[0]
        lon_idx = np.where((grid_lons >= lon_min - 1) & (grid_lons <= lon_max + 1))[0]
        u_lats = grid_lats[lat_idx]
        u_lons = grid_lons[lon_idx]
        # pred_flat is (G, C) with G = n_lat * n_lon, lat-major
        n_lat_total = len(grid_lats)
        pred_2d = pred_flat.reshape(len(grid_lons), n_lat_total, C)  # approx
        grid_2d = pred_2d[np.ix_(lon_idx, lat_idx)].transpose(1, 0, 2)  # (lat, lon, C)

    # Interpolate each channel to fine grid
    fine_lon_grid, fine_lat_grid = np.meshgrid(target_lons, target_lats)
    fine_points = np.stack([fine_lat_grid.ravel(), fine_lon_grid.ravel()], axis=-1)

    result = np.zeros((len(target_lats), len(target_lons), C), dtype=np.float32)
    for c in range(C):
        interp = RegularGridInterpolator(
            (u_lats, u_lons), grid_2d[:, :, c],
            method="linear", bounds_error=False, fill_value=None
        )
        result[:, :, c] = interp(fine_points).reshape(len(target_lats), len(target_lons))

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gnn_experiment", help="GNN experiment dir (e.g. experiments/multires_nores_freeze6)")
    ap.add_argument("ds_experiment", help="Downscaler experiment dir")
    ap.add_argument("--downscaler-data", required=True, help="Downscaler dataset dir")
    ap.add_argument("--gnn-data", default=None, help="GNN dataset dir (override config)")
    ap.add_argument("--roi", nargs=4, type=float, default=[50, 60, 83, 98])
    ap.add_argument("--ar-steps", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--no-residual", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(42)

    roi = tuple(args.roi)

    # ── 1. Load GNN model ──
    gnn_dir = Path(args.gnn_experiment)
    gnn_cfg = ExperimentConfig(**load_from_json_file(gnn_dir / FileNames.EXPERIMENT_CONFIG))
    gnn_data_dir = args.gnn_data or getattr(gnn_cfg, 'data_dir', None)
    if gnn_data_dir is None:
        gnn_data_dir = f"data/datasets/{gnn_cfg.data.dataset_name.value}"

    _, _, test_ds, meta = load_chunked_datasets(
        data_path=gnn_data_dir,
        obs_window=gnn_cfg.data.obs_window_used,
        pred_steps=args.ar_steps,
        n_features=gnn_cfg.data.num_features_used,
    )

    flat_grid = getattr(meta, 'flat_grid', False)
    real_coords = getattr(meta, 'cordinates', None)
    gnn_model = load_model_from_experiment_config(
        gnn_cfg, device, meta,
        coordinates=real_coords, flat_grid=flat_grid,
    )

    # Load GNN weights
    gnn_ckpt = gnn_dir / "best_model.pth"
    if not gnn_ckpt.exists():
        gnn_ckpt = gnn_dir / "checkpoint.pth"
    state = torch.load(gnn_ckpt, map_location=device)
    gnn_model.load_state_dict(state, strict=False)
    gnn_model = gnn_model.to(device)  # move registered buffers (edge features) to GPU
    gnn_model.eval()
    print(f"[GNN] Loaded from {gnn_ckpt}")

    # Grid coordinates
    if flat_grid and real_coords is not None:
        grid_lats = real_coords[0].astype(np.float32)
        grid_lons = real_coords[1].astype(np.float32)
    else:
        lats = np.linspace(-90, 90, meta.num_latitudes, endpoint=True)
        lons = np.linspace(0, 360, meta.num_longitudes, endpoint=False)
        lon_g, lat_g = np.meshgrid(lons, lats)
        grid_lats = lat_g.flatten().astype(np.float32)
        grid_lons = lon_g.flatten().astype(np.float32)

    # ── 2. Load UNet downscaler ──
    ds_dir = Path(args.ds_experiment)
    ds_data_dir = Path(args.downscaler_data)

    with open(ds_data_dir / "dataset_info.json") as f:
        ds_info = json.load(f)
    ds_coords = np.load(ds_data_dir / "coords.npz")
    target_lats = ds_coords["latitude"].astype(np.float64)
    target_lons = ds_coords["longitude"].astype(np.float64)
    n_lat_fine = ds_info["n_lat"]
    n_lon_fine = ds_info["n_lon"]
    C = ds_info["n_feat"]

    # Load scalers
    sc = np.load(ds_data_dir / "scalers.npz")
    ds_mean = sc["mean"].astype(np.float32)
    ds_std = sc["std"].astype(np.float32)

    # Static fields
    static_fine = None
    static_path = ds_data_dir / "static_fine.npy"
    static_ch = ds_info.get("static_channels", [])
    n_static = 0
    if static_path.exists():
        static_fine = np.load(static_path)  # (n_lon, n_lat, N_static)
        static_fine = static_fine.transpose(1, 0, 2)  # (H, W, N_static)
        n_static = static_fine.shape[2]

    # Load UNet
    ds_cfg_path = ds_dir / "config.json"
    ds_cfg = json.load(open(ds_cfg_path)) if ds_cfg_path.exists() else {}
    obs_window = ds_cfg.get("obs_window", 2)
    base_filters = ds_cfg.get("base_filters", 64)
    ds_residual = ds_cfg.get("residual", False)

    in_channels = obs_window * C + n_static
    unet = WeatherUNet(in_channels=in_channels, out_channels=C,
                       base_filters=base_filters).to(device)
    unet_ckpt = ds_dir / "best_model.pth"
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
    unet.eval()
    print(f"[UNet] Loaded from {unet_ckpt}, params={unet.num_params:,}")

    # ── 3. Load real fine targets ──
    fine_data = np.memmap(ds_data_dir / "fine.npy", dtype=np.float16, mode="r",
                          shape=(ds_info["n_time"], n_lon_fine, n_lat_fine, C))

    # ── 4. Load variables ──
    vars_path = ds_data_dir / "variables.json"
    var_names = json.load(open(vars_path)) if vars_path.exists() else [f"ch{i}" for i in range(C)]

    # ── 5. Run inference ──
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    use_residual = not args.no_residual and getattr(gnn_cfg, 'use_residual', False)

    mse_gnn = np.zeros(C)
    mse_cascade = np.zeros(C)
    n_samples = 0

    print(f"\n[Inference] AR={args.ar_steps}, samples={min(args.max_samples, len(test_ds))}, "
          f"gnn_residual={use_residual}, unet_residual={ds_residual}")

    for i, (X, Y) in enumerate(test_loader):
        if i >= args.max_samples:
            break

        X, Y = X.to(device), Y.to(device)
        N, G, feat_dim = X.shape
        obs = gnn_cfg.data.obs_window_used
        C_feat = feat_dim // obs
        curr_state = X.view(N, G, obs, C_feat)

        # AR rollout
        for step in range(args.ar_steps):
            inp = curr_state.view(N, G, -1)
            with torch.no_grad():
                pred = gnn_model(X=inp, attention_threshold=0.0)
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)
            if use_residual:
                out = curr_state[:, :, -1, :] + pred
            else:
                out = pred
            curr_state = torch.cat([curr_state[:, :, 1:, :], out.unsqueeze(2)], dim=2)

        # GNN final prediction: (1, G, C)
        gnn_pred = out[0].cpu().numpy()  # (G, C)

        # Crop + upsample to fine grid
        coarse_up = crop_and_upsample_roi(
            gnn_pred, grid_lats, grid_lons, roi,
            target_lats, target_lons, flat_grid=flat_grid,
        )  # (H, W, C)

        # ── UNet downscaling ──
        # Normalize
        coarse_norm = (coarse_up - ds_mean) / (ds_std + 1e-8)
        # For obs_window: repeat the same frame (we only have 1 prediction)
        frames = [coarse_norm] * obs_window
        coarse_stack = np.concatenate(frames, axis=-1)  # (H, W, obs*C)
        x_unet = torch.from_numpy(coarse_stack).permute(2, 0, 1).unsqueeze(0).float().to(device)

        if static_fine is not None:
            static_t = torch.from_numpy(static_fine.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)
            x_unet = torch.cat([x_unet, static_t], dim=1)

        with torch.no_grad():
            unet_out = unet(x_unet)  # (1, C, H, W)

        if ds_residual:
            # UNet predicts delta — add last coarse frame (normalized)
            x_last = x_unet[:, (obs_window - 1) * C : obs_window * C, :, :]
            unet_out = x_last + unet_out

        cascade_norm = unet_out[0].cpu().numpy()  # (C, H, W)
        cascade_norm = cascade_norm.transpose(1, 2, 0)  # (H, W, C)

        # Denormalize
        cascade_phys = cascade_norm * (ds_std + 1e-8) + ds_mean  # (H, W, C)

        # ── Target: real fine ERA5 ──
        # We need to find the matching timestep in fine_data
        # Use the test dataset index to map back
        # For simplicity, use the aligned time step
        t_idx = test_ds.indices[i] if hasattr(test_ds, 'indices') else i
        # The target is the last AR step
        t_target = t_idx + obs - 1 + args.ar_steps
        if t_target >= ds_info["n_time"]:
            continue

        fine_target = np.array(fine_data[t_target], dtype=np.float32)  # (n_lon, n_lat, C)
        fine_target = fine_target.transpose(1, 0, 2)  # (H, W, C)

        # ── Metrics ──
        for c in range(C):
            if c in static_ch:
                continue
            mse_gnn[c] += np.mean((coarse_up[:, :, c] - fine_target[:, :, c]) ** 2)
            mse_cascade[c] += np.mean((cascade_phys[:, :, c] - fine_target[:, :, c]) ** 2)

        n_samples += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{min(args.max_samples, len(test_ds))}")

    if n_samples == 0:
        print("No valid samples!")
        return

    # ── Results ──
    mse_gnn /= n_samples
    mse_cascade /= n_samples
    rmse_gnn = np.sqrt(mse_gnn)
    rmse_cascade = np.sqrt(mse_cascade)
    improvement = 1.0 - rmse_cascade / (rmse_gnn + 1e-8)

    print(f"\n{'='*60}")
    print(f"Results ({n_samples} samples, +{args.ar_steps*6}h)")
    print(f"{'='*60}")
    print(f"{'Variable':>12}  {'RMSE_GNN':>10}  {'RMSE_cascade':>12}  {'Improv':>8}")
    print("-" * 48)
    for c in range(C):
        if c in static_ch:
            continue
        print(f"{var_names[c]:>12}  {rmse_gnn[c]:10.4f}  {rmse_cascade[c]:12.4f}  {improvement[c]:>7.1%}")

    mean_imp = np.mean([improvement[c] for c in range(C) if c not in static_ch])
    print(f"\n{'MEAN':>12}  {'':>10}  {'':>12}  {mean_imp:>7.1%}")


if __name__ == "__main__":
    main()

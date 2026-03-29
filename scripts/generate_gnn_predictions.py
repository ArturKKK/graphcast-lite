#!/usr/bin/env python3
"""
scripts/generate_gnn_predictions.py

Прогоняет замороженную GNN на обучающем периоде и сохраняет
предсказания в ROI как memmap-файл gnn_pred.npy.

Это нужно для обучения UNet downscaler'а на GNN-выходе
вместо ERA5 coarse, чтобы закрыть domain gap.

Для каждого timestep t (obs_window..T):
  1. GNN получает obs_window кадров (t-obs..t-1) из глобального датасета
  2. Предсказывает следующий шаг → (G, C)
  3. Кропаем ROI, билинейно интерполируем до 0.25° → (61, 41, C)
  4. Сохраняем в gnn_pred.npy

Результат можно использовать как input для train_downscaler.py --gnn-input.

Usage:
  python scripts/generate_gnn_predictions.py \
    experiments/multires_nores_freeze6 \
    --data-dir data/datasets/multires_krsk_19f \
    --downscaler-data data/datasets/downscaler_krsk_19f \
    --roi 50 60 83 98 \
    --subsample 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import FileNames
from src.config import ExperimentConfig
from src.utils import load_from_json_file
from src.data.dataloader_chunked import load_chunked_datasets
from src.main import load_model_from_experiment_config, set_random_seeds


def crop_and_upsample_roi(
    pred_flat: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    roi: tuple,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    flat_grid: bool = True,
):
    """Exactly same logic as predict_cascade.py."""
    lat_min, lat_max, lon_min, lon_max = roi
    C = pred_flat.shape[-1]

    if flat_grid:
        roi_mask = (
            (grid_lats >= lat_min) & (grid_lats <= lat_max) &
            (grid_lons >= lon_min) & (grid_lons <= lon_max)
        )
        roi_lats = grid_lats[roi_mask]
        roi_lons = grid_lons[roi_mask]
        roi_vals = pred_flat[roi_mask]

        u_lats = np.unique(roi_lats)
        u_lons = np.unique(roi_lons)

        grid_2d = np.full((len(u_lats), len(u_lons), C), np.nan, dtype=np.float32)
        lat_idx_map = {v: i for i, v in enumerate(u_lats)}
        lon_idx_map = {v: i for i, v in enumerate(u_lons)}
        for k in range(len(roi_lats)):
            li = lat_idx_map.get(roi_lats[k])
            lo = lon_idx_map.get(roi_lons[k])
            if li is not None and lo is not None:
                grid_2d[li, lo] = roi_vals[k]

        for c in range(C):
            mask = np.isnan(grid_2d[:, :, c])
            if mask.any():
                grid_2d[:, :, c][mask] = np.nanmean(grid_2d[:, :, c])
    else:
        n_lat_total = len(np.unique(grid_lats))
        lat_idx = np.where((grid_lats >= lat_min - 1) & (grid_lats <= lat_max + 1))[0]
        lon_idx = np.where((grid_lons >= lon_min - 1) & (grid_lons <= lon_max + 1))[0]
        u_lats = np.unique(grid_lats[lat_idx])
        u_lons = np.unique(grid_lons[lon_idx])
        # approximate reshape
        pred_2d = pred_flat.reshape(len(np.unique(grid_lons)), n_lat_total, C)
        grid_2d = pred_2d[np.ix_(
            np.searchsorted(np.unique(grid_lons), u_lons),
            np.searchsorted(np.unique(grid_lats), u_lats)
        )].transpose(1, 0, 2)

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
    ap.add_argument("gnn_experiment", help="GNN experiment dir")
    ap.add_argument("--data-dir", default=None, help="GNN dataset dir (override config)")
    ap.add_argument("--downscaler-data", required=True, help="Downscaler dataset dir (target)")
    ap.add_argument("--roi", nargs=4, type=float, default=[50, 60, 83, 98])
    ap.add_argument("--subsample", type=int, default=1, help="Use every Nth sample")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_random_seeds(42)
    roi = tuple(args.roi)

    # ── 1. Load GNN ──
    gnn_dir = Path(args.gnn_experiment)
    gnn_cfg = ExperimentConfig(**load_from_json_file(gnn_dir / FileNames.EXPERIMENT_CONFIG))
    gnn_data_dir = args.data_dir or getattr(gnn_cfg, 'data_dir', None)
    if gnn_data_dir is None:
        gnn_data_dir = f"data/datasets/{gnn_cfg.data.dataset_name.value}"

    obs = gnn_cfg.data.obs_window_used
    C_feat = gnn_cfg.data.num_features_used

    train_ds, val_ds, test_ds, meta = load_chunked_datasets(
        data_path=gnn_data_dir,
        obs_window=obs,
        pred_steps=1,  # only 1-step predictions
        n_features=C_feat,
    )

    flat_grid = getattr(meta, 'flat_grid', False)
    real_coords = getattr(meta, 'cordinates', None)
    gnn_model = load_model_from_experiment_config(
        gnn_cfg, device, meta,
        coordinates=real_coords, flat_grid=flat_grid,
    )
    gnn_ckpt = gnn_dir / "best_model.pth"
    if not gnn_ckpt.exists():
        gnn_ckpt = gnn_dir / "checkpoint.pth"
    state = torch.load(gnn_ckpt, map_location=device, weights_only=False)
    gnn_model.load_state_dict(state, strict=False)
    gnn_model = gnn_model.to(device)
    gnn_model.eval()
    print(f"[GNN] Loaded from {gnn_ckpt}, obs={obs}, C={C_feat}")

    use_residual = getattr(gnn_cfg, 'use_residual', False)

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

    # ── 2. Load downscaler dataset info ──
    ds_dir = Path(args.downscaler_data)
    with open(ds_dir / "dataset_info.json") as f:
        ds_info = json.load(f)
    ds_coords = np.load(ds_dir / "coords.npz")
    target_lats = ds_coords["latitude"].astype(np.float64)
    target_lons = ds_coords["longitude"].astype(np.float64)
    n_lat_fine = ds_info["n_lat"]
    n_lon_fine = ds_info["n_lon"]
    C_ds = ds_info["n_feat"]
    T = ds_info["n_time"]

    print(f"[Target] T={T}, grid={n_lon_fine}×{n_lat_fine}, C={C_ds}")

    # ── 3. Create output memmap ──
    out_path = ds_dir / "gnn_pred.npy"
    gnn_pred = np.memmap(out_path, dtype=np.float16, mode="w+",
                         shape=(T, n_lon_fine, n_lat_fine, C_ds))

    # ── 4. Process all datasets (train+val+test) ──
    # We need to map dataset indices → global time indices
    # The DownscalerDataset uses indices [obs_window-1..T], so we need to
    # generate GNN predictions for the same timesteps

    from torch.utils.data import DataLoader

    all_datasets = [("train", train_ds), ("val", val_ds), ("test", test_ds)]
    total_generated = 0

    for split_name, ds in all_datasets:
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        n_samples = len(ds)
        print(f"\n[{split_name}] {n_samples} samples (subsample={args.subsample})")

        for i, (X_batch, Y_batch) in enumerate(loader):
            if i % args.subsample != 0:
                continue

            X_batch = X_batch.to(device)  # (1, G, obs*C)
            N, G, feat_dim = X_batch.shape

            with torch.no_grad():
                pred = gnn_model(X=X_batch, attention_threshold=0.0)

            if pred.dim() == 2:
                pred = pred.unsqueeze(0)

            if use_residual:
                C_f = feat_dim // obs
                curr_last = X_batch.view(N, G, obs, C_f)[:, :, -1, :]
                out = curr_last + pred
            else:
                out = pred

            # Crop to ROI and upsample
            gnn_out_np = out[0].cpu().numpy()  # (G, C)
            roi_fine = crop_and_upsample_roi(
                gnn_out_np, grid_lats, grid_lons, roi,
                target_lats, target_lons, flat_grid=flat_grid,
            )  # (n_lat, n_lon, C) — lat-first

            # Map to global timestep index
            # The chunk dataset stores sample indices internally
            if hasattr(ds, '_sample_indices') and i < len(ds._sample_indices):
                _chunk, local_t = ds._sample_indices[i]
                t_pred = local_t + obs  # prediction target time
            else:
                # Fallback: sequential indexing
                t_pred = obs + i

            if t_pred < T:
                # Store as (n_lon, n_lat, C) — lon-first to match downscaler format
                gnn_pred[t_pred] = roi_fine.transpose(1, 0, 2).astype(np.float16)
                total_generated += 1

            if (i + 1) % 200 == 0:
                print(f"  [{split_name}] {i+1}/{n_samples}")

    gnn_pred.flush()
    print(f"\n[Done] Generated {total_generated} GNN predictions → {out_path}")
    print(f"  Shape: ({T}, {n_lon_fine}, {n_lat_fine}, {C_ds})")
    print(f"  Note: timesteps without predictions remain zero-filled")


if __name__ == "__main__":
    main()

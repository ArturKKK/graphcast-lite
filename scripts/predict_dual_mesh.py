#!/usr/bin/env python3
"""
scripts/predict_dual_mesh.py

Инференс DualMeshModel: глобальная + региональная модель.

Использование:
  python scripts/predict_dual_mesh.py experiments/dual_mesh_krsk \
    --pretrained experiments/multires_nores_freeze6/results/best_model.pth \
    --regional-ckpt experiments/dual_mesh_krsk/results/best_regional.pth \
    --roi 50 60 83 98 \
    --data-dir data/datasets/multires_krsk_19f
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import FileNames
from src.config import ExperimentConfig
from src.utils import load_from_json_file
from src.data.dataloader_chunked import load_chunked_datasets
from src.main import load_model_from_experiment_config
from src.dual_mesh import DualMeshModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--pretrained", required=True, help="Checkpoint глобальной модели")
    ap.add_argument("--regional-ckpt", default=None, help="Checkpoint региональных весов")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--roi", nargs=4, type=float, required=True,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--reg-mesh-level", type=int, default=7)
    ap.add_argument("--reg-steps", type=int, default=4)
    ap.add_argument("--cross-k", type=int, default=3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--per-channel", action="store_true")
    args = ap.parse_args()

    exp_dir = args.experiment_dir
    cfg_path = os.path.join(exp_dir, FileNames.EXPERIMENT_CONFIG)
    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset ---
    data_dir = args.data_dir or getattr(exp_cfg, 'data_dir', None)
    if data_dir is None:
        data_dir = f"data/datasets/{exp_cfg.data.dataset_name.value}"

    _, _, test_ds, meta = load_chunked_datasets(
        data_path=data_dir,
        obs_window=exp_cfg.data.obs_window_used,
        pred_steps=1,
        n_features=exp_cfg.data.num_features_used,
    )

    # --- Global model ---
    real_coords = getattr(meta, 'cordinates', None)
    flat_grid = getattr(meta, 'flat_grid', False)

    global_model = load_model_from_experiment_config(
        exp_cfg, device, meta,
        coordinates=real_coords,
        flat_grid=flat_grid,
    )
    state = torch.load(args.pretrained, map_location=device)
    global_model.load_state_dict(state, strict=False)
    global_model = global_model.to(device)

    for p in global_model.parameters():
        p.requires_grad = False

    # --- Grid coords ---
    if flat_grid and real_coords is not None:
        grid_lats = real_coords[0].astype(np.float32)
        grid_lons = real_coords[1].astype(np.float32)
    else:
        lats = np.linspace(-90, 90, meta.num_latitudes, endpoint=True)
        lons = np.linspace(0, 360, meta.num_longitudes, endpoint=False)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        grid_lats = lat_grid.flatten().astype(np.float32)
        grid_lons = lon_grid.flatten().astype(np.float32)

    # --- Dual model ---
    roi = tuple(args.roi)
    dual_model = DualMeshModel(
        global_model=global_model,
        roi=roi,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        device=device,
        reg_mesh_level=args.reg_mesh_level,
        reg_processor_steps=args.reg_steps,
        cross_k=args.cross_k,
        hidden_dim=args.hidden_dim,
    )

    # Load regional weights
    ckpt_path = args.regional_ckpt or os.path.join(exp_dir, "results", "best_regional.pth")
    if os.path.exists(ckpt_path):
        reg_state = torch.load(ckpt_path, map_location=device)
        # Load only regional keys
        missing, unexpected = dual_model.load_state_dict(reg_state, strict=False)
        n_loaded = len(reg_state) - len(unexpected)
        print(f"Loaded {n_loaded} regional weight tensors from {ckpt_path}")
    else:
        print(f"WARNING: Regional checkpoint not found: {ckpt_path}")

    dual_model = dual_model.to(device)
    dual_model.eval()

    # --- Inference ---
    G = len(grid_lats)
    C = exp_cfg.data.num_features_used
    OBS = exp_cfg.data.obs_window_used
    use_residual = not args.no_residual and getattr(exp_cfg, 'use_residual', True)

    roi_mask = dual_model.roi_mask.cpu().numpy()
    n_roi = int(roi_mask.sum())

    max_samples = min(args.max_samples, len(test_ds))
    print(f"\n[predict] {max_samples}/{len(test_ds)} test samples, ROI={n_roi} points")

    # Streaming metrics
    sum_se_global = 0.0
    sum_se_region = 0.0
    sum_se_base_region = 0.0
    n_elem_global = 0
    n_elem_region = 0
    acc_values = []

    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            if i >= max_samples:
                break

            X, y = X.to(device), y.to(device)
            y = y.squeeze(0) if y.dim() == 4 else y
            y_step0 = y.view(y.shape[0], G, -1, C)[:, :, 0, :].squeeze(0) if y.shape[-1] > C else y.squeeze(0)

            pred = dual_model(X)
            if pred.dim() == 2:
                pass  # (G, C)
            else:
                pred = pred.squeeze(0)

            if use_residual:
                X_sq = X.squeeze(0)
                X_last = X_sq[:, -C:]
                out = X_last + pred
            else:
                out = pred

            # Baseline (persistence)
            X_sq = X.squeeze(0)
            baseline = X_sq[:, -C:]

            err_sq = (out - y_step0).pow(2)
            base_err_sq = (baseline - y_step0).pow(2)

            sum_se_global += err_sq.sum().item()
            n_elem_global += err_sq.numel()

            sum_se_region += err_sq[roi_mask].sum().item()
            sum_se_base_region += base_err_sq[roi_mask].sum().item()
            n_elem_region += err_sq[roi_mask].numel()

            # Regional ACC
            pred_r = out[roi_mask]
            true_r = y_step0[roi_mask]
            p_a = pred_r - pred_r.mean(0, keepdim=True)
            t_a = true_r - true_r.mean(0, keepdim=True)
            corr = (p_a * t_a).sum(0) / (p_a.norm(dim=0) * t_a.norm(dim=0) + 1e-8)
            acc_values.append(corr.mean().item())

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{max_samples} done...")

    rmse_global = (sum_se_global / n_elem_global) ** 0.5
    rmse_region = (sum_se_region / n_elem_region) ** 0.5
    rmse_base_region = (sum_se_base_region / n_elem_region) ** 0.5
    skill = 1 - rmse_region / rmse_base_region
    acc = sum(acc_values) / len(acc_values)

    print(f"\n{'='*50}")
    print(f"Results ({max_samples} samples)")
    print(f"  Global RMSE:   {rmse_global:.4f}")
    print(f"  Regional RMSE: {rmse_region:.4f}")
    print(f"  Baseline RMSE: {rmse_base_region:.4f}")
    print(f"  Skill:         {skill*100:.2f}%")
    print(f"  Regional ACC:  {acc:.4f}")
    print(f"{'='*50}")

    results = {
        "rmse_global": rmse_global,
        "rmse_region": rmse_region,
        "rmse_baseline_region": rmse_base_region,
        "skill": skill,
        "regional_acc": acc,
        "n_samples": max_samples,
        "roi": list(roi),
    }

    results_path = os.path.join(exp_dir, "results", "predict_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()

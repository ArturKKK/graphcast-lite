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

    # ── 5. Run inference (per–AR-step) ──
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    use_residual = not args.no_residual and getattr(gnn_cfg, 'use_residual', False)
    AR = args.ar_steps
    obs = gnn_cfg.data.obs_window_used

    # Accumulators: per-step, per-channel
    mse_gnn   = np.zeros((AR, C))   # GNN bilinear upsampled
    mse_casc  = np.zeros((AR, C))   # GNN + UNet cascade
    mse_pers  = np.zeros((AR, C))   # persistence (last real 0.25° obs)
    n_per_step = np.zeros(AR, dtype=int)

    # Physical units & formatting helpers
    UNITS = {
        "t2m": "°C", "10u": "m/s", "10v": "m/s", "msl": "Pa",
        "tp": "m", "sp": "Pa", "tcwv": "kg/m²",
        "z_surf": "m²/s²", "lsm": "-",
        "t@850": "°C", "u@850": "m/s", "v@850": "m/s",
        "z@850": "gpm", "q@850": "kg/kg",
        "t@500": "°C", "u@500": "m/s", "v@500": "m/s",
        "z@500": "gpm", "q@500": "kg/kg",
    }

    def _fmt_rmse(val, name):
        if "z@" in name or name == "z_surf":
            return f"{val / 9.81:7.1f}m"
        elif name == "t2m" or name.startswith("t@"):
            return f"{val:6.2f}°C"
        elif name in ("10u", "10v") or name.startswith("u@") or name.startswith("v@"):
            return f"{val:6.2f}ms"
        elif name == "msl" or name == "sp":
            return f"{val / 100:6.1f}hPa"
        elif name == "tp":
            return f"{val * 1000:6.3f}mm"
        elif name == "tcwv":
            return f"{val:6.2f}kg"
        else:
            return f"{val:8.4f}"

    max_samples = min(args.max_samples, len(test_ds))
    print(f"\n[Inference] AR={AR}, samples={max_samples}, "
          f"gnn_residual={use_residual}, unet_residual={ds_residual}")

    # Pre-compute static tensor once
    static_t = None
    if static_fine is not None:
        static_t = torch.from_numpy(static_fine.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)

    for i, (X, Y) in enumerate(test_loader):
        if i >= max_samples:
            break

        # ── Time alignment: actual dataset time index ──
        if hasattr(test_ds, '_sample_indices') and i < len(test_ds._sample_indices):
            _chunk_idx, local_t = test_ds._sample_indices[i]
        else:
            local_t = i

        X = X.to(device)
        N, G, feat_dim = X.shape
        C_feat = feat_dim // obs
        curr_state = X.view(N, G, obs, C_feat)

        # Persistence baseline: real 0.25° ERA5 at last obs time
        t_persist = local_t + obs - 1
        if t_persist >= ds_info["n_time"]:
            continue
        persist_raw = np.array(fine_data[t_persist], dtype=np.float32)  # (lon, lat, C)
        persist_phys = persist_raw.transpose(1, 0, 2)  # (H, W, C)

        # ── AR rollout with per-step evaluation ──
        sample_ok = True
        for step in range(AR):
            # One GNN step
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

            # Fine target for this step
            t_fine = local_t + obs + step
            if t_fine >= ds_info["n_time"]:
                sample_ok = False
                break

            fine_target = np.array(fine_data[t_fine], dtype=np.float32)  # (lon, lat, C)
            fine_target = fine_target.transpose(1, 0, 2)  # (H, W, C)

            # Crop GNN output → fine grid (normalized space)
            gnn_pred_np = out[0].cpu().numpy()  # (G, C)
            coarse_norm = crop_and_upsample_roi(
                gnn_pred_np, grid_lats, grid_lons, roi,
                target_lats, target_lons, flat_grid=flat_grid,
            )  # (H, W, C) normalized

            # UNet refinement
            frames = [coarse_norm] * obs_window
            coarse_stack = np.concatenate(frames, axis=-1)
            x_unet = torch.from_numpy(coarse_stack).permute(2, 0, 1).unsqueeze(0).float().to(device)
            if static_t is not None:
                x_unet = torch.cat([x_unet, static_t], dim=1)
            with torch.no_grad():
                unet_out = unet(x_unet)
            if ds_residual:
                x_last = x_unet[:, (obs_window - 1) * C : obs_window * C, :, :]
                unet_out = x_last + unet_out
            cascade_norm = unet_out[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)

            # Denormalize to physical
            coarse_phys = coarse_norm * ds_std + ds_mean
            cascade_phys = cascade_norm * ds_std + ds_mean

            # Per-channel MSE (skip static)
            for c in range(C):
                if c in static_ch:
                    continue
                mse_gnn[step, c]  += np.mean((coarse_phys[:, :, c]  - fine_target[:, :, c]) ** 2)
                mse_casc[step, c] += np.mean((cascade_phys[:, :, c] - fine_target[:, :, c]) ** 2)
                mse_pers[step, c] += np.mean((persist_phys[:, :, c] - fine_target[:, :, c]) ** 2)
            n_per_step[step] += 1

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{max_samples}")

    # ═══════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════
    dyn_ch = [c for c in range(C) if c not in static_ch]

    # Per-step per-channel RMSE
    rmse_gnn  = np.zeros_like(mse_gnn)
    rmse_casc = np.zeros_like(mse_gnn)
    rmse_pers = np.zeros_like(mse_gnn)
    for s in range(AR):
        if n_per_step[s] > 0:
            rmse_gnn[s]  = np.sqrt(mse_gnn[s]  / n_per_step[s])
            rmse_casc[s] = np.sqrt(mse_casc[s] / n_per_step[s])
            rmse_pers[s] = np.sqrt(mse_pers[s] / n_per_step[s])

    # ── Compact per-horizon summary ──
    print(f"\n{'=' * 72}")
    print(f"Cascade results  ({int(n_per_step[0])} samples)")
    print(f"{'=' * 72}")

    header = f"  {'':>10}"
    for s in range(AR):
        header += f"  +{(s+1)*6:02d}h       "
    print(header)

    header2 = f"  {'':>10}"
    for s in range(AR):
        header2 += f"  {'GNN':>6} {'Casc':>6} {'Δ':>5}"
    print(header2)
    print("  " + "-" * (10 + AR * 19))

    for c in dyn_ch:
        name = var_names[c]
        row = f"  {name:>10}"
        for s in range(AR):
            g = _fmt_rmse(rmse_gnn[s, c], name).strip()
            ca = _fmt_rmse(rmse_casc[s, c], name).strip()
            imp = (1 - rmse_casc[s, c] / (rmse_gnn[s, c] + 1e-12)) * 100
            sign = "+" if imp >= 0 else ""
            row += f"  {g:>6} {ca:>6} {sign}{imp:4.1f}%"
        print(row)

    # Mean skill per horizon
    print("  " + "-" * (10 + AR * 19))
    row = f"  {'MEAN':>10}"
    for s in range(AR):
        skills = [(1 - rmse_casc[s, c] / (rmse_gnn[s, c] + 1e-12)) for c in dyn_ch]
        ms = np.mean(skills) * 100
        sign = "+" if ms >= 0 else ""
        row += f"  {'':>6} {'':>6} {sign}{ms:4.1f}%"
    print(row)

    # ── Per-horizon summary: Skill vs Persistence ──
    print(f"\n  Skill vs persistence (0.25° ERA5):")
    header = f"  {'':>10}"
    for s in range(AR):
        header += f"  +{(s+1)*6:02d}h "
    print(header)
    print("  " + "-" * (10 + AR * 8))
    for c in dyn_ch:
        name = var_names[c]
        row = f"  {name:>10}"
        for s in range(AR):
            sk = (1 - rmse_casc[s, c] / (rmse_pers[s, c] + 1e-12)) * 100
            row += f"  {sk:+5.1f}% "
        print(row)

    row = f"  {'MEAN':>10}"
    for s in range(AR):
        skills = [(1 - rmse_casc[s, c] / (rmse_pers[s, c] + 1e-12)) for c in dyn_ch]
        row += f"  {np.mean(skills)*100:+5.1f}% "
    print(row)

    # ── Detailed table: last horizon ──
    s_last = AR - 1
    print(f"\n  Detailed +{AR*6}h:")
    print(f"  {'Variable':>10} {'Unit':>6}  {'Persist':>8}  {'GNN_bln':>8}  {'Cascade':>8}  {'Sk_pers':>7}  {'Sk_GNN':>7}")
    print("  " + "-" * 62)
    for c in dyn_ch:
        name = var_names[c]
        unit = UNITS.get(name, "?")
        rp = rmse_pers[s_last, c]
        rg = rmse_gnn[s_last, c]
        rc = rmse_casc[s_last, c]
        sk_p = (1 - rc / (rp + 1e-12)) * 100
        sk_g = (1 - rc / (rg + 1e-12)) * 100

        def _val(v, nm):
            if "z@" in nm or nm == "z_surf":
                return f"{v/9.81:7.1f}"
            elif nm == "msl" or nm == "sp":
                return f"{v/100:7.2f}"
            elif nm == "tp":
                return f"{v*1000:7.4f}"
            else:
                return f"{v:7.3f}"

        print(f"  {name:>10} {unit:>6}  {_val(rp, name)}  {_val(rg, name)}  {_val(rc, name)}  {sk_p:+6.1f}%  {sk_g:+6.1f}%")

    mean_sk_p = np.mean([(1 - rmse_casc[s_last, c] / (rmse_pers[s_last, c] + 1e-12)) for c in dyn_ch]) * 100
    mean_sk_g = np.mean([(1 - rmse_casc[s_last, c] / (rmse_gnn[s_last, c] + 1e-12)) for c in dyn_ch]) * 100
    print(f"  {'MEAN':>10} {'':>6}  {'':>8}  {'':>8}  {'':>8}  {mean_sk_p:+6.1f}%  {mean_sk_g:+6.1f}%")

    # ── Save results.json ──
    results = {}
    for s in range(AR):
        h = (s + 1) * 6
        results[f"+{h}h"] = {
            "n_samples": int(n_per_step[s]),
            "per_channel": {
                var_names[c]: {
                    "rmse_persist": float(rmse_pers[s, c]),
                    "rmse_gnn": float(rmse_gnn[s, c]),
                    "rmse_cascade": float(rmse_casc[s, c]),
                }
                for c in dyn_ch
            },
        }
    out_path = ds_dir / "cascade_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()

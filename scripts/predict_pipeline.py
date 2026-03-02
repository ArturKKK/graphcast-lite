#!/usr/bin/env python3
"""
Plan B: Pipeline inference — Global model + Regional model + Blending.

1. Глобальная модель (512×256) → прогноз на глобальной сетке
2. Интерполируем глобальный прогноз на региональную сетку 61×41 → background
3. Региональная модель (61×41) → свой прогноз
4. Blending: taper-маска плавно переходит от региональной к глобальной модели на краях

Использование:
    python scripts/predict_pipeline.py \
        --global-exp experiments/wb2_512x256_19f_ar_v2 \
        --region-exp experiments/region_krsk_cds_19f \
        --global-data data/datasets/wb2_512x256_19f_jan2023 \
        --region-data data/datasets/region_krsk_cds_19f \
        --max-samples 3 --per-channel --taper-width 3
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import FileNames
from src.config import ExperimentConfig
from src.utils import load_from_json_file
from src.data.dataloader_chunked import load_chunked_datasets
from src.main import load_model_from_experiment_config


# ============================================================
# Helpers
# ============================================================

def load_experiment(exp_dir, data_dir, device, ckpt=None, prune=False, mesh_buf=15.0):
    """Load config, dataset, model, weights for one experiment."""
    cfg_path = os.path.join(exp_dir, FileNames.EXPERIMENT_CONFIG)
    ckpt_path = ckpt or os.path.join(exp_dir, FileNames.SAVED_MODEL)
    assert os.path.exists(cfg_path), f"нет конфига: {cfg_path}"
    assert os.path.exists(ckpt_path), f"нет чекпойнта: {ckpt_path}"

    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))

    data_path = Path(data_dir) if data_dir else Path(exp_cfg.data_dir or "")
    assert data_path.exists(), f"нет данных: {data_path}"

    ar_steps = max(exp_cfg.max_ar_steps, 1)
    _, _, test_ds, meta = load_chunked_datasets(
        data_path=str(data_path),
        obs_window=exp_cfg.data.obs_window_used,
        pred_steps=ar_steps,
        n_features=exp_cfg.data.num_features_used,
    )

    # Определяем координаты (flat_grid, prune-mesh)
    coords = getattr(meta, 'cordinates', None)
    region_bounds = None
    flat_grid = getattr(meta, 'flat_grid', False)

    coords_file = data_path / "coords.npz"
    if prune and coords_file.exists():
        z = np.load(coords_file)
        real_lats = z["latitude"].astype(np.float32)
        real_lons = z["longitude"].astype(np.float32)
        coords = (real_lats, real_lons)
        region_bounds = (float(real_lats.min()), float(real_lats.max()),
                         float(real_lons.min()), float(real_lons.max()))

    model = load_model_from_experiment_config(
        exp_cfg, device, meta,
        coordinates=coords,
        region_bounds=region_bounds,
        mesh_buffer=mesh_buf,
        flat_grid=flat_grid,
    )

    state = torch.load(ckpt_path, map_location=device)
    if region_bounds is not None:
        state = {k: v for k, v in state.items()
                 if not k.startswith("_processing_edge_features")}
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    return exp_cfg, test_ds, meta, model, data_path


def interpolate_global_to_region(global_pred, global_lats, global_lons,
                                 region_lats, region_lons):
    """Bilinear interpolation of global grid prediction onto regional grid.
    
    global_pred: (G_global, C*P) tensor
    global_lats: (n_lat_g,) 1D axis  
    global_lons: (n_lon_g,) 1D axis
    region_lats: (n_lat_r,) 1D axis
    region_lons: (n_lon_r,) 1D axis
    
    Returns: (G_region, C*P) tensor
    """
    from scipy.interpolate import RegularGridInterpolator

    n_lat_g = len(global_lats)
    n_lon_g = len(global_lons)
    n_lat_r = len(region_lats)
    n_lon_r = len(region_lons)
    CP = global_pred.shape[1]

    # Reshape global from (n_lon*n_lat, CP) → (n_lon, n_lat, CP)
    # Our convention: node order is lon-major, i.e. idx = j*n_lat + i
    global_3d = global_pred.numpy().reshape(n_lon_g, n_lat_g, CP)

    # Create meshgrid for target region
    reg_lon_grid, reg_lat_grid = np.meshgrid(region_lons, region_lats, indexing='ij')
    target_points = np.stack([reg_lon_grid.ravel(), reg_lat_grid.ravel()], axis=1)

    result = np.zeros((n_lon_r * n_lat_r, CP), dtype=np.float32)

    for c in range(CP):
        interp = RegularGridInterpolator(
            (global_lons, global_lats), global_3d[:, :, c],
            method='linear', bounds_error=False, fill_value=None
        )
        result[:, c] = interp(target_points).astype(np.float32)

    return torch.from_numpy(result)


def build_taper_mask_2d(n_lat, n_lon, taper_width):
    """Create a 2D taper mask: 1.0 inside, smooth decay to 0.0 at borders.
    
    Returns tensor of shape (n_lon * n_lat, 1)  —  node order: lon-major.
    """
    mask = np.ones((n_lon, n_lat), dtype=np.float32)
    for w in range(taper_width):
        alpha = (w + 1) / (taper_width + 1)
        # lat boundaries
        mask[:, w] = np.minimum(mask[:, w], alpha)
        mask[:, -(w + 1)] = np.minimum(mask[:, -(w + 1)], alpha)
        # lon boundaries
        mask[w, :] = np.minimum(mask[w, :], alpha)
        mask[-(w + 1), :] = np.minimum(mask[-(w + 1), :], alpha)
    return torch.from_numpy(mask.ravel()).unsqueeze(1)  # (G, 1)


# ============================================================
# StreamingMetrics (copied from predict.py)
# ============================================================

class StreamingMetrics:
    def __init__(self, C):
        self.C = C; self.n = 0; self.total_elem = 0
        self.sum_se = 0.0; self.sum_ae = 0.0
        self.sum_se_per_ch = np.zeros(C, dtype=np.float64)
        self.elem_per_ch = np.zeros(C, dtype=np.int64)
        self.sum_acc = np.zeros(C, dtype=np.float64)
        self.acc_count = np.zeros(C, dtype=np.int64)

    def update(self, y_true, y_pred):
        err = y_pred.float() - y_true.float()
        self.sum_se += err.pow(2).sum().item()
        self.sum_ae += err.abs().sum().item()
        self.total_elem += y_true.numel()
        CP = y_true.shape[1]; eps = 1e-8
        for c in range(CP):
            yt, yp = y_true[:, c].float(), y_pred[:, c].float()
            ch = c % self.C
            self.sum_se_per_ch[ch] += (yp - yt).pow(2).sum().item()
            self.elem_per_ch[ch] += yt.numel()
            yt_a, yp_a = yt - yt.mean(), yp - yp.mean()
            corr = (yt_a * yp_a).sum() / (yt_a.norm() * yp_a.norm() + eps)
            self.sum_acc[ch] += corr.item()
            self.acc_count[ch] += 1
        self.n += 1

    @property
    def rmse(self): return float(np.sqrt(self.sum_se / max(self.total_elem, 1)))
    @property
    def acc(self): return float((self.sum_acc / np.maximum(self.acc_count, 1)).mean())
    @property
    def rmse_per_channel(self):
        return np.sqrt(self.sum_se_per_ch / np.maximum(self.elem_per_ch, 1))
    @property
    def acc_per_channel(self):
        return self.sum_acc / np.maximum(self.acc_count, 1)


# ============================================================
# Main pipeline
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Plan B: Global + Regional pipeline inference")
    ap.add_argument("--global-exp", required=True, help="experiments/wb2_512x256_19f_ar_v2")
    ap.add_argument("--region-exp", required=True, help="experiments/region_krsk_cds_19f")
    ap.add_argument("--global-data", default=None, help="data dir for global test")
    ap.add_argument("--region-data", default=None, help="data dir for regional test")
    ap.add_argument("--global-ckpt", default=None)
    ap.add_argument("--region-ckpt", default=None)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--taper-width", type=int, default=3,
                    help="Ширина taper-зоны (в точках) для blending")
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--prune-mesh", action="store_true",
                    help="Для региональной модели: обрезать mesh до bbox данных")
    ap.add_argument("--mesh-buffer", type=float, default=15.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] device={device}")

    # --- Load global model ---
    print("\n" + "=" * 60)
    print("[1/4] Loading global model...")
    g_cfg, g_test, g_meta, g_model, g_data_path = load_experiment(
        args.global_exp, args.global_data, device, ckpt=args.global_ckpt)
    G_g = g_meta.num_longitudes * g_meta.num_latitudes
    C = g_cfg.data.num_features_used
    P_g = max(g_cfg.max_ar_steps, g_cfg.data.pred_window_used)
    OBS_g = g_cfg.data.obs_window_used
    g_lats = np.linspace(-90, 90, g_meta.num_latitudes, endpoint=True)
    g_lons = np.linspace(0, 360, g_meta.num_longitudes, endpoint=False)
    print(f"  Global: {g_meta.num_longitudes}×{g_meta.num_latitudes} ({G_g} nodes), C={C}, P={P_g}")

    # --- Load regional model ---
    print("\n[2/4] Loading regional model...")
    r_cfg, r_test, r_meta, r_model, r_data_path = load_experiment(
        args.region_exp, args.region_data, device, ckpt=args.region_ckpt,
        prune=args.prune_mesh, mesh_buf=args.mesh_buffer)
    G_r = r_meta.num_longitudes * r_meta.num_latitudes
    P_r = max(r_cfg.max_ar_steps, r_cfg.data.pred_window_used)
    OBS_r = r_cfg.data.obs_window_used

    # Regional coordinates
    r_coords_file = r_data_path / "coords.npz"
    if r_coords_file.exists():
        rz = np.load(r_coords_file)
        r_lats = rz["latitude"].astype(np.float32)
        r_lons = rz["longitude"].astype(np.float32)
    else:
        r_lats = np.linspace(-90, 90, r_meta.num_latitudes, endpoint=True)
        r_lons = np.linspace(0, 360, r_meta.num_longitudes, endpoint=False)
    print(f"  Region: {r_meta.num_longitudes}×{r_meta.num_latitudes} ({G_r} nodes), "
          f"lat=[{r_lats.min():.2f},{r_lats.max():.2f}], lon=[{r_lons.min():.2f},{r_lons.max():.2f}]")

    P = min(P_g, P_r)
    print(f"  Common horizons: {P}")

    # --- Taper mask ---
    taper_mask = build_taper_mask_2d(r_meta.num_latitudes, r_meta.num_longitudes, args.taper_width)
    # taper_mask: (G_r, 1) — 1.0 centre → 0.0 border
    print(f"\n[3/4] Taper mask: tw={args.taper_width}, shape={taper_mask.shape}")

    # --- Variables and scalers for per-channel ---
    var_path = r_data_path / "variables.json"
    var_order = json.loads(var_path.read_text()) if var_path.exists() else [f"ch{c}" for c in range(C)]
    scalers_path = r_data_path / "scalers.npz"
    std = None
    if scalers_path.exists():
        std = np.load(scalers_path)["std"].astype(np.float64)[:C]

    UNITS = {
        "t2m": "K", "10u": "m/s", "10v": "m/s", "msl": "Pa",
        "tp": "m", "sp": "Pa", "tcwv": "kg/m²",
        "z_surf": "m²/s²", "lsm": "-",
        "t@850": "K", "u@850": "m/s", "v@850": "m/s",
        "z@850": "m²/s²", "q@850": "kg/kg",
        "t@500": "K", "u@500": "m/s", "v@500": "m/s",
        "z@500": "m²/s²", "q@500": "kg/kg",
    }

    # --- Metrics ---
    # Regional model only (after blending)
    sm_blend = StreamingMetrics(C)
    sm_region_only = StreamingMetrics(C)
    sm_global_interp = StreamingMetrics(C)
    sm_base = StreamingMetrics(C)

    sm_blend_h, sm_region_h, sm_global_h, sm_base_h = [], [], [], []
    if P > 1:
        for _ in range(P):
            sm_blend_h.append(StreamingMetrics(C))
            sm_region_h.append(StreamingMetrics(C))
            sm_global_h.append(StreamingMetrics(C))
            sm_base_h.append(StreamingMetrics(C))

    # --- Inference ---
    print(f"\n[4/4] Running pipeline inference ({args.max_samples} samples)...\n")
    N = min(args.max_samples, len(g_test), len(r_test))

    g_loader = torch.utils.data.DataLoader(g_test, batch_size=1, shuffle=False)
    r_loader = torch.utils.data.DataLoader(r_test, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, ((gX, gy), (rX, ry)) in enumerate(zip(g_loader, r_loader)):
            if i >= N:
                break

            # ---- Global model inference ----
            gy = gy.squeeze(0)
            if gy.dim() == 3: gy = gy.squeeze(-2)
            gX_dev = gX.to(device)

            g_out = g_model(gX_dev, attention_threshold=0.0).cpu()  # (G_g, C*P_g)

            # ---- Interpolate global → regional grid ----
            g_interp = interpolate_global_to_region(
                g_out[:, :P * C], g_lats, g_lons, r_lats, r_lons
            )  # (G_r, C*P)

            # ---- Regional model inference ----
            ry = ry.squeeze(0)
            if ry.dim() == 3: ry = ry.squeeze(-2)
            rX_dev = rX.to(device)

            r_out = r_model(rX_dev, attention_threshold=0.0).cpu()  # (G_r, C*P_r)
            r_out = r_out[:, :P * C]  # trim to common P

            # ---- Blending ----
            blended = taper_mask * r_out + (1.0 - taper_mask) * g_interp

            # ---- Baseline: persistence (last observation repeated) ----
            rX_last = rX.squeeze(0)[:, -C:].repeat(1, P)

            # ---- Ground truth ----
            ry_gt = ry[:, :P * C]

            # ---- Update metrics ----
            sm_blend.update(ry_gt, blended)
            sm_region_only.update(ry_gt, r_out)
            sm_global_interp.update(ry_gt, g_interp)
            sm_base.update(ry_gt, rX_last)

            if P > 1:
                for p in range(P):
                    sl = slice(p * C, (p + 1) * C)
                    sm_blend_h[p].update(ry_gt[:, sl], blended[:, sl])
                    sm_region_h[p].update(ry_gt[:, sl], r_out[:, sl])
                    sm_global_h[p].update(ry_gt[:, sl], g_interp[:, sl])
                    sm_base_h[p].update(ry_gt[:, sl], rX_last[:, sl])

            if (i + 1) % 5 == 0 or i == N - 1:
                print(f"  [{i+1}/{N}] blend_RMSE={sm_blend.rmse:.6f} region_RMSE={sm_region_only.rmse:.6f} "
                      f"global_interp_RMSE={sm_global_interp.rmse:.6f}")

    # ============================================================
    # Results
    # ============================================================
    print()
    print("=" * 70)
    print(f"=== Pipeline Results ({N} samples, {P} horizons) ===")
    print(f"Regional grid: {r_meta.num_longitudes}×{r_meta.num_latitudes} ({G_r} nodes)")
    print(f"Taper width: {args.taper_width}")
    print()

    methods = [
        ("Blended (region+global)", sm_blend, sm_blend_h),
        ("Regional model only",     sm_region_only, sm_region_h),
        ("Global interpolated",     sm_global_interp, sm_global_h),
        ("Persistence baseline",    sm_base, sm_base_h),
    ]

    # Overall table
    print(f"{'Method':>30s} {'RMSE':>10s} {'ACC':>8s} {'Skill':>8s}")
    for name, sm, _ in methods:
        sk = 1.0 - (sm.rmse / (sm_base.rmse + 1e-12))
        print(f"{name:>30s} {sm.rmse:10.6f} {sm.acc:8.4f} {sk*100:7.2f}%")

    # Per-horizon table
    if P > 1:
        print(f"\nPer-horizon RMSE:")
        header = f"  {'Method':>30s}"
        for p in range(P):
            header += f" {'+'+ str((p+1)*6) + 'h':>8s}"
        print(header)
        for name, _, sm_h in methods:
            row = f"  {name:>30s}"
            for p in range(P):
                row += f" {sm_h[p].rmse:8.6f}"
            print(row)

    # Per-channel (key vars)
    if args.per_channel and std is not None:
        key_vars = ["t2m", "10u", "10v", "msl", "t@850", "z@500"]
        key_idx = [i for i, v in enumerate(var_order[:C]) if v in key_vars]

        for name, sm, sm_h in methods:
            print(f"\n  [{name}] Per-channel RMSE (physical units):")
            if P > 1 and sm_h:
                header = f"    {'var':>10s} {'unit':>6s}"
                for p in range(P):
                    header += f" {'+'+ str((p+1)*6) + 'h':>8s}"
                header += f" {'AVG':>8s}"
                print(header)
                for c in key_idx:
                    vname = var_order[c]
                    unit = UNITS.get(vname, "?")
                    row = f"    {vname:>10s} {unit:>6s}"
                    vals = []
                    for p in range(P):
                        phys_rmse = sm_h[p].rmse_per_channel[c] * std[c]
                        vals.append(phys_rmse)
                        if "z@" in vname:
                            row += f" {phys_rmse/9.81:7.1f}m"
                        elif vname in ("t2m",) or vname.startswith("t@"):
                            row += f" {phys_rmse:7.2f}K"
                        else:
                            row += f" {phys_rmse:8.2f}"
                    avg = np.mean(vals)
                    if "z@" in vname:
                        row += f" {avg/9.81:7.1f}m"
                    elif vname in ("t2m",) or vname.startswith("t@"):
                        row += f" {avg:7.2f}K"
                    else:
                        row += f" {avg:8.2f}"
                    print(row)
            else:
                rmse_pc = sm.rmse_per_channel
                for c in key_idx:
                    vname = var_order[c]
                    unit = UNITS.get(vname, "?")
                    phys = rmse_pc[c] * std[c]
                    print(f"    {vname:>10s}: {phys:.4f} {unit}")

    print("=" * 70)


if __name__ == "__main__":
    main()

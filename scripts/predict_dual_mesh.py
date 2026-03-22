#!/usr/bin/env python3
"""
scripts/predict_dual_mesh.py

Инференс DualMeshModel: глобальная + региональная модель.
Per-channel метрики, физические единицы, per-horizon (AR rollout).

Использование:
  python scripts/predict_dual_mesh.py experiments/dual_mesh_krsk \
    --pretrained experiments/multires_nores_freeze6/best_model.pth \
    --regional-ckpt experiments/dual_mesh_krsk/results/best_regional.pth \
    --roi 50 60 83 98 \
    --data-dir data/datasets/multires_krsk_19f \
    --per-channel --max-samples 200 --ar-steps 4
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

UNITS = {
    "t2m": "K", "10u": "m/s", "10v": "m/s", "msl": "Pa",
    "tp": "m", "sp": "Pa", "tcwv": "kg/m²",
    "z_surf": "m²/s²", "lsm": "-",
    "t@850": "K", "u@850": "m/s", "v@850": "m/s",
    "z@850": "m²/s²", "q@850": "kg/kg",
    "t@500": "K", "u@500": "m/s", "v@500": "m/s",
    "z@500": "m²/s²", "q@500": "kg/kg",
}


class StreamingMetrics:
    """Накапливает MSE/MAE/ACC потоково — без хранения всех сэмплов в RAM."""

    def __init__(self, num_channels: int):
        self.C = num_channels
        self.n = 0
        self.total_elem = 0
        self.sum_se = 0.0
        self.sum_ae = 0.0
        self.sum_se_per_ch = np.zeros(num_channels, dtype=np.float64)
        self.elem_per_ch = np.zeros(num_channels, dtype=np.int64)
        self.sum_acc = np.zeros(num_channels, dtype=np.float64)
        self.acc_count = np.zeros(num_channels, dtype=np.int64)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """y_true, y_pred: [G, C]"""
        err = y_pred.float() - y_true.float()
        CP = y_true.shape[1]
        eps = 1e-8
        for c in range(CP):
            yt = y_true[:, c].float()
            yp = y_pred[:, c].float()
            ch = c % self.C
            se_c = (yp - yt).pow(2).sum().item()
            self.sum_se_per_ch[ch] += se_c
            self.elem_per_ch[ch] += yt.numel()
            yt_a = yt - yt.mean()
            yp_a = yp - yp.mean()
            corr = (yt_a * yp_a).sum() / (yt_a.norm() * yp_a.norm() + eps)
            self.sum_acc[ch] += corr.item()
            self.acc_count[ch] += 1

        self.sum_se += err.pow(2).sum().item()
        self.sum_ae += err.abs().sum().item()
        self.total_elem += err.numel()
        self.n += 1

    @property
    def mse(self):
        return self.sum_se / max(self.total_elem, 1)

    @property
    def rmse(self):
        return float(np.sqrt(self.mse))

    @property
    def mae(self):
        return self.sum_ae / max(self.total_elem, 1)

    @property
    def rmse_per_channel(self):
        mse_pc = self.sum_se_per_ch / np.maximum(self.elem_per_ch, 1)
        return np.sqrt(mse_pc)

    @property
    def acc_per_channel(self):
        return self.sum_acc / np.maximum(self.acc_count, 1)

    @property
    def acc(self):
        return float(self.acc_per_channel.mean())


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
    ap.add_argument("--ar-steps", type=int, default=1,
                    help="Число AR-шагов. 1=+6h, 4=+24h. Выход подаётся обратно на вход.")
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
        pred_steps=args.ar_steps,
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
        missing, unexpected = dual_model.load_state_dict(reg_state, strict=False)
        n_loaded = len(reg_state) - len(unexpected)
        print(f"Loaded {n_loaded} regional weight tensors from {ckpt_path}")
    else:
        print(f"WARNING: Regional checkpoint not found: {ckpt_path}")

    dual_model = dual_model.to(device)
    dual_model.eval()

    # --- Inference setup ---
    G = len(grid_lats)
    C = exp_cfg.data.num_features_used
    OBS = exp_cfg.data.obs_window_used
    AR_STEPS = args.ar_steps
    use_residual = not args.no_residual and getattr(exp_cfg, 'use_residual', True)

    roi_mask = dual_model.roi_mask.cpu().numpy()
    n_roi = int(roi_mask.sum())

    max_samples = min(args.max_samples, len(test_ds))
    print(f"\n[predict] {max_samples}/{len(test_ds)} test samples, ROI={n_roi} points")
    print(f"  use_residual={use_residual}, C={C}, G={G}, AR={AR_STEPS} steps (+{AR_STEPS*6}h)")

    # Streaming metrics: overall (across all horizons)
    sm_region = StreamingMetrics(C)
    sm_base_region = StreamingMetrics(C)

    # Per-horizon metrics (regional only — that's what we care about)
    sm_region_h = [StreamingMetrics(C) for _ in range(AR_STEPS)]
    sm_base_h = [StreamingMetrics(C) for _ in range(AR_STEPS)]

    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            if i >= max_samples:
                break

            X, y = X.to(device), y.to(device)
            y = y.squeeze(0) if y.dim() == 4 else y
            y = y.squeeze(0) if y.dim() == 3 else y
            # y: (G, P*C) where P = pred_steps from dataset

            y_total_steps = y.shape[-1] // C
            effective_P = min(AR_STEPS, y_total_steps)

            X_sq = X.squeeze(0)  # (G, OBS*C)
            baseline_last = X_sq[:, -C:]  # persistence: last obs

            # --- AR rollout ---
            curr_state = X_sq.clone()  # (G, OBS*C)
            ar_outs = []

            for ar_step in range(effective_P):
                inp = curr_state.unsqueeze(0)  # (1, G, OBS*C)
                pred = dual_model(inp)
                if pred.dim() == 3:
                    pred = pred.squeeze(0)
                # pred: (G, C)

                if use_residual:
                    step_out = curr_state[:, -C:] + pred
                else:
                    step_out = pred

                ar_outs.append(step_out.cpu())

                # Shift observation window: drop oldest, append prediction
                if OBS > 1:
                    curr_state = torch.cat([curr_state[:, C:], step_out], dim=-1)
                else:
                    curr_state = step_out

            # --- Metrics ---
            bl_cpu = baseline_last.cpu()

            for p in range(effective_P):
                y_p = y[:, p*C:(p+1)*C].cpu()  # ground truth for horizon p
                out_p = ar_outs[p]

                # Per-horizon
                sm_region_h[p].update(y_p[roi_mask], out_p[roi_mask])
                sm_base_h[p].update(y_p[roi_mask], bl_cpu[roi_mask])

                # Overall (all horizons combined)
                sm_region.update(y_p[roi_mask], out_p[roi_mask])
                sm_base_region.update(y_p[roi_mask], bl_cpu[roi_mask])

            if (i + 1) % 50 == 0:
                sk_r = 1.0 - sm_region.rmse / (sm_base_region.rmse + 1e-12)
                print(f"  [{i+1}/{max_samples}] region RMSE={sm_region.rmse:.6f} ACC={sm_region.acc:.4f} skill={sk_r*100:.2f}%")

    # === RESULTS ===
    skill_region = 1.0 - sm_region.rmse / (sm_base_region.rmse + 1e-12)

    print()
    print("=" * 60)
    print(f"=== Inference summary ({sm_region.n} samples, AR={AR_STEPS}) ===")
    print(f"Grid: G={G} | C={C} | ROI={n_roi} points")
    print()
    print(f"  REGIONAL overall (avg across {AR_STEPS} horizons):")
    print(f"    RMSE={sm_region.rmse:.6f} | MAE={sm_region.mae:.6f}")
    print(f"    Baseline RMSE={sm_base_region.rmse:.6f}")
    print(f"    Skill={skill_region*100:.2f}%")
    print(f"    ACC={sm_region.acc:.4f} | base ACC={sm_base_region.acc:.4f}")

    # Per-horizon summary
    print(f"\n  Per-horizon (regional):")
    for p in range(AR_STEPS):
        sp, sb = sm_region_h[p], sm_base_h[p]
        if sp.n == 0:
            continue
        sk = 1.0 - sp.rmse / (sb.rmse + 1e-12)
        print(f"    +{(p+1)*6:02d}h: RMSE={sp.rmse:.6f} | base={sb.rmse:.6f} | skill={sk*100:.2f}% | ACC={sp.acc:.4f} (base {sb.acc:.4f})")
    print("=" * 60)

    # === PER-CHANNEL ===
    if args.per_channel:
        data_dir_path = Path(data_dir)
        var_path = data_dir_path / "variables.json"
        var_order = json.loads(var_path.read_text()) if var_path.exists() else [f"ch{c}" for c in range(C)]

        scalers_path = data_dir_path / "scalers.npz"
        std = None
        if scalers_path.exists():
            scl = np.load(scalers_path)
            std = scl["std"].astype(np.float64)[:C]

        # --- Per-horizon per-channel table (ключевая) ---
        if AR_STEPS > 1 and std is not None:
            key_vars = ["t2m", "10u", "10v", "msl", "z@500", "t@850", "u@850", "v@850", "z@850"]
            key_idx = [i for i, v in enumerate(var_order[:C]) if v in key_vars]

            print(f"\nPer-horizon per-channel RMSE (physical units, regional):")
            header = f"  {'var':>10s} {'unit':>6s}"
            for p in range(AR_STEPS):
                header += f" {'+'+ str((p+1)*6) + 'h':>8s}"
            print(header)

            for c in key_idx:
                name = var_order[c]
                unit = UNITS.get(name, "?")
                row = f"  {name:>10s} {unit:>6s}"
                for p in range(AR_STEPS):
                    if sm_region_h[p].n == 0:
                        row += f" {'N/A':>8s}"
                        continue
                    phys_rmse = sm_region_h[p].rmse_per_channel[c] * std[c]
                    if "z@" in name or name == "z_surf":
                        row += f" {phys_rmse/9.81:7.1f}m"
                    elif name == "t2m" or name.startswith("t@"):
                        row += f" {phys_rmse:6.2f}°C"
                    else:
                        row += f" {phys_rmse:8.4f}"
                print(row)

            # Per-horizon per-channel ACC
            print(f"\nPer-horizon per-channel ACC (regional):")
            header = f"  {'var':>10s}"
            for p in range(AR_STEPS):
                header += f" {'+'+ str((p+1)*6) + 'h':>8s}"
            print(header)
            for c in key_idx:
                name = var_order[c]
                row = f"  {name:>10s}"
                for p in range(AR_STEPS):
                    if sm_region_h[p].n == 0:
                        row += f" {'N/A':>8s}"
                        continue
                    row += f" {sm_region_h[p].acc_per_channel[c]:8.4f}"
                print(row)

            # Per-horizon per-channel Skill
            print(f"\nPer-horizon per-channel Skill (regional):")
            header = f"  {'var':>10s}"
            for p in range(AR_STEPS):
                header += f" {'+'+ str((p+1)*6) + 'h':>8s}"
            print(header)
            for c in key_idx:
                name = var_order[c]
                row = f"  {name:>10s}"
                for p in range(AR_STEPS):
                    sp, sb = sm_region_h[p], sm_base_h[p]
                    if sp.n == 0:
                        row += f" {'N/A':>8s}"
                        continue
                    r_pred = sp.rmse_per_channel[c]
                    r_base = sb.rmse_per_channel[c]
                    sk_c = (1 - r_pred / (r_base + 1e-12)) * 100
                    row += f" {sk_c:7.2f}%"
                print(row)

        # --- Overall per-channel (averaged across horizons) ---
        rmse_rg = sm_region.rmse_per_channel
        rmse_bl = sm_base_region.rmse_per_channel
        acc_rg = sm_region.acc_per_channel

        if std is not None:
            print(f"\nPer-channel overall (physical units, avg over {AR_STEPS} horizons):")
            print(f"  {'#':>3s} {'var':>10s} {'unit':>8s}  {'ACC':>8s}  {'RMSE':>10s} {'RMSE_base':>10s} {'Skill':>7s}")
            for c, name in enumerate(var_order[:C]):
                unit = UNITS.get(name, "?")
                phys_rg = rmse_rg[c] * std[c]
                phys_bl = rmse_bl[c] * std[c]
                sk_c = (1 - phys_rg / (phys_bl + 1e-12)) * 100
                extra = ""
                if "z@" in name or name == "z_surf":
                    extra = f"  ({phys_rg/9.81:.1f} gpm)"
                elif name == "t2m" or name.startswith("t@"):
                    extra = f"  ({phys_rg:.2f} °C)"
                print(f"  {c:3d} {name:>10s} {unit:>8s}  {acc_rg[c]:8.4f}  {phys_rg:10.4f} {phys_bl:10.4f} {sk_c:6.2f}%{extra}")

    # Save results
    results = {
        "n_samples": sm_region.n,
        "ar_steps": AR_STEPS,
        "roi": list(roi),
        "regional_overall": {
            "rmse": sm_region.rmse, "mae": sm_region.mae, "acc": sm_region.acc,
            "baseline_rmse": sm_base_region.rmse, "skill": skill_region,
        },
        "per_horizon": [],
    }
    for p in range(AR_STEPS):
        sp, sb = sm_region_h[p], sm_base_h[p]
        if sp.n == 0:
            continue
        sk = 1.0 - sp.rmse / (sb.rmse + 1e-12)
        results["per_horizon"].append({
            "horizon_h": (p+1)*6,
            "rmse": sp.rmse, "acc": sp.acc,
            "baseline_rmse": sb.rmse, "skill": sk,
        })
    if args.per_channel:
        results["per_channel"] = {
            "variables": var_order[:C],
            "regional_rmse_norm": sm_region.rmse_per_channel.tolist(),
            "regional_acc": sm_region.acc_per_channel.tolist(),
        }

    results_path = os.path.join(exp_dir, "results", "predict_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()

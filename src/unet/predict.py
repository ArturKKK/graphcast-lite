"""
U-Net inference script.

Usage:
    python -m src.unet.predict experiments/unet_region_krsk
    python -m src.unet.predict experiments/unet_region_krsk --ar-steps 4 --max-samples 200
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.unet.model import WeatherUNet
from src.data.dataloader_chunked import TimeseriesChunkDataset


# ─────────────────────────────────────────────────────────
class Grid2DDataset(Dataset):
    def __init__(self, chunk_ds: TimeseriesChunkDataset, n_lon: int, n_lat: int):
        self.ds = chunk_ds
        self.H = n_lat
        self.W = n_lon

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        X, Y = self.ds[idx]
        X = X.view(self.H, self.W, -1).permute(2, 0, 1)
        Y = Y.view(self.H, self.W, -1).permute(2, 0, 1)
        return X, Y


# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", help="e.g. experiments/unet_region_krsk")
    ap.add_argument("--ckpt", default=None, help="path to .pth (default: best_model.pth)")
    ap.add_argument("--ar-steps", type=int, default=None, help="AR rollout steps (default: from config)")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--data-dir", default=None, help="override data_dir from config")
    args = ap.parse_args()

    # Load config
    cfg_path = os.path.join(args.experiment_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] device={device}")

    data_dir = args.data_dir or cfg["data_dir"]
    C = cfg["num_features"]
    obs_window = cfg.get("obs_window", 2)
    pred_steps = cfg.get("pred_steps", 4)
    base_filters = cfg.get("base_filters", 64)
    static_ch = cfg.get("static_channels", [])
    forcing_ch = cfg.get("forcing_channels", [])
    no_loss_ch = sorted(set(static_ch) | set(forcing_ch))

    ar_steps_eval = args.ar_steps or cfg.get("max_ar_steps", pred_steps)

    # Load dataset
    test_ds_raw = TimeseriesChunkDataset(
        data_dir, obs_window, pred_steps, split="test_only", n_features=C
    )
    n_lon = test_ds_raw.n_lon
    n_lat = test_ds_raw.n_lat
    assert n_lon and n_lat, "U-Net requires regular grid data"

    test_ds = Grid2DDataset(test_ds_raw, n_lon, n_lat)
    max_samples = min(args.max_samples, len(test_ds))

    H, W = n_lat, n_lon
    print(f"[predict] Grid: {H}×{W} | C={C} | obs={obs_window} | AR={ar_steps_eval}")
    if no_loss_ch:
        print(f"[predict] Excluded from aggregate: {no_loss_ch}")

    # Load model
    ckpt_path = args.ckpt or os.path.join(args.experiment_dir, "best_model.pth")
    model = WeatherUNet(in_channels=obs_window * C, out_channels=C, base_filters=base_filters)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"[predict] Loaded {ckpt_path} ({model.num_params:,} params)")

    # Scalers
    scalers = np.load(os.path.join(data_dir, "scalers.npz"))
    mean = scalers["mean"][:C].astype(np.float32)
    std = scalers["std"][:C].astype(np.float32)

    # Variable names
    var_path = os.path.join(data_dir, "variables.json")
    var_names = json.load(open(var_path)) if os.path.exists(var_path) else [f"ch{i}" for i in range(C)]

    # Per-horizon accumulators (physical units)
    dyn_count = C - len(no_loss_ch)
    sum_se_h = [np.zeros(C, dtype=np.float64) for _ in range(ar_steps_eval)]
    sum_se_base_h = [np.zeros(C, dtype=np.float64) for _ in range(ar_steps_eval)]
    # Per-horizon accumulators (normalized units — fair across channels)
    sum_nse_h = [np.zeros(C, dtype=np.float64) for _ in range(ar_steps_eval)]
    sum_nse_base_h = [np.zeros(C, dtype=np.float64) for _ in range(ar_steps_eval)]
    # Per-horizon ACC accumulators
    sum_acc_h = [np.zeros(C, dtype=np.float64) for _ in range(ar_steps_eval)]
    acc_count_h = [np.zeros(C, dtype=np.int64) for _ in range(ar_steps_eval)]
    count_h = [0] * ar_steps_eval

    with torch.no_grad():
        for i in range(max_samples):
            X, Y = test_ds[i]
            # X: (obs*C, H, W),  Y: (pred*C, H, W) — already channel-first from Grid2DDataset
            X = X.unsqueeze(0).to(device)  # (1, obs*C, H, W)

            curr_state = X.view(1, obs_window, C, H, W)
            Y_steps = Y.reshape(pred_steps, C, H, W)  # (pred, C, H, W)
            baseline = curr_state[0, -1].cpu()  # (C, H, W)

            for ar_step in range(min(ar_steps_eval, pred_steps)):
                inp = curr_state.view(1, obs_window * C, H, W)
                delta = model(inp)
                x_last = curr_state[:, -1]
                out = x_last + delta

                # Carry-forward
                if static_ch:
                    for ch in static_ch:
                        out[:, ch] = x_last[:, ch]
                if forcing_ch and ar_step < pred_steps:
                    target_step = Y_steps[ar_step].unsqueeze(0).to(device)
                    for ch in forcing_ch:
                        out[:, ch] = target_step[:, ch]

                # Metrics in physical units
                out_np = out[0].cpu().numpy()    # (C, H, W)
                gt_np = Y_steps[ar_step].numpy() # (C, H, W)
                bl_np = baseline.numpy()          # (C, H, W)

                for c in range(C):
                    pred_phys = out_np[c] * std[c] + mean[c]
                    gt_phys = gt_np[c] * std[c] + mean[c]
                    bl_phys = bl_np[c] * std[c] + mean[c]
                    sum_se_h[ar_step][c] += ((pred_phys - gt_phys) ** 2).sum()
                    sum_se_base_h[ar_step][c] += ((bl_phys - gt_phys) ** 2).sum()
                    # Normalized (unit-agnostic) — all channels contribute equally
                    sum_nse_h[ar_step][c] += ((out_np[c] - gt_np[c]) ** 2).sum()
                    sum_nse_base_h[ar_step][c] += ((bl_np[c] - gt_np[c]) ** 2).sum()

                    # Spatial ACC (normalized, no denorm needed)
                    p = out_np[c].flatten()
                    t = gt_np[c].flatten()
                    p = p - p.mean()
                    t = t - t.mean()
                    pn = np.sqrt((p ** 2).sum())
                    tn = np.sqrt((t ** 2).sum())
                    if pn > 1e-8 and tn > 1e-8:
                        corr = (p * t).sum() / (pn * tn)
                        sum_acc_h[ar_step][c] += corr
                        acc_count_h[ar_step][c] += 1

                count_h[ar_step] += H * W

                # Shift window
                curr_state = torch.cat([curr_state[:, 1:], out.unsqueeze(1)], dim=1)

            if (i + 1) % 50 == 0:
                # Quick intermediate RMSE for t2m
                t2m_rmse = np.sqrt(sum_se_h[0][0] / max(count_h[0], 1))
                print(f"  [{i+1}/{max_samples}] t2m@+6h={t2m_rmse:.2f}°C")

    # ─────────── Print results ───────────
    hours = [6 * (h + 1) for h in range(ar_steps_eval)]

    print(f"\n{'='*70}")
    print(f"=== U-Net Inference ({max_samples} samples, AR={ar_steps_eval}) ===")
    print(f"Grid: {H}×{W} | C={C} (dynamic={dyn_count})")

    # Per-horizon skill (normalized — fair across channels)
    print(f"\nPer-horizon (dynamic channels, normalized skill):")
    for h in range(ar_steps_eval):
        # Per-channel skill, then average (each channel contributes equally)
        skills_h = []
        accs_h = []
        for c in range(C):
            if c in no_loss_ch:
                continue
            n = max(count_h[h], 1)
            rmse_c = np.sqrt(sum_nse_h[h][c] / n)
            rmse_b = np.sqrt(sum_nse_base_h[h][c] / n)
            sk = (1.0 - rmse_c / rmse_b) * 100 if rmse_b > 1e-12 else 0
            skills_h.append(sk)
            if acc_count_h[h][c] > 0:
                accs_h.append(sum_acc_h[h][c] / acc_count_h[h][c])
        avg_skill = sum(skills_h) / max(len(skills_h), 1)
        avg_acc = sum(accs_h) / max(len(accs_h), 1)
        print(f"  +{hours[h]:02d}h: avg_skill={avg_skill:.1f}% | ACC={avg_acc:.4f}")

    # Per-horizon per-channel RMSE (physical)
    print(f"\nPer-horizon per-channel RMSE (physical units):")
    header = f"{'var':>10} {'unit':>6}" + "".join(f" {f'+{h}h':>8}" for h in hours)
    print(header)

    for c in range(C):
        if c in no_loss_ch:
            continue
        name = var_names[c] if c < len(var_names) else f"ch{c}"
        unit = _guess_unit(name)

        vals = []
        for h in range(ar_steps_eval):
            rmse = np.sqrt(sum_se_h[h][c] / max(count_h[h], 1))
            if "t2m" in name or "t@" in name:
                vals.append(f"{rmse:.2f}°C")
            elif "z@" in name or "z_" in name:
                gpm = rmse / 9.80665
                vals.append(f"{gpm:.1f}m")
            else:
                vals.append(f"{rmse:.2f}")
        line = f"{name:>10} {unit:>6}" + "".join(f" {v:>8}" for v in vals)
        print(line)

    # Per-channel overall (avg over horizons)
    print(f"\nPer-channel metrics overall (avg over {ar_steps_eval} horizons):")
    print(f"  {'#':>3} {'var':>10} {'ACC':>8} {'skill%':>8} {'RMSE_phys':>12} {'unit':>8}")
    overall_skills = []
    overall_accs = []
    for c in range(C):
        name = var_names[c] if c < len(var_names) else f"ch{c}"
        unit = _guess_unit(name)
        # Average metrics over horizons
        rmse_vals = []
        skill_vals = []
        acc_vals = []
        for h in range(ar_steps_eval):
            if count_h[h] > 0:
                rmse_vals.append(np.sqrt(sum_se_h[h][c] / count_h[h]))
                rmse_c = np.sqrt(sum_nse_h[h][c] / count_h[h])
                rmse_b = np.sqrt(sum_nse_base_h[h][c] / count_h[h])
                sk = (1.0 - rmse_c / rmse_b) * 100 if rmse_b > 1e-12 else 0
                skill_vals.append(sk)
            if acc_count_h[h][c] > 0:
                acc_vals.append(sum_acc_h[h][c] / acc_count_h[h][c])
        avg_rmse = sum(rmse_vals) / max(len(rmse_vals), 1)
        avg_skill = sum(skill_vals) / max(len(skill_vals), 1)
        avg_acc = sum(acc_vals) / max(len(acc_vals), 1)
        tag = " (static)" if c in static_ch else " (forcing)" if c in forcing_ch else ""
        print(f"  {c:3d} {name:>10}   {avg_acc:6.4f}   {avg_skill:+6.1f}%   {avg_rmse:10.4f}   {unit:>6}{tag}")
        if c not in no_loss_ch:
            overall_skills.append(avg_skill)
            overall_accs.append(avg_acc)

    # Summary line
    mean_skill = sum(overall_skills) / max(len(overall_skills), 1)
    mean_acc = sum(overall_accs) / max(len(overall_accs), 1)
    print(f"\n  >>> MEAN (dynamic): skill={mean_skill:.1f}% | ACC={mean_acc:.4f}")
    print(f"{'='*70}")


def _guess_unit(name: str) -> str:
    name = name.lower()
    if "t2m" in name or "t@" in name:
        return "K"
    if "u" in name or "v" in name:
        return "m/s"
    if "msl" in name or "sp" in name:
        return "Pa"
    if "z@" in name or "z_" in name:
        return "m²/s²"
    if "tp" in name:
        return "m"
    if "tcwv" in name or "q@" in name:
        return "kg/m²"
    if "lsm" in name:
        return "-"
    return "?"


if __name__ == "__main__":
    main()

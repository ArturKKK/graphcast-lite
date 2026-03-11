#!/usr/bin/env python3
"""
scripts/plot_region_multires.py

Conference-quality regional scatter maps for multires predictions.
Loads predictions.pt (dict with "predictions" + "ground_truth"),
denormalizes to physical units, and produces:
  1) 1×3 panel per variable: Truth | Prediction | Error (scatter on lat/lon)
  2) Summary bar chart: RMSE per variable per horizon

Works with FLAT grids (multires) — uses scatter, not imshow.

Example:
  python scripts/plot_region_multires.py \
    experiments/multires_nores_nofreeze/predictions.pt \
    --data-dir data/datasets/multires_krsk_19f \
    --region 55.5 56.5 92 94 \
    --out-dir experiments/multires_nores_nofreeze/plots \
    --vars t2m 10u 10v \
    --sample -1 --step 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── Variable display config ─────────────────────────────────────────
DISPLAY = {
    "t2m":  {"label": "Temperature 2m", "unit": "°C",   "to_phys": lambda x: x - 273.15, "cmap": "RdYlBu_r", "err_cmap": "bwr"},
    "10u":  {"label": "U-wind 10m",     "unit": "m/s",  "to_phys": lambda x: x,            "cmap": "coolwarm",  "err_cmap": "bwr"},
    "10v":  {"label": "V-wind 10m",     "unit": "m/s",  "to_phys": lambda x: x,            "cmap": "coolwarm",  "err_cmap": "bwr"},
    "sp":   {"label": "Surface Pressure","unit": "hPa", "to_phys": lambda x: x / 100.0,    "cmap": "viridis",   "err_cmap": "bwr"},
    "msl":  {"label": "Mean Sea Level P","unit": "hPa", "to_phys": lambda x: x / 100.0,    "cmap": "viridis",   "err_cmap": "bwr"},
    "t@850":{"label": "Temperature 850hPa","unit": "°C","to_phys": lambda x: x - 273.15,   "cmap": "RdYlBu_r", "err_cmap": "bwr"},
    "t@500":{"label": "Temperature 500hPa","unit": "°C","to_phys": lambda x: x - 273.15,   "cmap": "RdYlBu_r", "err_cmap": "bwr"},
    "u@850":{"label": "U-wind 850hPa",  "unit": "m/s",  "to_phys": lambda x: x,            "cmap": "coolwarm",  "err_cmap": "bwr"},
    "v@850":{"label": "V-wind 850hPa",  "unit": "m/s",  "to_phys": lambda x: x,            "cmap": "coolwarm",  "err_cmap": "bwr"},
    "z@500":{"label": "Geopotential 500hPa","unit": "dam","to_phys": lambda x: x / 9.80665 / 10, "cmap": "terrain", "err_cmap": "bwr"},
}
DEFAULT_DISPLAY = {"label": "", "unit": "", "to_phys": lambda x: x, "cmap": "viridis", "err_cmap": "bwr"}


def load_data(pred_path, data_dir):
    """Load predictions.pt dict + scalers + variable names + coords."""
    pred_path = Path(pred_path)
    data_dir = Path(data_dir)

    bundle = torch.load(pred_path, map_location="cpu")
    if isinstance(bundle, dict):
        preds = bundle["predictions"]    # (N, G, C*P)
        truth = bundle["ground_truth"]   # (N, G, C*P)
        C = bundle.get("n_features", None)
        AR = bundle.get("ar_steps", 1)
    else:
        raise ValueError("Expected predictions.pt to be a dict with 'predictions' and 'ground_truth' keys")

    # Variable names
    var_path = data_dir / "variables.json"
    if var_path.exists():
        var_names = json.loads(var_path.read_text())
    else:
        var_names = [f"ch{c}" for c in range(C)]

    if C is None:
        C = len(var_names)
    P = preds.shape[-1] // C

    # Scalers
    sc = np.load(data_dir / "scalers.npz")
    if "mean" in sc:
        mean = sc["mean"].astype(np.float32)[:C]
        std = sc["std"].astype(np.float32)[:C]
    elif "y_mean" in sc:
        mean = sc["y_mean"].astype(np.float32)[:C]
        std = sc["y_scale"].astype(np.float32)[:C]
    else:
        mean = np.zeros(C, dtype=np.float32)
        std = np.ones(C, dtype=np.float32)

    # Coords
    coords = np.load(data_dir / "coords.npz")
    lats = coords["latitude"].astype(np.float32)
    lons = coords["longitude"].astype(np.float32)

    return preds, truth, var_names, C, P, mean, std, lats, lons


def denormalize(tensor_flat, mean, std, C, P):
    """
    tensor_flat: (N, G, C*P)  →  (N, G, P, C) in physical units
    """
    N, G, CP = tensor_flat.shape
    arr = tensor_flat.numpy().reshape(N, G, P, C)
    arr = arr * std[np.newaxis, np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, np.newaxis, :]
    return arr


def region_mask(lats, lons, bbox):
    """Return boolean mask for nodes inside bbox [lat_min, lat_max, lon_min, lon_max]."""
    lat_min, lat_max, lon_min, lon_max = bbox
    return ((lats >= lat_min) & (lats <= lat_max) &
            (lons >= lon_min) & (lons <= lon_max))


def plot_trio(lats, lons, truth_2d, pred_2d, var_name, step_h, out_path, marker_size=12):
    """
    Plot 1×3 panel: Truth | Prediction | Error.
    truth_2d, pred_2d: (n_nodes,) in physical display units.
    """
    d = DISPLAY.get(var_name, {**DEFAULT_DISPLAY, "label": var_name})
    error = pred_2d - truth_2d
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    bias = np.mean(error)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # Common value range for truth/pred
    vmin = min(truth_2d.min(), pred_2d.min())
    vmax = max(truth_2d.max(), pred_2d.max())

    # 1. Ground Truth
    sc0 = axes[0].scatter(lons, lats, c=truth_2d, cmap=d["cmap"], s=marker_size,
                          edgecolors="none", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth (ERA5)\n{d['label']} [{d['unit']}]", fontsize=12)
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Prediction
    sc1 = axes[1].scatter(lons, lats, c=pred_2d, cmap=d["cmap"], s=marker_size,
                          edgecolors="none", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Prediction (+{step_h}h)\n{d['label']} [{d['unit']}]", fontsize=12)
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Error
    elim = max(abs(error.min()), abs(error.max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-elim, vcenter=0, vmax=elim)
    sc2 = axes[2].scatter(lons, lats, c=error, cmap=d["err_cmap"], s=marker_size,
                          edgecolors="none", norm=norm)
    axes[2].set_title(f"Error (Pred − Truth)\nRMSE={rmse:.2f} MAE={mae:.2f} bias={bias:+.2f} [{d['unit']}]",
                      fontsize=11)
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{var_name} — +{step_h}h forecast", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [Save] {out_path}")


def plot_rmse_bars(var_metrics, out_path):
    """
    Bar chart: RMSE per variable per horizon.
    var_metrics: dict{var_name: list[rmse_per_horizon]}
    """
    n_vars = len(var_metrics)
    if n_vars == 0:
        return
    var_names = list(var_metrics.keys())
    n_h = len(var_metrics[var_names[0]])
    horizons = [f"+{(h+1)*6}h" for h in range(n_h)]

    x = np.arange(n_vars)
    width = 0.8 / n_h

    fig, ax = plt.subplots(figsize=(max(8, n_vars * 1.5), 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_h))

    for h in range(n_h):
        vals = [var_metrics[v][h] for v in var_names]
        offset = (h - n_h / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=horizons[h], color=colors[h])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    d_labels = []
    for v in var_names:
        d = DISPLAY.get(v, DEFAULT_DISPLAY)
        unit = d["unit"] or ""
        d_labels.append(f"{v}\n[{unit}]" if unit else v)
    ax.set_xticklabels(d_labels)
    ax.set_ylabel("RMSE (physical units)")
    ax.set_title("Regional RMSE by Variable and Forecast Horizon", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [Save] {out_path}")


def main():
    import torch  # deferred import so argparse is fast

    ap = argparse.ArgumentParser(description="Conference-quality regional scatter maps for multires predictions")
    ap.add_argument("predictions", help="Path to predictions.pt")
    ap.add_argument("--data-dir", required=True, help="Dataset directory (scalers, coords, variables)")
    ap.add_argument("--region", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    help="Regional bbox for zoom (default: all nodes)")
    ap.add_argument("--out-dir", default=None, help="Output directory for PNGs")
    ap.add_argument("--vars", nargs="*", default=None, help="Variables to plot (default: t2m 10u 10v)")
    ap.add_argument("--sample", type=int, default=-1, help="Sample index (-1 = last)")
    ap.add_argument("--step", type=int, default=None, help="Single horizon step (0-indexed). If omitted, plots all.")
    ap.add_argument("--marker-size", type=float, default=12, help="Scatter marker size")
    args = ap.parse_args()

    # Defaults
    pred_path = Path(args.predictions)
    exp_dir = pred_path.parent
    out_dir = Path(args.out_dir or (exp_dir / "plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    target_vars = args.vars or ["t2m", "10u", "10v"]

    # Load
    preds, truth, var_names, C, P, mean, std, lats, lons = load_data(pred_path, args.data_dir)

    # Denormalize
    pred_phys = denormalize(preds, mean, std, C, P)   # (N, G, P, C)
    truth_phys = denormalize(truth, mean, std, C, P)

    N = pred_phys.shape[0]
    sample_idx = args.sample if args.sample >= 0 else N + args.sample

    print(f"[Info] N={N} samples, G={pred_phys.shape[1]} nodes, P={P} horizons, C={C} channels")
    print(f"[Info] Sample: {sample_idx}, Variables: {target_vars}")

    # Region mask
    if args.region:
        rmask = region_mask(lats, lons, args.region)
        r_lats = lats[rmask]
        r_lons = lons[rmask]
        pred_r = pred_phys[:, rmask, :, :]
        truth_r = truth_phys[:, rmask, :, :]
        print(f"[Region] {rmask.sum()} nodes in [{args.region[0]:.1f},{args.region[1]:.1f}]N × [{args.region[2]:.1f},{args.region[3]:.1f}]E")
    else:
        r_lats, r_lons = lats, lons
        pred_r, truth_r = pred_phys, truth_phys
        print(f"[Region] All {len(lats)} nodes (no --region filter)")

    # Determine horizons to plot
    steps = [args.step] if args.step is not None else list(range(P))

    # Per-variable per-horizon RMSE for summary chart
    var_metrics = {}

    for vname in target_vars:
        if vname not in var_names:
            print(f"  [SKIP] {vname} not in dataset")
            continue

        vi = var_names.index(vname)
        d = DISPLAY.get(vname, {**DEFAULT_DISPLAY, "label": vname})
        to_phys = d["to_phys"]

        horizon_rmses = []

        for step in steps:
            step_h = (step + 1) * 6  # hours

            t_vals = to_phys(truth_r[sample_idx, :, step, vi])
            p_vals = to_phys(pred_r[sample_idx, :, step, vi])

            # Scatter trio
            fname = f"{vname}_+{step_h:02d}h_sample{sample_idx}.png"
            plot_trio(r_lats, r_lons, t_vals, p_vals, vname, step_h,
                      out_dir / fname, marker_size=args.marker_size)

            # RMSE for bar chart (over ALL samples, not just one)
            all_t = to_phys(truth_r[:, :, step, vi])   # (N, n_nodes)
            all_p = to_phys(pred_r[:, :, step, vi])
            rmse_h = np.sqrt(np.mean((all_p - all_t) ** 2))
            horizon_rmses.append(rmse_h)

        var_metrics[vname] = horizon_rmses

    # Summary bar chart
    if var_metrics:
        plot_rmse_bars(var_metrics, out_dir / "rmse_summary.png")

    # Print summary table
    print("\n" + "=" * 70)
    print("REGIONAL RMSE SUMMARY (physical units, all samples)")
    print("=" * 70)
    header = f"{'Variable':<10}"
    for step in steps:
        header += f"  +{(step+1)*6:02d}h"
    header += "    AVG"
    print(header)
    print("-" * len(header))

    for vname, rmses in var_metrics.items():
        d = DISPLAY.get(vname, DEFAULT_DISPLAY)
        row = f"{vname:<10}"
        for r in rmses:
            row += f"  {r:6.3f}"
        row += f"   {np.mean(rmses):6.3f} {d['unit']}"
        print(row)

    print("=" * 70)
    print(f"\nAll plots saved to: {out_dir}")


# need torch at module level for load_data
import torch

if __name__ == "__main__":
    main()

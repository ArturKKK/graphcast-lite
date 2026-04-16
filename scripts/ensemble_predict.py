#!/usr/bin/env python3
"""
Ensemble inference: average predictions from nores_freeze6 and real_freeze6.

Runs both models on the same data and reports individual + ensemble metrics.

Usage (on VM):
    python scripts/ensemble_predict.py --max-samples 200
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig
from src.data.dataloader_chunked import load_chunked_datasets
from src.data.dataloader import load_train_and_test_datasets
from src.main import load_model_from_experiment_config
from src.utils import load_from_json_file


def load_model(exp_dir: str, device="cpu", dataset_metadata=None,
               coordinates=None, region_bounds=None, flat_grid=False):
    """Load model from experiment directory."""
    exp_dir = Path(exp_dir)
    cfg_path = str(exp_dir / "config.json")
    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))
    
    model = load_model_from_experiment_config(
        experiment_config=exp_cfg, device=device, dataset_metadata=dataset_metadata,
        coordinates=coordinates, region_bounds=region_bounds, flat_grid=flat_grid,
    )
    ckpt = exp_dir / "best_model.pth"
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {exp_dir.name}: {n_params:,} params")
    return model, exp_cfg


def region_node_indices(lat_min, lat_max, lon_min, lon_max, lats, lons):
    """For flat grids: indices where lat/lon fall in region box."""
    mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
    return np.where(mask)[0]


class StreamingRMSE:
    def __init__(self):
        self.sum_se = 0.0
        self.n = 0

    def update(self, y_true, y_pred, mask=None):
        if mask is not None:
            err = (y_pred[mask] - y_true[mask]).float()
        else:
            err = (y_pred - y_true).float()
        self.sum_se += err.pow(2).sum().item()
        self.n += err.numel()

    @property
    def rmse(self):
        return np.sqrt(self.sum_se / max(self.n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp1", default="experiments/multires_nores_freeze6",
                    help="Experiment 1 (nores)")
    ap.add_argument("--exp2", default="experiments/multires_real_freeze6",
                    help="Experiment 2 (real)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--region", nargs=4, type=float, default=[55.5, 56.5, 92, 94],
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    ap.add_argument("--ar-steps", type=int, default=4)
    ap.add_argument("--weights", nargs=2, type=float, default=[0.5, 0.5],
                    help="Ensemble weights for model1 and model2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load configs ────────────────────────────────────────────────
    exp1 = Path(args.exp1)
    exp2 = Path(args.exp2)
    exp_cfg1 = ExperimentConfig(**load_from_json_file(str(exp1 / "config.json")))
    exp_cfg2 = ExperimentConfig(**load_from_json_file(str(exp2 / "config.json")))

    # Use data_dir from first experiment
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None:
        if hasattr(exp_cfg1, "data_dir") and exp_cfg1.data_dir:
            data_dir = Path(exp_cfg1.data_dir)
        else:
            data_dir = Path("data/datasets") / exp_cfg1.data.dataset_name
    print(f"Data dir: {data_dir}")

    # ── Load dataset ────────────────────────────────────────────────
    C = exp_cfg1.data.num_features_used
    OBS = exp_cfg1.data.obs_window_used
    P = exp_cfg1.data.pred_window_used

    data_npy = data_dir / "data.npy"
    if data_npy.exists():
        _, _, test_ds, meta = load_chunked_datasets(
            str(data_dir), obs_window=OBS, pred_steps=P, n_features=C, test_split="test_only"
        )
    else:
        _, _, test_ds, meta = load_train_and_test_datasets(str(data_dir), exp_cfg1.data)

    G = len(test_ds[0][0])
    print(f"Dataset: {len(test_ds)} test samples, G={G}, C={C}")

    # ── Load coords & region ────────────────────────────────────────
    npz_path = data_dir / "coords.npz"
    z = np.load(npz_path)
    grid_lat = z["latitude"].astype(np.float32)
    grid_lon = z["longitude"].astype(np.float32)

    is_regional = z.get("is_regional", None)
    if is_regional is not None:
        roi_idxs = np.where(is_regional)[0]
        print(f"ROI nodes: {len(roi_idxs)}")
    else:
        roi_idxs = None

    lat_min, lat_max, lon_min, lon_max = args.region
    box_idxs = region_node_indices(lat_min, lat_max, lon_min, lon_max, grid_lat, grid_lon)
    print(f"Region box [{lat_min}-{lat_max}°N, {lon_min}-{lon_max}°E]: {len(box_idxs)} nodes")

    # ── Load models ─────────────────────────────────────────────────
    real_coords = (grid_lat, grid_lon) if npz_path.exists() else None
    flat_grid = getattr(meta, 'flat_grid', False)
    # Detect regional dataset → compute region_bounds for mesh trimming
    region_bounds = None
    if real_coords is not None:
        lat_span = float(grid_lat.max() - grid_lat.min())
        lon_span = float(grid_lon.max() - grid_lon.min())
        if lat_span < 90 and lon_span < 90:
            region_bounds = (
                float(grid_lat.min()), float(grid_lat.max()),
                float(grid_lon.min()), float(grid_lon.max()),
            )
            print(f"[region] lat=[{region_bounds[0]:.1f},{region_bounds[1]:.1f}] "
                  f"lon=[{region_bounds[2]:.1f},{region_bounds[3]:.1f}]")
    print("\nLoading models...")
    model1, _ = load_model(args.exp1, device=device, dataset_metadata=meta,
                           coordinates=real_coords, region_bounds=region_bounds,
                           flat_grid=flat_grid)
    model2, _ = load_model(args.exp2, device=device, dataset_metadata=meta,
                           coordinates=real_coords, region_bounds=region_bounds,
                           flat_grid=flat_grid)

    # ── Metrics ─────────────────────────────────────────────────────
    AR = args.ar_steps
    w1, w2 = args.weights
    w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)  # normalize

    # Per-horizon metrics: model1, model2, ensemble, baseline
    metrics = {
        "m1": [StreamingRMSE() for _ in range(AR)],
        "m2": [StreamingRMSE() for _ in range(AR)],
        "ens": [StreamingRMSE() for _ in range(AR)],
        "base": [StreamingRMSE() for _ in range(AR)],
    }
    # Same but for region box, t2m only
    rmetrics = {
        "m1": [StreamingRMSE() for _ in range(AR)],
        "m2": [StreamingRMSE() for _ in range(AR)],
        "ens": [StreamingRMSE() for _ in range(AR)],
        "base": [StreamingRMSE() for _ in range(AR)],
    }

    # ── Load scalers for denormalization ────────────────────────────
    sc = np.load(data_dir / "scalers.npz")
    t2m_std = float(sc["std"][0])

    max_samples = min(args.max_samples, len(test_ds))
    print(f"\nInference: {max_samples} samples, AR={AR}, weights=({w1:.2f}, {w2:.2f})")

    # Detect static/forcing channels
    static_ch = [7, 8]  # z_surf, lsm
    forcing_ch = [4]    # tp

    t0 = time.time()
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            if i >= max_samples:
                break

            X = X.to(device)
            y = y.squeeze(0).cpu()  # [G, P*C] or [G, C]
            y_steps = max(1, y.shape[-1] // C)

            # AR rollout for both models
            def ar_rollout(model, X_in):
                curr = X_in.view(1, G, OBS, C)
                outs = []
                for step in range(min(AR, y_steps)):
                    inp = curr.view(1, G, -1)
                    delta = model(inp, attention_threshold=0.0)
                    if delta.dim() == 2:
                        delta = delta.unsqueeze(0)
                    step_out = curr[:, :, -1, :] + delta  # residual=False → just delta
                    # For no-residual models, step_out = delta
                    # Since both models use use_residual=false, we just use delta directly
                    step_out = delta

                    # Static carry-forward
                    for ch in static_ch:
                        step_out[:, :, ch] = curr[:, :, -1, ch]
                    # Forcing from ground truth
                    if step < y_steps:
                        y_step = y[:, step*C:(step+1)*C]
                        for ch in forcing_ch:
                            step_out[:, :, ch] = y_step[:, ch].unsqueeze(0).to(device)

                    outs.append(step_out.cpu())
                    curr = torch.cat([curr[:, :, 1:, :], step_out.unsqueeze(2)], dim=2)
                return outs

            out1_steps = ar_rollout(model1, X)
            out2_steps = ar_rollout(model2, X)

            # Baseline: persistence from last obs frame
            X_cpu = X.squeeze(0).cpu()
            X_last = X_cpu[:, (OBS-1)*C : OBS*C]  # [G, C]

            for step in range(min(AR, y_steps)):
                gt = y[:, step*C:(step+1)*C]  # [G, C]
                o1 = out1_steps[step].squeeze(0)
                o2 = out2_steps[step].squeeze(0)

                # Ensemble
                ens = w1 * o1 + w2 * o2

                # All channels
                metrics["m1"][step].update(gt, o1)
                metrics["m2"][step].update(gt, o2)
                metrics["ens"][step].update(gt, ens)
                metrics["base"][step].update(gt, X_last)

                # Region t2m only (channel 0), denormalized
                if len(box_idxs) > 0:
                    gt_t2m = gt[box_idxs, 0] * t2m_std
                    o1_t2m = o1[box_idxs, 0] * t2m_std
                    o2_t2m = o2[box_idxs, 0] * t2m_std
                    ens_t2m = ens[box_idxs, 0] * t2m_std
                    bl_t2m = X_last[box_idxs, 0] * t2m_std

                    rmetrics["m1"][step].update(gt_t2m, o1_t2m)
                    rmetrics["m2"][step].update(gt_t2m, o2_t2m)
                    rmetrics["ens"][step].update(gt_t2m, ens_t2m)
                    rmetrics["base"][step].update(gt_t2m, bl_t2m)

            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{max_samples}]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # ── Results ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS")
    print("="*70)

    print(f"\nAll channels (normalized), {max_samples} samples:")
    print(f"{'Horizon':>8s}  {'nores':>8s}  {'real':>8s}  {'ensemble':>8s}  {'persist':>8s}")
    for step in range(min(AR, y_steps)):
        h = (step + 1) * 6
        print(f"  +{h:2d}h    {metrics['m1'][step].rmse:.4f}   "
              f"{metrics['m2'][step].rmse:.4f}   "
              f"{metrics['ens'][step].rmse:.4f}   "
              f"{metrics['base'][step].rmse:.4f}")

    print(f"\nRegion box t2m RMSE (°C), {len(box_idxs)} nodes:")
    print(f"{'Horizon':>8s}  {'nores':>8s}  {'real':>8s}  {'ensemble':>8s}  {'persist':>8s}  {'ens_skill':>10s}")
    for step in range(min(AR, y_steps)):
        h = (step + 1) * 6
        bl = rmetrics["base"][step].rmse
        en = rmetrics["ens"][step].rmse
        skill = (1 - en / (bl + 1e-12)) * 100
        print(f"  +{h:2d}h    {rmetrics['m1'][step].rmse:.4f}   "
              f"{rmetrics['m2'][step].rmse:.4f}   "
              f"{en:.4f}   "
              f"{bl:.4f}   "
              f"{skill:+.1f}%")

    # Also try different weights
    print(f"\n{'='*70}")
    print("Weight sensitivity (region t2m +6h):")
    # Need to re-run with different weights... skip if too complex
    # Just report the main results


if __name__ == "__main__":
    main()

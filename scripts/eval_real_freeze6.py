#!/usr/bin/env python3
"""Quick eval of multires_real_freeze6 on merge dataset."""

import json, os, sys, time
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

VAR_ORDER = [
    "t2m","10u","10v","msl","tp","sp","tcwv",
    "z_surf","lsm",
    "t@850","u@850","v@850","z@850","q@850",
    "t@500","u@500","v@500","z@500","q@500",
]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-exp", default="experiments/multires_real_freeze6")
    ap.add_argument("--ar-steps", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AR = args.ar_steps

    # ── Load config ──
    exp_dir = args.gnn_exp
    cfg_path = os.path.join(exp_dir, FileNames.EXPERIMENT_CONFIG)
    ckpt_path = os.path.join(exp_dir, "best_model.pth")
    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))

    data_dir = Path(exp_cfg.data_dir)
    assert data_dir.exists(), f"data_dir not found: {data_dir}"

    # ── Load dataset ──
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
    N = len(flat_lats)
    C = ds_info["n_feat"]
    OBS = 2

    print(f"N_total={N}, N_regional={n_regional}, C={C}, mode={ds_info.get('mode','?')}")

    sc = np.load(data_dir / "scalers.npz")
    y_mean = sc["mean"][:C].astype(np.float32)
    y_std = sc["std"][:C].astype(np.float32)

    train_ds, val_ds, test_ds, metadata = load_chunked_datasets(
        data_path=str(data_dir), obs_window=OBS, pred_steps=1,
        n_features=C, test_split="test_only",
    )

    # ── Load model ──
    print("Loading GNN model...")
    ROI = (50.0, 60.0, 83.0, 98.0)
    gnn_model = load_model_from_experiment_config(
        exp_cfg, device, metadata,
        coordinates=(flat_lats, flat_lons),
        region_bounds=ROI, mesh_buffer=15.0, flat_grid=True,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k: v for k, v in state.items()
             if not k.startswith("_processing_edge_features")}
    gnn_model.load_state_dict(state, strict=False)
    gnn_model = gnn_model.to(device)
    gnn_model.eval()
    n_params = sum(p.numel() for p in gnn_model.parameters())
    print(f"  GNN loaded on {device}, params={n_params:,}")

    use_residual = getattr(exp_cfg, 'use_residual', False)

    # ── Eval loop ──
    max_samples = min(args.max_samples, len(test_ds))
    step_sz = max(1, len(test_ds) // max_samples)
    sample_indices = [i * step_sz for i in range(max_samples)]
    print(f"Evaluating {max_samples} samples, AR={AR}")

    t2m_idx = VAR_ORDER.index("t2m")

    # Per-horizon accumulators
    mse_region = [np.zeros(C, np.float64) for _ in range(AR)]
    mse_persist = [np.zeros(C, np.float64) for _ in range(AR)]
    count = 0
    t0 = time.time()

    for si, idx in enumerate(sample_indices):
        sample = test_ds[idx]
        x_tensor = sample[0].unsqueeze(0).to(device)  # (1, OBS, N, C)
        y_true_all = sample[1].cpu().numpy()           # (1, N, C) flat

        # Reshape y_true_all if needed
        if y_true_all.ndim == 2:
            y_true_all = y_true_all[np.newaxis, :]  # (1, N, C)

        # AR rollout
        current_input = x_tensor
        for ar in range(AR):
            with torch.no_grad():
                pred = gnn_model(current_input)  # (1, N, C)
            pred_np = pred.cpu().numpy().squeeze(0)  # (N, C)

            if ar == 0:
                # GT for step 0 is y_true_all[0]
                gt = y_true_all[0] if y_true_all.shape[0] > 0 else y_true_all.squeeze(0)
            else:
                # For AR > 0, we need next timestep GT
                next_idx = idx + ar
                if next_idx >= len(test_ds):
                    break
                next_sample = test_ds[next_idx]
                gt = next_sample[1].cpu().numpy()
                if gt.ndim == 3:
                    gt = gt[0]

            # Physical units
            pred_phys = pred_np * y_std + y_mean
            gt_phys = gt * y_std + y_mean

            # Persistence baseline (last obs frame)
            if ar == 0:
                persist_norm = x_tensor[0, -1].cpu().numpy()  # (N, C)
            persist_phys = persist_norm * y_std + y_mean

            # Regional MSE
            pred_reg = pred_phys[region_idx]
            gt_reg = gt_phys[region_idx]
            persist_reg = persist_phys[region_idx]

            mse_region[ar] += ((pred_reg - gt_reg) ** 2).mean(axis=0)
            mse_persist[ar] += ((persist_reg - gt_reg) ** 2).mean(axis=0)

            # Prepare next step input
            if use_residual:
                full_pred = current_input[0, -1] + pred.squeeze(0)
            else:
                full_pred = pred.squeeze(0)

            current_input = torch.cat([
                current_input[:, 1:, :, :],
                full_pred.unsqueeze(0).unsqueeze(1)
            ], dim=1)

        count += 1
        if (si + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{si+1}/{max_samples}] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {count} samples in {elapsed:.1f}s\n")

    # ── Results ──
    print("=" * 70)
    print(f"RESULTS: {args.gnn_exp}")
    print(f"  Dataset mode: {ds_info.get('mode','?')}, samples={count}")
    print("=" * 70)

    for ar in range(AR):
        rmse = np.sqrt(mse_region[ar] / count)
        rmse_p = np.sqrt(mse_persist[ar] / count)
        skill = 1.0 - rmse / np.clip(rmse_p, 1e-9, None)

        h = (ar + 1) * 6
        t2m_rmse = rmse[t2m_idx]
        t2m_rmse_p = rmse_p[t2m_idx]
        t2m_skill = skill[t2m_idx]

        # Mean over dynamic channels (first 7, skip static z_surf/lsm)
        dyn_skill = skill[:7].mean()

        print(f"\n+{h}h:")
        print(f"  t2m:  RMSE={t2m_rmse:.3f}°C  persist={t2m_rmse_p:.3f}°C  skill={t2m_skill*100:.2f}%")
        print(f"  dynamic (7var): mean_skill={dyn_skill*100:.2f}%")

        if ar == 0:
            print(f"\n  Per-variable +6h skill:")
            for vi, vname in enumerate(VAR_ORDER[:7]):
                print(f"    {vname:8s}: RMSE={rmse[vi]:.4f}  skill={skill[vi]*100:.2f}%")

if __name__ == "__main__":
    main()

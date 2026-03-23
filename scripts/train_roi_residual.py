#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig
from src.constants import FileNames
from src.data.dataloader_chunked import load_chunked_datasets
from src.main import load_model_from_experiment_config, set_random_seeds
from src.roi_residual import ROIResidualModel
from src.train import spatial_corr, weighted_mse_loss
from src.utils import load_from_json_file, save_to_json_file


def overfit_test(model, dataloader, device, n_steps=200, lr=0.01):
    print("\n" + "=" * 60)
    print(f"[OverfitTest] 1 sample, {n_steps} steps, lr={lr}...")
    print("=" * 60)

    roi_mask = model.roi_mask
    X, y = next(iter(dataloader))
    X, y = X.to(device), y.to(device)
    y = y.squeeze(0) if y.dim() == 4 else y
    N, G, feat_dim = X.shape
    C = feat_dim // model.obs_window
    total_target_steps = y.shape[-1] // C
    if total_target_steps > 1:
        y_step0 = y.view(N, G, total_target_steps, C)[:, :, 0, :]
    else:
        y_step0 = y

    saved_state = deepcopy({
        k: v for k, v in model.state_dict().items()
        if not k.startswith("global_model.")
    })
    trainable_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("global_model.") and p.requires_grad
    ]
    opt = Adam(trainable_params, lr=lr)

    model.train()
    model.global_model.eval()

    for step in range(n_steps):
        opt.zero_grad()
        pred = model(X)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        out_roi = pred[:, roi_mask, :]
        y_roi = y_step0[:, roi_mask, :]
        loss = weighted_mse_loss(out_roi, y_roi, None)
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == n_steps - 1:
            with torch.no_grad():
                global_pred = model.global_model(X=X, attention_threshold=0.0)
                if global_pred.dim() == 2:
                    global_pred = global_pred.unsqueeze(0)
                correction = out_roi - global_pred[:, roi_mask, :]
            print(
                f"    step {step:3d}: loss={loss.item():.6f}  "
                f"|correction|={correction.abs().mean().item():.6f}  "
                f"max={correction.abs().max().item():.4f}"
            )

    current_state = model.state_dict()
    for k, v in saved_state.items():
        current_state[k].copy_(v)

    print("=" * 60)
    print("[OverfitTest] DONE. Model weights RESTORED to initial state.")
    print("=" * 60 + "\n")


def train_epoch(model, dataloader, optimizer, device, check_grads=False):
    model.train()
    model.global_model.eval()
    roi_mask = model.roi_mask

    total_loss = 0.0
    n_batches = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y = y.squeeze(0) if y.dim() == 4 else y

        N, G, feat_dim = X.shape
        C = feat_dim // model.obs_window
        total_target_steps = y.shape[-1] // C
        if total_target_steps > 1:
            y_step0 = y.view(N, G, total_target_steps, C)[:, :, 0, :]
        else:
            y_step0 = y

        optimizer.zero_grad()
        pred = model(X)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        out_roi = pred[:, roi_mask, :]
        y_roi = y_step0[:, roi_mask, :]
        loss = weighted_mse_loss(out_roi, y_roi, None)
        loss.backward()

        if check_grads and n_batches == 0:
            print("[GradCheck] out.requires_grad:", pred.requires_grad)
            module_grads = {}
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grad_norm = p.grad.norm().item()
                    module = name.split(".")[0]
                    module_grads.setdefault(module, []).append((name, grad_norm))
            for module, grads in sorted(module_grads.items()):
                max_gn = max(g for _, g in grads)
                mean_gn = sum(g for _, g in grads) / len(grads)
                top_param = max(grads, key=lambda item: item[1])
                print(f"  [{module}] {len(grads)} params, max_grad={max_gn:.6f}, mean_grad={mean_gn:.6f}")
                print(f"    top: {top_param[0]} = {top_param[1]:.6f}")

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        last_X = X
        last_out_roi = out_roi.detach()
        last_y_roi = y_roi.detach()

    if n_batches > 0:
        with torch.no_grad():
            model.eval()
            global_pred = model.global_model(X=last_X, attention_threshold=0.0)
            if global_pred.dim() == 2:
                global_pred = global_pred.unsqueeze(0)
            global_roi = global_pred[:, model.roi_mask, :]
            correction = last_out_roi - global_roi
            baseline_loss = weighted_mse_loss(global_roi, last_y_roi, None).item()
            model.train()
            model.global_model.eval()
        print(
            f"  [Correction] mean_abs={correction.abs().mean().item():.6f}, "
            f"max_abs={correction.abs().max().item():.4f}, "
            f"baseline_mse={baseline_loss:.6f}"
        )

    return total_loss / max(n_batches, 1)


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    acc_values = []
    roi_mask = model.roi_mask

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.squeeze(0) if y.dim() == 4 else y

            N, G, feat_dim = X.shape
            C = feat_dim // model.obs_window
            total_target_steps = y.shape[-1] // C
            if total_target_steps > 1:
                y_step0 = y.view(N, G, total_target_steps, C)[:, :, 0, :]
            else:
                y_step0 = y

            pred = model(X)
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)
            out_roi = pred[:, roi_mask, :]
            y_roi = y_step0[:, roi_mask, :]
            loss = weighted_mse_loss(out_roi, y_roi, None)
            total_loss += loss.item()
            n_batches += 1
            acc_values.append(spatial_corr(out_roi, y_roi))

    return total_loss / max(n_batches, 1), sum(acc_values) / max(len(acc_values), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--processor-steps", type=int, default=6)
    parser.add_argument("--roi-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        required=True,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
    )
    parser.add_argument("--overfit-only", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    cfg_path = exp_dir / FileNames.EXPERIMENT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = ExperimentConfig(**load_from_json_file(str(cfg_path)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_random_seeds(42)

    data_dir = args.data_dir or getattr(exp_cfg, "data_dir", None)
    if data_dir is None:
        raise ValueError("data_dir must be provided either in config or via --data-dir")

    train_ds, val_ds, test_ds, meta = load_chunked_datasets(
        data_path=data_dir,
        obs_window=exp_cfg.data.obs_window_used,
        pred_steps=1,
        n_features=exp_cfg.data.num_features_used,
    )

    use_cuda = device.type == "cuda"
    loader_kwargs = dict(
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    real_coords = getattr(meta, "cordinates", None)
    flat_grid = getattr(meta, "flat_grid", False)
    global_model = load_model_from_experiment_config(
        exp_cfg,
        device,
        meta,
        coordinates=real_coords,
        flat_grid=flat_grid,
    )

    print(f"\n>>> Loading pretrained global weights: {args.pretrained}")
    state = torch.load(args.pretrained, map_location=device)
    missing, unexpected = global_model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    global_model = global_model.to(device)
    for p in global_model.parameters():
        p.requires_grad = False
    print("[ROIResidualTrain] Global model FROZEN.")

    if flat_grid and real_coords is not None:
        grid_lats = real_coords[0].astype(np.float32)
        grid_lons = real_coords[1].astype(np.float32)
    else:
        lats = np.linspace(-90, 90, meta.num_latitudes, endpoint=True)
        lons = np.linspace(0, 360, meta.num_longitudes, endpoint=False)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        grid_lats = lat_grid.flatten().astype(np.float32)
        grid_lons = lon_grid.flatten().astype(np.float32)

    roi = tuple(args.roi)
    print(f"\n>>> ROI: {roi}")
    model = ROIResidualModel(
        global_model=global_model,
        roi=roi,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        device=device,
        hidden_dim=args.hidden_dim,
        processor_steps=args.processor_steps,
        roi_k=args.roi_k,
    ).to(device)

    trainable_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("global_model.") and p.requires_grad
    ]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"\n[Optimizer] {n_params:,} trainable ROI parameters, lr={args.lr}")
    optimizer = Adam(trainable_params, lr=args.lr)

    overfit_test(model, train_loader, device)
    if args.overfit_only:
        print("\n[--overfit-only] Done. Exiting without full training.")
        return

    num_epochs = args.epochs or exp_cfg.num_epochs
    best_val_loss = float("inf")
    patience = 0
    max_patience = 10
    log_path = results_dir / "training_log.txt"

    def log(msg: str):
        print(msg)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(msg + "\n")

    log(f"{'epoch':>5}  {'train_loss':>10}  {'val_loss':>10}  {'val_ACC':>8}  {'best':>10}")
    log("-" * 55)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, check_grads=(epoch == 0))
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            head_state = {
                k: v for k, v in model.state_dict().items()
                if not k.startswith("global_model.")
            }
            torch.save(head_state, results_dir / "best_head.pth")
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            improved = " *"
        else:
            patience += 1

        log(f"{epoch + 1:5d}  {train_loss:10.6f}  {val_loss:10.6f}  {val_acc:8.4f}  {best_val_loss:10.6f}{improved}")
        if patience >= max_patience:
            log(f"Early stopping at epoch {epoch + 1}")
            break

    best_ckpt = results_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    test_loss, test_acc = eval_epoch(model, test_loader, device)
    log(f"\n[TEST] loss={test_loss:.6f}  regional_ACC={test_acc:.4f}")

    save_to_json_file(
        {
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "test_regional_acc": test_acc,
            "roi": list(roi),
            "hidden_dim": args.hidden_dim,
            "processor_steps": args.processor_steps,
            "roi_k": args.roi_k,
            "lr": args.lr,
            "pretrained": args.pretrained,
            "data_dir": data_dir,
        },
        str(results_dir / "results.json"),
    )
    log(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
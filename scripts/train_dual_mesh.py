#!/usr/bin/env python3
"""
scripts/train_dual_mesh.py

Обучение DualMeshModel: глобальная pretrained модель (frozen) + региональный refined меш.

Использование:
  python scripts/train_dual_mesh.py experiments/dual_mesh \
    --pretrained experiments/multires_nores_freeze6/results/best_model.pth \
    --data-dir data/datasets/multires_krsk_19f

Аргументы:
  experiment_dir    — папка с config.json (+ dual_config.json опционально)
  --pretrained      — чекпоинт глобальной модели (обязательно)
  --data-dir        — путь к датасету (override data_dir из конфига)
  --epochs          — число эпох (override)
  --lr              — learning rate для regional модулей (default: 5e-4)
  --freeze-global   — полностью заморозить глобальную модель (default: True)
  --reg-mesh-level  — уровень икосаэдра для регионального меша (default: 7)
  --reg-steps       — число шагов регионального процессора (default: 4)
  --cross-k         — число cross-edge соседей (default: 3)
  --resume          — возобновить из checkpoint.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import FileNames
from src.config import ExperimentConfig, DatasetNames
from src.utils import load_from_json_file, save_to_json_file
from src.data.dataloader_chunked import load_chunked_datasets
from src.main import load_model_from_experiment_config, set_random_seeds
from src.dual_mesh import DualMeshModel
from copy import deepcopy
from src.train import get_lat_weights, weighted_mse_loss, spatial_corr


def apply_special_channels(out, prev_state, target_step, static_channels=None, forcing_channels=None):
    """Carry forward static channels and inject known forcing channels."""
    if static_channels:
        prev_last = prev_state[:, :, -1, :]
        for ch in static_channels:
            out[:, :, ch] = prev_last[:, :, ch]
    if forcing_channels and target_step is not None:
        for ch in forcing_channels:
            out[:, :, ch] = target_step[:, :, ch]
    return out


def overfit_test(model, dataloader, device, use_residual=False, n_steps=200, lr=0.01):
    """Overfit на 1 сэмпл — если loss падает, модель МОЖЕТ учиться."""
    print("\n" + "="*60)
    print("[OverfitTest] 1 sample, {} steps, lr={}...".format(n_steps, lr))
    print("="*60)

    roi_mask = model.roi_mask

    # Берём первый батч
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

    # Сохраняем state dict и optimizer
    saved_state = deepcopy({k: v for k, v in model.state_dict().items()
                            if not k.startswith("global_model.")})

    regional_params = [p for n, p in model.named_parameters()
                       if not n.startswith("global_model.") and p.requires_grad]
    opt = Adam(regional_params, lr=lr)

    model.train()
    model.global_model.eval()

    for step in range(n_steps):
        opt.zero_grad()
        pred = model(X)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if use_residual:
            X_reshaped = X.view(N, G, model.obs_window, C)
            out = X_reshaped[:, :, -1, :] + pred
        else:
            out = pred
        out_roi = out[:, roi_mask, :]
        y_roi = y_step0[:, roi_mask, :]
        loss = weighted_mse_loss(out_roi, y_roi, None)
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == n_steps - 1:
            with torch.no_grad():
                # Correction magnitude
                global_pred = model.global_model(X=X, attention_threshold=0.0)
                if global_pred.dim() == 2:
                    global_pred = global_pred.unsqueeze(0)
                if use_residual:
                    X_reshaped = X.view(N, G, model.obs_window, C)
                    global_out = X_reshaped[:, :, -1, :] + global_pred
                else:
                    global_out = global_pred
                global_roi = global_out[:, roi_mask, :]
                correction = out_roi - global_roi
            print(f"    step {step:3d}: loss={loss.item():.6f}  "
                  f"|correction|={correction.abs().mean().item():.6f}  "
                  f"max={correction.abs().max().item():.4f}")

    # Восстанавливаем веса
    current_state = model.state_dict()
    for k, v in saved_state.items():
        current_state[k].copy_(v)

    loss_start = None  # just for clarity
    print("="*60)
    print("[OverfitTest] DONE. Model weights RESTORED to initial state.")
    print("="*60 + "\n")


def train_dual_epoch(
    model: DualMeshModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lat_weights=None,
    use_residual: bool = False,
    current_ar_steps: int = 1,
    static_channels=None,
    forcing_channels=None,
    check_grads: bool = False,
    cache: list = None,
):
    """Один проход обучения DualMeshModel.

    loss считается ТОЛЬКО по ROI точкам.

    If cache is not None: uses precomputed global outputs (forward_cached),
    skipping the global model entirely. Each epoch is ~30s instead of hours.
    """
    model.train()
    model.global_model.eval()

    roi_mask = model.roi_mask

    total_loss = 0
    n_batches = 0

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.squeeze(0) if y.dim() == 4 else y

        N, G, feat_dim = X.shape
        C = feat_dim // model.obs_window
        total_target_steps = y.shape[-1] // C
        y_steps = y.view(N, G, total_target_steps, C)

        optimizer.zero_grad()

        if cache is not None:
            # ── Cached mode: skip global model, AR=1 only ──
            roi_raw = X[0][roi_mask]  # (n_roi, T*F)
            c = cache[batch_idx]
            out_roi = model.forward_cached(
                roi_raw=roi_raw,
                global_pred_roi=c['global_pred_roi'].to(device),
                roi_grid_latent=c['roi_grid_latent'].to(device),
                cross_sender_feat=c['cross_sender_feat'].to(device),
            )
            if use_residual:
                x_last_roi = X[0, roi_mask, -C:]
                out_roi = x_last_roi + out_roi
            target = y_steps[:, :, 0, :]
            y_roi = target[:, roi_mask, :]
            loss = weighted_mse_loss(out_roi.unsqueeze(0), y_roi, None)
        else:
            # ── Standard mode: full forward with global model ──
            obs = model.obs_window
            curr_state = X.view(N, G, obs, C)
            steps_to_run = min(current_ar_steps, total_target_steps)
            loss = 0.0
            out_roi = None
            y_roi = None

            for step in range(steps_to_run):
                inp = curr_state.view(N, G, -1)
                pred = model(inp)
                if pred.dim() == 2:
                    pred = pred.unsqueeze(0)

                if use_residual:
                    x_last = curr_state[:, :, -1, :]
                    out = x_last + pred
                else:
                    out = pred

                target = y_steps[:, :, step, :]
                out = apply_special_channels(
                    out, curr_state, target,
                    static_channels=static_channels,
                    forcing_channels=forcing_channels,
                )

                out_roi = out[:, roi_mask, :]
                y_roi = target[:, roi_mask, :]
                loss = loss + weighted_mse_loss(out_roi, y_roi, None)

                curr_state = torch.cat([curr_state[:, :, 1:, :], out.unsqueeze(2)], dim=2)

            loss = loss / max(steps_to_run, 1)

        loss.backward()

        if check_grads and n_batches == 0:
            print("[GradCheck] out_roi.requires_grad:", out_roi.requires_grad)
            module_grads = {}
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    gn = p.grad.norm().item()
                    module = name.split('.')[0]
                    if module not in module_grads:
                        module_grads[module] = []
                    module_grads[module].append((name, gn))
            for module, grads in sorted(module_grads.items()):
                max_gn = max(g for _, g in grads)
                mean_gn = sum(g for _, g in grads) / len(grads)
                top_param = max(grads, key=lambda x: x[1])
                print(f"  [{module}] {len(grads)} params, max_grad={max_gn:.6f}, mean_grad={mean_gn:.6f}")
                print(f"    top: {top_param[0]} = {top_param[1]:.6f}")
            if not module_grads:
                print("  WARNING: все градиенты нулевые!")

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

    return total_loss / max(n_batches, 1)


def eval_dual(
    model: DualMeshModel,
    dataloader: DataLoader,
    device: torch.device,
    lat_weights=None,
    use_residual: bool = False,
    current_ar_steps: int = 1,
    static_channels=None,
    forcing_channels=None,
    cache: list = None,
):
    """Оценка на val/test. Loss считается ТОЛЬКО по ROI."""
    model.eval()
    total_loss = 0
    total_global_loss = 0
    n_batches = 0

    roi_mask = model.roi_mask

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y = y.squeeze(0) if y.dim() == 4 else y

            N, G, feat_dim = X.shape
            C = feat_dim // model.obs_window
            total_target_steps = y.shape[-1] // C
            y_steps = y.view(N, G, total_target_steps, C)

            if cache is not None:
                # ── Cached mode: AR=1, skip global model ──
                roi_raw = X[0][roi_mask]
                c = cache[batch_idx]
                out_roi = model.forward_cached(
                    roi_raw=roi_raw,
                    global_pred_roi=c['global_pred_roi'].to(device),
                    roi_grid_latent=c['roi_grid_latent'].to(device),
                    cross_sender_feat=c['cross_sender_feat'].to(device),
                )
                if use_residual:
                    x_last_roi = X[0, roi_mask, -C:]
                    out_roi = x_last_roi + out_roi
                target = y_steps[:, :, 0, :]
                y_roi = target[:, roi_mask, :]
                global_pred_roi = c['global_pred_roi'].to(device)
                if use_residual:
                    global_pred_roi = X[0, roi_mask, -C:] + global_pred_roi
                loss = weighted_mse_loss(out_roi.unsqueeze(0), y_roi, None)
                global_loss = weighted_mse_loss(global_pred_roi.unsqueeze(0), y_roi, None)
            else:
                # ── Standard mode ──
                obs = model.obs_window
                corr_state = X.view(N, G, obs, C)
                glob_state = X.view(N, G, obs, C).clone()
                steps_to_run = min(current_ar_steps, total_target_steps)
                loss = 0.0
                global_loss = 0.0

                for step in range(steps_to_run):
                    target = y_steps[:, :, step, :]

                    corr_inp = corr_state.view(N, G, -1)
                    pred = model(corr_inp)
                    if pred.dim() == 2:
                        pred = pred.unsqueeze(0)
                    if use_residual:
                        corr_out = corr_state[:, :, -1, :] + pred
                    else:
                        corr_out = pred
                    corr_out = apply_special_channels(
                        corr_out, corr_state, target,
                        static_channels=static_channels,
                        forcing_channels=forcing_channels,
                    )

                    global_inp = glob_state.view(N, G, -1)
                    global_pred = model.global_model(X=global_inp, attention_threshold=0.0)
                    if global_pred.dim() == 2:
                        global_pred = global_pred.unsqueeze(0)
                    if use_residual:
                        global_out = glob_state[:, :, -1, :] + global_pred
                    else:
                        global_out = global_pred
                    global_out = apply_special_channels(
                        global_out, glob_state, target,
                        static_channels=static_channels,
                        forcing_channels=forcing_channels,
                    )

                    corr_roi = corr_out[:, roi_mask, :]
                    global_roi = global_out[:, roi_mask, :]
                    y_roi = target[:, roi_mask, :]
                    loss = loss + weighted_mse_loss(corr_roi, y_roi, None)
                    global_loss = global_loss + weighted_mse_loss(global_roi, y_roi, None)

                    corr_state = torch.cat([corr_state[:, :, 1:, :], corr_out.unsqueeze(2)], dim=2)
                    glob_state = torch.cat([glob_state[:, :, 1:, :], global_out.unsqueeze(2)], dim=2)

                loss = loss / max(steps_to_run, 1)
                global_loss = global_loss / max(steps_to_run, 1)

            total_loss += loss.item()
            total_global_loss += global_loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_global_loss = total_global_loss / max(n_batches, 1)
    correction_skill = 1.0 - avg_loss / avg_global_loss if avg_global_loss > 0 else 0.0
    return avg_loss, correction_skill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--pretrained", required=True, help="Checkpoint глобальной модели")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--freeze-global", action="store_true", default=True)
    ap.add_argument("--reg-mesh-level", type=int, default=7)
    ap.add_argument("--reg-steps", type=int, default=4)
    ap.add_argument("--cross-k", type=int, default=3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--reg-buffer", type=float, default=2.0)
    ap.add_argument("--roi", nargs=4, type=float, required=True,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    help="Region of Interest: lat_min lat_max lon_min lon_max")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--ar-steps", type=int, default=None,
                    help="AR rollout horizon for train/val/test. Defaults to config max_ar_steps.")
    ap.add_argument("--overfit-only", action="store_true",
                    help="Run only overfit test (1 sample, 200 steps) and exit")
    ap.add_argument("--precompute", action="store_true",
                    help="Precompute global outputs and cache in RAM. Requires --ar-steps 1. "
                         "~4 MB/sample. Eliminates global model from training loop.")
    ap.add_argument("--subsample", type=int, default=1,
                    help="Use every Nth sample (default: 1 = all). E.g. --subsample 4 → 25%% of data.")
    args = ap.parse_args()

    exp_dir = args.experiment_dir
    cfg_path = os.path.join(exp_dir, FileNames.EXPERIMENT_CONFIG)
    assert os.path.exists(cfg_path), f"Config not found: {cfg_path}"

    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_random_seeds(42)

    # --- Load dataset ---
    data_dir = args.data_dir or getattr(exp_cfg, 'data_dir', None)
    if data_dir is None:
        data_dir = f"data/datasets/{exp_cfg.data.dataset_name.value}"

    ar_steps = args.ar_steps or max(getattr(exp_cfg, 'max_ar_steps', 1), getattr(exp_cfg.data, 'pred_window_used', 1))

    train_ds, val_ds, test_ds, meta = load_chunked_datasets(
        data_path=data_dir,
        obs_window=exp_cfg.data.obs_window_used,
        pred_steps=ar_steps,
        n_features=exp_cfg.data.num_features_used,
    )

    # --- Subsample datasets if requested ---
    if args.subsample > 1:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(0, len(train_ds), args.subsample)))
        val_ds = Subset(val_ds, list(range(0, len(val_ds), args.subsample)))
        print(f"[Subsample] every {args.subsample}th sample: train={len(train_ds)}, val={len(val_ds)}")

    # --- Validate precompute mode ---
    if args.precompute:
        effective_ar = ar_steps
        if effective_ar != 1:
            print(f"WARNING: --precompute requires --ar-steps 1, got {effective_ar}. Forcing ar_steps=1.")
            ar_steps = 1

    batch_size = args.batch_size or exp_cfg.batch_size
    use_cuda = device.type == "cuda"
    loader_kwargs = dict(
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
    )
    # With --precompute, cache is indexed sequentially → must disable shuffle
    train_shuffle = not args.precompute
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # --- Load global model ---
    real_coords = getattr(meta, 'cordinates', None)
    flat_grid = getattr(meta, 'flat_grid', False)

    global_model = load_model_from_experiment_config(
        exp_cfg, device, meta,
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

    # Freeze global model
    if args.freeze_global:
        for p in global_model.parameters():
            p.requires_grad = False
        print("[DualTrain] Global model FROZEN.")

    # --- Grid coordinates ---
    if flat_grid and real_coords is not None:
        grid_lats = real_coords[0].astype(np.float32)
        grid_lons = real_coords[1].astype(np.float32)
    else:
        lats = np.linspace(-90, 90, meta.num_latitudes, endpoint=True)
        lons = np.linspace(0, 360, meta.num_longitudes, endpoint=False)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        grid_lats = lat_grid.flatten().astype(np.float32)
        grid_lons = lon_grid.flatten().astype(np.float32)

    # --- Create DualMeshModel ---
    roi = tuple(args.roi)
    print(f"\n>>> ROI: {roi}")

    dual_model = DualMeshModel(
        global_model=global_model,
        roi=roi,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        device=device,
        reg_mesh_level=args.reg_mesh_level,
        reg_mesh_buffer=args.reg_buffer,
        reg_processor_steps=args.reg_steps,
        cross_k=args.cross_k,
        hidden_dim=args.hidden_dim,
    )
    dual_model = dual_model.to(device)

    # --- Optimizer: only regional parameters ---
    regional_params = [p for n, p in dual_model.named_parameters()
                       if not n.startswith("global_model.") and p.requires_grad]
    n_params = sum(p.numel() for p in regional_params)
    print(f"\n[Optimizer] {n_params:,} trainable regional parameters, lr={args.lr}")

    optimizer = Adam(regional_params, lr=args.lr)

    # --- Latitude weights ---
    use_residual = getattr(exp_cfg, 'use_residual', False)
    static_channels = getattr(exp_cfg, 'static_channels', [])
    forcing_channels = getattr(exp_cfg, 'forcing_channels', [])
    lat_weights = None
    if getattr(exp_cfg, 'use_latitude_weighting', False) and meta:
        if flat_grid and real_coords is not None:
            lat_weights = get_lat_weights(0, 0, device, flat_lats=real_coords[0])
        else:
            lat_weights = get_lat_weights(meta.num_latitudes, meta.num_longitudes, device)

    # --- Training loop ---
    num_epochs = args.epochs or exp_cfg.num_epochs
    best_val_loss = float("inf")
    patience = 0
    max_patience = 10

    log_path = os.path.join(results_dir, "training_log.txt")

    def _log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # --- Overfit test: ДОКАЗАТЕЛЬСТВО что модель может учиться ---
    overfit_test(dual_model, train_loader, device, use_residual=use_residual)

    if args.overfit_only:
        print("\n[--overfit-only] Done. Exiting without full training.")
        return

    # --- Precompute global outputs (if requested) ---
    train_cache = None
    val_cache = None
    if args.precompute:
        import time as _time
        print("\n" + "="*60)
        print("[Precompute] Caching global model outputs...")
        print("="*60)
        dual_model.eval()

        def _build_cache(loader, name):
            cache = []
            t0 = _time.time()
            for i, (X, y) in enumerate(loader):
                X = X.to(device)
                c = dual_model.precompute_global(X)
                cache.append(c)
                if (i + 1) % 500 == 0:
                    elapsed = _time.time() - t0
                    print(f"  {name}: {i+1}/{len(loader)} ({elapsed:.0f}s)")
            elapsed = _time.time() - t0
            mem_mb = sum(
                sum(v.numel() * v.element_size() for v in c.values())
                for c in cache
            ) / 1e6
            print(f"  {name}: {len(cache)} samples cached in {elapsed:.0f}s ({mem_mb:.0f} MB)")
            return cache

        train_cache = _build_cache(train_loader, "train")
        val_cache = _build_cache(val_loader, "val")
        print("[Precompute] Done. Training will skip global model from now on.\n")
        dual_model.train()
        dual_model.global_model.eval()

    _log(f"[AR] training/eval horizon = {ar_steps} step(s) (+{ar_steps*6}h)")
    _log(f"{'epoch':>5}  {'train_loss':>10}  {'val_loss':>10}  {'corr_skill':>10}  {'best':>10}")
    _log("-" * 59)

    for epoch in range(num_epochs):
        train_loss = train_dual_epoch(
            dual_model, train_loader, optimizer, device,
            lat_weights=lat_weights, use_residual=use_residual,
            current_ar_steps=ar_steps,
            static_channels=static_channels,
            forcing_channels=forcing_channels,
            check_grads=(epoch == 0),
            cache=train_cache,
        )

        val_loss, correction_skill = eval_dual(
            dual_model, val_loader, device,
            lat_weights=lat_weights, use_residual=use_residual,
            current_ar_steps=ar_steps,
            static_channels=static_channels,
            forcing_channels=forcing_channels,
            cache=val_cache,
        )

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            # Save best model (only regional weights)
            regional_state = {
                k: v for k, v in dual_model.state_dict().items()
                if not k.startswith("global_model.")
            }
            torch.save(regional_state, os.path.join(results_dir, "best_regional.pth"))
            # Also save full model for convenience
            torch.save(dual_model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            improved = " *"
        else:
            patience += 1

        _log(f"{epoch+1:5d}  {train_loss:10.6f}  {val_loss:10.6f}  {correction_skill:>9.2%}  {best_val_loss:10.6f}{improved}")

        if patience >= max_patience:
            _log(f"Early stopping at epoch {epoch+1}")
            break

    # --- Test ---
    # Load best weights
    best_ckpt = os.path.join(results_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        dual_model.load_state_dict(torch.load(best_ckpt, map_location=device))

    test_loss, test_skill = eval_dual(
        dual_model, test_loader, device,
        lat_weights=lat_weights, use_residual=use_residual,
        current_ar_steps=ar_steps,
        static_channels=static_channels,
        forcing_channels=forcing_channels,
    )
    _log(f"\n[TEST] loss={test_loss:.6f}  correction_skill={test_skill:.2%}")

    # Save results
    results = {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_correction_skill": test_skill,
        "roi": list(roi),
        "reg_mesh_level": args.reg_mesh_level,
        "reg_steps": args.reg_steps,
        "cross_k": args.cross_k,
        "hidden_dim": args.hidden_dim,
        "ar_steps": ar_steps,
    }
    save_to_json_file(results, os.path.join(results_dir, "results.json"))
    _log(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()

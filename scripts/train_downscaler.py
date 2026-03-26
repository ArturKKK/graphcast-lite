#!/usr/bin/env python3
"""
scripts/train_downscaler.py

Обучение UNet-downscaler'а: coarse (0.7° upsampled) → fine (real 0.25°).

Каскадная схема:
  1. GNN предсказывает на глобальном гриде 512×256
  2. Кропаем ROI и ресайзим до 41×61
  3. UNet восстанавливает мелкомасштабные детали

UNet обучается на парах (coarse ERA5, real ERA5) — не нужно генерировать
предсказания GNN для всех сэмплов. При инференсе вместо coarse ERA5
подставляется прогноз GNN.

Usage:
  python scripts/train_downscaler.py experiments/downscaler_krsk \
    --data-dir /data/datasets/downscaler_krsk_19f
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.unet.model import WeatherUNet


# ──────────────────────────────────────────────────────────
# Dataset: пары (coarse_upsampled, real_fine)
# ──────────────────────────────────────────────────────────
class DownscalerDataset(Dataset):
    """
    Загружает пары (coarse, fine) из memmap файлов.

    Для temporal context: вход = obs_window последовательных coarse кадров.
    Target = fine кадр для последнего (или obs_window-го) timestep.

    X: (obs_window * C, H, W) — coarse
    Y: (C, H, W) — real fine
    """

    def __init__(self, data_dir: str, obs_window: int = 2, split: str = "train",
                 static_context: bool = True):
        data_dir = Path(data_dir)

        with open(data_dir / "dataset_info.json") as f:
            info = json.load(f)

        self.T = info["n_time"]
        self.n_lon = info["n_lon"]  # 61
        self.n_lat = info["n_lat"]  # 41
        self.C = info["n_feat"]     # 19
        self.obs_window = obs_window
        self.static_channels = info.get("static_channels", [])

        shape = (self.T, self.n_lon, self.n_lat, self.C)
        self.coarse = np.memmap(data_dir / "coarse.npy", dtype=np.float16,
                                mode="r", shape=shape)
        self.fine = np.memmap(data_dir / "fine.npy", dtype=np.float16,
                              mode="r", shape=shape)

        # Scalers
        sc = np.load(data_dir / "scalers.npz")
        self.mean = sc["mean"].astype(np.float32)  # (C,)
        self.std = sc["std"].astype(np.float32)     # (C,)

        # Static fine-resolution fields (extra input channels)
        self.static_fine = None
        if static_context and (data_dir / "static_fine.npy").exists():
            self.static_fine = np.load(data_dir / "static_fine.npy")  # (n_lon, n_lat, N_static)

        # Train/val/test split (chronological, same as main training)
        valid_indices = list(range(obs_window - 1, self.T))
        n = len(valid_indices)
        n_test = int(n * 0.2)
        n_val = n_test // 2
        n_test_only = n_test - n_val
        n_train = n - n_test

        if split == "train":
            self.indices = valid_indices[:n_train]
        elif split == "val":
            self.indices = valid_indices[n_train:n_train + n_val]
        elif split == "test":
            self.indices = valid_indices[n_train + n_val:]
        else:
            self.indices = valid_indices

        print(f"[DownscalerDataset] split={split}, samples={len(self.indices)}, "
              f"grid={self.n_lat}×{self.n_lon}, C={self.C}, obs={obs_window}")

    def __len__(self):
        return len(self.indices)

    def _normalize(self, x):
        """x: (..., C) → normalized"""
        return (x - self.mean) / (self.std + 1e-8)

    def __getitem__(self, idx):
        t = self.indices[idx]

        # Coarse: obs_window кадров
        coarse_frames = []
        for i in range(self.obs_window):
            t_i = t - self.obs_window + 1 + i
            frame = np.array(self.coarse[t_i], dtype=np.float32)  # (n_lon, n_lat, C)
            frame = self._normalize(frame)
            # Transpose to (n_lat, n_lon, C) = (H, W, C) for CNN
            frame = frame.transpose(1, 0, 2)  # (H, W, C)
            coarse_frames.append(frame)

        # Stack obs → (obs*C, H, W)
        coarse_stack = np.concatenate(coarse_frames, axis=-1)  # (H, W, obs*C)
        X = torch.from_numpy(coarse_stack).permute(2, 0, 1)   # (obs*C, H, W)

        # Fine target: single frame
        fine = np.array(self.fine[t], dtype=np.float32)  # (n_lon, n_lat, C)
        fine = self._normalize(fine)
        fine = fine.transpose(1, 0, 2)  # (H, W, C)
        Y = torch.from_numpy(fine).permute(2, 0, 1)      # (C, H, W)

        # Append static fine fields as extra input channels
        if self.static_fine is not None:
            static = self.static_fine.transpose(1, 0, 2)  # (H, W, N_static)
            static_t = torch.from_numpy(static.copy()).permute(2, 0, 1).float()  # (N_static, H, W)
            # Normalize static: z_surf is in meters (÷1000), lsm is 0-1
            # Simple: just append raw — UNet will figure it out
            X = torch.cat([X, static_t], dim=0)  # (obs*C + N_static, H, W)

        return X, Y


# ──────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────
def masked_mse_loss(pred, target, channel_mask=None):
    diff = (pred - target) ** 2
    if channel_mask is not None:
        diff = diff * channel_mask.view(1, -1, 1, 1)
    return diff.mean()


# ──────────────────────────────────────────────────────────
# Training / validation
# ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, channel_mask, residual):
    model.train()
    total_loss = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()

        out = model(X)  # (B, C, H, W)

        if residual:
            # Add to last coarse frame: channels [(obs-1)*C : obs*C]
            C = Y.shape[1]
            obs_end = (X.shape[1] // C) * C  # skip static channels at end
            x_last = X[:, obs_end - C:obs_end, :, :]
            out = x_last + out

        loss = masked_mse_loss(out, Y, channel_mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, channel_mask, residual, C):
    model.eval()
    total_loss = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        out = model(X)
        if residual:
            obs_channels = (X.shape[1] // C) * C
            x_last = X[:, obs_channels - C:obs_channels, :, :]
            out = x_last + out
        loss = masked_mse_loss(out, Y, channel_mask)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def compute_metrics(model, loader, device, channel_mask, residual, C, std_tensor,
                    exclude_channels=None):
    """Compute per-channel RMSE (in physical units) and overall skill."""
    model.eval()
    mse_sum = torch.zeros(C, device=device)
    mse_coarse_sum = torch.zeros(C, device=device)
    n = 0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        out = model(X)
        if residual:
            obs_channels = (X.shape[1] // C) * C
            x_last = X[:, obs_channels - C:obs_channels, :, :]
            out = x_last + out

        # Per-channel MSE
        diff = (out - Y) ** 2  # (B, C, H, W)
        mse_sum += diff.mean(dim=(0, 2, 3))  # (C,)

        # Baseline: coarse (= last input frame, already upsampled)
        obs_channels = (X.shape[1] // C) * C
        coarse_last = X[:, obs_channels - C:obs_channels, :, :]
        diff_coarse = (coarse_last - Y) ** 2
        mse_coarse_sum += diff_coarse.mean(dim=(0, 2, 3))

        n += 1

    mse_model = mse_sum / n     # (C,) normalized
    mse_coarse = mse_coarse_sum / n

    # RMSE in physical units: rmse_physical = rmse_normalized * std
    rmse_model = (mse_model.sqrt() * std_tensor).cpu().numpy()
    rmse_coarse = (mse_coarse.sqrt() * std_tensor).cpu().numpy()

    # Skill per channel: 1 - rmse_model / rmse_coarse
    skill = 1.0 - rmse_model / (rmse_coarse + 1e-8)

    return rmse_model, rmse_coarse, skill


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", help="e.g. experiments/downscaler_krsk")
    ap.add_argument("--data-dir", required=True, help="Path to downscaler dataset")
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = exp_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = {}

    # Defaults
    C = cfg.get("num_features", 19)
    obs_window = cfg.get("obs_window", 2)
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("learning_rate", 1e-3)
    num_epochs = cfg.get("num_epochs", 50)
    patience = cfg.get("patience", 15)
    base_filters = cfg.get("base_filters", 64)
    static_context = cfg.get("static_context", True)
    residual = cfg.get("residual", False)
    static_ch = cfg.get("static_channels", [7, 8])
    seed = cfg.get("random_seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datasets ──
    train_ds = DownscalerDataset(args.data_dir, obs_window, "train", static_context)
    val_ds = DownscalerDataset(args.data_dir, obs_window, "val", static_context)
    test_ds = DownscalerDataset(args.data_dir, obs_window, "test", static_context)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── Model ──
    # Input: obs_window * C + N_static (if static_context)
    n_static = len(static_ch) if static_context else 0
    in_channels = obs_window * C + n_static
    out_channels = C

    model = WeatherUNet(in_channels=in_channels, out_channels=out_channels,
                        base_filters=base_filters).to(device)
    print(f"WeatherUNet: {model.num_params:,} params | in={in_channels} out={out_channels} "
          f"filters={base_filters}")
    print(f"Grid: {train_ds.n_lat}×{train_ds.n_lon} (H×W) | obs={obs_window}")

    # Channel mask (exclude static from loss)
    channel_mask = torch.ones(C, device=device)
    for ch in static_ch:
        channel_mask[ch] = 0.0
    dyn = int(channel_mask.sum().item())
    print(f"Channels: {dyn}/{C} dynamic (static excluded: {static_ch})")

    # Std for RMSE conversion
    std_tensor = torch.from_numpy(train_ds.std).to(device)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ── Training ──
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = exp_dir / "best_model.pth"

    log_path = exp_dir / "training_log.txt"
    def _log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    _log(f"=== Downscaler training started: {datetime.now().isoformat()} ===")
    _log(f"data_dir={args.data_dir}  epochs={num_epochs}  batch={batch_size}  lr={lr}")
    _log(f"{'epoch':>5}  {'train_loss':>10}  {'val_loss':>10}  {'best':>10}  {'lr':>8}")
    _log("-" * 55)

    for epoch in range(num_epochs):
        t0 = datetime.now()

        train_loss = train_epoch(model, train_loader, optimizer, device,
                                 channel_mask, residual)
        val_loss = validate(model, val_loader, device, channel_mask, residual, C)
        scheduler.step()

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            improved = " *"
        else:
            patience_counter += 1

        cur_lr = optimizer.param_groups[0]['lr']
        dt = (datetime.now() - t0).total_seconds()
        _log(f"{epoch+1:5d}  {train_loss:10.6f}  {val_loss:10.6f}  "
             f"{best_val_loss:10.6f}  {cur_lr:.1e}{improved}  ({dt:.0f}s)")

        if patience_counter >= patience:
            _log(f"Early stopping at epoch {epoch+1}")
            break

    # ── Test evaluation ──
    _log("\n" + "=" * 55)
    _log("[TEST] Loading best model...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_loss = validate(model, test_loader, device, channel_mask, residual, C)
    rmse_model, rmse_coarse, skill = compute_metrics(
        model, test_loader, device, channel_mask, residual, C, std_tensor,
        exclude_channels=set(static_ch)
    )

    # Load variable names
    vars_path = Path(args.data_dir) / "variables.json"
    var_names = json.load(open(vars_path)) if vars_path.exists() else [f"ch{i}" for i in range(C)]

    _log(f"\n[TEST] loss={test_loss:.6f}")
    _log(f"\n{'Variable':>12}  {'RMSE_model':>10}  {'RMSE_coarse':>11}  {'Skill':>8}")
    _log("-" * 48)
    for i in range(C):
        if i in static_ch:
            continue
        _log(f"{var_names[i]:>12}  {rmse_model[i]:10.4f}  {rmse_coarse[i]:11.4f}  {skill[i]:>7.1%}")

    mean_skill = np.mean([skill[i] for i in range(C) if i not in static_ch])
    _log(f"\n{'MEAN':>12}  {'':>10}  {'':>11}  {mean_skill:>7.1%}")

    # ── Save results ──
    results = {
        "test_loss": float(test_loss),
        "mean_skill_vs_coarse": float(mean_skill),
        "per_channel_rmse": {var_names[i]: float(rmse_model[i]) for i in range(C)},
        "per_channel_skill": {var_names[i]: float(skill[i]) for i in range(C) if i not in static_ch},
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _log(f"\nResults saved to {exp_dir}")


if __name__ == "__main__":
    main()

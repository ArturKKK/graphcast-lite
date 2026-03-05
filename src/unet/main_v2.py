"""
U-Net V2 training — serious architecture with attention + spectral + multi-scale loss.

Usage:
    python -m src.unet.main_v2 experiments/unet_v2_region_krsk
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.unet.model_v2 import WeatherUNetV2
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
# Losses
# ─────────────────────────────────────────────────────────
def masked_mse_loss(pred, target, channel_mask=None):
    diff = (pred - target) ** 2
    if channel_mask is not None:
        diff = diff * channel_mask.view(1, -1, 1, 1)
    return diff.mean()


def spectral_loss(pred, target, channel_mask=None):
    """
    FFT-based loss — penalizes differences in frequency domain.
    Prevents blurring on longer horizons.
    """
    if channel_mask is not None:
        mask = channel_mask.view(1, -1, 1, 1)
        pred = pred * mask
        target = target * mask
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    tgt_fft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(tgt_fft))


def gradient_loss(pred, target, channel_mask=None):
    """
    Sobel-like gradient loss — preserves sharp edges / fronts.
    Weather fronts = sharp gradients in t2m, wind, pressure.
    """
    # Spatial gradients (finite differences)
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    if channel_mask is not None:
        mask = channel_mask.view(1, -1, 1, 1)
        loss_dy = ((pred_dy - tgt_dy) ** 2 * mask).mean()
        loss_dx = ((pred_dx - tgt_dx) ** 2 * mask).mean()
    else:
        loss_dy = ((pred_dy - tgt_dy) ** 2).mean()
        loss_dx = ((pred_dx - tgt_dx) ** 2).mean()
    return loss_dy + loss_dx


# ─────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────
def spatial_acc(pred, target, exclude_channels=None):
    if pred.dim() == 4:
        pred = pred.mean(0)
        target = target.mean(0)
    C = pred.shape[0]
    eps = 1e-8
    accs = []
    for c in range(C):
        if exclude_channels and c in exclude_channels:
            continue
        p = pred[c].flatten().float()
        t = target[c].flatten().float()
        p = p - p.mean()
        t = t - t.mean()
        corr = (p * t).sum() / (p.norm() * t.norm() + eps)
        accs.append(corr.item())
    return sum(accs) / max(len(accs), 1)


# ─────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, channel_mask,
                current_ar_steps, C, static_channels, forcing_channels,
                spectral_weight=0.1, gradient_weight=0.05, scheduler=None):
    model.train()
    total_loss = 0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        B, _, H, W = X.shape
        obs = X.shape[1] // C

        total_pred = Y.shape[1] // C
        Y_steps = Y.view(B, total_pred, C, H, W)
        curr_state = X.view(B, obs, C, H, W)

        optimizer.zero_grad()
        loss_batch = 0
        steps = min(current_ar_steps, total_pred)

        for step in range(steps):
            inp = curr_state.reshape(B, obs * C, H, W)
            delta = model(inp)
            x_last = curr_state[:, -1]
            out = x_last + delta
            target = Y_steps[:, step]

            # Combined loss: MSE + spectral + gradient
            l_mse = masked_mse_loss(out, target, channel_mask)
            l_spec = spectral_loss(out, target, channel_mask)
            l_grad = gradient_loss(out, target, channel_mask)
            loss_batch += l_mse + spectral_weight * l_spec + gradient_weight * l_grad

            # Carry-forward
            if static_channels:
                for ch in static_channels:
                    out = out.clone()
                    out[:, ch] = x_last[:, ch]
            if forcing_channels and step < total_pred:
                for ch in forcing_channels:
                    out = out.clone()
                    out[:, ch] = target[:, ch]

            curr_state = torch.cat([curr_state[:, 1:], out.unsqueeze(1)], dim=1)

        loss_batch = loss_batch / steps
        loss_batch.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss_batch.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, channel_mask, C,
             static_channels, forcing_channels, exclude_channels):
    model.eval()
    total_loss = 0
    acc_values = []

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        B, _, H, W = X.shape
        obs = X.shape[1] // C

        total_pred = Y.shape[1] // C
        Y_steps = Y.view(B, total_pred, C, H, W)
        curr_state = X.view(B, obs, C, H, W)

        inp = curr_state.reshape(B, obs * C, H, W)
        delta = model(inp)
        x_last = curr_state[:, -1]
        out = x_last + delta
        target = Y_steps[:, 0]

        if static_channels:
            for ch in static_channels:
                out[:, ch] = x_last[:, ch]
        if forcing_channels:
            for ch in forcing_channels:
                out[:, ch] = target[:, ch]

        loss = masked_mse_loss(out, target, channel_mask)
        total_loss += loss.item()
        acc_values.append(spatial_acc(out, target, exclude_channels))

    return total_loss / len(loader), sum(acc_values) / max(len(acc_values), 1)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    args = parser.parse_args()

    cfg_path = os.path.join(args.experiment_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = cfg["data_dir"]
    C = cfg["num_features"]
    obs_window = cfg.get("obs_window", 4)
    pred_steps = cfg.get("pred_steps", 4)
    batch_size = cfg.get("batch_size", 8)
    lr = cfg.get("learning_rate", 5e-4)
    num_epochs = cfg.get("num_epochs", 80)
    patience = cfg.get("patience", 15)
    base_filters = cfg.get("base_filters", 64)
    max_ar = cfg.get("max_ar_steps", 4)
    static_ch = cfg.get("static_channels", [])
    forcing_ch = cfg.get("forcing_channels", [])
    attn_heads = cfg.get("attn_heads", 4)
    spectral_modes = cfg.get("spectral_modes", 4)
    spectral_weight = cfg.get("spectral_weight", 0.1)
    gradient_weight = cfg.get("gradient_weight", 0.05)
    seed = cfg.get("random_seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    train_ds_raw = TimeseriesChunkDataset(data_dir, obs_window, pred_steps, split="train", n_features=C)
    val_ds_raw = TimeseriesChunkDataset(data_dir, obs_window, pred_steps, split="val", n_features=C)

    n_lon = train_ds_raw.n_lon
    n_lat = train_ds_raw.n_lat
    assert n_lon and n_lat, "U-Net V2 requires regular grid"

    train_ds = Grid2DDataset(train_ds_raw, n_lon, n_lat)
    val_ds = Grid2DDataset(val_ds_raw, n_lon, n_lat)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    in_channels = obs_window * C
    model = WeatherUNetV2(
        in_channels=in_channels, out_channels=C,
        base_filters=base_filters, attn_heads=attn_heads,
        spectral_modes=spectral_modes
    ).to(device)

    print(f"WeatherUNetV2: {model.num_params:,} params")
    print(f"  in={in_channels} out={C} filters={base_filters} attn_heads={attn_heads}")
    print(f"  Grid: {n_lat}×{n_lon} | obs={obs_window} | pred={pred_steps} | max_ar={max_ar}")
    print(f"  Loss weights: spectral={spectral_weight} gradient={gradient_weight}")

    # Channel mask
    no_loss_ch = sorted(set(static_ch) | set(forcing_ch))
    channel_mask = torch.ones(C, device=device)
    for ch in no_loss_ch:
        channel_mask[ch] = 0.0
    dyn_count = int(channel_mask.sum().item())
    print(f"  Channels: {dyn_count}/{C} dynamic (no-loss: {no_loss_ch})")
    exclude_set = set(no_loss_ch)

    # Optimizer: OneCycleLR for aggressive training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=num_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    # Curriculum
    epochs_per_stage = max(num_epochs // max_ar, 5) if max_ar > 1 else num_epochs

    # Training
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(args.experiment_dir, "best_model.pth")
    log_path = os.path.join(args.experiment_dir, "train_log.txt")

    def _log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    _log(f"{'epoch':>5}  {'ar':>2}  {'train_loss':>10}  {'val_loss':>10}  {'val_ACC':>8}  timestamp")

    val_loss, val_acc = validate(model, val_loader, device, channel_mask, C,
                                  static_ch, forcing_ch, exclude_set)
    print(f"[Init] val_loss={val_loss:.5f} val_ACC={val_acc:.4f}")
    _log(f"{'init':>5}  {'--':>2}  {'--':>10}  {val_loss:10.5f}  {val_acc:8.4f}  {datetime.now().strftime('%H:%M:%S')}")

    ar_steps = 1

    for epoch in range(num_epochs):
        t0 = time.time()

        correct_ar = min(1 + epoch // epochs_per_stage, max_ar)
        if correct_ar > ar_steps:
            ar_steps = correct_ar
            print(f"\n--- AR increased to {ar_steps} ---\n")

        train_loss = train_epoch(model, train_loader, optimizer, device, channel_mask,
                                  ar_steps, C, static_ch, forcing_ch,
                                  spectral_weight, gradient_weight, scheduler)
        val_loss, val_acc = validate(model, val_loader, device, channel_mask, C,
                                      static_ch, forcing_ch, exclude_set)

        elapsed = time.time() - t0
        print(f"[Epoch {epoch+1}] AR={ar_steps} train_loss={train_loss:.5f} "
              f"val_loss={val_loss:.5f} val_ACC={val_acc:.4f} ({elapsed:.0f}s)")
        _log(f"{epoch+1:5d}  {ar_steps:2d}  {train_loss:10.5f}  {val_loss:10.5f}  {val_acc:8.4f}  {datetime.now().strftime('%H:%M:%S')}")

        if val_loss < best_val_loss - 1e-5:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved (improved by {improvement:.5f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping.")
                break

    print(f"\nDone. Best val_loss={best_val_loss:.5f}. Model: {ckpt_path}")
    print("Run inference: python -m src.unet.predict_v2", args.experiment_dir)


if __name__ == "__main__":
    main()

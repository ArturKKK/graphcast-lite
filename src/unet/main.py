"""
U-Net regional weather training & evaluation.

Usage:
    python -m src.unet.main experiments/unet_region_krsk
    
Reads config.json from experiment dir, trains U-Net, saves best_model.pth.
Data from existing chunked datasets — only reshaped to (B, C, H, W).
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
from torch.utils.data import DataLoader, Dataset

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.unet.model import WeatherUNet
from src.data.dataloader_chunked import TimeseriesChunkDataset


# ─────────────────────────────────────────────────────────
# Dataset wrapper: flatten (G, obs*C) → 2D (obs*C, H, W)
# ─────────────────────────────────────────────────────────
class Grid2DDataset(Dataset):
    """
    Wraps TimeseriesChunkDataset to return 2D grids for CNN.
    
    Original returns: X (G, obs*C), Y (G, pred*C)  where G = H*W, lat-major
    We reshape to:    X (obs*C, H, W), Y (pred*C, H, W)
    """
    def __init__(self, chunk_ds: TimeseriesChunkDataset, n_lon: int, n_lat: int):
        self.ds = chunk_ds
        self.H = n_lat   # lat = height
        self.W = n_lon   # lon = width
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        X, Y = self.ds[idx]
        # X: (H*W, obs*C) → (H, W, obs*C) → (obs*C, H, W) 
        X = X.view(self.H, self.W, -1).permute(2, 0, 1)
        Y = Y.view(self.H, self.W, -1).permute(2, 0, 1)
        return X, Y


# ─────────────────────────────────────────────────────────
# Loss with channel mask
# ─────────────────────────────────────────────────────────
def masked_mse_loss(pred, target, channel_mask=None):
    """
    MSE with optional channel mask.
    pred, target: (B, C, H, W)
    channel_mask: (C,) with 0.0 for excluded channels
    """
    diff = (pred - target) ** 2
    if channel_mask is not None:
        # (C,) → (1, C, 1, 1)
        diff = diff * channel_mask.view(1, -1, 1, 1)
    return diff.mean()


# ─────────────────────────────────────────────────────────
# Spatial correlation (ACC)
# ─────────────────────────────────────────────────────────
def spatial_acc(pred, target, exclude_channels=None):
    """
    ACC averaged over channels (excluding specified ones).
    pred, target: (B, C, H, W) or (C, H, W)
    """
    if pred.dim() == 4:
        pred = pred.mean(0)
        target = target.mean(0)
    # pred, target: (C, H, W)
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
                current_ar_steps, C, static_channels, forcing_channels):
    model.train()
    total_loss = 0
    
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        # X: (B, obs*C, H, W), Y: (B, pred*C, H, W)
        B, _, H, W = X.shape
        obs = X.shape[1] // C
        
        # Reshape Y into steps: (B, pred, C, H, W)
        total_pred = Y.shape[1] // C
        Y_steps = Y.view(B, total_pred, C, H, W)
        
        # Reshape X into steps: (B, obs, C, H, W)
        curr_state = X.view(B, obs, C, H, W)
        
        optimizer.zero_grad()
        loss_batch = 0
        steps = min(current_ar_steps, total_pred)
        
        for step in range(steps):
            # Input: flatten obs dim → (B, obs*C, H, W)
            inp = curr_state.view(B, obs * C, H, W)
            
            # Model predicts delta
            delta = model(inp)  # (B, C, H, W)
            
            # Residual: add to last obs step
            x_last = curr_state[:, -1, :, :, :]  # (B, C, H, W)
            out = x_last + delta
            
            # Target
            target = Y_steps[:, step, :, :, :]  # (B, C, H, W)
            
            loss_batch += masked_mse_loss(out, target, channel_mask)
            
            # Carry-forward for next AR step
            if static_channels:
                for ch in static_channels:
                    out[:, ch] = x_last[:, ch]
            if forcing_channels and step < total_pred:
                for ch in forcing_channels:
                    out[:, ch] = target[:, ch]
            
            # Shift window: drop oldest, append prediction
            curr_state = torch.cat([curr_state[:, 1:], out.unsqueeze(1)], dim=1)
        
        loss_batch = loss_batch / steps
        loss_batch.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
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
        
        # Validate on 1 step only (like GNN test())
        inp = curr_state.view(B, obs * C, H, W)
        delta = model(inp)
        x_last = curr_state[:, -1]
        out = x_last + delta
        target = Y_steps[:, 0]
        
        # Carry-forward before ACC
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
    parser.add_argument("experiment_dir", help="e.g. experiments/unet_region_krsk")
    args = parser.parse_args()
    
    # Load config
    cfg_path = os.path.join(args.experiment_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Config values
    data_dir = cfg["data_dir"]
    C = cfg["num_features"]
    obs_window = cfg.get("obs_window", 2)
    pred_steps = cfg.get("pred_steps", 4)
    batch_size = cfg.get("batch_size", 8)
    lr = cfg.get("learning_rate", 1e-3)
    num_epochs = cfg.get("num_epochs", 50)
    patience = cfg.get("patience", 10)
    base_filters = cfg.get("base_filters", 64)
    max_ar = cfg.get("max_ar_steps", 4)
    static_ch = cfg.get("static_channels", [])
    forcing_ch = cfg.get("forcing_channels", [])
    seed = cfg.get("random_seed", 42)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load datasets
    train_ds_raw = TimeseriesChunkDataset(data_dir, obs_window, pred_steps, split="train", n_features=C)
    val_ds_raw = TimeseriesChunkDataset(data_dir, obs_window, pred_steps, split="val", n_features=C)
    
    n_lon = train_ds_raw.n_lon
    n_lat = train_ds_raw.n_lat
    assert n_lon is not None and n_lat is not None, "U-Net requires regular grid data (not flat)"
    
    train_ds = Grid2DDataset(train_ds_raw, n_lon, n_lat)
    val_ds = Grid2DDataset(val_ds_raw, n_lon, n_lat)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    # Model
    in_channels = obs_window * C
    model = WeatherUNet(in_channels=in_channels, out_channels=C, base_filters=base_filters)
    model = model.to(device)
    print(f"WeatherUNet: {model.num_params:,} params | in={in_channels} out={C} filters={base_filters}")
    print(f"Grid: {n_lat}×{n_lon} (H×W) | obs={obs_window} | pred_steps={pred_steps} | max_ar={max_ar}")
    
    # Channel mask
    no_loss_ch = sorted(set(static_ch) | set(forcing_ch))
    channel_mask = torch.ones(C, device=device)
    for ch in no_loss_ch:
        channel_mask[ch] = 0.0
    dyn_count = int(channel_mask.sum().item())
    print(f"Channels: {dyn_count}/{C} dynamic (no-loss: {no_loss_ch})")
    
    exclude_set = set(no_loss_ch)
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Curriculum
    epochs_per_stage = max(num_epochs // max_ar, 5) if max_ar > 1 else num_epochs
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(args.experiment_dir, "best_model.pth")
    log_path = os.path.join(args.experiment_dir, "train_log.txt")
    
    def _log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    
    _log(f"{'epoch':>5}  {'ar':>2}  {'train_loss':>10}  {'val_loss':>10}  {'val_ACC':>8}  timestamp")
    
    # Initial val
    val_loss, val_acc = validate(model, val_loader, device, channel_mask, C, 
                                  static_ch, forcing_ch, exclude_set)
    print(f"[Init] val_loss={val_loss:.5f} val_ACC={val_acc:.4f}")
    _log(f"{'init':>5}  {'--':>2}  {'--':>10}  {val_loss:10.5f}  {val_acc:8.4f}  {datetime.now().strftime('%H:%M:%S')}")
    
    ar_steps = 1
    
    for epoch in range(num_epochs):
        t0 = time.time()
        
        # Curriculum AR
        correct_ar = min(1 + epoch // epochs_per_stage, max_ar)
        if correct_ar > ar_steps:
            ar_steps = correct_ar
            print(f"\n--- AR increased to {ar_steps} ---\n")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, channel_mask,
                                  ar_steps, C, static_ch, forcing_ch)
        val_loss, val_acc = validate(model, val_loader, device, channel_mask, C,
                                      static_ch, forcing_ch, exclude_set)
        scheduler.step()
        
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
    
    # Quick inference on test
    print("\n=== Quick eval on test set ===")
    test_ds_raw = TimeseriesChunkDataset(data_dir, obs_window, pred_steps, split="test_only", n_features=C)
    test_ds = Grid2DDataset(test_ds_raw, n_lon, n_lat)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Load best
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load scalers for denormalization
    scalers = np.load(os.path.join(data_dir, "scalers.npz"))
    mean = scalers["mean"][:C].astype(np.float32)
    std = scalers["std"][:C].astype(np.float32)
    
    # Load variable names
    var_path = os.path.join(data_dir, "variables.json")
    var_names = json.load(open(var_path)) if os.path.exists(var_path) else [f"ch{i}" for i in range(C)]
    
    # AR rollout on test samples
    max_samples = min(200, len(test_ds))
    ar_steps_eval = max_ar
    
    # Per-horizon stats
    sum_se_h = [np.zeros(C, dtype=np.float64) for _ in range(ar_steps_eval)]
    sum_se_base_h = [np.zeros(C, dtype=np.float64) for _ in range(ar_steps_eval)]
    count_h = [0] * ar_steps_eval
    
    with torch.no_grad():
        for i in range(max_samples):
            X, Y = test_ds[i]
            X = X.unsqueeze(0).to(device)  # (1, obs*C, H, W)
            Y = Y.view(n_lat, n_lon, -1)   # (H, W, pred*C)
            
            B, _, H, W = X.shape
            obs = obs_window
            curr_state = X.view(1, obs, C, H, W)
            
            Y_steps = Y.permute(2, 0, 1).view(-1, C, H, W)  # (pred, C, H, W)
            total_pred = Y_steps.shape[0] // C if Y_steps.dim() == 3 else Y_steps.shape[0]
            # Actually Y is (pred*C, H, W) → reshape to (pred, C, H, W)
            Y_steps = Y.permute(2, 0, 1).view(pred_steps, C, H, W)
            
            baseline = curr_state[0, -1].cpu()  # (C, H, W) — persistence
            
            for ar_step in range(min(ar_steps_eval, pred_steps)):
                inp = curr_state.view(1, obs * C, H, W)
                delta = model(inp)
                x_last = curr_state[:, -1]
                out = x_last + delta
                
                if static_ch:
                    for ch in static_ch:
                        out[:, ch] = x_last[:, ch]
                if forcing_ch and ar_step < pred_steps:
                    target_step = Y_steps[ar_step].unsqueeze(0).to(device)
                    for ch in forcing_ch:
                        out[:, ch] = target_step[:, ch]
                
                # Denormalize for physical metrics
                out_np = out[0].cpu().numpy()  # (C, H, W)
                gt_np = Y_steps[ar_step].numpy()  # (C, H, W)
                bl_np = baseline.numpy()  # (C, H, W)
                
                for c in range(C):
                    if c in no_loss_ch:
                        continue
                    pred_phys = out_np[c] * std[c] + mean[c]
                    gt_phys = gt_np[c] * std[c] + mean[c]
                    bl_phys = bl_np[c] * std[c] + mean[c]
                    sum_se_h[ar_step][c] += ((pred_phys - gt_phys) ** 2).sum()
                    sum_se_base_h[ar_step][c] += ((bl_phys - gt_phys) ** 2).sum()
                count_h[ar_step] += H * W
                
                # Shift window
                curr_state = torch.cat([curr_state[:, 1:], out.unsqueeze(1)], dim=1)
            
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{max_samples}]")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"=== U-Net Inference ({max_samples} samples, AR={ar_steps_eval}) ===")
    print(f"Grid: {n_lat}×{n_lon} | C={C} (dynamic={dyn_count})")
    
    hours = [6 * (h + 1) for h in range(ar_steps_eval)]
    
    print(f"\nPer-horizon per-channel RMSE (physical units):")
    header = f"{'var':>10} {'unit':>6}" + "".join(f" {f'+{h}h':>8}" for h in hours)
    print(header)
    
    for c in range(C):
        if c in no_loss_ch:
            continue
        name = var_names[c] if c < len(var_names) else f"ch{c}"
        unit = "K" if "t2m" in name or "t@" in name else "m/s" if "u" in name or "v" in name else "Pa" if "msl" in name or "sp" in name else "?"
        
        vals = []
        for h in range(ar_steps_eval):
            rmse = np.sqrt(sum_se_h[h][c] / max(count_h[h], 1))
            if "t2m" in name or "t@" in name:
                vals.append(f"{rmse:.2f}°C")
            elif "u" in name or "v" in name:
                vals.append(f"{rmse:.2f}")
            else:
                vals.append(f"{rmse:.2f}")
        line = f"{name:>10} {unit:>6}" + "".join(f" {v:>8}" for v in vals)
        print(line)
    
    # Overall skill
    print(f"\nPer-horizon skill (dynamic channels only):")
    for h in range(ar_steps_eval):
        dyn_se = sum(sum_se_h[h][c] for c in range(C) if c not in no_loss_ch)
        dyn_se_base = sum(sum_se_base_h[h][c] for c in range(C) if c not in no_loss_ch)
        rmse_pred = np.sqrt(dyn_se / max(count_h[h] * dyn_count, 1))
        rmse_base = np.sqrt(dyn_se_base / max(count_h[h] * dyn_count, 1))
        skill = (1.0 - rmse_pred / rmse_base) * 100 if rmse_base > 0 else 0
        print(f"  +{hours[h]:02d}h: RMSE={rmse_pred:.4f} | base={rmse_base:.4f} | skill={skill:.1f}%")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

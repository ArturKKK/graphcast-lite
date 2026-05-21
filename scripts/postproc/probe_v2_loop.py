"""200-step v2 training probe with NaN tracking + per-step diagnostics."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.postprocessing.neural.dataset import StationCorpusDataset, build_balanced_sampler
from src.postprocessing.neural.losses import compute_total_loss
from src.postprocessing.neural.models import StationLeadAwareResidualMLP

DEFAULT_FEATURES_V2 = [
    "gnn_t2m","gnn_u10","gnn_v10","gnn_msl","gnn_sp",
    "gnn_t850","gnn_t500","gnn_q850","gnn_z500",
    "gnn_u850","gnn_v850","gnn_u1000","gnn_v1000",
    "lapse_t850_1000","dewpoint_depression","solar_zen",
    "lat","lon","elev","z_surf","lsm",
    "sin_hour","cos_hour","sin_doy","cos_doy","lead_norm",
]

train_p = sys.argv[1] if len(sys.argv) > 1 else "data/postproc/corpus_v2_train.parquet"
val_p = sys.argv[2] if len(sys.argv) > 2 else "data/postproc/corpus_v2_val.parquet"

import pandas as pd
seen = set()
for p in (train_p, val_p):
    seen.update(pd.read_parquet(p, columns=["station_usaf"])["station_usaf"].astype(str).unique())
station_to_idx = {s: i for i, s in enumerate(sorted(seen))}

ds = StationCorpusDataset(train_p, feature_cols=DEFAULT_FEATURES_V2, station_to_idx=station_to_idx)

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
sampler = build_balanced_sampler(ds)
loader = DataLoader(ds, batch_size=4096, sampler=sampler, num_workers=0, drop_last=True)

model = StationLeadAwareResidualMLP(
    feature_dim=len(DEFAULT_FEATURES_V2),
    num_stations=len(station_to_idx),
    station_emb_dim=16,
).to(dev)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for step, batch in enumerate(loader):
    if step >= 250:
        break
    features = batch["features"].to(dev)
    station_idx = batch["station_idx"].to(dev)
    lead_norm = batch["lead_norm"].to(dev)
    gnn = {"t2m": batch["gnn_t2m"].to(dev),
           "u10": batch["gnn_u10"].to(dev),
           "v10": batch["gnn_v10"].to(dev)}
    targets = {"t2m": batch["t2m"].to(dev),
               "u10": batch["u10"].to(dev),
               "v10": batch["v10"].to(dev)}

    optim.zero_grad(set_to_none=True)
    out = model(features, station_idx=station_idx, lead_norm=lead_norm, gnn_targets=gnn)
    losses = compute_total_loss(out, targets, probabilistic=False, w_t2m=1.0, w_wind=1.0,
                                huber_delta=1.0, wind_alpha=0.5)
    loss = losses["loss"]
    if not torch.isfinite(loss):
        print(f"\n!!! NaN/Inf loss at step {step}: {loss.item()}")
        print(f"  features finite={torch.isfinite(features).all().item()} max={features.abs().max().item():.4g}")
        for k in ("t2m","u10","v10"):
            t = targets[k]; o = out[k]
            print(f"  target[{k}] finite={torch.isfinite(t).all().item()} min={t.min().item():.4g} max={t.max().item():.4g}")
            print(f"  out[{k}] finite={torch.isfinite(o).all().item()} min={o.min().item():.4g} max={o.max().item():.4g}")
        print(f"  loss_t2m={losses['loss_t2m'].item():.4g} loss_wind={losses['loss_wind'].item():.4g}")
        # check weights
        for n, p in model.named_parameters():
            if not torch.isfinite(p).all():
                print(f"  WEIGHT NaN: {n}  abs max={p.abs().max().item():.4g}")
        break
    loss.backward()
    bad = []
    for n, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append((n, p.grad.abs().max().item()))
    if bad:
        print(f"\n!!! step {step}: NaN/Inf in grads BEFORE clip:")
        for n, m in bad:
            print(f"     {n}: max={m:.4g}")
        # still check weight norm before clip
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    if step % 20 == 0 or step < 5:
        gn = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
        wm = max((p.abs().max().item() for p in model.parameters()), default=0)
        print(f"step {step:4d} loss={loss.item():.4g} t2m={losses['loss_t2m'].item():.4g} "
              f"wind={losses['loss_wind'].item():.4g} grad_norm={gn:.4g} wmax={wm:.4g}")
print("done")

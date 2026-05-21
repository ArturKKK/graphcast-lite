"""Forward/backward probe: locate source of NaN in v2 training.

Loads first batch, runs forward, checks each intermediate, then backward,
inspects grads.
"""
import json, sys
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
print(f"num_stations={len(station_to_idx)}")

ds = StationCorpusDataset(train_p, feature_cols=DEFAULT_FEATURES_V2, station_to_idx=station_to_idx)
print(f"len={len(ds)}  X_norm shape={ds.X_norm.shape}")
print(f"X_norm finite: {np.isfinite(ds.X_norm).all()}  any inf: {np.isinf(ds.X_norm).any()}")
print(f"Y finite: {np.isfinite(ds.Y).all()}")
print(f"G finite: {np.isfinite(ds.G).all()}")
print(f"feature_std min: {ds.feature_std.min()}  any zero-like: {(ds.feature_std < 1e-5).any()}")
print(f"X_norm abs max: {np.abs(ds.X_norm).max():.4g}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0, drop_last=True)
batch = next(iter(loader))

for k, v in batch.items():
    if v.dtype == torch.long:
        print(f"  batch[{k}] dtype={v.dtype} min={v.min().item()} max={v.max().item()}")
    else:
        f = torch.isfinite(v).all().item()
        print(f"  batch[{k}] dtype={v.dtype} finite={f} min={v.min().item():.4g} max={v.max().item():.4g}")

model = StationLeadAwareResidualMLP(
    feature_dim=len(DEFAULT_FEATURES_V2),
    num_stations=len(station_to_idx),
    station_emb_dim=16,
).to(dev)

features = batch["features"].to(dev)
station_idx = batch["station_idx"].to(dev)
lead_norm = batch["lead_norm"].to(dev)
gnn = {"t2m": batch["gnn_t2m"].to(dev),
       "u10": batch["gnn_u10"].to(dev),
       "v10": batch["gnn_v10"].to(dev)}
targets = {"t2m": batch["t2m"].to(dev),
           "u10": batch["u10"].to(dev),
           "v10": batch["v10"].to(dev)}

print(f"\nstation_idx range: [{station_idx.min().item()}, {station_idx.max().item()}]  vs num_stations={len(station_to_idx)}")

out = model(features, station_idx=station_idx, lead_norm=lead_norm, gnn_targets=gnn)
for k, v in out.items():
    print(f"  out[{k}] finite={torch.isfinite(v).all().item()} min={v.min().item():.4g} max={v.max().item():.4g}")

losses = compute_total_loss(out, targets, probabilistic=False, w_t2m=1.0, w_wind=1.0,
                            huber_delta=1.0, wind_alpha=0.5)
for k, v in losses.items():
    print(f"  loss[{k}] = {v.item():.6g}")

losses["loss"].backward()
for n, p in model.named_parameters():
    if p.grad is None:
        print(f"  GRAD None: {n}")
        continue
    f = torch.isfinite(p.grad).all().item()
    if not f:
        print(f"  GRAD NaN/Inf in: {n}  max={p.grad.abs().max().item():.4g}")

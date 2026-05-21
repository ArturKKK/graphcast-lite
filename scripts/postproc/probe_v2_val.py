"""Check val dataset for finiteness post-normalization."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.postprocessing.neural.dataset import StationCorpusDataset

DEFAULT_FEATURES_V2 = [
    "gnn_t2m","gnn_u10","gnn_v10","gnn_msl","gnn_sp",
    "gnn_t850","gnn_t500","gnn_q850","gnn_z500",
    "gnn_u850","gnn_v850","gnn_u1000","gnn_v1000",
    "lapse_t850_1000","dewpoint_depression","solar_zen",
    "lat","lon","elev","z_surf","lsm",
    "sin_hour","cos_hour","sin_doy","cos_doy","lead_norm",
]

train_p = "data/postproc/corpus_v2_train.parquet"
val_p = "data/postproc/corpus_v2_val.parquet"

seen = set()
for p in (train_p, val_p):
    seen.update(pd.read_parquet(p, columns=["station_usaf"])["station_usaf"].astype(str).unique())
station_to_idx = {s: i for i, s in enumerate(sorted(seen))}
print(f"num_stations={len(station_to_idx)}")

train_ds = StationCorpusDataset(train_p, feature_cols=DEFAULT_FEATURES_V2, station_to_idx=station_to_idx)
val_ds = StationCorpusDataset(val_p, feature_cols=DEFAULT_FEATURES_V2, station_to_idx=station_to_idx, scalers=train_ds.export_scalers())

print(f"\nTRAIN: len={len(train_ds)}")
print(f"  X_norm finite: {np.isfinite(train_ds.X_norm).all()}  abs max: {np.abs(train_ds.X_norm).max():.4g}")
print(f"  Y finite: {np.isfinite(train_ds.Y).all()}")
print(f"  G finite: {np.isfinite(train_ds.G).all()}")

print(f"\nVAL: len={len(val_ds)}")
print(f"  X_norm finite: {np.isfinite(val_ds.X_norm).all()}  abs max: {np.abs(val_ds.X_norm).max():.4g}")
print(f"  Y finite: {np.isfinite(val_ds.Y).all()}")
print(f"  G finite: {np.isfinite(val_ds.G).all()}")
print(f"  any NaN in Y: {np.isnan(val_ds.Y).any()}  Inf: {np.isinf(val_ds.Y).any()}")
print(f"  any NaN in G: {np.isnan(val_ds.G).any()}  Inf: {np.isinf(val_ds.G).any()}")
print(f"  any NaN in X_norm: {np.isnan(val_ds.X_norm).any()}  Inf: {np.isinf(val_ds.X_norm).any()}")

# Per-column for val
for i, c in enumerate(DEFAULT_FEATURES_V2):
    x = val_ds.X_norm[:, i]
    nn = np.isnan(x).sum(); inf = np.isinf(x).sum()
    if nn or inf:
        print(f"  VAL.X[{c}] NaN={nn} Inf={inf}")
for i, c in enumerate(["t2m","u10","v10"]):
    y = val_ds.Y[:, i]
    nn = np.isnan(y).sum(); inf = np.isinf(y).sum()
    if nn or inf:
        print(f"  VAL.Y[{c}] NaN={nn} Inf={inf}")
for i, c in enumerate(["t2m","u10","v10"]):
    g = val_ds.G[:, i]
    nn = np.isnan(g).sum(); inf = np.isinf(g).sum()
    if nn or inf:
        print(f"  VAL.G[{c}] NaN={nn} Inf={inf}")

# Also raw val dataframe NaN check on relevant cols
df = pd.read_parquet(val_p)
print(f"\nRAW VAL parquet rows={len(df)}")
for c in ["dewpoint_depression","obs_t2m_K","obs_u10","obs_v10","gnn_t2m","gnn_u10","gnn_v10"]:
    if c in df.columns:
        nn = df[c].isna().sum(); inf = np.isinf(df[c]).sum()
        if nn or inf:
            print(f"  raw[{c}] NaN={nn} Inf={inf}")

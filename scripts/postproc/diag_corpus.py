"""Diagnose corpus_v2_train.parquet for NaN/Inf features and target ranges."""
import pandas as pd, numpy as np, sys

p = sys.argv[1] if len(sys.argv) > 1 else "data/postproc/corpus_v2_train.parquet"
df = pd.read_parquet(p)
feats = [
    "gnn_t2m","gnn_u10","gnn_v10","gnn_msl","gnn_sp","gnn_t850","gnn_t500",
    "gnn_q850","gnn_z500","gnn_u850","gnn_v850","gnn_u1000","gnn_v1000",
    "lapse_t850_1000","dewpoint_depression","solar_zen",
    "station_lat","station_lon","station_elev","z_surf","lsm",
    "sin_hour","cos_hour","sin_doy","cos_doy","lead_norm",
]
print(f"rows={len(df):,} cols={len(df.columns)}")
print("--- NaN/Inf per feature (only nonzero) ---")
for c in feats:
    if c not in df.columns:
        print(f"  MISSING: {c}")
        continue
    nn = df[c].isna().sum()
    inf = np.isinf(df[c]).sum()
    if nn or inf:
        print(f"  {c}: NaN={nn} Inf={inf}")

print("--- targets ---")
for c in ("obs_t2m_K", "obs_u10", "obs_v10"):
    if c in df.columns:
        x = df[c]
        print(f"  {c}: NaN={x.isna().sum()} Inf={np.isinf(x).sum()} "
              f"min={x.min():.3g} max={x.max():.3g} mean={x.mean():.3g}")
    else:
        print(f"  MISSING: {c}")

print("--- feature stats ---")
for c in feats:
    if c not in df.columns:
        continue
    x = df[c].to_numpy()
    print(f"  {c:24s} min={np.nanmin(x):+.4g} max={np.nanmax(x):+.4g} "
          f"mean={np.nanmean(x):+.4g} std={np.nanstd(x):.4g}")

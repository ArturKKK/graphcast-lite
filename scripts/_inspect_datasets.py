#!/usr/bin/env python3
"""Quick inspection of global Jan2023 and regional CDS datasets."""
import json, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "datasets"

for name in ["wb2_512x256_19f_jan2023", "region_krsk_cds_19f"]:
    d = ROOT / name
    if not d.exists():
        print(f"--- {name}: NOT FOUND ---")
        continue
    info = json.loads((d / "dataset_info.json").read_text())
    print(f"\n=== {name} ===")
    print(json.dumps(info, indent=2))

    c = np.load(d / "coords.npz")
    lat, lon = c["latitude"], c["longitude"]
    print(f"  lat: {lat.shape} [{lat.min():.4f} .. {lat.max():.4f}]")
    print(f"  lon: {lon.shape} [{lon.min():.4f} .. {lon.max():.4f}]")

    s = np.load(d / "scalers.npz")
    print(f"  scalers mean: {s['mean'].shape}  std: {s['std'].shape}")

    if (d / "variables.json").exists():
        v = json.loads((d / "variables.json").read_text())
        print(f"  variables: {v}")

    data = np.memmap(str(d / "data.npy"), dtype=np.float16, mode="r")
    n_f = info.get("n_features") or info.get("n_feat", 19)
    n_lon = info.get("n_lon", 512)
    n_lat = info.get("n_lat", 256)
    n_t = len(data) // (n_lon * n_lat * n_f)
    print(f"  data: {len(data)} elems => {n_t} timesteps x {n_lon} x {n_lat} x {n_f}")
    print(f"  data bytes: {data.nbytes / 1e6:.1f} MB")

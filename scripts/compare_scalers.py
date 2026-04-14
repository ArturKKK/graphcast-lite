#!/usr/bin/env python3
"""Compare scalers between merge and interpolate datasets."""
import numpy as np, json, os

datasets = {
    "merge (real)": "data/datasets/multires_krsk_19f_real",
    "interpolate (nores)": "data/datasets/multires_krsk_19f",
}

for name, path in datasets.items():
    sc = np.load(os.path.join(path, "scalers.npz"))
    vr = json.load(open(os.path.join(path, "variables.json")))
    print(f"\n=== {name} ===")
    print(f"Path: {path}")
    for i, v in enumerate(vr):
        print(f"  {i:2d} {v:>8s}  mean={sc['mean'][i]:12.4f}  std={sc['std'][i]:12.4f}")

# Also check global 512x256 if exists
glob_path = "data/datasets/global_512x256_19f_2010-2021_07deg"
if os.path.exists(os.path.join(glob_path, "scalers.npz")):
    sc = np.load(os.path.join(glob_path, "scalers.npz"))
    vr = json.load(open(os.path.join(glob_path, "variables.json")))
    print(f"\n=== global 512x256 ===")
    for i, v in enumerate(vr[:19]):
        print(f"  {i:2d} {v:>8s}  mean={sc['mean'][i]:12.4f}  std={sc['std'][i]:12.4f}")
else:
    print(f"\n[global 512x256 not found at {glob_path}]")

# Check 64x32
for gp in ["data/datasets/wb2_64x32_15f", "data/datasets/2010-2020_15var_4obs_1pred_t_50ep"]:
    if os.path.exists(os.path.join(gp, "scalers.npz")):
        sc = np.load(os.path.join(gp, "scalers.npz"))
        print(f"\n=== {gp} ===")
        print(f"  t2m  mean={sc['mean'][0]:12.4f}  std={sc['std'][0]:12.4f}")
        break

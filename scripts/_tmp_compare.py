"""Compare interp vs merge in ROI Krsk.

Builds 'interp' on the fly: linear-interpolate global (256,512) onto region (41,61) grid
for the same timestamps, then compute per-channel RMSE/MAE in raw units and normalized
by merge scalers std.
"""
import numpy as np, json, os
from scipy.interpolate import RegularGridInterpolator

GLOBAL_DIR = "/data/datasets/wb2_512x256_19f_ar"
REGION_DIR = "/data/datasets/region_krsk_61x41_19f_2010-2020_025deg"
MERGE_DIR  = "/data/datasets/multires_krsk_19f_merge"

gi = json.load(open(os.path.join(GLOBAL_DIR, "dataset_info.json")))
ri = json.load(open(os.path.join(REGION_DIR, "dataset_info.json")))
variables = json.load(open(os.path.join(MERGE_DIR, "variables.json")))

g_lat = np.load(os.path.join(GLOBAL_DIR, "coords.npz"))["latitude"].astype(np.float32)
g_lon = np.load(os.path.join(GLOBAL_DIR, "coords.npz"))["longitude"].astype(np.float32)
r_lat = np.load(os.path.join(REGION_DIR, "coords.npz"))["latitude"].astype(np.float32)
r_lon = np.load(os.path.join(REGION_DIR, "coords.npz"))["longitude"].astype(np.float32)

# Global lat is 89.65..-89.65 descending? sort ascending for interpolator
g_lat_sorted_idx = np.argsort(g_lat)
g_lat_s = g_lat[g_lat_sorted_idx]
print("g_lat asc range:", g_lat_s[0], g_lat_s[-1])
print("g_lon range    :", g_lon[0], g_lon[-1])
print("r_lat range    :", r_lat[0], r_lat[-1])
print("r_lon range    :", r_lon[0], r_lon[-1])

N_global_t = gi["n_time"]
N_region_t = ri["n_time"]
F = gi["n_feat"]
assert F == len(variables)

g = np.memmap(os.path.join(GLOBAL_DIR, "data.npy"), dtype=np.float16, mode="r",
              shape=(N_global_t, 256, 512, F))
r = np.memmap(os.path.join(REGION_DIR, "data.npy"), dtype=np.float16, mode="r",
              shape=(N_region_t, 41, 61, F))

# scalers from merge (model space)
sc = np.load(os.path.join(MERGE_DIR, "scalers.npz"))
mean = sc["mean"].astype(np.float64)
std  = sc["std"].astype(np.float64)
std_safe = np.where(std > 1e-9, std, 1.0)

# build region mesh for interpolation target
rlat_grid, rlon_grid = np.meshgrid(r_lat, r_lon, indexing="ij")  # (41,61)
pts_target = np.stack([rlat_grid.ravel(), rlon_grid.ravel()], axis=-1)  # (41*61, 2)

rng = np.random.default_rng(42)
N_sample = 300
t_idx = rng.choice(min(N_global_t, N_region_t), size=N_sample, replace=False)
t_idx.sort()

sq_err = np.zeros(F, dtype=np.float64)
abs_err = np.zeros(F, dtype=np.float64)
n_total = 0

for ti, t in enumerate(t_idx):
    g_slice = np.asarray(g[t], dtype=np.float32)            # (256,512,19)
    g_slice_sorted = g_slice[g_lat_sorted_idx, :, :]        # lat ascending
    r_slice = np.asarray(r[t], dtype=np.float32)            # (41,61,19)
    # interpolate per-channel
    interp_at_region = np.empty((41 * 61, F), dtype=np.float32)
    for f in range(F):
        rgi = RegularGridInterpolator(
            (g_lat_s, g_lon), g_slice_sorted[:, :, f],
            method="linear", bounds_error=False, fill_value=None)
        interp_at_region[:, f] = rgi(pts_target)
    r_flat = r_slice.reshape(41 * 61, F)
    diff = interp_at_region - r_flat
    sq_err += (diff ** 2).sum(axis=0)
    abs_err += np.abs(diff).sum(axis=0)
    n_total += r_flat.shape[0]
    if ti % 50 == 0:
        print(f"  progressed t={ti}/{N_sample}")

rmse = np.sqrt(sq_err / n_total)
mae  = abs_err / n_total
rmse_norm = rmse / std_safe

print("\n=== Interp (global→region) vs MERGE (real region) — ROI Krsk 50..60N, 83..98E ===")
print(f"samples: {N_sample} timestamps × {n_total//N_sample} nodes each ({n_total} total per channel)")
print(f"\n{'var':<8s} {'RMSE raw':>12s} {'MAE raw':>12s} {'std (merge)':>14s} {'RMSE σ-norm':>14s}")
for i, v in enumerate(variables):
    print(f"{v:<8s} {rmse[i]:>12.5f} {mae[i]:>12.5f} {std[i]:>14.5f} {rmse_norm[i]:>14.4f}")

print(f"\nMean normalized RMSE across 19 channels: {rmse_norm.mean():.4f} σ")
print(f"Median normalized RMSE              : {np.median(rmse_norm):.4f} σ")
print(f"Max normalized RMSE                 : {rmse_norm.max():.4f} σ  (var: {variables[int(np.argmax(rmse_norm))]})")

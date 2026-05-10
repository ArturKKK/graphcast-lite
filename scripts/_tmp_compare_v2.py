"""Compare interp vs merge in ROI Krsk — CORRECT global shape (T, lon, lat, feat)."""
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
# Both ascending, regular grids

F = gi["n_feat"]
# Global data shape: (time, lon=512, lat=256, feat)
g = np.memmap(os.path.join(GLOBAL_DIR, "data.npy"), dtype=np.float16, mode="r",
              shape=(gi["n_time"], 512, 256, F))
# Region data shape: (time, lat=41, lon=61, feat) — verified via probe
r = np.memmap(os.path.join(REGION_DIR, "data.npy"), dtype=np.float16, mode="r",
              shape=(ri["n_time"], 41, 61, F))

sc = np.load(os.path.join(MERGE_DIR, "scalers.npz"))
std = sc["std"].astype(np.float64)
std_safe = np.where(std > 1e-9, std, 1.0)

# region mesh target: (lat, lon) pairs flattened
rlat_grid, rlon_grid = np.meshgrid(r_lat, r_lon, indexing="ij")    # (41,61)
pts_target = np.stack([rlat_grid.ravel(), rlon_grid.ravel()], -1)  # (2501, 2)

rng = np.random.default_rng(42)
N_sample = 300
T_common = min(gi["n_time"], ri["n_time"])
t_idx = np.sort(rng.choice(T_common, size=N_sample, replace=False))

sq_err = np.zeros(F, dtype=np.float64)
abs_err = np.zeros(F, dtype=np.float64)
n_total = 0
for ti, t in enumerate(t_idx):
    # global slice shape (lon=512, lat=256, feat=19); for interpolator need (lat,lon,feat)
    g_slice = np.asarray(g[t], dtype=np.float32).transpose(1, 0, 2)  # -> (lat, lon, feat)
    r_slice = np.asarray(r[t], dtype=np.float32)                     # (41, 61, 19)
    interp_at_region = np.empty((41 * 61, F), dtype=np.float32)
    for f in range(F):
        rgi = RegularGridInterpolator((g_lat, g_lon), g_slice[:, :, f],
                                      method="linear", bounds_error=False, fill_value=None)
        interp_at_region[:, f] = rgi(pts_target)
    r_flat = r_slice.reshape(41 * 61, F)
    diff = interp_at_region - r_flat
    sq_err  += (diff ** 2).sum(axis=0)
    abs_err += np.abs(diff).sum(axis=0)
    n_total += r_flat.shape[0]
    if ti % 50 == 0:
        print(f"  t={ti}/{N_sample}")

rmse = np.sqrt(sq_err / n_total)
mae  = abs_err / n_total
rmse_norm = rmse / std_safe

print("\n=== Interp (global→region, bilinear) vs MERGE (real region) — ROI Krsk ===")
print(f"samples: {N_sample} timestamps × 2501 regional nodes ({n_total} total per channel)")
print(f"\n{'var':<8s} {'RMSE raw':>12s} {'MAE raw':>12s} {'std (merge)':>14s} {'RMSE σ-norm':>14s}")
for i, v in enumerate(variables):
    print(f"{v:<8s} {rmse[i]:>12.5f} {mae[i]:>12.5f} {std[i]:>14.5f} {rmse_norm[i]:>14.4f}")

print(f"\nMean normalized RMSE: {rmse_norm.mean():.4f} σ")
print(f"Median normalized RMSE: {np.median(rmse_norm):.4f} σ")
top = np.argsort(rmse_norm)[::-1][:5]
print("Top-5 σ-RMSE channels:")
for i in top:
    print(f"  {variables[i]:<8s} {rmse_norm[i]:.4f} σ")

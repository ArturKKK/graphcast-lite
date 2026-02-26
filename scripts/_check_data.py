#!/usr/bin/env python3
"""Quick check: compare normalized regional vs global data distributions."""
import numpy as np

vars19 = ['t2m','10u','10v','msl','tp','sp','tcwv','z_surf','lsm',
          't@850','u@850','v@850','z@850','q@850',
          't@500','u@500','v@500','z@500','q@500']

# --- Regional ---
d = np.memmap('data/datasets/region_krsk_cds_19f/data.npy',
              dtype=np.float16, mode='r').reshape(-1, 61, 41, 19).astype(np.float32)
sc = np.load('data/datasets/region_krsk_cds_19f/scalers.npz')
mean, std = sc['mean'], sc['std']
normed = (d - mean) / std

print("=== REGIONAL (normalized with global scalers) ===")
for i, v in enumerate(vars19):
    ch = normed[:,:,:,i]
    print(f"  {i:2d}: {v:8s}  min={ch.min():+7.2f}  max={ch.max():+7.2f}  "
          f"mean={ch.mean():+7.4f}  std={ch.std():.4f}")

# --- Global (first 10 timesteps) ---
print()
info_g = np.load('data/datasets/wb2_512x256_19f_ar/coords.npz')
nlat, nlon = len(info_g['latitude']), len(info_g['longitude'])
total_elems = 19 * nlon * nlat
dg_raw = np.memmap('data/datasets/wb2_512x256_19f_ar/data.npy',
                   dtype=np.float16, mode='r')
n_time = dg_raw.size // total_elems
print(f"Global: {n_time} timesteps, {nlon}x{nlat}, 19 features")
dg = dg_raw[:10*total_elems].reshape(10, nlon, nlat, 19).astype(np.float32)
sg = np.load('data/datasets/wb2_512x256_19f_ar/scalers.npz')
normed_g = (dg - sg['mean']) / sg['std']

print("=== GLOBAL (first 10 timesteps, normalized) ===")
for i, v in enumerate(vars19):
    ch = normed_g[:,:,:,i]
    print(f"  {i:2d}: {v:8s}  min={ch.min():+7.2f}  max={ch.max():+7.2f}  "
          f"mean={ch.mean():+7.4f}  std={ch.std():.4f}")

# --- Scalers comparison ---
print()
print("=== SCALERS IDENTICAL?", np.allclose(sc['mean'], sg['mean']) and np.allclose(sc['std'], sg['std']), "===")

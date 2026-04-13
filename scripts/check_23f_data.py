"""Quick check of 23f dataset raw values vs scalers."""
import numpy as np
ds = "data/datasets/region_krsk_61x41_23f_2010-2020_025deg"
scl = np.load(ds + "/scalers.npz")
print("mean[:10]:", scl["mean"][:10])
print("std[:10]:", scl["std"][:10])
data = np.memmap(ds + "/data.npy", dtype=np.float16, mode="r", shape=(16072, 61, 41, 23))
print("shape:", data.shape)
print("t=0 center:", data[0, 30, 20, :10].astype(float))
print("t=100 center:", data[100, 30, 20, :10].astype(float))
print("t=8000 center:", data[8000, 30, 20, :10].astype(float))
t2m = data[:1000, :, :, 0].astype(np.float32)
print(f"t2m: min={t2m.min():.1f} max={t2m.max():.1f} mean={t2m.mean():.1f}")
msl = data[:1000, :, :, 3].astype(np.float32)
print(f"msl: min={msl.min():.0f} max={msl.max():.0f} mean={msl.mean():.0f}")
sp = data[:1000, :, :, 5].astype(np.float32)
print(f"sp: min={sp.min():.0f} max={sp.max():.0f} mean={sp.mean():.0f}")
z500 = data[:1000, :, :, 17].astype(np.float32)
print(f"z@500: min={z500.min():.0f} max={z500.max():.0f} mean={z500.mean():.0f}")

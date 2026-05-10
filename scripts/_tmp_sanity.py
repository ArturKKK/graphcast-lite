import numpy as np, json, os
GLOBAL_DIR = "/data/datasets/wb2_512x256_19f_ar"
REGION_DIR = "/data/datasets/region_krsk_61x41_19f_2010-2020_025deg"
MERGE_DIR  = "/data/datasets/multires_krsk_19f_merge"

g_lat = np.load(os.path.join(GLOBAL_DIR, "coords.npz"))["latitude"].astype(np.float32)
g_lon = np.load(os.path.join(GLOBAL_DIR, "coords.npz"))["longitude"].astype(np.float32)
r_lat = np.load(os.path.join(REGION_DIR, "coords.npz"))["latitude"].astype(np.float32)
r_lon = np.load(os.path.join(REGION_DIR, "coords.npz"))["longitude"].astype(np.float32)
print("g_lat[:5]", g_lat[:5], "g_lat[-5:]", g_lat[-5:])
print("g_lon[:5]", g_lon[:5], "g_lon[-5:]", g_lon[-5:])
print("r_lat[:5]", r_lat[:5], "r_lat[-5:]", r_lat[-5:])
print("r_lon[:5]", r_lon[:5], "r_lon[-5:]", r_lon[-5:])

g = np.memmap(os.path.join(GLOBAL_DIR, "data.npy"), dtype=np.float16, mode="r",
              shape=(17532, 256, 512, 19))
r = np.memmap(os.path.join(REGION_DIR, "data.npy"), dtype=np.float16, mode="r",
              shape=(16072, 41, 61, 19))
m = np.memmap(os.path.join(MERGE_DIR, "data.npy"), dtype=np.float16, mode="r",
              shape=(16072, 133279, 19))
mc = np.load(os.path.join(MERGE_DIR, "coords.npz"))
is_reg = mc["is_regional"]

# t2m at t=0 central region point
print("\n-- t=0, t2m at a few points --")
# global node nearest to lat=55, lon=90
gi_lat = int(np.argmin(np.abs(g_lat - 55)))
gi_lon = int(np.argmin(np.abs(g_lon - 90)))
print("global[0, lat~55, lon~90, t2m] =", float(g[0, gi_lat, gi_lon, 0]))
# region node at lat=55, lon=90
ri_lat = int(np.argmin(np.abs(r_lat - 55)))
ri_lon = int(np.argmin(np.abs(r_lon - 90)))
print("region[0, lat=55, lon=90, t2m] =", float(r[0, ri_lat, ri_lon, 0]))
# merge: mask to regional, find nearest
reg_idx = np.where(is_reg)[0]
m_lat = mc["latitude"][reg_idx]
m_lon = mc["longitude"][reg_idx]
mi = int(np.argmin(np.abs(m_lat - 55) + np.abs(m_lon - 90)))
print("merge[0, regional ~ 55/90, t2m] =", float(m[0, reg_idx[mi], 0]))

print("\n-- ranges of t2m in a central slice at t=0 --")
print("global slice t2m:", float(g[0, :, :, 0].min()), float(g[0, :, :, 0].max()), "mean", float(np.asarray(g[0,:,:,0]).mean()))
print("region t2m     :", float(r[0, :, :, 0].min()), float(r[0, :, :, 0].max()), "mean", float(np.asarray(r[0,:,:,0]).mean()))
print("merge regional t2m:", float(m[0, reg_idx, 0].min()), float(m[0, reg_idx, 0].max()))
print("merge globalpart t2m:", float(m[0, ~is_reg, 0].min()), float(m[0, ~is_reg, 0].max()))

sc_m = np.load(os.path.join(MERGE_DIR, "scalers.npz"))
sc_g = np.load(os.path.join(GLOBAL_DIR, "scalers.npz"))
sc_r = np.load(os.path.join(REGION_DIR, "scalers.npz"))
print("\nscalers merge mean[0] (t2m):", float(sc_m["mean"][0]), "std:", float(sc_m["std"][0]))
print("scalers global mean[0] (t2m):", float(sc_g["mean"][0]), "std:", float(sc_g["std"][0]))
print("scalers region mean[0] (t2m):", float(sc_r["mean"][0]), "std:", float(sc_r["std"][0]))

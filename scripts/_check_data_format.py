import numpy as np, json
r = "data/datasets/region_krsk_61x41_19f_2010-2020_025deg"
i = json.load(open(r + "/dataset_info.json"))
shape = (i["n_time"], i["n_lon"], i["n_lat"], i["n_feat"])
d = np.memmap(r + "/data.npy", dtype=np.float16, mode="r", shape=shape)
sc = np.load(r + "/scalers.npz")
m, s = sc["mean"][:19].astype(np.float32), sc["std"][:19].astype(np.float32)

v = float(d[0, 30, 20, 0])
z = float(d[0, 30, 20, 7])
print(f"t2m raw={v:.4f}")
print(f"  if normalized: phys = {v * s[0] + m[0]:.2f} K = {v * s[0] + m[0] - 273.15:.2f} C")
print(f"  if raw: {v:.2f} K = {v - 273.15:.2f} C")
print(f"  mean={m[0]:.2f} std={s[0]:.2f}")
print(f"z_surf raw={z:.4f}")
print(f"  if normalized: phys = {z * s[7] + m[7]:.2f}")
print(f"  if raw: {z:.2f}")
print(f"  mean={m[7]:.2f} std={s[7]:.2f}")

# Also check global
g = "data/datasets/global_512x256_19f_2010-2021_07deg"
gi = json.load(open(g + "/dataset_info.json"))
gshape = (gi["n_time"], gi["n_lon"], gi["n_lat"], gi["n_feat"])
gd = np.memmap(g + "/data.npy", dtype=np.float16, mode="r", shape=gshape)
gsc = np.load(g + "/scalers.npz")
gm, gs = gsc["mean"][:19].astype(np.float32), gsc["std"][:19].astype(np.float32)
gv = float(gd[0, 250, 128, 0])
print(f"\nGlobal t2m raw={gv:.4f}")
print(f"  if normalized: {gv * gs[0] + gm[0]:.2f} K")
print(f"  if raw: {gv:.2f} K")
print(f"  global mean={gm[0]:.2f} std={gs[0]:.2f}")

import numpy as np
g = np.memmap("/data/datasets/wb2_512x256_19f_ar/data.npy",
              dtype=np.float16, mode="r", shape=(17532, 256, 512, 19))
r = np.memmap("/data/datasets/region_krsk_61x41_19f_2010-2020_025deg/data.npy",
              dtype=np.float16, mode="r", shape=(16072, 41, 61, 19))
# probe annual cycle at (55N, 90E) for global; t2m
for t in [0, 200, 500, 730, 1000, 1200, 1461, 1461+730]:
    vg = float(g[t, 206, 128, 0])
    vr = float(r[t, 20, 28, 0]) if t < 16072 else float("nan")
    print(f"t={t:5d}  global={vg:6.1f}K  region={vr:6.1f}K")
print()
# Also scan first week of t
for t in range(0, 8):
    print(f"t={t} global(55N,90E)={float(g[t,206,128,0]):6.1f} region(55N,90E)={float(r[t,20,28,0]):6.1f}")

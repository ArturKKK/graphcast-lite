import numpy as np
# Try alternative shape: (time, 512, 256, 19) i.e. (time, lon, lat, feat)
g = np.memmap("/data/datasets/wb2_512x256_19f_ar/data.npy",
              dtype=np.float16, mode="r", shape=(17532, 512, 256, 19))
r = np.memmap("/data/datasets/region_krsk_61x41_19f_2010-2020_025deg/data.npy",
              dtype=np.float16, mode="r", shape=(16072, 41, 61, 19))

# 55N ~ lat idx 206; 90E ~ lon idx 128. If shape is (T, lon=512, lat=256, F):
for t in [0, 730, 1461]:
    print(f"t={t}: try (lon=128,lat=206) -> {float(g[t, 128, 206, 0]):.1f}  "
          f"(lon=128,lat=49) -> {float(g[t, 128, 49, 0]):.1f}")
    print(f"       region(lat=20,lon=28) = {float(r[t, 20, 28, 0]):.1f}")

# Also consider flipped lat (data lat descending): lat_flipped_idx = 255-206 = 49
# Already covered above.

import numpy as np
g = np.memmap("/data/datasets/wb2_512x256_19f_ar/data.npy",
              dtype=np.float16, mode="r", shape=(17532, 256, 512, 19))
print("g[0,0,0,t2m]", float(g[0, 0, 0, 0]))
print("g[0,255,0,t2m]", float(g[0, 255, 0, 0]))
print("g[0,206,128,t2m]", float(g[0, 206, 128, 0]))
print("g[0,49,128,t2m]", float(g[0, 49, 128, 0]))

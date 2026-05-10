"""Compare interp vs merge datasets in ROI Krsk per-channel.

Builds 'interp' on the fly by nearest-neighbour mapping from the global 512x256 grid
onto regional 61x41 coordinates, then computes differences vs the real regional dataset
in raw physical units and normalized by merge scalers std.
"""
import numpy as np, json, os, sys

GLOBAL_DIR = "/data/datasets/wb2_512x256_19f_ar"
REGION_DIR = "/data/datasets/region_krsk_61x41_19f_2010-2020_025deg"
MERGE_DIR  = "/data/datasets/multires_krsk_19f_merge"

def info(d):
    with open(os.path.join(d, "dataset_info.json")) as f:
        return json.load(f)

gi = info(GLOBAL_DIR); ri = info(REGION_DIR); mi = info(MERGE_DIR)
print("global:", gi["n_time"], gi.get("n_nodes"), "mode", gi.get("mode"))
print("region:", ri["n_time"], ri.get("n_nodes"), "mode", ri.get("mode"))
print("merge :", mi["n_time"], mi.get("n_nodes"), "mode", mi.get("mode"), "regional nodes", mi.get("n_regional"))

gc = np.load(os.path.join(GLOBAL_DIR, "coords.npz"))
rc = np.load(os.path.join(REGION_DIR, "coords.npz"))
mc = np.load(os.path.join(MERGE_DIR,  "coords.npz"))
print("global coord keys:", list(gc.keys()))
print("region coord keys:", list(rc.keys()))
print("merge  coord keys:", list(mc.keys()))

# Figure out global lat/lon and region lat/lon arrays
def pick(c):
    keys = list(c.keys())
    lat_k = [k for k in keys if "lat" in k.lower()]
    lon_k = [k for k in keys if "lon" in k.lower()]
    return lat_k, lon_k

print(" global pick:", pick(gc))
print(" region pick:", pick(rc))
print(" merge  pick:", pick(mc))

for k in gc.keys(): print("global", k, gc[k].shape, gc[k].dtype, "min/max", np.asarray(gc[k]).min(), np.asarray(gc[k]).max())
for k in rc.keys(): print("region", k, rc[k].shape, rc[k].dtype, "min/max", np.asarray(rc[k]).min(), np.asarray(rc[k]).max())
for k in mc.keys(): print("merge ", k, mc[k].shape, mc[k].dtype, "min/max", np.asarray(mc[k]).min(), np.asarray(mc[k]).max())

variables = json.load(open(os.path.join(MERGE_DIR, "variables.json")))
print("vars:", variables)

# Load scalers for normalization (use merge scalers)
sc = np.load(os.path.join(MERGE_DIR, "scalers.npz"))
print("scaler keys:", list(sc.keys()), {k:sc[k].shape for k in sc.keys()})

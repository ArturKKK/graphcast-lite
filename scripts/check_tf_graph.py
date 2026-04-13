"""Diagnostic: compare checkpoint graph edges with rebuilt graph for timeforce model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.main import load_model_from_experiment_config
from src.config import ExperimentConfig
from src.data.data_configs import DatasetMetadata
from src.utils import load_from_json_file

cfg = ExperimentConfig(**load_from_json_file("experiments/region_krsk_23f_timeforce/config.json"))
coords = np.load("/data/datasets/region_krsk_61x41_23f_2010-2020_025deg/coords.npz")
lats = coords["latitude"].astype(np.float32)
lons = coords["longitude"].astype(np.float32)
print(f"lat: {lats.shape} [{lats.min():.2f}, {lats.max():.2f}]")
print(f"lon: {lons.shape} [{lons.min():.2f}, {lons.max():.2f}]")

rb = (float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max()))
meta = DatasetMetadata(
    flattened=True, num_latitudes=len(lats), num_longitudes=len(lons),
    num_features=23, obs_window=2, pred_window=1,
)
meta.flat_grid = False
meta.cordinates = (lats, lons)

print("Building model with region_bounds + mesh_buffer=15.0 ...")
m = load_model_from_experiment_config(
    cfg, torch.device("cpu"), meta,
    coordinates=(lats, lons), region_bounds=rb, mesh_buffer=15.0,
)

sd = m.state_dict()
pe = sd.get("_processing_edge_features")
print(f"REBUILT _processing_edge_features: {pe.shape if pe is not None else 'NOT FOUND'}")

# Compare with checkpoint
ckpt = torch.load("experiments/region_krsk_23f_timeforce/best_model.pth", map_location="cpu")
pe_ckpt = ckpt.get("_processing_edge_features")
print(f"CKPT   _processing_edge_features: {pe_ckpt.shape if pe_ckpt is not None else 'NOT FOUND'}")

if pe is not None and pe_ckpt is not None:
    if pe.shape == pe_ckpt.shape:
        diff = (pe - pe_ckpt).abs().max().item()
        print(f"MATCH! max diff = {diff:.6f}")
    else:
        print(f"MISMATCH: rebuilt={pe.shape} vs ckpt={pe_ckpt.shape}")

# Also check encoder/decoder edge counts
for k, v in sd.items():
    if "edge" in k.lower() and "features" in k.lower():
        print(f"  model buffer: {k} = {v.shape}")

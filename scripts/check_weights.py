#!/usr/bin/env python3
"""Check weight loading for real_freeze6 — debug missing/unexpected keys."""
import sys, torch
sys.path.insert(0, ".")
from src.config import ExperimentConfig
from src.main import load_model_from_experiment_config
from data.data_loading import load_chunked_datasets

cfg = ExperimentConfig.from_json("experiments/multires_real_freeze6/config.json")
_, _, _, meta = load_chunked_datasets(
    "data/datasets/multires_krsk_19f_real", obs_window=2, pred_steps=1, n_features=19
)

device = torch.device("cuda")
model = load_model_from_experiment_config(
    cfg, device, meta, coordinates=meta.cordinates, flat_grid=True
)

ckpt = torch.load("experiments/multires_real_freeze6/best_model.pth", map_location=device)
result = model.load_state_dict(ckpt, strict=False)

print("\n=== WEIGHT LOADING REPORT ===")
print(f"Total model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Total ckpt keys: {len(ckpt)}")
print(f"Total model keys: {len(dict(model.state_dict()))}")
print(f"\nMissing keys ({len(result.missing_keys)}):")
for k in result.missing_keys:
    print(f"  MISSING: {k}")
print(f"\nUnexpected keys ({len(result.unexpected_keys)}):")
for k in result.unexpected_keys:
    print(f"  UNEXPECTED: {k}")

# Check processor specifically
proc_ckpt = [k for k in ckpt if "processor" in k]
proc_model = [k for k in model.state_dict() if "processor" in k]
print(f"\nProcessor keys in ckpt: {len(proc_ckpt)}")
print(f"Processor keys in model: {len(proc_model)}")
if proc_ckpt[:5]:
    print(f"  ckpt sample: {proc_ckpt[:5]}")
if proc_model[:5]:
    print(f"  model sample: {proc_model[:5]}")

#!/usr/bin/env python3
# scripts/generate_assim_dataset.py
# Create a minimal "observations" package for nudging using y_test,
# and a mask over variables to assimilate.

import argparse, json, os
from pathlib import Path
import torch
from typing import List
from src.constants import FileNames

def parse_args():
    p = argparse.ArgumentParser("Generate assimilation observations (from y_test) and variable mask")
    p.add_argument("experiment_dir", type=str, help="Path to experiment folder (has config.json and data tensors)")
    p.add_argument("--vars", type=str, default="", help="Comma-separated variable names to assimilate (e.g. t2m,10u,10v). If omitted, all variables are used.")
    p.add_argument("--feature-list", type=str, default="", help="Optional path to a text or JSON file listing feature names in the *dataset order*. If omitted, tries <experiment_dir>/features.json.")
    p.add_argument("--indices", type=str, default="", help="Alternative: comma-separated indices (0..C-1) to assimilate if names are unknown.")
    p.add_argument("--out-name", type=str, default="assimilation", help="Subfolder name to create within experiment_dir")
    args = p.parse_args()
    return args

def _load_feature_list(path: str):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() in (".json",".jsn"):
            return json.loads(p.read_text())
        else:
            # treat as plain text: one name per line or comma-separated
            txt = p.read_text().strip()
            if not txt:
                return None
            if "\n" in txt and ("," not in txt):
                return [t.strip() for t in txt.splitlines() if t.strip()]
            else:
                return [t.strip() for t in txt.split(",") if t.strip()]
    except Exception:
        return None

def main():
    a = parse_args()
    exp = Path(a.experiment_dir)
    data_dir = exp

    # Load y_test
    y_path = exp / FileNames.TEST_Y
    if not y_path.exists():
        raise SystemExit(f"Not found: {y_path}")

    y_test = torch.load(y_path)  # expected [N, LON, LAT, P, C] OR flattened [N, G, P*C]
    # Try to coerce to [N, G, P*C]
    if y_test.dim() == 5:
        N,LON,LAT,P,C = y_test.shape
        y_flat = y_test.reshape(N, LON*LAT, P*C).contiguous()
        C_total = C
        P_used = P
    elif y_test.dim() == 3:
        N,G,PC = y_test.shape
        # We'll need P,C from config; try to infer by dividing by obs channels
        # Here we cannot be 100% sure; fallback: assume 'pred_window_used' and 'num_features_used' exist in config.json
        cfg = json.loads((exp/"config.json").read_text())
        P_used = int(cfg.get("data",{}).get("pred_window_used", cfg.get("data",{}).get("pred_window", 1)))
        C_total = PC // P_used
        y_flat = y_test
    else:
        raise SystemExit(f"Unexpected y_test shape: {tuple(y_test.shape)}")

    # Build mask
    feature_names = None
    if a.feature_list:
        feature_names = _load_feature_list(a.feature_list)
    else:
        fjson = exp/"features.json"
        feature_names = _load_feature_list(str(fjson))

    var_names = [v.strip() for v in a.vars.split(",") if v.strip()]
    idxs = [int(s) for s in a.indices.split(",") if s.strip().isdigit()]

    if var_names and not feature_names:
        print("[WARN] variable names provided but feature list not found; cannot map names -> indices. Mask will be empty. Consider --feature-list or --indices.")
        indices = []
    elif var_names and feature_names:
        name_to_idx = {n:i for i,n in enumerate(feature_names)}
        indices = []
        for vn in var_names:
            if vn not in name_to_idx:
                print(f"[WARN] variable {vn} not in feature list; skipping")
                continue
            indices.append(name_to_idx[vn])
    else:
        indices = idxs

    C = C_total
    if not indices:
        indices = list(range(C))  # "all"

    mask = torch.zeros(C, dtype=torch.bool)
    for i in indices:
        if 0 <= i < C:
            mask[i] = True
    mask_flat = mask.repeat(P_used)  # length = P*C

    # Save
    outdir = exp / a.out_name
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(y_flat, outdir/"y_obs.pt")
    torch.save(mask_flat, outdir/"mask_flat.pt")

    meta = {
        "feature_list_used": feature_names,
        "vars_requested": var_names,
        "indices_used": indices,
        "pred_window": P_used,
        "num_features": C_total,
        "y_shape": list(y_flat.shape),
        "notes": "y_obs.pt has shape [N, G, P*C]; mask_flat.pt length = P*C (booleans).",
    }
    (outdir/"meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved: {outdir/'y_obs.pt'}; {outdir/'mask_flat.pt'}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

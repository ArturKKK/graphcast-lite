#!/usr/bin/env python3
"""
Extract y_test.pt from a chunked dataset (data.npy) for use with create_obs.py.

Usage:
  python scripts/extract_y_test.py \
    --data-dir data/datasets/multires_krsk_19f \
    --obs-window 2 --pred-steps 4 --n-features 19 \
    --out data/datasets/multires_krsk_19f/y_test.pt
"""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataloader_chunked import load_chunked_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--obs-window", type=int, default=2)
    ap.add_argument("--pred-steps", type=int, default=4)
    ap.add_argument("--n-features", type=int, default=19)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["test", "test_only"])
    args = ap.parse_args()

    _, _, test_ds, meta = load_chunked_datasets(
        data_path=args.data_dir,
        obs_window=args.obs_window,
        pred_steps=args.pred_steps,
        n_features=args.n_features,
        test_split=args.split,
    )

    N = len(test_ds)
    print(f"Extracting Y from {N} test samples...")

    ys = []
    for i in range(N):
        _, y = test_ds[i]
        ys.append(y)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N}")

    y_test = torch.stack(ys)  # (N, G, P*C)
    print(f"y_test shape: {y_test.shape}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(y_test, str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

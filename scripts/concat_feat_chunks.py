#!/usr/bin/env python3
"""
Склейка двух data.npy по feature-axis: base (19f) + extra (4f time) -> 23f.
Не грузит в RAM — memmap чанками.

Использование:
  python scripts/concat_feat_chunks.py \
      --base  /data/datasets/wb2_512x256_19f_ar \
      --extra /data/datasets/wb2_512x256_4f_time \
      --out   /data/datasets/wb2_512x256_23f_v3

Обязано совпадать: n_time, n_lon, n_lat. Складываются n_feat.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    base = Path(args.base); extra = Path(args.extra); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ib = json.loads((base / "dataset_info.json").read_text())
    ie = json.loads((extra / "dataset_info.json").read_text())
    for k in ("n_time", "n_lon", "n_lat"):
        if ib[k] != ie[k]:
            raise SystemExit(f"{k} mismatch: base={ib[k]} extra={ie[k]}")
    T, X, Y = ib["n_time"], ib["n_lon"], ib["n_lat"]
    Fb, Fe = ib["n_feat"], ie["n_feat"]
    Fout = Fb + Fe
    print(f"Concat feat-axis: ({T},{X},{Y}, {Fb}+{Fe}={Fout})")

    bmm = np.memmap(base / "data.npy", dtype=np.float16, mode="r", shape=(T, X, Y, Fb))
    emm = np.memmap(extra / "data.npy", dtype=np.float16, mode="r", shape=(T, X, Y, Fe))
    dst = np.memmap(out / "data.npy", dtype=np.float16, mode="w+", shape=(T, X, Y, Fout))

    for t0 in range(0, T, args.chunk):
        t1 = min(t0 + args.chunk, T)
        dst[t0:t1, :, :, :Fb] = bmm[t0:t1]
        dst[t0:t1, :, :, Fb:] = emm[t0:t1]
        if (t0 // args.chunk) % 10 == 0:
            print(f"  {t1}/{T}")
    dst.flush(); del dst, bmm, emm
    print("data.npy written")

    shutil.copy2(base / "coords.npz", out / "coords.npz")

    # объединить variables.json
    vb = json.loads((base / "variables.json").read_text())
    ve = json.loads((extra / "variables.json").read_text())
    new_vars = vb + ve
    (out / "variables.json").write_text(json.dumps(new_vars, indent=2, ensure_ascii=False))

    # scalers: ре-используем mean/std в виде concat
    sb = np.load(base / "scalers.npz")
    se = np.load(extra / "scalers.npz")
    mean = np.concatenate([sb["mean"], se["mean"]])
    std = np.concatenate([sb["std"], se["std"]])
    n = int(sb.get("n", np.array(T * X * Y)))
    np.savez(out / "scalers.npz", mean=mean.astype(np.float32), std=std.astype(np.float32), n=np.array(n))

    new_info = dict(ib)
    new_info["n_feat"] = Fout
    new_info["variables"] = new_vars
    new_info["size_gb"] = round(T * X * Y * Fout * 2 / 1e9, 3)
    new_info["source"] = f"concat_feat: {base} + {extra}"
    (out / "dataset_info.json").write_text(json.dumps(new_info, indent=2, ensure_ascii=False))
    print(f"Done. {out}")


if __name__ == "__main__":
    main()

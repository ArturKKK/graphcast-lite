#!/usr/bin/env python3
"""
Склейка двух (или более) data.npy чанков по time-axis в один.
Не грузит всё в RAM — пишет через memmap чанками.

Использование:
  python scripts/concat_time_chunks.py \
      --parts /data/datasets/part1 /data/datasets/part2 \
      --out   /data/datasets/wb2_512x256_19f_full

Каждый --parts должен иметь data.npy, dataset_info.json, coords.npz, scalers.npz, variables.json.
Все части обязаны иметь одинаковые (n_lon, n_lat, n_feat). time_start берётся у первой,
time_end — у последней.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="dirs with data.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=500, help="timesteps per copy chunk")
    args = ap.parse_args()

    parts = [Path(p) for p in args.parts]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    infos = [json.loads((p / "dataset_info.json").read_text()) for p in parts]

    # sanity
    shape_xy = (infos[0]["n_lon"], infos[0]["n_lat"], infos[0]["n_feat"])
    for inf, p in zip(infos, parts):
        s = (inf["n_lon"], inf["n_lat"], inf["n_feat"])
        if s != shape_xy:
            raise SystemExit(f"shape mismatch: {p} has {s}, expected {shape_xy}")

    total_T = sum(inf["n_time"] for inf in infos)
    n_lon, n_lat, n_feat = shape_xy
    print(f"Concatenating {len(parts)} parts -> total T={total_T}, shape=({total_T},{n_lon},{n_lat},{n_feat})")

    dst = np.memmap(out / "data.npy", dtype=np.float16, mode="w+",
                    shape=(total_T, n_lon, n_lat, n_feat))
    offset = 0
    for p, inf in zip(parts, infos):
        T = inf["n_time"]
        src = np.memmap(p / "data.npy", dtype=np.float16, mode="r",
                        shape=(T, n_lon, n_lat, n_feat))
        for t0 in range(0, T, args.chunk):
            t1 = min(t0 + args.chunk, T)
            dst[offset + t0:offset + t1] = src[t0:t1]
            if (t0 // args.chunk) % 10 == 0:
                print(f"  part {p.name}: {t1}/{T}")
        del src
        offset += T
    dst.flush()
    del dst
    print("data.npy written")

    # coords / variables — берём с первой части (они должны совпадать)
    shutil.copy2(parts[0] / "coords.npz", out / "coords.npz")
    shutil.copy2(parts[0] / "variables.json", out / "variables.json")

    # Пересчёт scalers по всему объединённому массиву (по чанкам)
    print("Recomputing scalers...")
    mm = np.memmap(out / "data.npy", dtype=np.float16, mode="r",
                   shape=(total_T, n_lon, n_lat, n_feat))
    # Welford по feature
    n = 0
    mean = np.zeros(n_feat, dtype=np.float64)
    M2 = np.zeros(n_feat, dtype=np.float64)
    for t0 in range(0, total_T, args.chunk):
        t1 = min(t0 + args.chunk, total_T)
        block = mm[t0:t1].astype(np.float32).reshape(-1, n_feat)
        for row in block:
            n += 1
            delta = row - mean
            mean += delta / n
            M2 += delta * (row - mean)
    var = M2 / max(n - 1, 1)
    std = np.sqrt(var).astype(np.float32)
    std[std < 1e-8] = 1.0
    np.savez(out / "scalers.npz",
             mean=mean.astype(np.float32),
             std=std,
             n=np.array(n))
    print(f"  mean={mean[:5]}... std={std[:5]}...")

    # dataset_info.json
    new_info = dict(infos[0])
    new_info["n_time"] = total_T
    new_info["time_start"] = infos[0]["time_start"]
    new_info["time_end"] = infos[-1].get("time_end", new_info.get("time_end"))
    new_info["size_gb"] = round(total_T * n_lon * n_lat * n_feat * 2 / 1e9, 3)
    new_info["source"] = "concat_time: " + " + ".join(str(p) for p in parts)
    with open(out / "dataset_info.json", "w") as f:
        json.dump(new_info, f, indent=2, ensure_ascii=False)
    print(f"Done. {out}")


if __name__ == "__main__":
    main()

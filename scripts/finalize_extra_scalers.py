"""Finalize scalers/metadata for an already-downloaded data_extra.npy.

Use after extend_dataset_512x256_to_30f.py finished writing data_extra.npy
but crashed/ended before writing scalers (e.g. when base scalers.npz is
absent on this machine — base is on a different host).

Streams over data_extra.npy along time axis, computes per-channel mean/std
in float64, saves:
  - scalers_extra.npz       (mean, std, n)  shape (n_extra,)
  - variables_extra.json    (10 names)
  - dataset_info_extra.json (selfcontained metadata for the extra block)

Does NOT touch data.npy / scalers.npz / variables.json / dataset_info.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

NEW_VAR_ORDER = [
    "z@250", "t@250", "u@250", "v@250", "q@250",
    "z@1000", "t@1000", "u@1000", "v@1000", "q@1000",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, required=True,
                   help="Directory containing data_extra.npy + dataset_info.json")
    p.add_argument("--time-chunk", type=int, default=500,
                   help="Time-steps per streaming block")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir: Path = args.base_dir
    info = json.loads((base_dir / "dataset_info.json").read_text())
    n_time = int(info["n_time"])
    n_lon = int(info["n_lon"])
    n_lat = int(info["n_lat"])
    n_extra = len(NEW_VAR_ORDER)

    extra_path = base_dir / "data_extra.npy"
    if not extra_path.exists():
        raise SystemExit(f"data_extra.npy not found in {base_dir}")

    expected = n_time * n_lon * n_lat * n_extra * 2  # float16
    actual = extra_path.stat().st_size
    if actual != expected:
        raise SystemExit(
            f"Size mismatch: data_extra.npy is {actual} bytes, "
            f"expected {expected} bytes (T={n_time}, {n_lon}x{n_lat}, "
            f"feat={n_extra}, fp16)."
        )

    print(f"[INFO] Streaming {extra_path} for stats")
    print(f"       shape=({n_time},{n_lon},{n_lat},{n_extra})  "
          f"size={actual/1024**3:.1f} GB")

    arr = np.memmap(
        extra_path, dtype=np.float16, mode="r",
        shape=(n_time, n_lon, n_lat, n_extra),
    )

    sum_ = np.zeros(n_extra, dtype=np.float64)
    sumsq = np.zeros(n_extra, dtype=np.float64)
    n_total = 0

    t0 = time.time()
    for t_start in range(0, n_time, args.time_chunk):
        t_end = min(t_start + args.time_chunk, n_time)
        block = np.asarray(arr[t_start:t_end], dtype=np.float64)
        # block shape: (chunk, lon, lat, feat) → reduce over (chunk,lon,lat)
        sum_ += block.sum(axis=(0, 1, 2))
        sumsq += (block * block).sum(axis=(0, 1, 2))
        n_total += (t_end - t_start) * n_lon * n_lat
        elapsed = time.time() - t0
        pct = 100.0 * t_end / n_time
        print(f"  [{t_start:>6d}-{t_end:>6d}/{n_time}] {pct:5.1f}%  {elapsed/60:.1f} min")

    mean = (sum_ / n_total).astype(np.float32)
    var = sumsq / n_total - (sum_ / n_total) ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var).astype(np.float32)
    std = np.maximum(std, 1e-6)

    np.savez(
        base_dir / "scalers_extra.npz",
        mean=mean, std=std, n=np.int64(n_total),
    )
    (base_dir / "variables_extra.json").write_text(
        json.dumps(NEW_VAR_ORDER, indent=2, ensure_ascii=False)
    )
    extra_info = {
        "time_start": info["time_start"],
        "time_end": info["time_end"],
        "n_time": n_time,
        "n_lon": n_lon,
        "n_lat": n_lat,
        "n_feat_extra": n_extra,
        "variables_extra": NEW_VAR_ORDER,
        "dtype": "float16",
        "file": "data_extra.npy",
        "size_gb": round(actual / 1024**3, 2),
        "note": "Per-channel mean/std for data_extra.npy only. Merge with "
                "base scalers (19f) on the training host before training.",
    }
    (base_dir / "dataset_info_extra.json").write_text(
        json.dumps(extra_info, indent=2)
    )

    print()
    print("=" * 70)
    print("✓ Finalized extra-only metadata")
    for i, name in enumerate(NEW_VAR_ORDER):
        print(f"  {name:>8s}:  mean={mean[i]:+10.3f}  std={std[i]:10.3f}")
    print("=" * 70)
    print(f"  scalers_extra.npz       → {base_dir/'scalers_extra.npz'}")
    print(f"  variables_extra.json    → {base_dir/'variables_extra.json'}")
    print(f"  dataset_info_extra.json → {base_dir/'dataset_info_extra.json'}")


if __name__ == "__main__":
    main()

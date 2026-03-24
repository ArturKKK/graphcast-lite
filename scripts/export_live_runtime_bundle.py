#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a lightweight runtime bundle for live GDAS inference")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_dataset_info(data_dir: Path) -> dict:
    info_path = data_dir / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset info: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def export_bundle(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    required_files = ["coords.npz", "scalers.npz", "variables.json", "dataset_info.json", "data.npy"]
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required dataset file: {file_path}")

    info = load_dataset_info(data_dir)
    coords = np.load(data_dir / "coords.npz")
    scalers = np.load(data_dir / "scalers.npz")
    with (data_dir / "variables.json").open("r", encoding="utf-8") as handle:
        var_order = json.load(handle)

    np.savez_compressed(
        out_dir / "coords.npz",
        latitude=coords["latitude"].astype(np.float32),
        longitude=coords["longitude"].astype(np.float32),
        **({"is_regional": coords["is_regional"]} if "is_regional" in coords else {}),
    )

    if "mean" in scalers:
        np.savez_compressed(
            out_dir / "scalers.npz",
            mean=scalers["mean"].astype(np.float32),
            std=scalers["std"].astype(np.float32),
        )
    else:
        np.savez_compressed(
            out_dir / "scalers.npz",
            x_mean=scalers["x_mean"].astype(np.float32),
            x_scale=scalers["x_scale"].astype(np.float32),
            y_mean=scalers["y_mean"].astype(np.float32),
            y_scale=scalers["y_scale"].astype(np.float32),
        )

    (out_dir / "variables.json").write_text(json.dumps(var_order, ensure_ascii=False, indent=2), encoding="utf-8")

    dtype = np.float16
    if info.get("flat", False):
        shape = (info["n_time"], info["n_nodes"], info["n_feat"])
    else:
        shape = (info["n_time"], info["n_lon"], info["n_lat"], info["n_feat"])
    mmap_arr = np.memmap(str(data_dir / "data.npy"), dtype=dtype, mode="r", shape=shape)
    info_vars = info.get("variables", var_order)

    static_payload: dict[str, np.ndarray] = {}
    for name in ["z_surf", "lsm"]:
        if name not in info_vars:
            continue
        idx = info_vars.index(name)
        if info.get("flat", False):
            values = mmap_arr[0, :, idx].astype(np.float32)
        else:
            values = mmap_arr[0, :, :, idx].astype(np.float32).reshape(-1)
        static_payload[name] = values
    np.savez_compressed(out_dir / "static_fields.npz", **static_payload)

    bundle_meta = {
        "flat_grid": bool(info.get("flat", False)),
        "n_nodes": int(info.get("n_nodes", len(coords["latitude"]))),
        "n_feat": int(info["n_feat"]),
        "variables": var_order,
        "static_fields": sorted(static_payload.keys()),
    }
    (out_dir / "bundle_meta.json").write_text(json.dumps(bundle_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    size_bytes = sum(path.stat().st_size for path in out_dir.iterdir() if path.is_file())
    print(f"[bundle] Exported to {out_dir}")
    print(f"[bundle] Size: {size_bytes / (1024 ** 2):.2f} MiB")
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            print(f"  - {path.name}: {path.stat().st_size / (1024 ** 2):.2f} MiB")


def main() -> None:
    args = parse_args()
    export_bundle(Path(args.data_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
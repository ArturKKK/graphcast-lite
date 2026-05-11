#!/usr/bin/env python3
"""
scripts/build_v3_extra_with_time.py

Готовит датасет wb2_512x256_33f_v3 для v3 GLOBAL обучения:

ВХОД (на VM):
  /data/datasets/wb2_512x256_19f_ar/data.npy                                (T,512,256,19)  float16
  /data/datasets/wb2_512x256_19f_ar/scalers.npz                             mean,std (19,)
  /data/datasets/wb2_512x256_19f_ar/coords.npz                              lon,lat
  /data/datasets/global_512x256_extra_2010-2021_07deg/data_extra.npy        (T,512,256,10)  float16
  /data/datasets/global_512x256_extra_2010-2021_07deg/scalers_extra.npz     mean,std (10,)

ВЫХОД:
  /data/datasets/wb2_512x256_33f_v3/
    data.npy           — SYMLINK -> base 19f data.npy
    data_extra.npy     — НОВЫЙ файл (T,512,256,14) = [10 plev] + [4 time-forcing]
    scalers.npz        — (33,) mean/std для всех каналов
    variables.json     — 33 имени
    coords.npz         — копия из base
    dataset_info.json  — n_feat=33, n_feat_base=19, n_feat_extra=14, extra_file=data_extra.npy

Каналы (idx → имя):
   0..18   t2m, 10u, 10v, msl, tp, sp, tcwv, z_surf, lsm,
           t/u/v/z/q@850, t/u/v/z/q@500           (base 19f)
   19..23  z/t/u/v/q @250                          (new plev)
   24..28  z/t/u/v/q @1000                         (new plev)
   29..32  sin_hour, cos_hour, sin_doy, cos_doy   (time-forcing)

Время-forcing рассчитывается из time_start (2010-01-01, 6h step):
   sin/cos(2π * hour/24), sin/cos(2π * doy/365.25)
   → mean ≈ 0, std ≈ √0.5 ≈ 0.7071 (analytical for sin/cos uniform).

Запуск (на VM):
  python scripts/build_v3_extra_with_time.py \\
      --base-dir  /data/datasets/wb2_512x256_19f_ar \\
      --extra-dir /data/datasets/global_512x256_extra_2010-2021_07deg \\
      --out-dir   /data/datasets/wb2_512x256_33f_v3
"""

import argparse
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


BASE_VARS = [
    "t2m", "10u", "10v", "msl", "tp",
    "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
]
EXTRA_PLEV = [
    "z@250", "t@250", "u@250", "v@250", "q@250",
    "z@1000", "t@1000", "u@1000", "v@1000", "q@1000",
]
TIME_VARS = ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]
ALL_VARS = BASE_VARS + EXTRA_PLEV + TIME_VARS  # 19 + 10 + 4 = 33


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--extra-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--time-chunk", type=int, default=500)
    args = ap.parse_args()

    base = Path(args.base_dir)
    extra = Path(args.extra_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Read base info & scalers ---
    base_info = json.loads((base / "dataset_info.json").read_text())
    T, n_lon, n_lat = base_info["n_time"], base_info["n_lon"], base_info["n_lat"]
    assert base_info["n_feat"] == 19, f"base must be 19f, got {base_info['n_feat']}"
    base_sc = np.load(base / "scalers.npz")
    base_mean = base_sc["mean"].astype(np.float32)
    base_std = base_sc["std"].astype(np.float32)
    assert base_mean.shape == (19,)

    # --- 2. Read extra info & scalers ---
    ext_info_files = [extra / "dataset_info_extra.json", extra / "dataset_info.json"]
    ext_info = None
    for p in ext_info_files:
        if p.exists():
            ext_info = json.loads(p.read_text())
            if ext_info.get("n_feat_extra") or ext_info.get("variables_extra"):
                break
    assert ext_info is not None, f"no extra info json in {extra}"
    n_feat_extra_plev = ext_info.get("n_feat_extra", 10)
    assert n_feat_extra_plev == 10, f"expected 10 plev channels, got {n_feat_extra_plev}"
    assert ext_info["n_time"] == T, f"T mismatch base={T} extra={ext_info['n_time']}"

    ext_sc = np.load(extra / "scalers_extra.npz")
    ext_mean = ext_sc["mean"].astype(np.float32)
    ext_std = ext_sc["std"].astype(np.float32)
    assert ext_mean.shape == (10,)

    time_start = base_info["time_start"]  # "2010-01-01"
    print(f"[INFO] T={T}, lon={n_lon}, lat={n_lat}, time_start={time_start}")

    # --- 3. Compute time features (T, 4) ---
    t0 = datetime.strptime(time_start, "%Y-%m-%d")
    times = [t0 + timedelta(hours=6 * i) for i in range(T)]
    hours = np.array([t.hour + t.minute / 60 for t in times], dtype=np.float32)
    doys = np.array([t.timetuple().tm_yday for t in times], dtype=np.float32)
    sin_h = np.sin(2 * np.pi * hours / 24.0).astype(np.float16)
    cos_h = np.cos(2 * np.pi * hours / 24.0).astype(np.float16)
    sin_d = np.sin(2 * np.pi * doys / 365.25).astype(np.float16)
    cos_d = np.cos(2 * np.pi * doys / 365.25).astype(np.float16)
    time_feats_T4 = np.stack([sin_h, cos_h, sin_d, cos_d], axis=-1)  # (T, 4) float16
    print(f"[INFO] time_feats sin_h range: [{sin_h.min():.3f}, {sin_h.max():.3f}]")

    # Scalers for time features: empirical to keep loader honest.
    tf32 = time_feats_T4.astype(np.float32)
    time_mean = tf32.mean(axis=0)
    time_std = tf32.std(axis=0)
    time_std[time_std < 1e-6] = 1.0  # safety
    print(f"[INFO] time_mean={time_mean}, time_std={time_std}")

    # --- 4. Open inputs (memmap) ---
    plev_mm = np.memmap(extra / "data_extra.npy", dtype=np.float16, mode="r",
                        shape=(T, n_lon, n_lat, 10))

    # --- 5. Create output data_extra.npy (T, 512, 256, 14) ---
    out_extra_path = out / "data_extra.npy"
    expected_bytes = T * n_lon * n_lat * 14 * 2
    print(f"[WRITE] {out_extra_path} ≈ {expected_bytes / 1e9:.1f} GB")
    out_mm = np.memmap(out_extra_path, dtype=np.float16, mode="w+",
                       shape=(T, n_lon, n_lat, 14))

    # Fill plev channels (0..9) by chunk-copy from extra/data_extra.npy
    for t in range(0, T, args.time_chunk):
        t_e = min(t + args.time_chunk, T)
        out_mm[t:t_e, :, :, 0:10] = plev_mm[t:t_e, :, :, :]
        # Broadcast time features (4,) to (chunk, n_lon, n_lat, 4)
        out_mm[t:t_e, :, :, 10:14] = time_feats_T4[t:t_e, np.newaxis, np.newaxis, :]
        if t % (args.time_chunk * 10) == 0:
            print(f"  t={t}/{T}")
            out_mm.flush()
    out_mm.flush()
    del out_mm, plev_mm
    print(f"[DONE] {out_extra_path}: {expected_bytes / 1e9:.1f} GB")

    # --- 6. Symlink base data.npy ---
    out_data = out / "data.npy"
    if out_data.exists() or out_data.is_symlink():
        out_data.unlink()
    os.symlink(str((base / "data.npy").resolve()), str(out_data))
    print(f"[SYMLINK] {out_data} -> {base / 'data.npy'}")

    # --- 7. Write merged scalers.npz (33 channels) ---
    mean_all = np.concatenate([base_mean, ext_mean, time_mean]).astype(np.float32)
    std_all = np.concatenate([base_std, ext_std, time_std]).astype(np.float32)
    assert mean_all.shape == (33,) and std_all.shape == (33,)
    np.savez(out / "scalers.npz", mean=mean_all, std=std_all)
    print(f"[WRITE] scalers.npz: mean[0:5]={mean_all[:5]}, std[0:5]={std_all[:5]}")

    # --- 8. Copy coords.npz ---
    coords_src = base / "coords.npz"
    if coords_src.exists():
        shutil.copy2(coords_src, out / "coords.npz")
        print(f"[COPY] coords.npz")

    # --- 9. Write variables.json ---
    (out / "variables.json").write_text(json.dumps(ALL_VARS, indent=2))

    # --- 10. Write dataset_info.json ---
    out_info = {
        "time_start": time_start,
        "time_end": base_info.get("time_end", "2021-12-31"),
        "n_time": T,
        "n_lon": n_lon,
        "n_lat": n_lat,
        "n_feat": 33,
        "n_feat_base": 19,
        "n_feat_extra": 14,
        "extra_file": "data_extra.npy",
        "variables": ALL_VARS,
        "dtype": "float16",
        "file": "data.npy",
        "note": "data.npy is symlink to wb2_512x256_19f_ar/data.npy (19ch). data_extra.npy = 10 plev (250+1000hPa) + 4 time-forcing.",
    }
    (out / "dataset_info.json").write_text(json.dumps(out_info, indent=2))
    print(f"[WRITE] dataset_info.json")
    print("\nDone. Now run:")
    print(f"  python -m src.main experiments/wb2_512x256_33f_ar_v3")


if __name__ == "__main__":
    main()

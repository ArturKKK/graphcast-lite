#!/usr/bin/env python3
"""
scripts/build_multires_russia_33f.py

Расширяет существующий multires_russia_19f (flat-grid) до 33 каналов (как v3 GLOBAL):
  19 base + 10 plev@250/1000 + 4 time-forcing (sin/cos hour/doy).

Ключевая идея: data.npy остаётся как есть (символическая ссылка), а
data_extra.npy (T, N_nodes, 14) строится:
  - 10 plev: для каждого flat-узла билинейно интерполируем из глобального
    grid 512×256 экстры (/data/datasets/global_512x256_extra_2010-2021_07deg/data_extra.npy).
  - 4 time-forcing: одинаковые по всем узлам, рассчитываются из time_start.

ВХОД (на VM v4):
  /data/datasets/multires_russia_19f/
      data.npy           — (T, N_nodes, 19) float16
      coords.npz         — latitude (N,), longitude (N,)
      dataset_info.json
      scalers.npz        — (19,)
      variables.json
  /data/datasets/global_512x256_extra_2010-2021_07deg/
      data_extra.npy     — (T_glob, 512, 256, 10) float16
      coords.npz         — longitude (512,), latitude (256,)
      scalers_extra.npz  — (10,) mean/std
      dataset_info_extra.json

ВЫХОД:
  /data/datasets/multires_russia_33f/
      data.npy           — SYMLINK на multires_russia_19f/data.npy
      data_extra.npy     — НОВЫЙ (T, N_nodes, 14) float16
      scalers.npz        — (33,) объединённые
      variables.json     — 33 имени
      coords.npz         — копия
      dataset_info.json  — n_feat=33, n_feat_base=19, n_feat_extra=14, extra_file=data_extra.npy

Запуск:
  python scripts/build_multires_russia_33f.py \\
      --multires-dir /data/datasets/multires_russia_19f \\
      --extra-dir    /data/datasets/global_512x256_extra_2010-2021_07deg \\
      --out-dir      /data/datasets/multires_russia_33f
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
ALL_VARS = BASE_VARS + EXTRA_PLEV + TIME_VARS  # 33


def bilinear_sample(grid_data_lonlat, g_lons, g_lats, q_lon, q_lat):
    """
    Bilinear sample from regular grid (n_lon, n_lat) at irregular query points.

    grid_data_lonlat : (n_lon, n_lat) — single channel slice
    g_lons           : (n_lon,) монотонно возрастающие (0..360 или -180..180)
    g_lats           : (n_lat,) монотонно возрастающие
    q_lon, q_lat     : (N,)
    return           : (N,) float32
    """
    n_lon = len(g_lons)
    n_lat = len(g_lats)
    # Привести query lon к диапазону g_lons
    lon_min = g_lons[0]
    lon_max = g_lons[-1]
    q_lon_n = np.mod(q_lon - lon_min, lon_max - lon_min + (g_lons[1] - g_lons[0])) + lon_min

    # индексы по lon (циклично)
    di = (q_lon_n - lon_min) / (g_lons[1] - g_lons[0])
    i0 = np.floor(di).astype(np.int64) % n_lon
    i1 = (i0 + 1) % n_lon
    fx = di - np.floor(di)

    # индексы по lat (clamped)
    dj = (q_lat - g_lats[0]) / (g_lats[1] - g_lats[0])
    j0 = np.clip(np.floor(dj).astype(np.int64), 0, n_lat - 2)
    j1 = j0 + 1
    fy = np.clip(dj - j0, 0.0, 1.0)

    g = grid_data_lonlat.astype(np.float32)
    v00 = g[i0, j0]
    v10 = g[i1, j0]
    v01 = g[i0, j1]
    v11 = g[i1, j1]
    v = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11
    return v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multires-dir", required=True)
    ap.add_argument("--extra-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--time-chunk", type=int, default=200)
    args = ap.parse_args()

    multi = Path(args.multires_dir)
    extra = Path(args.extra_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === 1. читаем multires info ===
    m_info = json.loads((multi / "dataset_info.json").read_text())
    T = m_info["n_time"]
    N = m_info["n_nodes"]
    assert m_info["n_feat"] == 19, f"expected 19f base, got {m_info['n_feat']}"
    assert m_info.get("flat", False), "multires dataset must be flat"
    time_start = m_info["time_start"]
    print(f"[INFO] multires: T={T}, N_nodes={N}, time_start={time_start}")

    # base scalers (19,)
    m_sc = np.load(multi / "scalers.npz")
    m_mean = m_sc["mean"].astype(np.float32)
    m_std = m_sc["std"].astype(np.float32)
    assert m_mean.shape == (19,)

    # coords per node
    m_coords = np.load(multi / "coords.npz")
    node_lats = m_coords["latitude"].astype(np.float64)
    node_lons = m_coords["longitude"].astype(np.float64)
    assert node_lats.shape == (N,) and node_lons.shape == (N,)

    # === 2. читаем global extra ===
    # try both dataset_info_extra.json и dataset_info.json
    ext_info = None
    for cand in [extra / "dataset_info_extra.json", extra / "dataset_info.json"]:
        if cand.exists():
            ext_info = json.loads(cand.read_text())
            if ext_info.get("n_feat_extra") or ext_info.get("variables_extra"):
                break
    assert ext_info is not None, f"no extra info json in {extra}"
    T_ext = ext_info["n_time"]
    n_lon_ext = ext_info["n_lon"]
    n_lat_ext = ext_info["n_lat"]
    n_feat_ext = ext_info.get("n_feat_extra", 10)
    assert n_feat_ext == 10, f"expected 10 plev, got {n_feat_ext}"
    ext_time_start = ext_info["time_start"]
    print(f"[INFO] extra: T={T_ext}, lon={n_lon_ext}, lat={n_lat_ext}, time_start={ext_time_start}")

    # Время должно совпадать (multires Russia собран из тех же 2010-2021 что и extra)
    if time_start != ext_time_start:
        print(f"[WARN] time_start mismatch multires={time_start} extra={ext_time_start}")
    assert T == T_ext, f"T mismatch multires={T} extra={T_ext} — времена должны совпадать"

    ext_coords = np.load(extra / "coords.npz")
    g_lats = ext_coords["latitude"].astype(np.float64)
    g_lons = ext_coords["longitude"].astype(np.float64)
    # ensure monotonic increasing
    if g_lats[0] > g_lats[-1]:
        g_lats = g_lats[::-1]
        lat_flipped = True
    else:
        lat_flipped = False

    extra_mm = np.memmap(extra / "data_extra.npy", dtype=np.float16, mode="r",
                         shape=(T_ext, n_lon_ext, n_lat_ext, 10))

    # extra scalers (10,) — нормированные, поэтому интерполяция в нормированном пространстве
    ext_sc = np.load(extra / "scalers_extra.npz")
    ext_mean = ext_sc["mean"].astype(np.float32)
    ext_std = ext_sc["std"].astype(np.float32)
    assert ext_mean.shape == (10,)

    # === 3. time forcing per timestep ===
    t0 = datetime.strptime(time_start, "%Y-%m-%d")
    times = [t0 + timedelta(hours=6 * i) for i in range(T)]
    hours = np.array([t.hour + t.minute / 60 for t in times], dtype=np.float32)
    doys = np.array([t.timetuple().tm_yday for t in times], dtype=np.float32)
    sin_h = np.sin(2 * np.pi * hours / 24.0).astype(np.float16)
    cos_h = np.cos(2 * np.pi * hours / 24.0).astype(np.float16)
    sin_d = np.sin(2 * np.pi * doys / 365.25).astype(np.float16)
    cos_d = np.cos(2 * np.pi * doys / 365.25).astype(np.float16)
    time_feats_T4 = np.stack([sin_h, cos_h, sin_d, cos_d], axis=-1)  # (T, 4)

    # === 4. allocate output memmap ===
    out_path = out / "data_extra.npy"
    print(f"[INFO] allocating {out_path} shape ({T},{N},14) float16 = "
          f"{T * N * 14 * 2 / 1e9:.1f} GB")
    out_mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(T, N, 14))

    # === 5. fill plev (channels 0..9) via bilinear from global ===
    print(f"[INFO] filling plev channels via bilinear interp, chunks of {args.time_chunk} time steps")
    chunk = args.time_chunk
    for t_start in range(0, T, chunk):
        t_end = min(t_start + chunk, T)
        # читаем глобальный chunk: (chunk, 512, 256, 10) float16
        ext_chunk = extra_mm[t_start:t_end].astype(np.float32)
        if lat_flipped:
            ext_chunk = ext_chunk[:, :, ::-1, :]
        # для каждого канала и каждого t — bilinear по N узлам
        for ch in range(10):
            for ti in range(t_end - t_start):
                vals = bilinear_sample(ext_chunk[ti, :, :, ch], g_lons, g_lats, node_lons, node_lats)
                out_mm[t_start + ti, :, ch] = vals.astype(np.float16)
        print(f"  [plev] t={t_start}..{t_end-1} done")
    out_mm.flush()

    # === 6. fill time forcing (channels 10..13) — broadcast по всем узлам ===
    print("[INFO] filling time-forcing channels (broadcast)")
    for ti in range(T):
        out_mm[ti, :, 10:14] = time_feats_T4[ti][None, :]  # broadcast (1,4)→(N,4)
    out_mm.flush()
    del out_mm

    # === 7. symlink data.npy ===
    src_data = (multi / "data.npy").resolve()
    dst_data = out / "data.npy"
    if dst_data.exists() or dst_data.is_symlink():
        dst_data.unlink()
    os.symlink(src_data, dst_data)
    print(f"[INFO] symlink data.npy → {src_data}")

    # === 8. write scalers (33,) ===
    # time-forcing: sin/cos uniform → mean=0, std=1/√2≈0.7071
    time_mean = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    time_std = np.array([1/np.sqrt(2)] * 4, dtype=np.float32)
    all_mean = np.concatenate([m_mean, ext_mean, time_mean])
    all_std = np.concatenate([m_std, ext_std, time_std])
    assert all_mean.shape == (33,)
    np.savez(out / "scalers.npz", mean=all_mean, std=all_std)
    print(f"[INFO] scalers.npz written (33,)")

    # === 9. variables.json ===
    (out / "variables.json").write_text(json.dumps(ALL_VARS, indent=2))

    # === 10. coords.npz copy ===
    shutil.copy(multi / "coords.npz", out / "coords.npz")
    # если есть metadata о global_mask/region_mask — копируем
    for fname in ["node_metadata.npz", "metadata.npz"]:
        if (multi / fname).exists():
            shutil.copy(multi / fname, out / fname)

    # === 11. dataset_info.json ===
    out_info = dict(m_info)
    out_info["n_feat"] = 33
    out_info["n_feat_base"] = 19
    out_info["n_feat_extra"] = 14
    out_info["extra_file"] = "data_extra.npy"
    out_info["variables"] = ALL_VARS
    (out / "dataset_info.json").write_text(json.dumps(out_info, indent=2))
    print(f"[INFO] dataset_info.json written: n_feat=33 (19+14)")

    print("\n[DONE] multires Russia 33f assembled at:", out)


if __name__ == "__main__":
    main()

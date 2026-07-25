#!/usr/bin/env python3
"""
scripts/build_multires_russia_33f.py

Расширяет существующий multires_russia_19f (flat-grid) до 33 каналов (как v3 GLOBAL):
  19 base + 10 plev@250/1000 + 4 time-forcing (sin/cos hour/doy).

Ключевая идея: data.npy остаётся как есть (символическая ссылка), а
data_extra.npy (T, N_nodes, 14) строится:
  - 10 plev для каждого узла:
      * если --region-extra-dir задан: РЕГИОНАЛЬНЫЕ узлы (is_regional==True
        в coords.npz) bilinear-сэмплятся из РЕГИОНАЛЬНОГО 0.25° extra-датасета
        (honest native resolution, скачан с CDS); ГЛОБАЛЬНЫЕ узлы (is_regional==False)
        bilinear-сэмплятся из global 512x256 (~0.7°) extra-датасета;
      * если --region-extra-dir НЕ задан: legacy режим — все узлы из global extra.
  - 4 time-forcing (sin/cos hour/doy) одинаковые по всем узлам.

ВХОД (на VM v4):
  /data/datasets/multires_russia_19f/
      data.npy           — (T, N_nodes, 19) float16
      coords.npz         — latitude (N,), longitude (N,), is_regional (N,) bool
      dataset_info.json
      scalers.npz        — (19,)
      variables.json
  /data/datasets/global_512x256_extra_2010-2021_07deg/  (global extra)
      data_extra.npy     — (T_glob, 512, 256, 10) float16
      coords.npz         — longitude (512,), latitude (256,)
      scalers_extra.npz  — (10,) mean/std
      dataset_info_extra.json
  /data/datasets/region_russia_645x165_extra_2010-2021_025deg/  (опционально, honest 0.25°)
      data_extra.npy     — (T, 645, 165, 10) float16
      coords.npz         — longitude (645,), latitude (165,)
      scalers_extra.npz  — (10,)
      dataset_info_extra.json

ВЫХОД:
  /data/datasets/multires_russia_33f/
      data.npy           — SYMLINK на multires_russia_19f/data.npy
      data_extra.npy     — НОВЫЙ (T, N_nodes, 14) float16
      scalers.npz        — (33,) объединённые (берём GLOBAL extra scalers для plev,
                          т.к. v3 GLOBAL обучен с ними; channel semantics идентичны)
      variables.json     — 33 имени
      coords.npz         — копия (с is_regional)
      dataset_info.json  — n_feat=33, n_feat_base=19, n_feat_extra=14, extra_file=data_extra.npy

Запуск:
  python scripts/build_multires_russia_33f.py \\
      --multires-dir /data/datasets/multires_russia_19f \\
      --extra-dir    /data/datasets/global_512x256_extra_2010-2021_07deg \\
      --region-extra-dir /data/datasets/region_russia_645x165_extra_2010-2021_025deg \\
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
    ap.add_argument("--extra-dir", required=True,
                    help="Global 512x256 (~0.7°) extra: для глобальных узлов")
    ap.add_argument("--region-extra-dir", default=None,
                    help="Региональный 0.25° extra (honest, native): для регион-узлов. "
                         "Если не задан — все узлы сэмплятся из --extra-dir.")
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
    is_regional = m_coords["is_regional"] if "is_regional" in m_coords.files else None
    if is_regional is not None:
        is_regional = is_regional.astype(bool)
        assert is_regional.shape == (N,)
        n_reg = int(is_regional.sum())
        n_glob = int((~is_regional).sum())
        print(f"[INFO] node split: regional={n_reg}, global={n_glob}")
    else:
        print("[WARN] coords.npz has no 'is_regional' mask — все узлы пойдут как global")

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

    # Время: multires — источник истины. Если extra покрывает более длинный период
    # (напр. глобальный extra 2010-2021 против Krsk-merge 2010-2020), берём первые T
    # шагов — допустимо только при совпадающем time_start.
    if time_start != ext_time_start:
        print(f"[WARN] time_start mismatch multires={time_start} extra={ext_time_start}")
    if T_ext != T:
        assert time_start == ext_time_start, (
            f"T mismatch (multires={T}, extra={T_ext}) при разных time_start "
            f"({time_start} vs {ext_time_start}) — обрезка небезопасна")
        assert T_ext > T, f"extra короче multires: T={T} > T_ext={T_ext}"
        print(f"[INFO] extra длиннее ({T_ext} > {T}) → используем первые {T} шагов "
              f"(time_start совпадает: {time_start})")

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

    # extra scalers (10,) — global источник правды для нормализации (v3 GLOBAL обучен с ними)
    ext_sc = np.load(extra / "scalers_extra.npz")
    ext_mean = ext_sc["mean"].astype(np.float32)
    ext_std = ext_sc["std"].astype(np.float32)
    assert ext_mean.shape == (10,)

    # === 2b. (опц.) региональный 0.25° extra для регион-узлов ===
    use_region_extra = (args.region_extra_dir is not None) and (is_regional is not None)
    region_extra_mm = None
    rg_lats = rg_lons = None
    region_lat_flipped = False
    if use_region_extra:
        rex = Path(args.region_extra_dir)
        rex_info = None
        for cand in [rex / "dataset_info_extra.json", rex / "dataset_info.json"]:
            if cand.exists():
                rex_info = json.loads(cand.read_text())
                break
        assert rex_info is not None, f"no extra info json in {rex}"
        T_rex = rex_info["n_time"]
        n_lon_rex = rex_info["n_lon"]
        n_lat_rex = rex_info["n_lat"]
        n_feat_rex = rex_info.get("n_feat_extra", rex_info.get("n_feat", 10))
        assert n_feat_rex == 10, f"region extra: expected 10 plev, got {n_feat_rex}"
        rex_time_start = rex_info.get("time_start", "")
        print(f"[INFO] region-extra: T={T_rex}, lon={n_lon_rex}, lat={n_lat_rex}, time_start={rex_time_start}")
        if rex_time_start and rex_time_start != time_start:
            print(f"[WARN] region-extra time_start mismatch {rex_time_start} vs multires {time_start}")
        if T_rex != T:
            assert not rex_time_start or rex_time_start == time_start, (
                f"region-extra T mismatch ({T_rex} vs {T}) при разных time_start — обрезка небезопасна")
            assert T_rex > T, f"region-extra короче multires: T={T} > T_rex={T_rex}"
            print(f"[INFO] region-extra длиннее ({T_rex} > {T}) → используем первые {T} шагов")

        rex_coords = np.load(rex / "coords.npz")
        rg_lats = rex_coords["latitude"].astype(np.float64)
        rg_lons = rex_coords["longitude"].astype(np.float64)
        if rg_lats[0] > rg_lats[-1]:
            rg_lats = rg_lats[::-1]
            region_lat_flipped = True
        # sanity: lon ascending
        assert np.all(np.diff(rg_lons) > 0), "region-extra lon must be monotonic ascending"
        assert np.all(np.diff(rg_lats) > 0), "region-extra lat must be monotonic ascending (after flip)"

        region_extra_mm = np.memmap(rex / "data_extra.npy", dtype=np.float16, mode="r",
                                    shape=(T_rex, n_lon_rex, n_lat_rex, 10))

        # Sanity: проверим что все регион-узлы попадают в bbox региональной сетки
        reg_idx = np.where(is_regional)[0]
        rn_lon = node_lons[reg_idx]
        rn_lat = node_lats[reg_idx]
        # допустимо: lon в [rg_lons[0], rg_lons[-1]] (региональное окно, не циклично)
        in_box = (rn_lon >= rg_lons[0]) & (rn_lon <= rg_lons[-1]) & \
                 (rn_lat >= rg_lats[0]) & (rn_lat <= rg_lats[-1])
        out_of_box = (~in_box).sum()
        if out_of_box > 0:
            print(f"[WARN] {out_of_box}/{len(reg_idx)} regional nodes outside region-extra bbox — "
                  f"будут fallback на global extra")
        # save which regional nodes are inside region-extra bbox
        regional_use_region = np.zeros(N, dtype=bool)
        regional_use_region[reg_idx[in_box]] = True
        print(f"[INFO] region-extra coverage: {regional_use_region.sum()}/{N} nodes")
    else:
        regional_use_region = np.zeros(N, dtype=bool)
        print("[INFO] region-extra disabled — все узлы из global extra (legacy)")

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

    # === 5. fill plev (channels 0..9) ===
    # Routing:
    #   regional_use_region[i]==True  →  sample from region_extra_mm (0.25° native)
    #   else                          →  sample from extra_mm (global 0.7°)
    use_reg_mask = regional_use_region          # (N,) bool
    use_glob_mask = ~regional_use_region        # (N,) bool
    reg_nodes_lon = node_lons[use_reg_mask]
    reg_nodes_lat = node_lats[use_reg_mask]
    glob_nodes_lon = node_lons[use_glob_mask]
    glob_nodes_lat = node_lats[use_glob_mask]
    reg_idx_in_out = np.where(use_reg_mask)[0]
    glob_idx_in_out = np.where(use_glob_mask)[0]
    print(f"[INFO] plev routing: global-extra→{glob_idx_in_out.size} nodes, "
          f"region-extra→{reg_idx_in_out.size} nodes")

    print(f"[INFO] filling plev channels via bilinear, chunks of {args.time_chunk} time steps")
    chunk = args.time_chunk
    for t_start in range(0, T, chunk):
        t_end = min(t_start + chunk, T)
        # ----- global extra chunk -----
        ext_chunk = extra_mm[t_start:t_end].astype(np.float32)
        if lat_flipped:
            ext_chunk = ext_chunk[:, :, ::-1, :]
        # ----- region extra chunk (если нужен) -----
        if region_extra_mm is not None and reg_idx_in_out.size > 0:
            rex_chunk = region_extra_mm[t_start:t_end].astype(np.float32)
            if region_lat_flipped:
                rex_chunk = rex_chunk[:, :, ::-1, :]
        else:
            rex_chunk = None

        for ch in range(10):
            for ti in range(t_end - t_start):
                # global-sourced nodes
                if glob_idx_in_out.size > 0:
                    v_g = bilinear_sample(
                        ext_chunk[ti, :, :, ch], g_lons, g_lats,
                        glob_nodes_lon, glob_nodes_lat,
                    )
                    out_mm[t_start + ti, glob_idx_in_out, ch] = v_g.astype(np.float16)
                # region-sourced nodes
                if rex_chunk is not None and reg_idx_in_out.size > 0:
                    v_r = bilinear_sample(
                        rex_chunk[ti, :, :, ch], rg_lons, rg_lats,
                        reg_nodes_lon, reg_nodes_lat,
                    )
                    out_mm[t_start + ti, reg_idx_in_out, ch] = v_r.astype(np.float16)
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
    out_info["extra_source_global"] = str(extra)
    out_info["extra_source_region"] = str(args.region_extra_dir) if args.region_extra_dir else None
    out_info["extra_routing_region_nodes"] = int(regional_use_region.sum())
    out_info["extra_routing_global_nodes"] = int((~regional_use_region).sum())
    (out / "dataset_info.json").write_text(json.dumps(out_info, indent=2))
    print(f"[INFO] dataset_info.json written: n_feat=33 (19+14)")

    print("\n[DONE] multires Russia 33f assembled at:", out)


if __name__ == "__main__":
    main()

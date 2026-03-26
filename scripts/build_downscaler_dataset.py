#!/usr/bin/env python3
"""
scripts/build_downscaler_dataset.py

Строит датасет для обучения downscaler'а: пары (coarse ROI, fine ROI).

Входы:
  - Глобальный датасет 512×256 (~0.7°) → кропаем ROI → (T, 21, 14, C)
  - Региональный датасет 61×41 (0.25°) → (T, 61, 41, C)

Выход:
  - coarse.npy: (T, 61, 41, C) float16 — билинейно upsampled coarse данные
  - fine.npy:   (T, 61, 41, C) float16 — реальные 0.25° данные (target)
  - static_fine.npy: (61, 41, N_static) float32 — статические поля 0.25° (z_surf, lsm)
  - dataset_info.json, scalers.npz, variables.json

Коарс данные ресайзятся из ~0.7° до 0.25° билинейной интерполяцией —
это то, что GNN "видит". UNet учится восстанавливать мелкомасштабные детали.

Usage:
  python scripts/build_downscaler_dataset.py \
    --global-dir /data/datasets/global_512x256_19f_2010-2021_07deg \
    --region-dir /data/datasets/region_krsk_61x41_19f_2010-2020_025deg \
    --roi 50 60 83 98 \
    --out-dir /data/datasets/downscaler_krsk_19f
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-dir", required=True, help="Глобальный датасет 512×256")
    ap.add_argument("--region-dir", required=True, help="Региональный датасет 61×41 (real 0.25°)")
    ap.add_argument("--roi", nargs=4, type=float, default=[50, 60, 83, 98],
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--static-channels", nargs="*", type=int, default=[7, 8],
                    help="Indices of static channels (z_surf=7, lsm=8)")
    args = ap.parse_args()

    lat_min, lat_max, lon_min, lon_max = args.roi
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Загрузка глобального датасета ──
    g_dir = Path(args.global_dir)
    with open(g_dir / "dataset_info.json") as f:
        g_info = json.load(f)
    g_coords = np.load(g_dir / "coords.npz")
    g_lats = g_coords["latitude"].astype(np.float64)   # (n_lat,) sorted
    g_lons = g_coords["longitude"].astype(np.float64)   # (n_lon,) sorted
    g_shape = (g_info["n_time"], g_info["n_lon"], g_info["n_lat"], g_info["n_feat"])
    g_data = np.memmap(g_dir / "data.npy", dtype=np.float16, mode="r", shape=g_shape)
    # g_data: (T_global, n_lon, n_lat, C) — lon-first!
    g_scalers = np.load(g_dir / "scalers.npz")
    with open(g_dir / "variables.json") as f:
        g_vars = json.load(f)

    print(f"[Global] shape={g_shape}, lats=[{g_lats[0]:.2f}..{g_lats[-1]:.2f}], "
          f"lons=[{g_lons[0]:.2f}..{g_lons[-1]:.2f}]")

    # ── 2. Загрузка регионального датасета ──
    r_dir = Path(args.region_dir)
    with open(r_dir / "dataset_info.json") as f:
        r_info = json.load(f)
    r_coords = np.load(r_dir / "coords.npz")
    r_lats = r_coords["latitude"].astype(np.float64)   # (n_lat,)
    r_lons = r_coords["longitude"].astype(np.float64)   # (n_lon,)
    r_shape = (r_info["n_time"], r_info["n_lon"], r_info["n_lat"], r_info["n_feat"])
    r_data = np.memmap(r_dir / "data.npy", dtype=np.float16, mode="r", shape=r_shape)
    # r_data: (T_region, n_lon, n_lat, C) — lon-first!

    print(f"[Region] shape={r_shape}, lats=[{r_lats[0]:.2f}..{r_lats[-1]:.2f}], "
          f"lons=[{r_lons[0]:.2f}..{r_lons[-1]:.2f}]")

    # ── 3. Определяем общее количество timesteps ──
    T = min(g_info["n_time"], r_info["n_time"])
    print(f"[Align] Using first {T} timesteps (common to both datasets)")

    C = g_info["n_feat"]
    assert C == r_info["n_feat"], f"Feature mismatch: global={C}, region={r_info['n_feat']}"

    n_lon_fine = r_info["n_lon"]  # 61
    n_lat_fine = r_info["n_lat"]  # 41

    # ── 4. ROI индексы в глобальном гриде ──
    # Берём чуть шире ROI, чтобы интерполяция на краях была корректной
    buffer = 1.0  # 1° buffer
    lat_idx = np.where((g_lats >= lat_min - buffer) & (g_lats <= lat_max + buffer))[0]
    lon_idx = np.where((g_lons >= lon_min - buffer) & (g_lons <= lon_max + buffer))[0]

    roi_g_lats = g_lats[lat_idx]
    roi_g_lons = g_lons[lon_idx]
    print(f"[ROI coarse] {len(lon_idx)}×{len(lat_idx)} points, "
          f"lats=[{roi_g_lats[0]:.2f}..{roi_g_lats[-1]:.2f}], "
          f"lons=[{roi_g_lons[0]:.2f}..{roi_g_lons[-1]:.2f}]")

    # ── 5. Целевая fine-сетка для интерполяции ──
    # meshgrid regional coordinates: (n_lat_fine, n_lon_fine, 2)
    fine_lon_grid, fine_lat_grid = np.meshgrid(r_lons, r_lats)  # (n_lat, n_lon)
    fine_points = np.stack([fine_lon_grid.ravel(), fine_lat_grid.ravel()], axis=-1)
    # Points shape: (n_lat*n_lon, 2) — (lon, lat) pairs

    # ── 6. Создаём output memmaps ──
    # Храним в формате (T, n_lon, n_lat, C) — lon-first, как все остальные датасеты
    coarse_path = out_dir / "coarse.npy"
    fine_path = out_dir / "fine.npy"

    coarse_out = np.memmap(coarse_path, dtype=np.float16, mode="w+",
                           shape=(T, n_lon_fine, n_lat_fine, C))
    fine_out = np.memmap(fine_path, dtype=np.float16, mode="w+",
                         shape=(T, n_lon_fine, n_lat_fine, C))

    # ── 7. Заполняем fine (target) — просто копируем реальные данные ──
    print(f"[Fine] Copying real 0.25° data ({T} timesteps)...")
    chunk = 500
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        fine_out[t0:t1] = r_data[t0:t1].astype(np.float16)
        if (t0 // chunk) % 10 == 0:
            print(f"  fine: {t1}/{T}")
    fine_out.flush()
    print(f"  → {fine_path} ({T}, {n_lon_fine}, {n_lat_fine}, {C})")

    # ── 8. Заполняем coarse — интерполяция из глобального грида ──
    # Для каждого timestep: crop ROI из глобального, интерполировать на fine grid
    print(f"[Coarse] Interpolating from ~0.7° to 0.25° ({T} timesteps)...")

    for t in range(T):
        # Crop: g_data (T, n_lon, n_lat, C) → ROI
        # Axis 1 = lon, axis 2 = lat
        roi_slice = np.array(
            g_data[t, lon_idx[0]:lon_idx[-1]+1, lat_idx[0]:lat_idx[-1]+1, :],
            dtype=np.float32
        )
        # roi_slice: (n_lon_roi, n_lat_roi, C)

        # Интерполируем каждый канал отдельно
        for c in range(C):
            field = roi_slice[:, :, c]  # (n_lon_roi, n_lat_roi)
            # RegularGridInterpolator: points = (lon_axis, lat_axis) matching data axes
            interp = RegularGridInterpolator(
                (roi_g_lons, roi_g_lats), field,
                method="linear", bounds_error=False, fill_value=None
            )
            # Query на fine grid: нужны (lon, lat) пары
            fine_vals = interp(fine_points)  # (n_lat*n_lon,)
            # Reshape: (n_lat, n_lon) → transpose to (n_lon, n_lat) for lon-first storage
            coarse_out[t, :, :, c] = fine_vals.reshape(n_lat_fine, n_lon_fine).T.astype(np.float16)

        if t % 500 == 0:
            print(f"  coarse: {t}/{T}")

    coarse_out.flush()
    print(f"  → {coarse_path} ({T}, {n_lon_fine}, {n_lat_fine}, {C})")

    # ── 9. Извлекаем статические поля fine (z_surf, lsm) ──
    static_ch = args.static_channels
    if static_ch:
        # Берём из первого timestep реальных данных
        static_fine = np.array(r_data[0, :, :, :], dtype=np.float32)[:, :, static_ch]
        # static_fine: (n_lon, n_lat, N_static)
        np.save(out_dir / "static_fine.npy", static_fine)
        print(f"[Static] Saved {len(static_ch)} static channels: {static_fine.shape}")

    # ── 10. Сохраняем метаданные ──
    # Используем scalers от глобального датасета (данные нормализуются одинаково)
    np.savez(out_dir / "scalers.npz",
             mean=g_scalers["mean"], std=g_scalers["std"], n=g_scalers["n"])

    info = {
        "n_time": T,
        "n_lon": n_lon_fine,
        "n_lat": n_lat_fine,
        "n_feat": C,
        "roi": list(args.roi),
        "global_source": str(args.global_dir),
        "region_source": str(args.region_dir),
        "coarse_file": "coarse.npy",
        "fine_file": "fine.npy",
        "static_channels": static_ch,
    }
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(out_dir / "variables.json", "w") as f:
        json.dump(g_vars, f, indent=2)

    np.savez(out_dir / "coords.npz",
             latitude=r_lats.astype(np.float32),
             longitude=r_lons.astype(np.float32))

    print(f"\n[Done] Dataset saved to {out_dir}")
    print(f"  coarse.npy: ({T}, {n_lon_fine}, {n_lat_fine}, {C}) — bilinear upsampled from 0.7°")
    print(f"  fine.npy:   ({T}, {n_lon_fine}, {n_lat_fine}, {C}) — real ERA5 0.25°")
    print(f"  Total size: ~{(T * n_lon_fine * n_lat_fine * C * 2 * 2) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

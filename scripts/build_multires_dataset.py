#!/usr/bin/env python3
"""
Построение мультирезолюционного датасета (Variant B).

Идея:
  - Берём глобальные данные (512×256, ~0.7°)
  - В ROI (region of interest) убираем грубые глобальные точки
  - Вставляем мелкие региональные точки (0.25°)
  - Результат: FLAT массив (T, N_total, C) с нерегулярной сеткой

Два режима:
  1) --mode=interpolate (для ОБУЧЕНИЯ):
     У нас есть только глобальные данные за 2010-2021.
     Мелкие региональные значения ИНТЕРПОЛИРУЕМ из грубой глобальной сетки.
  2) --mode=merge (для ИНФЕРЕНСА/ТЕСТА):
     У нас есть и глобальные, и реальные региональные данные за один период.
     Совмещаем их по времени, в ROI берём реальные мелкие значения.

Usage:
  # Режим interpolate (обучение):
  python scripts/build_multires_dataset.py \\
    --global-dir data/datasets/wb2_512x256_19f_ar \\
    --region-coords data/datasets/region_krsk_cds_19f/coords.npz \\
    --roi 50 60 83 98 \\
    --mode interpolate \\
    --out-dir data/datasets/multires_krsk_19f

  # Режим merge (тест, Jan 2023):
  python scripts/build_multires_dataset.py \\
    --global-dir data/datasets/wb2_512x256_19f_jan2023 \\
    --region-dir data/datasets/region_krsk_cds_19f \\
    --roi 50 60 83 98 \\
    --mode merge \\
    --out-dir data/datasets/multires_krsk_jan2023
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_global_dataset(data_dir: str):
    """Загружает глобальный датасет как memmap + координаты."""
    data_dir = Path(data_dir)

    with open(data_dir / "dataset_info.json") as f:
        info = json.load(f)

    coords = np.load(data_dir / "coords.npz")
    g_lats = coords["latitude"].astype(np.float64)   # (n_lat,)
    g_lons = coords["longitude"].astype(np.float64)   # (n_lon,)

    shape = (info["n_time"], info["n_lon"], info["n_lat"], info["n_feat"])
    data = np.memmap(
        data_dir / "data.npy", dtype=np.float16, mode="r", shape=shape
    )  # (T, lon, lat, feat)

    scalers = np.load(data_dir / "scalers.npz")

    with open(data_dir / "variables.json") as f:
        variables = json.load(f)

    return data, g_lats, g_lons, info, scalers, variables


def load_regional_dataset(data_dir: str):
    """Загружает региональный датасет."""
    data_dir = Path(data_dir)

    with open(data_dir / "dataset_info.json") as f:
        info = json.load(f)

    coords = np.load(data_dir / "coords.npz")
    r_lats = coords["latitude"].astype(np.float64)
    r_lons = coords["longitude"].astype(np.float64)

    shape = (info["n_time"], info["n_lon"], info["n_lat"], info["n_feat"])
    data = np.memmap(
        data_dir / "data.npy", dtype=np.float16, mode="r", shape=shape
    )

    return data, r_lats, r_lons, info


def build_node_mapping(g_lats, g_lons, r_lats, r_lons, roi):
    """
    Строит мультирезолюционную сетку.

    Возвращает:
      flat_lats, flat_lons: координаты всех узлов (N_total,)
      global_mask: bool (N_total,) — True для глобальных узлов
      region_mask: bool (N_total,) — True для региональных узлов
      g_keep_lon_idx, g_keep_lat_idx: индексы глобальных узлов, которые сохраняем
      r_lon_grid, r_lat_grid: meshgrid региональных координат для индексации
    """
    lat_min, lat_max, lon_min, lon_max = roi

    # Индексы глобальных узлов ВНУТРИ ROI (будут удалены)
    g_lat_in_roi = (g_lats >= lat_min) & (g_lats <= lat_max)
    g_lon_in_roi = (g_lons >= lon_min) & (g_lons <= lon_max)

    # Глобальные узлы ВНЕ ROI — оставляем
    # Формируем полную сетку (meshgrid) и маску
    g_lon_mesh, g_lat_mesh = np.meshgrid(g_lons, g_lats)  # (nlat, nlon)
    in_roi = (
        (g_lat_mesh >= lat_min) & (g_lat_mesh <= lat_max) &
        (g_lon_mesh >= lon_min) & (g_lon_mesh <= lon_max)
    )
    keep_global = ~in_roi  # (nlat, nlon) — True для узлов, которые сохраняем

    # Flat координаты глобальных узлов (вне ROI)
    g_flat_lats = g_lat_mesh[keep_global]  # (N_kept,)
    g_flat_lons = g_lon_mesh[keep_global]  # (N_kept,)
    n_global_kept = g_flat_lats.shape[0]

    # Flat координаты региональных узлов (все)
    r_lon_mesh, r_lat_mesh = np.meshgrid(r_lons, r_lats)  # (r_nlat, r_nlon)
    r_flat_lats = r_lat_mesh.reshape(-1)
    r_flat_lons = r_lon_mesh.reshape(-1)
    n_regional = r_flat_lats.shape[0]

    # Конкатенация: [глобальные_вне_ROI, региональные]
    flat_lats = np.concatenate([g_flat_lats, r_flat_lats]).astype(np.float32)
    flat_lons = np.concatenate([g_flat_lons, r_flat_lons]).astype(np.float32)

    # Маски
    global_mask = np.zeros(len(flat_lats), dtype=bool)
    global_mask[:n_global_kept] = True
    region_mask = np.zeros(len(flat_lats), dtype=bool)
    region_mask[n_global_kept:] = True

    print(f"[build_node_mapping]")
    print(f"  Глобальная сетка: {len(g_lats)}×{len(g_lons)} = {len(g_lats)*len(g_lons)} узлов")
    print(f"  В ROI удалено: {in_roi.sum()} узлов")
    print(f"  Глобальных осталось: {n_global_kept}")
    print(f"  Региональных добавлено: {n_regional}")
    print(f"  Итого узлов: {len(flat_lats)}")

    return flat_lats, flat_lons, global_mask, region_mask, keep_global


def build_interpolate_mode(args):
    """
    Режим interpolate: берём глобальные данные, в ROI интерполируем на мелкую сетку.
    Для обучения на мультирезолюционной геометрии.
    """
    print("=== Режим: interpolate (обучение) ===")

    data, g_lats, g_lons, info, scalers, variables = load_global_dataset(args.global_dir)
    T, n_lon, n_lat, C = data.shape

    # Загружаем координаты региона
    r_coords = np.load(args.region_coords)
    r_lats = r_coords["latitude"].astype(np.float64)
    r_lons = r_coords["longitude"].astype(np.float64)

    roi = args.roi
    flat_lats, flat_lons, global_mask, region_mask, keep_global = \
        build_node_mapping(g_lats, g_lons, r_lats, r_lons, roi)

    N_total = len(flat_lats)
    n_global_kept = global_mask.sum()
    n_regional = region_mask.sum()

    # Создаём выходной каталог
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем координаты
    np.savez(
        out_dir / "coords.npz",
        latitude=flat_lats,
        longitude=flat_lons,
        is_regional=region_mask,
    )

    # Данные: пишем как memmap float16 (T, N_total, C)
    out_data = np.memmap(
        out_dir / "data.npy", dtype=np.float16, mode="w+",
        shape=(T, N_total, C),
    )

    # Для интерполяции: глобальная сетка как (lat, lon) regular grid
    # data shape: (T, lon, lat, C) → для RegularGridInterpolator нужен порядок (lat, lon)
    # RegularGridInterpolator принимает точки в порядке осей
    print(f"Записываем {T} временных шагов...")

    # Региональная meshgrid для интерполяции
    r_lon_mesh, r_lat_mesh = np.meshgrid(r_lons, r_lats)
    r_points = np.stack([r_lat_mesh.ravel(), r_lon_mesh.ravel()], axis=-1)  # (N_reg, 2)

    batch_size = 100  # кубиками по 100 шагов
    for t_start in range(0, T, batch_size):
        t_end = min(t_start + batch_size, T)
        batch = data[t_start:t_end].astype(np.float32)  # (batch, lon, lat, C)

        for t_local in range(t_end - t_start):
            t_global = t_start + t_local
            frame = batch[t_local]  # (lon, lat, C)

            # 1) Глобальные узлы вне ROI
            # frame shape is (n_lon, n_lat, C), keep_global is (n_lat, n_lon)
            # Нужно транспонировать frame → (n_lat, n_lon, C) чтобы совпало с keep_global
            frame_latlon = frame.transpose(1, 0, 2)  # (n_lat, n_lon, C)
            global_values = frame_latlon[keep_global]  # (n_kept, C)

            # 2) Региональные узлы: интерполяция
            regional_values = np.zeros((n_regional, C), dtype=np.float32)
            for c in range(C):
                # frame_latlon[:, :, c] — (n_lat, n_lon) с осями (g_lats, g_lons)
                interp = RegularGridInterpolator(
                    (g_lats, g_lons), frame_latlon[:, :, c],
                    method="linear", bounds_error=False, fill_value=None
                )
                regional_values[:, c] = interp(r_points)

            # 3) Конкатенация
            out_data[t_global, :n_global_kept, :] = global_values.astype(np.float16)
            out_data[t_global, n_global_kept:, :] = regional_values.astype(np.float16)

        if (t_start // batch_size) % 10 == 0:
            print(f"  t={t_start}..{t_end-1} / {T}")

    out_data.flush()
    print(f"Данные записаны: {out_dir / 'data.npy'} shape=({T}, {N_total}, {C})")

    # dataset_info.json
    out_info = {
        "time_start": info["time_start"],
        "time_end": info["time_end"],
        "n_time": T,
        "n_nodes": N_total,
        "n_feat": C,
        "flat": True,
        "n_global_kept": int(n_global_kept),
        "n_regional": int(n_regional),
        "roi": list(args.roi),
        "variables": variables,
        "dtype": "float16",
        "file": "data.npy",
        "source_global": str(args.global_dir),
        "source_region_coords": str(args.region_coords),
        "mode": "interpolate",
    }
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(out_info, f, indent=2, ensure_ascii=False)

    # scalers (те же, что и у глобального)
    np.savez(out_dir / "scalers.npz", **dict(scalers))

    # variables.json
    with open(out_dir / "variables.json", "w") as f:
        json.dump(variables, f)

    print(f"\nГотово! Датасет: {out_dir}")
    print(f"  Узлы: {N_total} ({n_global_kept} global + {n_regional} regional)")
    print(f"  Время: {T} шагов")
    print(f"  Признаки: {C}")


def build_merge_mode(args):
    """
    Режим merge: совмещаем глобальные и региональные данные за пересекающийся период.
    Для инференса/тестирования.
    """
    print("=== Режим: merge (тест/инференс) ===")

    g_data, g_lats, g_lons, g_info, g_scalers, variables = load_global_dataset(args.global_dir)
    r_data, r_lats, r_lons, r_info = load_regional_dataset(args.region_dir)

    T_g, n_lon_g, n_lat_g, C = g_data.shape
    T_r, n_lon_r, n_lat_r, C_r = r_data.shape
    assert C == C_r, f"Features mismatch: {C} vs {C_r}"

    roi = args.roi
    flat_lats, flat_lons, global_mask, region_mask, keep_global = \
        build_node_mapping(g_lats, g_lons, r_lats, r_lons, roi)

    N_total = len(flat_lats)
    n_global_kept = global_mask.sum()
    n_regional = region_mask.sum()

    # Определяем пересечение временных диапазонов
    # Для простоты используем минимум из длин (предполагаем выравнивание)
    T = min(T_g, T_r)
    if args.time_offset_global is not None:
        t_off_g = args.time_offset_global
    else:
        t_off_g = 0
    if args.time_offset_region is not None:
        t_off_r = args.time_offset_region
    else:
        t_off_r = 0

    T = min(T_g - t_off_g, T_r - t_off_r)
    print(f"Пересечение: {T} временных шагов")
    print(f"  Глобальные: offset={t_off_g}, всего={T_g}")
    print(f"  Региональные: offset={t_off_r}, всего={T_r}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "coords.npz",
        latitude=flat_lats,
        longitude=flat_lons,
        is_regional=region_mask,
    )

    out_data = np.memmap(
        out_dir / "data.npy", dtype=np.float16, mode="w+",
        shape=(T, N_total, C),
    )

    print(f"Записываем {T} шагов...")
    for t in range(T):
        g_frame = g_data[t + t_off_g].astype(np.float32)  # (lon_g, lat_g, C)
        r_frame = r_data[t + t_off_r].astype(np.float32)  # (lon_r, lat_r, C)

        # Глобальные (транспонируем lon,lat→lat,lon для совпадения с keep_global)
        g_frame_latlon = g_frame.transpose(1, 0, 2)  # (lat_g, lon_g, C)
        global_values = g_frame_latlon[keep_global]  # (n_kept, C)

        # Региональные (уже мелкие, берём как есть)
        # r_frame: (lon_r, lat_r, C) → flatten lat-major (как meshgrid lat,lon → ravel)
        r_frame_latlon = r_frame.transpose(1, 0, 2)  # (lat_r, lon_r, C)
        regional_values = r_frame_latlon.reshape(-1, C)  # (n_regional, C)

        out_data[t, :n_global_kept, :] = global_values.astype(np.float16)
        out_data[t, n_global_kept:, :] = regional_values.astype(np.float16)

        if t % 20 == 0:
            print(f"  t={t}/{T}")

    out_data.flush()
    print(f"Данные записаны: shape=({T}, {N_total}, {C})")

    out_info = {
        "time_start": g_info.get("time_start", ""),
        "time_end": g_info.get("time_end", ""),
        "n_time": T,
        "n_nodes": N_total,
        "n_feat": C,
        "flat": True,
        "n_global_kept": int(n_global_kept),
        "n_regional": int(n_regional),
        "roi": list(args.roi),
        "variables": variables,
        "dtype": "float16",
        "file": "data.npy",
        "source_global": str(args.global_dir),
        "source_region": str(args.region_dir),
        "mode": "merge",
    }
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(out_info, f, indent=2, ensure_ascii=False)

    np.savez(out_dir / "scalers.npz", **dict(g_scalers))

    with open(out_dir / "variables.json", "w") as f:
        json.dump(variables, f)

    print(f"\nГотово! Датасет: {out_dir}")
    print(f"  Узлы: {N_total} ({n_global_kept} global + {n_regional} regional)")
    print(f"  Время: {T} шагов")


def main():
    ap = argparse.ArgumentParser(description="Build multi-resolution dataset")
    ap.add_argument("--global-dir", required=True, help="Dir с глобальным датасетом")
    ap.add_argument("--region-dir", default=None, help="Dir с региональным датасетом (для merge)")
    ap.add_argument("--region-coords", default=None, help="coords.npz региона (для interpolate)")
    ap.add_argument("--roi", nargs=4, type=float, required=True,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    help="Bounding box региона")
    ap.add_argument("--mode", required=True, choices=["interpolate", "merge"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--time-offset-global", type=int, default=None,
                    help="Смещение по времени в глобальных данных (для merge)")
    ap.add_argument("--time-offset-region", type=int, default=None,
                    help="Смещение по времени в региональных данных (для merge)")
    args = ap.parse_args()

    if args.mode == "interpolate":
        if args.region_coords is None:
            # Пробуем взять из region-dir
            if args.region_dir:
                args.region_coords = str(Path(args.region_dir) / "coords.npz")
            else:
                ap.error("--region-coords required for interpolate mode")
        build_interpolate_mode(args)
    elif args.mode == "merge":
        if args.region_dir is None:
            ap.error("--region-dir required for merge mode")
        build_merge_mode(args)


if __name__ == "__main__":
    main()

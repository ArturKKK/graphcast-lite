#!/usr/bin/env python3
"""
Добавляет 4 временных канала к региональному датасету:
  - sin(hour_of_day / 24 * 2π)
  - cos(hour_of_day / 24 * 2π)
  - sin(day_of_year / 365.25 * 2π) 
  - cos(day_of_year / 365.25 * 2π)

Создаёт копию датасета с суффиксом _23f (или заданным).
Каналы добавляются в конец (индексы 19-22).
Они одинаковы для всех точек сетки (регион маленький, ~1 часовой пояс).

Использование:
  python scripts/add_time_features.py \
      --src data/datasets/region_krsk_61x41_19f_2010-2020_025deg \
      --dst data/datasets/region_krsk_61x41_23f_2010-2020_025deg
"""

import argparse
import json
import shutil
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Add time features to regional dataset")
    parser.add_argument("--src", required=True, help="Path to source dataset directory")
    parser.add_argument("--dst", required=True, help="Path to destination dataset directory")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # 1. Читаем метаданные источника
    with open(src / "dataset_info.json") as f:
        info = json.load(f)

    n_time = info["n_time"]
    n_lon = info["n_lon"]
    n_lat = info["n_lat"]
    n_feat_old = info["n_feat"]
    n_feat_new = n_feat_old + 4
    time_start = info["time_start"]  # "2010-01-01"

    print(f"Source: {src}")
    print(f"  shape: ({n_time}, {n_lon}, {n_lat}, {n_feat_old})")
    print(f"  time: {time_start} — {info['time_end']}, {n_time} steps (6h)")

    # 2. Генерируем временные метки (6h шаг)
    t0 = datetime.strptime(time_start, "%Y-%m-%d")
    times = [t0 + timedelta(hours=6 * i) for i in range(n_time)]

    # 3. Вычисляем 4 фичи для каждого timestep
    hours = np.array([t.hour + t.minute / 60 for t in times], dtype=np.float32)
    doys = np.array([t.timetuple().tm_yday for t in times], dtype=np.float32)

    sin_hour = np.sin(2 * np.pi * hours / 24.0)
    cos_hour = np.cos(2 * np.pi * hours / 24.0)
    sin_doy = np.sin(2 * np.pi * doys / 365.25)
    cos_doy = np.cos(2 * np.pi * doys / 365.25)

    # shape: (n_time, 4) → broadcast на (n_time, n_lon, n_lat, 4)
    time_feats = np.stack([sin_hour, cos_hour, sin_doy, cos_doy], axis=-1)  # (T, 4)
    time_feats_grid = time_feats[:, np.newaxis, np.newaxis, :]  # (T, 1, 1, 4)
    time_feats_grid = np.broadcast_to(
        time_feats_grid, (n_time, n_lon, n_lat, 4)
    )

    print(f"\nTime features computed:")
    print(f"  sin_hour range: [{sin_hour.min():.4f}, {sin_hour.max():.4f}]")
    print(f"  cos_hour range: [{cos_hour.min():.4f}, {cos_hour.max():.4f}]")
    print(f"  sin_doy  range: [{sin_doy.min():.4f}, {sin_doy.max():.4f}]")
    print(f"  cos_doy  range: [{cos_doy.min():.4f}, {cos_doy.max():.4f}]")

    # 4. Читаем исходные данные
    print(f"\nReading source data...")
    src_data = np.memmap(
        src / "data.npy", dtype=np.float16, mode="r",
        shape=(n_time, n_lon, n_lat, n_feat_old)
    )

    # 5. Создаём директорию назначения
    dst.mkdir(parents=True, exist_ok=True)

    # 6. Пишем новый data.npy
    print(f"Writing new data.npy ({n_time}, {n_lon}, {n_lat}, {n_feat_new})...")
    dst_data = np.memmap(
        dst / "data.npy", dtype=np.float16, mode="w+",
        shape=(n_time, n_lon, n_lat, n_feat_new)
    )

    # Копируем чанками по 1000 timesteps чтобы не убить память
    chunk = 1000
    for t0_idx in range(0, n_time, chunk):
        t1_idx = min(t0_idx + chunk, n_time)
        # Старые каналы
        dst_data[t0_idx:t1_idx, :, :, :n_feat_old] = src_data[t0_idx:t1_idx]
        # Новые каналы (time features)
        dst_data[t0_idx:t1_idx, :, :, n_feat_old:] = time_feats_grid[t0_idx:t1_idx].astype(np.float16)
        if (t0_idx // chunk) % 5 == 0:
            print(f"  {t1_idx}/{n_time} timesteps written")

    dst_data.flush()
    del dst_data, src_data
    print("  data.npy written.")

    # 7. Пересчитываем scalers
    print("Computing scalers for new dataset...")
    new_data = np.memmap(
        dst / "data.npy", dtype=np.float16, mode="r",
        shape=(n_time, n_lon, n_lat, n_feat_new)
    )
    # Для mean/std: flatten пространство, считаем по времени+пространству
    flat = new_data.reshape(-1, n_feat_new).astype(np.float32)
    new_mean = flat.mean(axis=0)
    new_std = flat.std(axis=0)
    # Защита от деления на 0
    new_std[new_std < 1e-8] = 1.0
    n_samples = flat.shape[0]
    del flat, new_data

    old_scalers = np.load(src / "scalers.npz")
    np.savez(
        dst / "scalers.npz",
        mean=new_mean.astype(np.float32),
        std=new_std.astype(np.float32),
        n=np.array(n_samples),
    )
    print(f"  scalers: mean_new_ch=[{new_mean[-4]:.4f}, {new_mean[-3]:.4f}, {new_mean[-2]:.4f}, {new_mean[-1]:.4f}]")
    print(f"           std_new_ch=[{new_std[-4]:.4f}, {new_std[-3]:.4f}, {new_std[-2]:.4f}, {new_std[-1]:.4f}]")

    # 8. Копируем coords.npz (без изменений)
    shutil.copy2(src / "coords.npz", dst / "coords.npz")

    # 9. Обновляем variables.json
    old_vars = info["variables"]
    new_vars = old_vars + ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    with open(dst / "variables.json", "w") as f:
        json.dump(new_vars, f, indent=2)

    # 10. Обновляем dataset_info.json
    new_info = dict(info)
    new_info["n_feat"] = n_feat_new
    new_info["variables"] = new_vars
    new_info["size_gb"] = round(n_time * n_lon * n_lat * n_feat_new * 2 / 1e9, 3)
    new_info["source"] = info.get("source", "") + " + time features (sin/cos hour, doy)"
    with open(dst / "dataset_info.json", "w") as f:
        json.dump(new_info, f, indent=2, ensure_ascii=False)

    print(f"\nDone! New dataset: {dst}")
    print(f"  Features: {n_feat_old} → {n_feat_new}")
    print(f"  Variables: {new_vars}")
    print(f"  New channels: 19=sin_hour, 20=cos_hour, 21=sin_doy, 22=cos_doy")
    print(f"  static_channels for config: [7, 8] (time channels are NOT static — they change per timestep)")


if __name__ == "__main__":
    main()

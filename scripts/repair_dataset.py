#!/usr/bin/env python3
"""
scripts/repair_dataset.py
Ремонт датасета 512×256: перекачка каналов с float16-переполнением.

Проблема:
  float16 max = 65504. Переменные давления (msl ~101325 Pa, sp ~98000 Pa)
  и геопотенциал поверхности (z_surf до ~86740 м²/с²) превышают этот предел
  и были записаны как inf.

Решение:
  1. Перекачиваем ТОЛЬКО сломанные каналы (msl, sp, z_surf) из GCS.
  2. Масштабируем: Pa → hPa (*0.01), м²/с² → м (/9.80665).
  3. z@850 и z@500 не переполнились (макс ~17000 и ~58000), но для
     единообразия тоже пересчитываем в метры.
  4. Перезаписываем столбцы в существующем data.npy.
  5. Пересчитываем scalers.npz поканально (без full_block, экономим RAM).

Время: ~2-3 часа (качаем 5 каналов из 19 + пересчёт стат по 14 каналам с диска).
RAM: ~1 GB (поканальная обработка, без аллокации full_block).
"""

import argparse
import json
import gc
import time
from pathlib import Path

import dask
import numpy as np
import xarray as xr
import gcsfs

WB2_ZARR = "gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr"

# Каналы которые нужно починить: (channel_name, zarr_var, level_or_None, scale_factor)
REPAIR_CHANNELS = [
    ("msl",    "mean_sea_level_pressure",  None, 0.01),         # Pa -> hPa
    ("sp",     "surface_pressure",         None, 0.01),         # Pa -> hPa
    ("z_surf", "geopotential_at_surface",  None, 1/9.80665),    # m²/s² -> m
    ("z@850",  "geopotential",             850,  1/9.80665),    # m²/s² -> m
    ("z@500",  "geopotential",             500,  1/9.80665),    # m²/s² -> m
]


def welford_update(running_mean, running_m2, total_n, block_sum, block_sumsq, block_n):
    block_mean = block_sum / block_n
    block_var  = block_sumsq / block_n - block_mean ** 2
    block_var  = np.maximum(block_var, 0.0)
    delta  = block_mean - running_mean
    new_n  = total_n + block_n
    running_mean += delta * (block_n / new_n)
    running_m2  += block_var * block_n + (delta ** 2) * total_n * block_n / new_n
    return running_mean, running_m2, new_n


def main():
    parser = argparse.ArgumentParser(description="Repair float16-overflowed channels.")
    parser.add_argument("--data-dir", type=str, default="data/datasets/wb2_512x256_19f_ar")
    parser.add_argument("--time-chunk", type=int, default=500)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    info = json.loads((data_dir / "dataset_info.json").read_text())
    n_time, n_lon, n_lat = info["n_time"], info["n_lon"], info["n_lat"]
    var_list = info["variables"]
    n_feat   = len(var_list)
    ch_idx   = {name: i for i, name in enumerate(var_list)}

    # Какие столбцы чиним
    repair_indices = set()
    for ch_name, _, _, _ in REPAIR_CHANNELS:
        if ch_name in ch_idx:
            repair_indices.add(ch_idx[ch_name])
    good_indices = sorted(set(range(n_feat)) - repair_indices)

    print("=" * 60)
    print(f"REPAIR DATASET: {data_dir}")
    print(f"  Shape: ({n_time}, {n_lon}, {n_lat}, {n_feat})")
    print(f"  Repairing channels: {[c[0] for c in REPAIR_CHANNELS]}")
    print(f"  Good channels (stats from disk): {len(good_indices)}")
    print("=" * 60)

    # Open zarr
    fs    = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB2_ZARR)
    ds    = xr.open_zarr(store, consolidated=True)
    ds    = ds.sel(time=slice(info["time_start"], info["time_end"]))

    # Подготовим DataArrays для ремонтных каналов
    repair_das = {}  # ch_name -> (DataArray, scale)
    for ch_name, zarr_var, level, scale in REPAIR_CHANNELS:
        if ch_name not in ch_idx:
            print(f"  [SKIP] {ch_name} not in dataset")
            continue
        da = ds[zarr_var]
        if level is not None:
            da = da.sel(level=level, drop=True)
        # убираем лишние coords
        non_idx = [c for c in da.coords if c not in da.dims]
        if non_idx:
            da = da.reset_coords(non_idx, drop=True)
        if "time" not in da.dims:
            da = da.expand_dims(time=ds.time)
        da = da.transpose("time", "longitude", "latitude")
        repair_das[ch_name] = (da, scale)

    # Open memmap
    npy_path = data_dir / "data.npy"
    fp = np.memmap(str(npy_path), dtype=np.float16, mode="r+",
                   shape=(n_time, n_lon, n_lat, n_feat))

    # Stats — поканальные аккумуляторы
    total_n      = 0
    running_mean = np.zeros(n_feat, dtype=np.float64)
    running_m2   = np.zeros(n_feat, dtype=np.float64)

    t0 = time.time()
    downloaded_steps = 0

    for t_start in range(0, n_time, args.time_chunk):
        t_end     = min(t_start + args.time_chunk, n_time)
        chunk_len = t_end - t_start
        t_slice   = slice(t_start, t_end)
        block_n   = chunk_len * n_lon * n_lat

        pct = t_end / n_time * 100
        elapsed = time.time() - t0
        if downloaded_steps > 0:
            speed   = downloaded_steps / max(elapsed, 1)
            eta_min = (n_time - t_end) / speed / 60
            print(f"\n  [{t_start:>6d}–{t_end:>6d} / {n_time}]  {pct:5.1f}%  "
                  f"ETA {eta_min:.0f} min")
        else:
            print(f"\n  [{t_start:>6d}–{t_end:>6d} / {n_time}]  {pct:5.1f}%  starting...")

        block_sum   = np.zeros(n_feat, dtype=np.float64)
        block_sumsq = np.zeros(n_feat, dtype=np.float64)

        # ── 1) Good channels: stats from disk (no download) ──
        ts0 = time.time()
        for j in good_indices:
            arr = fp[t_start:t_end, :, :, j].astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            block_sum[j]   = arr.sum(dtype=np.float64)
            block_sumsq[j] = (arr * arr).sum(dtype=np.float64)
            del arr
        dt_good = time.time() - ts0
        print(f"    [disk] {len(good_indices)} good channels: {dt_good:.1f}s")

        # ── 2) Repair channels: download + scale + overwrite ──
        # Группируем surface vars для параллельной загрузки
        surf_repair = [(name, da, scale) for name, (da, scale) in repair_das.items()
                       if "@" not in name]  # msl, sp, z_surf
        plev_repair = [(name, da, scale) for name, (da, scale) in repair_das.items()
                       if "@" in name]      # z@850, z@500

        if surf_repair:
            ts1 = time.time()
            lazy = [da.isel(time=t_slice) for _, da, _ in surf_repair]
            results = dask.compute(*lazy, scheduler="threads")

            for (name, _, scale), arr_xr in zip(surf_repair, results):
                j = ch_idx[name]
                arr = np.asarray(arr_xr, dtype=np.float32) * scale
                fp[t_start:t_end, :, :, j] = arr.astype(np.float16)
                block_sum[j]   = arr.sum(dtype=np.float64)
                block_sumsq[j] = (arr * arr).sum(dtype=np.float64)
                del arr

            dt_surf = time.time() - ts1
            print(f"    [fix]  surf ({', '.join(n for n,_,_ in surf_repair)}): {dt_surf:.1f}s")
            del results

        for name, da, scale in plev_repair:
            j = ch_idx[name]
            ts2 = time.time()
            arr = da.isel(time=t_slice).values.astype(np.float32) * scale
            fp[t_start:t_end, :, :, j] = arr.astype(np.float16)
            block_sum[j]   = arr.sum(dtype=np.float64)
            block_sumsq[j] = (arr * arr).sum(dtype=np.float64)
            dt_plev = time.time() - ts2
            print(f"    [fix]  {name}: {dt_plev:.1f}s")
            del arr

        fp.flush()
        gc.collect()

        # ── Welford update (all channels) ──
        running_mean, running_m2, total_n = welford_update(
            running_mean, running_m2, total_n, block_sum, block_sumsq, block_n
        )
        downloaded_steps += chunk_len

    del fp
    elapsed_total = time.time() - t0
    print(f"\n[DONE] Repair complete in {elapsed_total/60:.1f} min")

    # Save new scalers
    mean_f = running_mean.astype(np.float32)
    std_f  = np.sqrt(running_m2 / total_n).astype(np.float32)
    std_f  = np.maximum(std_f, 1e-6)

    np.savez(data_dir / "scalers.npz", mean=mean_f, std=std_f, n=total_n)
    print("[STAT] Updated scalers.npz")
    for i, name in enumerate(var_list):
        tag = " ← FIXED" if i in repair_indices else ""
        print(f"  {name:>8s}:  mean={mean_f[i]:+12.3f}  std={std_f[i]:12.3f}{tag}")


if __name__ == "__main__":
    main()

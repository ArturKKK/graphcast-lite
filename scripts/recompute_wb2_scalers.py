#!/usr/bin/env python3
"""
Пересчёт scalers.npz для датасета wb2_512x256_19f_ar из WeatherBench2 Zarr.

Скейлеры = mean/std по каждому из 19 каналов (с учётом SCALE_FACTORS).
Не скачиваем data.npy — только читаем чанками и считаем Welford-статистику.

Использование:
    python scripts/recompute_wb2_scalers.py --out scalers_wb2.npz
    python scripts/recompute_wb2_scalers.py --out scalers_wb2.npz --start-year 2010 --end-year 2022
"""

import argparse
import gc
import time

import dask
import gcsfs
import numpy as np
import xarray as xr

WB2_ZARR = "gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr"

# Порядок переменных — ТОЧНО как в build_dataset_512x256.py
VAR_ORDER = [
    "t2m", "10u", "10v", "msl", "tp",
    "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
]

SURF_DYNAMIC = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_column_water_vapour",
]
STATIC_VARS = ["geopotential_at_surface", "land_sea_mask"]
PLEV_VARS = ["temperature", "u_component_of_wind", "v_component_of_wind",
             "geopotential", "specific_humidity"]
LEVELS = [850, 500]

RENAME = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "total_column_water_vapour": "tcwv",
    "geopotential_at_surface": "z_surf",
    "land_sea_mask": "lsm",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "specific_humidity": "q",
}

SCALE_FACTORS = {
    "msl":    0.01,
    "sp":     0.01,
    "z_surf": 1/9.80665,
    "z@850":  1/9.80665,
    "z@500":  1/9.80665,
}


def welford_update(running_mean, running_m2, total_n, block_sum, block_sumsq, block_n):
    block_mean = block_sum / block_n
    block_var = block_sumsq / block_n - block_mean ** 2
    block_var = np.maximum(block_var, 0.0)
    delta = block_mean - running_mean
    new_n = total_n + block_n
    running_mean += delta * (block_n / new_n)
    running_m2 += block_var * block_n + (delta ** 2) * total_n * block_n / new_n
    return running_mean, running_m2, new_n


def resolve_tp_name(ds):
    for cand in ["total_precipitation_6hr", "total_precipitation", "tp"]:
        if cand in ds.data_vars:
            return cand
    raise RuntimeError("Precipitation variable not found!")


def main():
    parser = argparse.ArgumentParser(description="Recompute WB2 scalers without downloading full data")
    parser.add_argument("--out", type=str, default="scalers_wb2.npz")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2022, help="End year (exclusive)")
    parser.add_argument("--time-chunk", type=int, default=500)
    args = parser.parse_args()

    time_start = f"{args.start_year}-01-01"
    time_end = f"{args.end_year - 1}-12-31"

    print(f"Opening WB2 Zarr: {time_start} → {time_end}")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB2_ZARR)
    ds = xr.open_zarr(store, consolidated=True)
    ds = ds.sel(time=slice(time_start, time_end))
    print(f"Dims: {dict(ds.dims)}")

    # Build channel parts (same logic as build_dataset_512x256.py)
    surf_parts = {}
    for v in SURF_DYNAMIC:
        if v in ds.data_vars:
            surf_parts[RENAME[v]] = ds[v].transpose("time", "longitude", "latitude")
    tp_name = resolve_tp_name(ds)
    surf_parts["tp"] = ds[tp_name].transpose("time", "longitude", "latitude")
    for v in STATIC_VARS:
        if v in ds.data_vars:
            da = ds[v]
            if "time" not in da.dims:
                da = da.expand_dims(time=ds.time)
            surf_parts[RENAME[v]] = da.transpose("time", "longitude", "latitude")

    plev_groups = []
    for v in PLEV_VARS:
        if v in ds.data_vars:
            short = RENAME[v]
            lev_map = {lev: f"{short}@{lev}" for lev in LEVELS}
            da = ds[v].sel(level=LEVELS).transpose("time", "level", "longitude", "latitude")
            non_idx = [c for c in da.coords if c not in da.dims]
            if non_idx:
                da = da.reset_coords(non_idx, drop=True)
            plev_groups.append((v, lev_map, da))

    var_names = [k for k in VAR_ORDER if k in
                 set(surf_parts.keys()) | {ch for _, lm, _ in plev_groups for ch in lm.values()}]
    ch_idx = {name: i for i, name in enumerate(var_names)}
    n_feat = len(var_names)
    surf_names = [n for n in var_names if n in surf_parts]
    print(f"Channels ({n_feat}): {var_names}")

    ref = surf_parts[next(k for k in var_names if k in surf_parts)]
    n_time = ref.sizes["time"]
    n_lon = ref.sizes["longitude"]
    n_lat = ref.sizes["latitude"]
    print(f"Shape: {n_time} timesteps × {n_lon} × {n_lat}")

    # Welford accumulators
    total_n = 0
    running_mean = np.zeros(n_feat, dtype=np.float64)
    running_m2 = np.zeros(n_feat, dtype=np.float64)

    t0 = time.time()
    for t_start in range(0, n_time, args.time_chunk):
        t_end = min(t_start + args.time_chunk, n_time)
        chunk_steps = t_end - t_start
        pct = t_end / n_time * 100

        elapsed = time.time() - t0
        speed = t_end / max(elapsed, 1)
        eta = (n_time - t_end) / max(speed, 0.01) / 60
        print(f"  [{t_start:>6d}–{t_end:>6d} / {n_time}]  {pct:5.1f}%  ETA {eta:.0f} min")

        t_slice = slice(t_start, t_end)
        block_sum = np.zeros(n_feat, dtype=np.float64)
        block_sumsq = np.zeros(n_feat, dtype=np.float64)

        # Surface vars
        lazy = [surf_parts[name].isel(time=t_slice) for name in surf_names]
        results = dask.compute(*lazy, scheduler="threads")
        for arr_xr, name in zip(results, surf_names):
            j = ch_idx[name]
            arr = np.asarray(arr_xr, dtype=np.float32)
            if name in SCALE_FACTORS:
                arr *= SCALE_FACTORS[name]
            block_sum[j] = arr.sum(dtype=np.float64)
            block_sumsq[j] = (arr * arr).sum(dtype=np.float64)
            del arr
        del results

        # Plev vars
        for src_var, lev_map, da in plev_groups:
            block_3d = da.isel(time=t_slice).values.astype(np.float32, copy=False)
            levels_arr = da.level.values
            for li, lev_val in enumerate(levels_arr):
                ch_name = lev_map[int(lev_val)]
                if ch_name not in ch_idx:
                    continue
                j = ch_idx[ch_name]
                arr_2d = block_3d[:, li, :, :]
                if ch_name in SCALE_FACTORS:
                    arr_2d = arr_2d * SCALE_FACTORS[ch_name]
                block_sum[j] = arr_2d.sum(dtype=np.float64)
                block_sumsq[j] = (arr_2d * arr_2d).sum(dtype=np.float64)
            del block_3d

        gc.collect()

        block_n = chunk_steps * n_lon * n_lat
        running_mean, running_m2, total_n = welford_update(
            running_mean, running_m2, total_n, block_sum, block_sumsq, block_n
        )

    std = np.sqrt(running_m2 / max(total_n, 1))
    std = np.maximum(std, 1e-6)
    data_mean = running_mean.astype(np.float32)
    data_std = std.astype(np.float32)

    np.savez(args.out, mean=data_mean, std=data_std, n=total_n)

    total_min = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"✓ Saved {args.out}  (n={total_n}, {total_min:.1f} min)")
    for i, name in enumerate(var_names):
        print(f"  {name:>8s}:  mean={data_mean[i]:+12.5f}  std={data_std[i]:12.5f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

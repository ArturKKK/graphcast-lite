#!/usr/bin/env python3
"""
scripts/build_region_russia_19f.py

Скачивает мелкую 0.25° сетку ERA5 для прямоугольника России (bbox задаётся
флагами) из WeatherBench2 zarr `1959-2022-6h-1440x721.zarr` и сохраняет в
chunked-формате (data.npy memmap + scalers/coords/variables/dataset_info),
полностью совместимом с `build_multires_dataset.py --mode merge`.

Аналог `build_dataset_512x256_19f.py`, но:
  - источник 1440x721 (вместо 512x256);
  - выход — только подсетка bbox;
  - 19 каналов (тот же VAR_ORDER, что у russia_19f).

Статические каналы (z_surf, lsm) при необходимости берутся ИЗ ГЛОБАЛЬНОГО
датасета (--static-from) и интерполируются на 0.25° региональную сетку
ближайшим соседом, т.к. в WB2 1440x721 их обычно нет.

Пример (вся Россия, 5 лет):
  python scripts/build_region_russia_19f.py \
      --out-dir data/datasets/region_russia_645x165_19f_2017-2021_025deg \
      --start-year 2017 --end-year 2022 \
      --lon-min 19 --lon-max 180 --lat-min 41 --lat-max 82 \
      --static-from data/datasets/global_512x256_19f_2010-2021_07deg

Размеры (lat 41-82, lon 19-180): 645x165 = 106 425 узлов
  3y → ~16 GB, 5y → ~28 GB, 7y → ~38 GB, 12y → ~66 GB
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

WB2_ZARR_FINE = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

SURF_DYNAMIC = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_column_water_vapour",
]

# В 1440x721 zarr статиков обычно нет → берём из --static-from
STATIC_VARS = ["geopotential_at_surface", "land_sea_mask"]

PLEV_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "specific_humidity",
]
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

VAR_ORDER = [
    "t2m", "10u", "10v", "msl", "tp",
    "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
]
assert len(VAR_ORDER) == 19

SCALE_FACTORS = {
    "msl":    0.01,
    "sp":     0.01,
    "z_surf": 1 / 9.80665,
    "z@850":  1 / 9.80665,
    "z@500":  1 / 9.80665,
}


def open_wb2_fine(time_start, time_end, lon_min, lon_max, lat_min, lat_max):
    print(f"[INFO] Opening {WB2_ZARR_FINE}")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB2_ZARR_FINE)
    ds = xr.open_zarr(store, consolidated=True)
    print(f"[INFO] Time range: {time_start} → {time_end}")
    print(f"[INFO] Bbox: lon [{lon_min},{lon_max}], lat [{lat_min},{lat_max}]")
    ds = ds.sel(time=slice(time_start, time_end))
    # WB2 latitude descends → slice(max, min)
    ds = ds.sel(longitude=slice(lon_min, lon_max),
                latitude=slice(lat_max, lat_min))
    print(f"[INFO] Region dims: {dict(ds.sizes)}")
    if "level" in ds.dims:
        print(f"[INFO] Levels: {ds.level.values.tolist()}")
    return ds


def resolve_tp_name(ds):
    for cand in ["total_precipitation_6hr", "total_precipitation", "tp"]:
        if cand in ds.data_vars:
            return cand
    raise RuntimeError("Precipitation var not found in zarr!")


def save_progress(out_dir: Path, t_end: int, chunk_size: int):
    (out_dir / "progress.json").write_text(json.dumps({
        "last_completed_timestep": t_end,
        "chunk_size": chunk_size,
    }))


def load_progress(out_dir: Path):
    p = out_dir / "progress.json"
    if p.exists():
        return json.loads(p.read_text()).get("last_completed_timestep", None)
    return None


def welford_update(running_mean, running_m2, total_n, block_sum, block_sumsq, block_n):
    block_mean = block_sum / block_n
    block_var = np.maximum(block_sumsq / block_n - block_mean ** 2, 0.0)
    delta = block_mean - running_mean
    new_n = total_n + block_n
    running_mean += delta * (block_n / new_n)
    running_m2 += block_var * block_n + (delta ** 2) * total_n * block_n / new_n
    return running_mean, running_m2, new_n


def load_static_from_global(static_name: str, global_dir: Path,
                            target_lons: np.ndarray, target_lats: np.ndarray):
    """Берём статический канал из глобального chunked-датасета и интерполируем
    ближайшим соседом на региональную сетку. Возвращает (n_lon, n_lat) float32."""
    info = json.loads((global_dir / "dataset_info.json").read_text())
    var_idx = info["variables"].index(static_name)
    shape = (info["n_time"], info["n_lon"], info["n_lat"], info["n_feat"])
    mm = np.memmap(global_dir / "data.npy", dtype=np.float16, mode="r", shape=shape)
    field = mm[0, :, :, var_idx].astype(np.float32)  # (g_lon, g_lat)
    coords = np.load(global_dir / "coords.npz")
    g_lons = coords["longitude"].astype(np.float64)
    g_lats = coords["latitude"].astype(np.float64)

    # Нужны строго возрастающие оси для RegularGridInterpolator
    lons_sorted = np.argsort(g_lons)
    lats_sorted = np.argsort(g_lats)
    field_sorted = field[lons_sorted][:, lats_sorted]
    g_lons_s = g_lons[lons_sorted]
    g_lats_s = g_lats[lats_sorted]

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (g_lons_s, g_lats_s), field_sorted,
        method="nearest", bounds_error=False, fill_value=None,
    )
    mesh_lon, mesh_lat = np.meshgrid(
        target_lons.astype(np.float64),
        target_lats.astype(np.float64),
        indexing="ij",
    )
    out = interp((mesh_lon, mesh_lat)).astype(np.float32)
    del mm
    return out  # (n_lon, n_lat)


def build_channel_parts(ds, out_lons, out_lats, static_from: Path | None):
    """Сетим surf_parts (DataArray) + plev_groups + статику (precomputed 2D)."""
    surf_parts = {}
    for v in SURF_DYNAMIC:
        if v not in ds.data_vars:
            print(f"[WARN] {v} not in zarr — skip")
            continue
        surf_parts[RENAME[v]] = ds[v].transpose("time", "longitude", "latitude")

    tp_name = resolve_tp_name(ds)
    surf_parts["tp"] = ds[tp_name].transpose("time", "longitude", "latitude")

    static_2d = {}
    for sv in STATIC_VARS:
        if sv in ds.data_vars:
            da = ds[sv]
            if "time" in da.dims:
                da = da.isel(time=0)
            arr = da.transpose("longitude", "latitude").values.astype(np.float32)
            static_2d[RENAME[sv]] = arr
            print(f"[STAT] {RENAME[sv]} взят из zarr: {arr.shape}")
        elif static_from is not None:
            print(f"[STAT] {RENAME[sv]}: интерполируем из {static_from}")
            arr = load_static_from_global(sv, static_from, out_lons, out_lats)
            static_2d[RENAME[sv]] = arr
            print(f"[STAT]  → {arr.shape}  range=[{arr.min():.3f}, {arr.max():.3f}]")
        else:
            raise RuntimeError(
                f"Static var {sv} not in zarr and --static-from не задан"
            )

    plev_groups = []
    for v in PLEV_VARS:
        if v not in ds.data_vars:
            print(f"[WARN] {v} not in zarr — skip")
            continue
        short = RENAME[v]
        lev_map = {lev: f"{short}@{lev}" for lev in LEVELS}
        da = ds[v].sel(level=LEVELS).transpose("time", "level", "longitude", "latitude")
        non_idx = [c for c in da.coords if c not in da.dims]
        if non_idx:
            da = da.reset_coords(non_idx, drop=True)
        plev_groups.append((v, lev_map, da))

    all_channels = set(surf_parts) | set(static_2d)
    for _, lev_map, _ in plev_groups:
        all_channels.update(lev_map.values())
    var_names = [k for k in VAR_ORDER if k in all_channels]
    missing = set(VAR_ORDER) - all_channels
    if missing:
        raise RuntimeError(f"Не хватает каналов: {missing}")
    print(f"[VARS] {len(var_names)} channels: {var_names}")
    return surf_parts, static_2d, plev_groups, var_names


def download_loop(surf_parts, static_2d, plev_groups, var_names,
                  out_dir: Path, time_chunk: int, resume_from):
    first = next(k for k in var_names if k in surf_parts)
    ref = surf_parts[first]
    n_time = ref.sizes["time"]
    n_lon = ref.sizes["longitude"]
    n_lat = ref.sizes["latitude"]
    n_feat = len(var_names)
    ch_idx = {n: i for i, n in enumerate(var_names)}
    surf_names = [n for n in var_names if n in surf_parts]
    static_names = [n for n in var_names if n in static_2d]

    out_path = out_dir / "data.npy"
    est_gb = n_time * n_lon * n_lat * n_feat * 2 / (1024 ** 3)
    print(f"[SAVE] {out_path}  shape=({n_time},{n_lon},{n_lat},{n_feat}) ≈ {est_gb:.1f} GB")

    actual_start = resume_from if (resume_from and resume_from > 0) else 0
    fp = np.memmap(out_path, dtype=np.float16,
                   mode="r+" if actual_start > 0 else "w+",
                   shape=(n_time, n_lon, n_lat, n_feat))

    # Записываем статику сразу для всех t (тиражируем 2D)
    if actual_start == 0:
        for sn in static_names:
            j = ch_idx[sn]
            arr2d = static_2d[sn].copy()
            if sn in SCALE_FACTORS:
                arr2d *= SCALE_FACTORS[sn]
            arr2d_f16 = arr2d.astype(np.float16)
            # broadcast write
            for t_s in range(0, n_time, time_chunk):
                t_e = min(t_s + time_chunk, n_time)
                fp[t_s:t_e, :, :, j] = arr2d_f16
            print(f"[STAT] записан {sn} (broadcast по времени)")
        fp.flush()

    total_n = 0
    running_mean = np.zeros(n_feat, dtype=np.float64)
    running_m2 = np.zeros(n_feat, dtype=np.float64)

    if actual_start > 0:
        print(f"[RESUME] Re-stat {actual_start} steps from disk...")
        t0 = time.time()
        for t_s in range(0, actual_start, time_chunk):
            t_e = min(t_s + time_chunk, actual_start)
            block_n = (t_e - t_s) * n_lon * n_lat
            bs = np.zeros(n_feat, dtype=np.float64)
            bsq = np.zeros(n_feat, dtype=np.float64)
            for j in range(n_feat):
                arr = np.nan_to_num(fp[t_s:t_e, :, :, j].astype(np.float32), nan=0.0)
                bs[j] = arr.sum(dtype=np.float64)
                bsq[j] = (arr * arr).sum(dtype=np.float64)
                del arr
            running_mean, running_m2, total_n = welford_update(
                running_mean, running_m2, total_n, bs, bsq, block_n
            )
            gc.collect()
        print(f"[RESUME] stat done in {time.time()-t0:.0f}s")

    download_t0 = time.time()
    downloaded = 0
    for t_start in range(actual_start, n_time, time_chunk):
        t_end = min(t_start + time_chunk, n_time)
        chunk_steps = t_end - t_start
        pct = t_end / n_time * 100

        if downloaded > 0:
            elapsed = time.time() - download_t0
            speed = downloaded / max(elapsed, 1)
            eta = (n_time - t_end) / speed / 60
            print(f"\n  [{t_start:>6d}–{t_end:>6d}/{n_time}] {pct:5.1f}%  "
                  f"ETA {eta:.0f} min ({speed:.1f} steps/s)")
        else:
            print(f"\n  [{t_start:>6d}–{t_end:>6d}/{n_time}] {pct:5.1f}%  starting...")

        t_slice = slice(t_start, t_end)
        block_sum = np.zeros(n_feat, dtype=np.float64)
        block_sumsq = np.zeros(n_feat, dtype=np.float64)

        # surface (parallel)
        ts0 = time.time()
        lazy = [surf_parts[n].isel(time=t_slice) for n in surf_names]
        results = dask.compute(*lazy, scheduler="threads")
        dt_surf = time.time() - ts0
        for arr_xr, name in zip(results, surf_names):
            j = ch_idx[name]
            arr = np.asarray(arr_xr, dtype=np.float32)
            if name in SCALE_FACTORS:
                arr *= SCALE_FACTORS[name]
            fp[t_start:t_end, :, :, j] = arr.astype(np.float16)
            block_sum[j] = arr.sum(dtype=np.float64)
            block_sumsq[j] = (arr * arr).sum(dtype=np.float64)
            del arr
        surf_mb = sum(r.nbytes for r in results) / 1e6
        print(f"    [surf] {len(surf_names)} vars: {dt_surf:.1f}s "
              f"({surf_mb:.0f} MB, {surf_mb/max(dt_surf,0.1):.1f} MB/s)")
        del results

        # plev
        for src_var, lev_map, da in plev_groups:
            short = RENAME[src_var]
            ts1 = time.time()
            block_3d = da.isel(time=t_slice).values.astype(np.float32, copy=False)
            levels_arr = da.level.values
            dt_dl = time.time() - ts1
            mb = block_3d.nbytes / 1e6
            for li, lev_val in enumerate(levels_arr):
                ch_name = lev_map[int(lev_val)]
                if ch_name not in ch_idx:
                    continue
                j = ch_idx[ch_name]
                arr2d = block_3d[:, li, :, :]
                if ch_name in SCALE_FACTORS:
                    arr2d = arr2d * SCALE_FACTORS[ch_name]
                fp[t_start:t_end, :, :, j] = arr2d.astype(np.float16)
                block_sum[j] = arr2d.sum(dtype=np.float64)
                block_sumsq[j] = (arr2d * arr2d).sum(dtype=np.float64)
            print(f"    [plev] {short}@{[int(l) for l in levels_arr]}: "
                  f"{dt_dl:.1f}s ({mb:.0f} MB, {mb/max(dt_dl,0.1):.1f} MB/s)")
            del block_3d

        # статику добавляем в block_sum через сохранённую плоскость (она уже на диске)
        for sn in static_names:
            j = ch_idx[sn]
            arr = fp[t_start:t_end, :, :, j].astype(np.float32)
            block_sum[j] = arr.sum(dtype=np.float64)
            block_sumsq[j] = (arr * arr).sum(dtype=np.float64)
            del arr

        gc.collect()
        fp.flush()

        block_n = chunk_steps * n_lon * n_lat
        running_mean, running_m2, total_n = welford_update(
            running_mean, running_m2, total_n, block_sum, block_sumsq, block_n
        )
        downloaded += chunk_steps
        save_progress(out_dir, t_end, time_chunk)

    fp.flush()
    del fp
    elapsed = time.time() - download_t0
    print(f"\n[DONE] data.npy ({est_gb:.1f} GB) in {elapsed/60:.0f} min")

    std = np.maximum(np.sqrt(running_m2 / max(total_n, 1)), 1e-6)
    return (running_mean.astype(np.float32), std.astype(np.float32),
            total_n, n_time, n_lon, n_lat)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year",   type=int, default=2022, help="exclusive")
    p.add_argument("--lon-min", type=float, default=19.0)
    p.add_argument("--lon-max", type=float, default=180.0)
    p.add_argument("--lat-min", type=float, default=41.0)
    p.add_argument("--lat-max", type=float, default=82.0)
    p.add_argument("--time-chunk", type=int, default=200,
                   help="меньше чем у глобала: чанк по 0.25° тяжелее")
    p.add_argument("--static-from", type=str, default=None,
                   help="Глобальный chunked датасет, откуда брать z_surf/lsm "
                        "(например: data/datasets/global_512x256_19f_2010-2021_07deg)")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_from = Path(args.static_from) if args.static_from else None
    if static_from is not None and not (static_from / "dataset_info.json").exists():
        raise SystemExit(f"--static-from {static_from}: нет dataset_info.json")

    time_start = f"{args.start_year}-01-01"
    time_end   = f"{args.end_year - 1}-12-31"

    print("=" * 70)
    print("BUILD REGION DATASET — Russia 0.25° — WB2 1440x721")
    print(f"  Period:  {args.start_year}-{args.end_year - 1}")
    print(f"  Bbox:    lon [{args.lon_min},{args.lon_max}]  lat [{args.lat_min},{args.lat_max}]")
    print(f"  Output:  {out_dir}")
    print("=" * 70)

    ds = open_wb2_fine(time_start, time_end,
                       args.lon_min, args.lon_max, args.lat_min, args.lat_max)

    # Подготовим целевые координаты (нужны для интерполяции статики)
    out_lons = ds.longitude.values.astype(np.float32)
    out_lats = ds.latitude.values.astype(np.float32)

    surf_parts, static_2d, plev_groups, var_names = build_channel_parts(
        ds, out_lons, out_lats, static_from
    )

    n_time_est = surf_parts[next(k for k in var_names if k in surf_parts)].sizes["time"]
    n_lon_est = len(out_lons); n_lat_est = len(out_lats)
    est_gb = n_time_est * n_lon_est * n_lat_est * len(var_names) * 2 / (1024 ** 3)
    print(f"\n[INFO] {n_time_est} steps × {n_lon_est}×{n_lat_est} × {len(var_names)} feat "
          f"= {est_gb:.1f} GB float16")

    resume_from = None
    if args.resume:
        saved = load_progress(out_dir)
        if saved and saved > 0:
            resume_from = max(0, saved - args.time_chunk)
            print(f"[RESUME] last_completed={saved} → restart from {resume_from}")

    mean, std, total_n, n_time, n_lon, n_lat = download_loop(
        surf_parts, static_2d, plev_groups, var_names,
        out_dir, args.time_chunk, resume_from,
    )

    np.savez(out_dir / "scalers.npz", mean=mean, std=std, n=total_n)
    np.savez(out_dir / "coords.npz",
             longitude=out_lons, latitude=out_lats)
    (out_dir / "variables.json").write_text(
        json.dumps(var_names, indent=2, ensure_ascii=False))
    info = {
        "time_start": time_start, "time_end": time_end,
        "n_time": int(n_time), "n_lon": int(n_lon), "n_lat": int(n_lat),
        "n_feat": len(var_names), "variables": var_names,
        "dtype": "float16", "file": "data.npy",
        "size_gb": round(est_gb, 1),
        "bbox": {
            "lon_min": args.lon_min, "lon_max": args.lon_max,
            "lat_min": args.lat_min, "lat_max": args.lat_max,
        },
        "source_zarr": WB2_ZARR_FINE,
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))

    progress_path = out_dir / "progress.json"
    if progress_path.exists():
        progress_path.unlink()

    print("=" * 70)
    print(f"✓ Region dataset ready: {out_dir}")
    print(f"  data.npy ≈ {est_gb:.1f} GB, {n_time} × {n_lon}×{n_lat} × {len(var_names)}")
    print()
    print("Next: build merge multires dataset")
    print(f"  python scripts/build_multires_dataset.py \\")
    print(f"      --global-dir <global_19f_dir> \\")
    print(f"      --region-dir {out_dir} \\")
    print(f"      --roi {args.lat_min} {args.lat_max} {args.lon_min} {args.lon_max} \\")
    print(f"      --mode merge \\")
    print(f"      --out-dir data/datasets/multires_russia_19f_merge")
    print("=" * 70)


if __name__ == "__main__":
    main()

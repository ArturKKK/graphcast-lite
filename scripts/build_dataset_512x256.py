#!/usr/bin/env python3
"""
scripts/build_dataset_512x256.py
Сборка датасета 512×256 (≈0.7°) из WeatherBench2 ERA5 Zarr (6h).

Ключевые решения:
- Читаем каждый канал отдельно → пишем в memmap по feature (нет concat-зависаний)
- Plev-переменные группируем по source var (один .values → оба уровня)
- Surface vars качаем параллельно через dask.compute()
- Прогресс пишем в progress.json → надёжный resume с перекачкой последнего чанка
- Статистику при resume считаем из memmap (с диска, без сети)
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

SURF_DYNAMIC = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_column_water_vapour",
]

STATIC_VARS = [
    "geopotential_at_surface",
    "land_sea_mask",
]

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

# float16 max = 65504.  Давления (Pa) и геопотенциал (м²/с²) превышают этот предел.
# Масштабируем ДО cast-а в float16.
SCALE_FACTORS = {
    "msl":    0.01,        # Pa  → hPa   (~101325 → ~1013)
    "sp":     0.01,        # Pa  → hPa   (~98000  → ~980)
    "z_surf": 1/9.80665,   # м²/с² → м   (~86740  → ~8845)
    "z@850":  1/9.80665,   # м²/с² → м   (~13500  → ~1377)
    "z@500":  1/9.80665,   # м²/с² → м   (~54000  → ~5507)
}

# ─── helpers ───────────────────────────────────────────────────────────

def open_wb2(time_start: str, time_end: str) -> xr.Dataset:
    print(f"[INFO] Opening {WB2_ZARR}")
    print(f"[INFO] Time range: {time_start} → {time_end}")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB2_ZARR)
    ds = xr.open_zarr(store, consolidated=True)
    ds = ds.sel(time=slice(time_start, time_end))
    print(f"[INFO] Dims: {dict(ds.dims)}")
    if "level" in ds.dims:
        print(f"[INFO] Levels available: {ds.level.values.tolist()}")
    return ds


def resolve_tp_name(ds: xr.Dataset) -> str:
    for cand in ["total_precipitation_6hr", "total_precipitation", "tp"]:
        if cand in ds.data_vars:
            return cand
    raise RuntimeError("Precipitation variable not found!")


def save_progress(out_dir: Path, t_end: int, chunk_size: int):
    """Записываем последний полностью скачанный таймстеп."""
    (out_dir / "progress.json").write_text(json.dumps({
        "last_completed_timestep": t_end,
        "chunk_size": chunk_size,
    }))


def load_progress(out_dir: Path) -> int | None:
    """Читаем последний полностью скачанный таймстеп из progress.json."""
    p = out_dir / "progress.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d.get("last_completed_timestep", None)
    return None


# ─── welford helpers ───────────────────────────────────────────────────

def welford_update(running_mean, running_m2, total_n, block_sum, block_sumsq, block_n):
    """Обновить running Welford stats по блоку sum/sumsq."""
    block_mean = block_sum / block_n
    block_var = block_sumsq / block_n - block_mean ** 2
    block_var = np.maximum(block_var, 0.0)

    delta = block_mean - running_mean
    new_n = total_n + block_n
    running_mean += delta * (block_n / new_n)
    running_m2 += block_var * block_n + (delta ** 2) * total_n * block_n / new_n
    return running_mean, running_m2, new_n


# ─── build channel parts ──────────────────────────────────────────────

def build_channel_parts(ds: xr.Dataset):
    """
    Возвращаем:
      surf_parts  — dict[channel_name] -> DataArray(time, lon, lat)
      plev_groups — list[ (source_var, {lev: ch_name}, DataArray(time,level,lon,lat)) ]
      var_names   — canonical order
    """
    surf_parts = {}

    for v in SURF_DYNAMIC:
        if v not in ds.data_vars:
            print(f"[WARN] Surface var {v} not found! Skipping.")
            continue
        surf_parts[RENAME[v]] = ds[v].transpose("time", "longitude", "latitude")

    tp_name = resolve_tp_name(ds)
    surf_parts["tp"] = ds[tp_name].transpose("time", "longitude", "latitude")

    for v in STATIC_VARS:
        if v not in ds.data_vars:
            print(f"[WARN] Static var {v} not found!")
            continue
        da = ds[v]
        if "time" not in da.dims:
            da = da.expand_dims(time=ds.time)
        surf_parts[RENAME[v]] = da.transpose("time", "longitude", "latitude")

    plev_groups = []
    for v in PLEV_VARS:
        if v not in ds.data_vars:
            print(f"[WARN] Skipping {v} (not in zarr)")
            continue
        short = RENAME[v]
        lev_map = {lev: f"{short}@{lev}" for lev in LEVELS}
        da = ds[v].sel(level=LEVELS).transpose("time", "level", "longitude", "latitude")
        non_idx = [c for c in da.coords if c not in da.dims]
        if non_idx:
            da = da.reset_coords(non_idx, drop=True)
        plev_groups.append((v, lev_map, da))

    all_channels = set(surf_parts.keys())
    for _, lev_map, _ in plev_groups:
        all_channels.update(lev_map.values())

    var_names = [k for k in VAR_ORDER if k in all_channels]
    missing = set(VAR_ORDER) - all_channels
    if missing:
        print(f"[WARN] Missing: {missing}")
    print(f"[VARS] {len(var_names)} channels: {var_names}")
    return surf_parts, plev_groups, var_names


# ─── main download loop ───────────────────────────────────────────────

def download_compute_save(
    surf_parts: dict,
    plev_groups: list,
    var_names: list,
    out_dir: Path,
    time_chunk: int = 500,
    resume_from: int | None = None,
):
    first_key = next(k for k in var_names if k in surf_parts)
    ref = surf_parts[first_key]
    n_time = ref.sizes["time"]
    n_lon = ref.sizes["longitude"]
    n_lat = ref.sizes["latitude"]
    n_feat = len(var_names)

    ch_idx = {name: i for i, name in enumerate(var_names)}

    # Имена surface каналов в порядке var_names
    surf_names = [n for n in var_names if n in surf_parts]

    out_path = out_dir / "data.npy"
    estimated_gb = n_time * n_lon * n_lat * n_feat * 2 / (1024 ** 3)
    print(f"[SAVE] Target: {out_path}")
    print(f"[SAVE] Shape: ({n_time}, {n_lon}, {n_lat}, {n_feat}) float16 ≈ {estimated_gb:.1f} GB")

    # --- memmap ---
    actual_start = resume_from if (resume_from and resume_from > 0) else 0
    fp = np.memmap(
        str(out_path), dtype=np.float16,
        mode="r+" if actual_start > 0 else "w+",
        shape=(n_time, n_lon, n_lat, n_feat),
    )

    # --- welford accumulators ---
    total_n = 0
    running_mean = np.zeros(n_feat, dtype=np.float64)
    running_m2 = np.zeros(n_feat, dtype=np.float64)

    # --- Если resume: пересчитать статистику по уже записанным данным ---
    if actual_start > 0:
        print(f"[RESUME] Re-computing stats for timesteps 0–{actual_start} from disk...")
        stat_t0 = time.time()

        for t_s in range(0, actual_start, time_chunk):
            t_e = min(t_s + time_chunk, actual_start)
            block_n = (t_e - t_s) * n_lon * n_lat
            block_sum = np.zeros(n_feat, dtype=np.float64)
            block_sumsq = np.zeros(n_feat, dtype=np.float64)

            for j in range(n_feat):
                arr = np.nan_to_num(fp[t_s:t_e, :, :, j].astype(np.float32), nan=0.0)
                block_sum[j] = arr.sum(dtype=np.float64)
                block_sumsq[j] = (arr * arr).sum(dtype=np.float64)
                del arr

            running_mean, running_m2, total_n = welford_update(
                running_mean, running_m2, total_n, block_sum, block_sumsq, block_n
            )
            gc.collect()

        stat_dt = time.time() - stat_t0
        print(f"[RESUME] Stats done in {stat_dt:.0f}s. Continuing from timestep {actual_start}...")

    # --- Основной цикл скачивания ---
    download_t0 = time.time()
    downloaded_steps = 0  # сколько шагов скачали В ЭТОМ запуске (для ETA)

    for t_start in range(actual_start, n_time, time_chunk):
        t_end = min(t_start + time_chunk, n_time)
        chunk_steps = t_end - t_start
        pct = t_end / n_time * 100

        # ETA считаем только по скачанным В ЭТОМ запуске шагам
        elapsed = time.time() - download_t0
        if downloaded_steps > 0:
            speed = downloaded_steps / max(elapsed, 1)
            remaining = n_time - t_end
            eta_min = remaining / speed / 60
            print(f"\n  [{t_start:>6d}–{t_end:>6d} / {n_time}]  {pct:5.1f}%  "
                  f"ETA {eta_min:.0f} min  ({speed:.1f} steps/s)")
        else:
            print(f"\n  [{t_start:>6d}–{t_end:>6d} / {n_time}]  {pct:5.1f}%  starting...")

        t_slice = slice(t_start, t_end)
        block_sum = np.zeros(n_feat, dtype=np.float64)
        block_sumsq = np.zeros(n_feat, dtype=np.float64)

        # ── 1) Surface vars: параллельная загрузка через dask.compute() ──
        ts0 = time.time()
        lazy = [surf_parts[name].isel(time=t_slice) for name in surf_names]
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
        print(f"    [surf 1–{len(surf_names):02d}] {len(surf_names)} vars parallel: "
              f"{dt_surf:.1f}s  ({surf_mb:.0f} MB, {surf_mb/max(dt_surf,0.1):.1f} MB/s)")
        del results

        # ── 2) Plev vars: группой (один .values → оба уровня) ──
        for src_var, lev_map, da in plev_groups:
            short = RENAME[src_var]
            ts1 = time.time()

            block_3d = da.isel(time=t_slice).values.astype(np.float32, copy=False)
            levels_arr = da.level.values
            dt_dl = time.time() - ts1

            plev_mb = block_3d.nbytes / 1e6
            for li, lev_val in enumerate(levels_arr):
                ch_name = lev_map[int(lev_val)]
                if ch_name not in ch_idx:
                    continue
                j = ch_idx[ch_name]
                arr_2d = block_3d[:, li, :, :]
                if ch_name in SCALE_FACTORS:
                    arr_2d = arr_2d * SCALE_FACTORS[ch_name]

                fp[t_start:t_end, :, :, j] = arr_2d.astype(np.float16)
                block_sum[j] = arr_2d.sum(dtype=np.float64)
                block_sumsq[j] = (arr_2d * arr_2d).sum(dtype=np.float64)

            print(f"    [plev] {short}@{list(lev_map.keys())}: "
                  f"{dt_dl:.1f}s  ({plev_mb:.0f} MB, {plev_mb/max(dt_dl,0.1):.1f} MB/s)")
            del block_3d

        gc.collect()
        fp.flush()

        # ── Welford update ──
        block_n = chunk_steps * n_lon * n_lat
        running_mean, running_m2, total_n = welford_update(
            running_mean, running_m2, total_n, block_sum, block_sumsq, block_n
        )

        downloaded_steps += chunk_steps

        # ── Сохраняем прогресс ──
        save_progress(out_dir, t_end, time_chunk)

    fp.flush()
    del fp

    elapsed_total = time.time() - download_t0
    print(f"\n[DONE] ✓ data.npy ({estimated_gb:.1f} GB)  in {elapsed_total / 60:.0f} min")

    std = np.sqrt(running_m2 / max(total_n, 1))
    std = np.maximum(std, 1e-6)

    return running_mean.astype(np.float32), std.astype(np.float32), total_n, n_time, n_lon, n_lat


# ─── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build 512×256 dataset (19 vars) from WeatherBench2 ERA5."
    )
    parser.add_argument("--out-dir", type=str, default="data/datasets/wb2_512x256_19f_ar")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2022,
                        help="End year (exclusive)")
    parser.add_argument("--time-chunk", type=int, default=500,
                        help="Timesteps per download batch")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from progress.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    time_start = f"{args.start_year}-01-01"
    time_end = f"{args.end_year - 1}-12-31"

    print("=" * 70)
    print("BUILD DATASET — 512×256 — ERA5 — WeatherBench2")
    print(f"  Period:   {args.start_year}–{args.end_year - 1}")
    print(f"  Grid:     512 × 256")
    print(f"  Features: {len(VAR_ORDER)} channels")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    ds = open_wb2(time_start, time_end)
    surf_parts, plev_groups, var_names = build_channel_parts(ds)

    n_time = surf_parts[next(k for k in var_names if k in surf_parts)].sizes["time"]
    est_gb = n_time * 512 * 256 * len(var_names) * 2 / (1024 ** 3)
    print(f"\n[INFO] {n_time} timesteps × {len(var_names)} vars = {est_gb:.1f} GB (float16)")

    # --- Resume ---
    resume_from = None
    if args.resume:
        saved = load_progress(out_dir)
        if saved is not None and saved > 0:
            # Откатываем на 1 чанк назад — перекачиваем последний (мог быть неполным)
            safe_start = max(0, saved - args.time_chunk)
            print(f"\n[RESUME] progress.json: last_completed={saved}")
            print(f"[RESUME] Re-downloading from {safe_start} (1 chunk back for safety)")
            resume_from = safe_start
        elif (out_dir / "data.npy").exists():
            print(f"\n[RESUME] No progress.json, but data.npy exists.")
            print(f"[RESUME] Starting from 0 (will overwrite).")
        else:
            print(f"\n[RESUME] Nothing to resume, starting fresh.")

    print("\n[RUN] Downloading...")
    data_mean, data_std, total_n, n_time, n_lon, n_lat = download_compute_save(
        surf_parts, plev_groups, var_names, out_dir,
        time_chunk=args.time_chunk, resume_from=resume_from,
    )

    # --- Save metadata ---
    np.savez(out_dir / "scalers.npz", mean=data_mean, std=data_std, n=total_n)
    print(f"[STAT] ✓ Saved scalers.npz (n={total_n})")
    for i, name in enumerate(var_names):
        print(f"  {name:>8s}:  mean={data_mean[i]:+10.3f}  std={data_std[i]:10.3f}")

    first_surf = surf_parts[next(k for k in var_names if k in surf_parts)]
    np.savez(out_dir / "coords.npz",
             longitude=first_surf.longitude.values.astype(np.float32),
             latitude=first_surf.latitude.values.astype(np.float32))

    (out_dir / "variables.json").write_text(
        json.dumps(var_names, indent=2, ensure_ascii=False)
    )

    dataset_info = {
        "time_start": time_start, "time_end": time_end,
        "n_time": int(n_time), "n_lon": int(n_lon), "n_lat": int(n_lat),
        "n_feat": len(var_names), "variables": var_names,
        "dtype": "float16", "file": "data.npy",
        "size_gb": round(est_gb, 1),
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))

    # Удаляем progress.json — скачка завершена
    progress_path = out_dir / "progress.json"
    if progress_path.exists():
        progress_path.unlink()

    print()
    print("=" * 70)
    print("✓ Dataset complete!")
    print(f"  {out_dir / 'data.npy'} — {est_gb:.1f} GB")
    print(f"  {len(var_names)} variables × {n_time} timesteps × 512×256 grid")
    print("=" * 70)


if __name__ == "__main__":
    main()

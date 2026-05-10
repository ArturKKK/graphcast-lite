#!/usr/bin/env python3
"""
scripts/extend_dataset_512x256_to_30f.py
In-place расширение существующего _19f до 29f путём ДОЗАГРУЗКИ 10 новых каналов
в отдельный файл `data_extra.npy` рядом с `data.npy`.

NB: 2m_dewpoint_temperature ОТСУТСТВУЕТ в WB2 zarr 512x256_equiangular_conservative
(там только 2m_temperature). Поэтому в extend идут только plev-каналы
z/t/u/v/q @ 250 и 1000 hPa — итого 10 новых каналов.

Старый data.npy НЕ ТРОГАЕМ (нет места на перекладку).
Loader (src/data/dataloader_chunked.py) умеет склеивать data.npy + data_extra.npy
по последней оси прозрачно для модели → датасет ведёт себя как (T, lon, lat, 30).

Каналы (idx → имя):
   0..18   уже в data.npy   (старые 19, не трогаем)
   19..23  z/t/u/v/q @250
   24..28  z/t/u/v/q @1000

Запуск (на ноуте):
  python scripts/extend_dataset_512x256_to_30f.py \\
      --base-dir data/datasets/global_512x256_19f_2010-2021_07deg
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

# Каналы, которые добавляем (в порядке индексов 19..28).
# 2m_dewpoint_temperature НЕ доступен в этом zarr — пропускаем.
NEW_SURF: list[str] = []
NEW_PLEV = [
    ("geopotential",         [250, 1000]),
    ("temperature",          [250, 1000]),
    ("u_component_of_wind",  [250, 1000]),
    ("v_component_of_wind",  [250, 1000]),
    ("specific_humidity",    [250, 1000]),
]

RENAME = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
}

# Финальный порядок 10 новых каналов в data_extra.npy:
NEW_VAR_ORDER = [
    "z@250", "t@250", "u@250", "v@250", "q@250",
    "z@1000", "t@1000", "u@1000", "v@1000", "q@1000",
]
assert len(NEW_VAR_ORDER) == 10

SCALE_FACTORS = {
    "z@250":  1 / 9.80665,
    "z@1000": 1 / 9.80665,
}


def open_wb2(time_start: str, time_end: str) -> xr.Dataset:
    print(f"[INFO] Opening {WB2_ZARR}")
    print(f"[INFO] Time range: {time_start} → {time_end}")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB2_ZARR)
    ds = xr.open_zarr(store, consolidated=True)
    ds = ds.sel(time=slice(time_start, time_end))
    print(f"[INFO] Dims: {dict(ds.dims)}")
    return ds


def welford_update(running_mean, running_m2, total_n, block_sum, block_sumsq, block_n):
    block_mean = block_sum / block_n
    block_var = block_sumsq / block_n - block_mean ** 2
    block_var = np.maximum(block_var, 0.0)
    delta = block_mean - running_mean
    new_n = total_n + block_n
    running_mean += delta * (block_n / new_n)
    running_m2 += block_var * block_n + (delta ** 2) * total_n * block_n / new_n
    return running_mean, running_m2, new_n


def save_progress(out_dir: Path, t_end: int, chunk_size: int):
    (out_dir / "progress_extra.json").write_text(json.dumps({
        "last_completed_timestep": t_end,
        "chunk_size": chunk_size,
    }))


def load_progress(out_dir: Path):
    p = out_dir / "progress_extra.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extend existing _19f dataset to 29f by downloading 10 extra channels into data_extra.npy"
    )
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Existing _19f dataset dir (contains data.npy, dataset_info.json, scalers.npz)")
    parser.add_argument("--time-chunk", type=int, default=500)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Auto-resume from progress_extra.json (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    info_path = base_dir / "dataset_info.json"
    if not info_path.exists():
        raise SystemExit(f"--base-dir {base_dir}: нет dataset_info.json")

    info = json.loads(info_path.read_text())
    n_time = int(info["n_time"])
    n_lon = int(info["n_lon"])
    n_lat = int(info["n_lat"])
    base_vars = info["variables"]
    n_base = len(base_vars)
    time_start = info["time_start"]
    time_end = info["time_end"]

    print("=" * 70)
    print("EXTEND DATASET → 29f (extra channels into data_extra.npy)")
    print(f"  Base:   {base_dir}  ({n_base} channels, {n_time} timesteps)")
    print(f"  Period: {time_start} → {time_end}")
    print(f"  Grid:   {n_lon} × {n_lat}")
    print(f"  Add:    {len(NEW_VAR_ORDER)} channels → {NEW_VAR_ORDER}")
    n_extra = len(NEW_VAR_ORDER)
    extra_gb = n_time * n_lon * n_lat * n_extra * 2 / (1024 ** 3)
    print(f"  Size:   data_extra.npy ≈ {extra_gb:.1f} GB")
    print("=" * 70)

    # Sanity: новые каналы не должны конфликтовать со старыми
    overlap = set(base_vars) & set(NEW_VAR_ORDER)
    if overlap:
        raise SystemExit(f"Overlap with base channels: {overlap} — abort.")

    # --- открываем zarr и фильтруем нужное ---
    ds = open_wb2(time_start, time_end)

    # sanity: длина по времени совпадает
    n_time_zarr = ds.sizes.get("time", n_time)
    if n_time_zarr != n_time:
        raise SystemExit(f"Time mismatch: base={n_time} vs zarr={n_time_zarr}")

    surf_parts = {}
    for v in NEW_SURF:
        if v not in ds.data_vars:
            raise SystemExit(f"[FATAL] Surface var {v} not in zarr — list: "
                             f"{[k for k in ds.data_vars if '2m' in k or 'dew' in k]}")
        surf_parts[RENAME[v]] = ds[v].transpose("time", "longitude", "latitude")

    plev_groups = []
    for src_var, levels in NEW_PLEV:
        if src_var not in ds.data_vars:
            raise SystemExit(f"[FATAL] Plev var {src_var} not in zarr")
        short = RENAME[src_var]
        lev_map = {lev: f"{short}@{lev}" for lev in levels}
        da = ds[src_var].sel(level=levels).transpose("time", "level", "longitude", "latitude")
        non_idx = [c for c in da.coords if c not in da.dims]
        if non_idx:
            da = da.reset_coords(non_idx, drop=True)
        plev_groups.append((src_var, lev_map, da))

    # --- создаём/открываем data_extra.npy ---
    extra_path = base_dir / "data_extra.npy"
    fresh = not extra_path.exists()
    fp = np.memmap(str(extra_path), dtype=np.float16,
                   mode="w+" if fresh else "r+",
                   shape=(n_time, n_lon, n_lat, n_extra))

    ch_idx = {name: i for i, name in enumerate(NEW_VAR_ORDER)}
    surf_names = [n for n in NEW_VAR_ORDER if n in surf_parts]
    print(f"[DL] surface (NEW): {surf_names}")
    for src_var, lev_map, _ in plev_groups:
        print(f"[DL] plev (NEW) {RENAME[src_var]}: {list(lev_map.values())}")

    # --- welford ---
    total_n = 0
    running_mean = np.zeros(n_extra, dtype=np.float64)
    running_m2 = np.zeros(n_extra, dtype=np.float64)

    # --- resume ---
    actual_start = 0
    progress = load_progress(base_dir) if args.resume else None
    if progress:
        saved_t = int(progress.get("last_completed_timestep", 0))
        actual_start = max(0, saved_t - args.time_chunk)
        print(f"[RESUME] last_completed={saved_t}, restart from {actual_start}")

        # пересчёт стат с диска для уже скачанных шагов
        if actual_start > 0:
            print(f"[RESUME] Re-computing stats for timesteps 0–{actual_start} from disk...")
            t0 = time.time()
            for t_s in range(0, actual_start, args.time_chunk):
                t_e = min(t_s + args.time_chunk, actual_start)
                block_n = (t_e - t_s) * n_lon * n_lat
                block_sum = np.zeros(n_extra, dtype=np.float64)
                block_sumsq = np.zeros(n_extra, dtype=np.float64)
                for j in range(n_extra):
                    arr = np.nan_to_num(fp[t_s:t_e, :, :, j].astype(np.float32), nan=0.0)
                    block_sum[j] = arr.sum(dtype=np.float64)
                    block_sumsq[j] = (arr * arr).sum(dtype=np.float64)
                    del arr
                running_mean, running_m2, total_n = welford_update(
                    running_mean, running_m2, total_n,
                    block_sum, block_sumsq, block_n
                )
                gc.collect()
            print(f"[RESUME] Stats done in {time.time()-t0:.0f}s")

    # --- основной download цикл ---
    download_t0 = time.time()
    downloaded_steps = 0

    for t_start in range(actual_start, n_time, args.time_chunk):
        t_end = min(t_start + args.time_chunk, n_time)
        chunk_steps = t_end - t_start
        pct = t_end / n_time * 100

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
        block_sum = np.zeros(n_extra, dtype=np.float64)
        block_sumsq = np.zeros(n_extra, dtype=np.float64)

        # surface
        if surf_names:
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
            print(f"    [surf] {surf_names}: {dt_surf:.1f}s "
                  f"({surf_mb:.0f} MB, {surf_mb/max(dt_surf,0.1):.1f} MB/s)")
            del results

        # plev
        for src_var, lev_map, da in plev_groups:
            short = RENAME[src_var]
            ts1 = time.time()
            block_3d = da.isel(time=t_slice).values.astype(np.float32, copy=False)
            levels_arr = da.level.values
            dt_dl = time.time() - ts1
            plev_mb = block_3d.nbytes / 1e6
            for li, lev_val in enumerate(levels_arr):
                ch_name = lev_map[int(lev_val)]
                j = ch_idx[ch_name]
                arr_2d = block_3d[:, li, :, :]
                if ch_name in SCALE_FACTORS:
                    arr_2d = arr_2d * SCALE_FACTORS[ch_name]
                fp[t_start:t_end, :, :, j] = arr_2d.astype(np.float16)
                block_sum[j] = arr_2d.sum(dtype=np.float64)
                block_sumsq[j] = (arr_2d * arr_2d).sum(dtype=np.float64)
            print(f"    [plev] {short}@{[int(l) for l in levels_arr]}: "
                  f"{dt_dl:.1f}s ({plev_mb:.0f} MB, {plev_mb/max(dt_dl,0.1):.1f} MB/s)")
            del block_3d

        gc.collect()
        fp.flush()

        block_n = chunk_steps * n_lon * n_lat
        running_mean, running_m2, total_n = welford_update(
            running_mean, running_m2, total_n, block_sum, block_sumsq, block_n
        )
        downloaded_steps += chunk_steps
        save_progress(base_dir, t_end, args.time_chunk)

    fp.flush()
    del fp

    elapsed_total = time.time() - download_t0
    print(f"\n[DONE] ✓ data_extra.npy ({extra_gb:.1f} GB) в {elapsed_total/60:.0f} min")

    std = np.sqrt(running_m2 / max(total_n, 1))
    std = np.maximum(std, 1e-6)
    extra_mean = running_mean.astype(np.float32)
    extra_std = std.astype(np.float32)

    # --- мерджим scalers: base + extra ---
    base_scalers = np.load(base_dir / "scalers.npz")
    base_mean = base_scalers["mean"].astype(np.float32)
    base_std = base_scalers["std"].astype(np.float32)
    base_n = int(base_scalers["n"]) if "n" in base_scalers.files else \
             n_time * n_lon * n_lat

    full_mean = np.concatenate([base_mean, extra_mean], axis=0)
    full_std = np.concatenate([base_std, extra_std], axis=0)
    # n берём из base (для extra то же значение, т.к. T*lon*lat одинаково)
    full_n = base_n

    # бэкап старого scalers и пишем новый
    backup_path = base_dir / "scalers_19f.npz"
    if not backup_path.exists():
        import shutil
        shutil.copy(base_dir / "scalers.npz", backup_path)
        print(f"[STAT] Backup → {backup_path}")

    np.savez(base_dir / "scalers.npz", mean=full_mean, std=full_std, n=full_n)
    print(f"[STAT] ✓ scalers.npz updated (n={full_n}, channels={len(full_mean)})")
    for i, name in enumerate(base_vars + NEW_VAR_ORDER):
        src = "base" if i < len(base_vars) else " new"
        print(f"  [{src}] {name:>8s}:  mean={full_mean[i]:+10.3f}  std={full_std[i]:10.3f}")

    # --- variables.json ---
    full_vars = list(base_vars) + list(NEW_VAR_ORDER)
    backup_vars = base_dir / "variables_19f.json"
    if not backup_vars.exists():
        import shutil
        shutil.copy(base_dir / "variables.json", backup_vars)
    (base_dir / "variables.json").write_text(
        json.dumps(full_vars, indent=2, ensure_ascii=False)
    )

    # --- dataset_info.json (обновляем in-place, бэкапим оригинал) ---
    backup_info = base_dir / "dataset_info_19f.json"
    if not backup_info.exists():
        import shutil
        shutil.copy(base_dir / "dataset_info.json", backup_info)

    new_info = {
        "time_start": time_start, "time_end": time_end,
        "n_time": n_time, "n_lon": n_lon, "n_lat": n_lat,
        "n_feat": len(full_vars),
        "variables": full_vars,
        "dtype": "float16",
        "file": "data.npy",
        "extra_file": "data_extra.npy",
        "n_feat_base": n_base,
        "n_feat_extra": n_extra,
        "size_gb": round((n_time * n_lon * n_lat * len(full_vars) * 2) / (1024 ** 3), 1),
    }
    (base_dir / "dataset_info.json").write_text(json.dumps(new_info, indent=2))

    # удаляем progress
    progress_path = base_dir / "progress_extra.json"
    if progress_path.exists():
        progress_path.unlink()

    print()
    print("=" * 70)
    print("✓ Dataset extended to 29f!")
    print(f"  data.npy        — старый (T,{n_lon},{n_lat},{n_base})  не тронут")
    print(f"  data_extra.npy  — новый  (T,{n_lon},{n_lat},{n_extra})  {extra_gb:.1f} GB")
    print(f"  scalers.npz     — обновлён (29 каналов)")
    print(f"  Loader склеит автоматически по последней оси.")
    print("=" * 70)


if __name__ == "__main__":
    main()

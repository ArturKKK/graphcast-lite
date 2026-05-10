#!/usr/bin/env python3
"""
scripts/build_dataset_512x256_30f.py
Сборка датасета 512×256 (≈0.7°) из WeatherBench2 ERA5 Zarr (6h) — 30 каналов.

Расширение _19f до _30f:
  Старые 19:  t2m, 10u, 10v, msl, tp, sp, tcwv, z_surf, lsm,
              t/u/v/z/q@850, t/u/v/z/q@500
  Новые 11:   2d (2-метровая точка росы),
              z/t/u/v/q@250 (jet stream + влага верхней тропосферы),
              z/t/u/v/q@1000 (граничный слой)

Ключевое: --base-dir <старый _19f>  →  каналы которые уже есть в base
копируются С ДИСКА (без сети), скачивается только дельта (11 новых).

Если --base-dir не задан — качаем все 30 с нуля.
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
    "2m_dewpoint_temperature",   # NEW в 30f
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
LEVELS = [250, 500, 850, 1000]   # 30f: добавили 250 и 1000

RENAME = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "total_column_water_vapour": "tcwv",
    "2m_dewpoint_temperature": "2d",
    "geopotential_at_surface": "z_surf",
    "land_sea_mask": "lsm",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "specific_humidity": "q",
}

# Сохраняем порядок старого 19f для первых 19 каналов (важно для совместимости
# с уже обученными моделями: они обучались на этом порядке индексов).
VAR_ORDER = [
    # surface (8)
    "t2m", "10u", "10v", "msl", "tp", "sp", "tcwv", "2d",
    # static (2) — индексы 8, 9 ?  Нет, держим старые индексы 7, 8 — иначе
    # ломаем все обученные модели. См. ниже фактический порядок.
]
# ВАЖНО: чтобы сохранить совместимость с существующими v1/v2 моделями
# (multires_russia_19f, multires_nores_freeze6 и др.), первые 19 индексов
# должны быть точь-в-точь как в старом _19f. Новые каналы — с конца.
VAR_ORDER = [
    "t2m", "10u", "10v", "msl", "tp",
    "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
    # --- v3 additions (11) ---
    "2d",
    "z@250", "t@250", "u@250", "v@250", "q@250",
    "z@1000", "t@1000", "u@1000", "v@1000", "q@1000",
]
assert len(VAR_ORDER) == 30, f"VAR_ORDER must have 30 channels, got {len(VAR_ORDER)}"

# float16 max = 65504. Большие давления и геопотенциал требуют scale.
SCALE_FACTORS = {
    "msl":     0.01,
    "sp":      0.01,
    "z_surf":  1 / 9.80665,
    "z@850":   1 / 9.80665,
    "z@500":   1 / 9.80665,
    "z@1000":  1 / 9.80665,
    "z@250":   1 / 9.80665,
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


def save_progress(out_dir: Path, t_end: int, chunk_size: int, channels_dl: list[str]):
    (out_dir / "progress.json").write_text(json.dumps({
        "last_completed_timestep": t_end,
        "chunk_size": chunk_size,
        "downloaded_channels": channels_dl,
    }))


def load_progress(out_dir: Path) -> dict | None:
    p = out_dir / "progress.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def welford_update(running_mean, running_m2, total_n, block_sum, block_sumsq, block_n):
    block_mean = block_sum / block_n
    block_var = block_sumsq / block_n - block_mean ** 2
    block_var = np.maximum(block_var, 0.0)
    delta = block_mean - running_mean
    new_n = total_n + block_n
    running_mean += delta * (block_n / new_n)
    running_m2 += block_var * block_n + (delta ** 2) * total_n * block_n / new_n
    return running_mean, running_m2, new_n


# ─── base dataset (для копирования существующих каналов) ──────────────

class BaseDataset:
    """Старый _19f датасет, из которого копируем пересекающиеся каналы."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        info = json.loads((base_dir / "dataset_info.json").read_text())
        self.n_time = info["n_time"]
        self.n_lon = info["n_lon"]
        self.n_lat = info["n_lat"]
        self.variables = info["variables"]
        self.var_to_idx = {v: i for i, v in enumerate(self.variables)}

        scalers = np.load(base_dir / "scalers.npz")
        self.mean = scalers["mean"]
        self.std = scalers["std"]
        # n из scalers: число элементов welford'а (n_time*n_lon*n_lat)
        self.welford_n = int(scalers["n"]) if "n" in scalers.files else \
                         self.n_time * self.n_lon * self.n_lat

        self.fp = np.memmap(
            str(base_dir / "data.npy"),
            dtype=np.float16, mode="r",
            shape=(self.n_time, self.n_lon, self.n_lat, len(self.variables)),
        )
        print(f"[BASE] {base_dir}: {len(self.variables)} channels, "
              f"shape=({self.n_time},{self.n_lon},{self.n_lat}), "
              f"vars={self.variables}")

    def has(self, ch: str) -> bool:
        return ch in self.var_to_idx

    def copy_channel(self, ch: str, dst_fp: np.memmap, dst_idx: int,
                     time_chunk: int = 1000):
        """Копирует один канал из base в dst (по time-чанкам, без RAM-выкачки)."""
        src_idx = self.var_to_idx[ch]
        for t in range(0, self.n_time, time_chunk):
            t_e = min(t + time_chunk, self.n_time)
            dst_fp[t:t_e, :, :, dst_idx] = self.fp[t:t_e, :, :, src_idx]
        dst_fp.flush()

    def channel_stats(self, ch: str):
        """Возвращает (mean, std, n) для канала из base/scalers.npz."""
        i = self.var_to_idx[ch]
        return float(self.mean[i]), float(self.std[i]), self.welford_n


# ─── build channel parts (только для каналов, которые надо качать) ───

def build_channel_parts_filtered(ds: xr.Dataset, want_channels: set[str]):
    """Возвращает surf_parts и plev_groups ТОЛЬКО для каналов в want_channels."""
    surf_parts = {}

    for v in SURF_DYNAMIC:
        if v not in ds.data_vars:
            print(f"[WARN] Surface var {v} not in zarr! Skipping.")
            continue
        ch = RENAME[v]
        if ch not in want_channels:
            continue
        surf_parts[ch] = ds[v].transpose("time", "longitude", "latitude")

    if "tp" in want_channels:
        tp_name = resolve_tp_name(ds)
        surf_parts["tp"] = ds[tp_name].transpose("time", "longitude", "latitude")

    for v in STATIC_VARS:
        if v not in ds.data_vars:
            print(f"[WARN] Static var {v} not in zarr!")
            continue
        ch = RENAME[v]
        if ch not in want_channels:
            continue
        da = ds[v]
        if "time" not in da.dims:
            da = da.expand_dims(time=ds.time)
        surf_parts[ch] = da.transpose("time", "longitude", "latitude")

    plev_groups = []
    for v in PLEV_VARS:
        if v not in ds.data_vars:
            print(f"[WARN] Skipping {v} (not in zarr)")
            continue
        short = RENAME[v]
        # Какие уровни нужны для этой переменной?
        wanted_levs = [lev for lev in LEVELS
                       if f"{short}@{lev}" in want_channels]
        if not wanted_levs:
            continue
        lev_map = {lev: f"{short}@{lev}" for lev in wanted_levs}
        da = ds[v].sel(level=wanted_levs).transpose("time", "level", "longitude", "latitude")
        non_idx = [c for c in da.coords if c not in da.dims]
        if non_idx:
            da = da.reset_coords(non_idx, drop=True)
        plev_groups.append((v, lev_map, da))

    return surf_parts, plev_groups


# ─── main download loop (для дельты каналов) ───────────────────────────

def download_delta(
    ds_open_args: tuple,
    out_path: Path,
    n_time: int, n_lon: int, n_lat: int, n_feat: int,
    var_names: list[str], dl_channels: list[str],
    time_chunk: int = 500,
    resume_from: int | None = None,
):
    """Качает только каналы из dl_channels, остальное в файле уже есть (из base)."""
    if not dl_channels:
        print("[DL] Нечего скачивать — все каналы скопированы из base.")
        return None  # статистика будет посчитана отдельно

    ch_idx = {name: i for i, name in enumerate(var_names)}
    want = set(dl_channels)

    time_start, time_end = ds_open_args
    ds = open_wb2(time_start, time_end)
    surf_parts, plev_groups = build_channel_parts_filtered(ds, want)

    surf_names = [n for n in var_names if n in surf_parts]
    print(f"[DL] Скачиваем {len(dl_channels)} каналов: {dl_channels}")
    print(f"[DL]   surface: {surf_names}")
    for src_var, lev_map, _ in plev_groups:
        print(f"[DL]   plev {RENAME[src_var]}: {list(lev_map.values())}")

    fp = np.memmap(str(out_path), dtype=np.float16, mode="r+",
                   shape=(n_time, n_lon, n_lat, n_feat))

    # --- welford только по dl_channels ---
    dl_idx = [ch_idx[c] for c in dl_channels]
    n_dl = len(dl_channels)
    total_n = 0
    running_mean = np.zeros(n_dl, dtype=np.float64)
    running_m2 = np.zeros(n_dl, dtype=np.float64)
    dl_idx_to_local = {ch_idx[c]: li for li, c in enumerate(dl_channels)}

    actual_start = resume_from if (resume_from and resume_from > 0) else 0

    # Если resume — досчитываем стат с диска для уже скачанных шагов
    if actual_start > 0:
        print(f"[RESUME] Re-computing stats for timesteps 0–{actual_start} (dl channels only)...")
        for t_s in range(0, actual_start, time_chunk):
            t_e = min(t_s + time_chunk, actual_start)
            block_n = (t_e - t_s) * n_lon * n_lat
            block_sum = np.zeros(n_dl, dtype=np.float64)
            block_sumsq = np.zeros(n_dl, dtype=np.float64)
            for li, j in enumerate(dl_idx):
                arr = np.nan_to_num(fp[t_s:t_e, :, :, j].astype(np.float32), nan=0.0)
                block_sum[li] = arr.sum(dtype=np.float64)
                block_sumsq[li] = (arr * arr).sum(dtype=np.float64)
                del arr
            running_mean, running_m2, total_n = welford_update(
                running_mean, running_m2, total_n, block_sum, block_sumsq, block_n
            )
            gc.collect()

    download_t0 = time.time()
    downloaded_steps = 0

    for t_start in range(actual_start, n_time, time_chunk):
        t_end = min(t_start + time_chunk, n_time)
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
        block_sum = np.zeros(n_dl, dtype=np.float64)
        block_sumsq = np.zeros(n_dl, dtype=np.float64)

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
                li = dl_idx_to_local[j]
                block_sum[li] = arr.sum(dtype=np.float64)
                block_sumsq[li] = (arr * arr).sum(dtype=np.float64)
                del arr
            surf_mb = sum(r.nbytes for r in results) / 1e6
            print(f"    [surf] {len(surf_names)} vars: "
                  f"{dt_surf:.1f}s  ({surf_mb:.0f} MB, {surf_mb/max(dt_surf,0.1):.1f} MB/s)")
            del results

        # plev
        for src_var, lev_map, da in plev_groups:
            short = RENAME[src_var]
            ts1 = time.time()
            block_3d = da.isel(time=t_slice).values.astype(np.float32, copy=False)
            levels_arr = da.level.values
            dt_dl = time.time() - ts1
            plev_mb = block_3d.nbytes / 1e6
            for li_lev, lev_val in enumerate(levels_arr):
                ch_name = lev_map[int(lev_val)]
                j = ch_idx[ch_name]
                arr_2d = block_3d[:, li_lev, :, :]
                if ch_name in SCALE_FACTORS:
                    arr_2d = arr_2d * SCALE_FACTORS[ch_name]
                fp[t_start:t_end, :, :, j] = arr_2d.astype(np.float16)
                local = dl_idx_to_local[j]
                block_sum[local] = arr_2d.sum(dtype=np.float64)
                block_sumsq[local] = (arr_2d * arr_2d).sum(dtype=np.float64)
            print(f"    [plev] {short}@{[int(l) for l in levels_arr]}: "
                  f"{dt_dl:.1f}s  ({plev_mb:.0f} MB, {plev_mb/max(dt_dl,0.1):.1f} MB/s)")
            del block_3d

        gc.collect()
        fp.flush()

        block_n = chunk_steps * n_lon * n_lat
        running_mean, running_m2, total_n = welford_update(
            running_mean, running_m2, total_n, block_sum, block_sumsq, block_n
        )
        downloaded_steps += chunk_steps
        save_progress(out_path.parent, t_end, time_chunk, dl_channels)

    fp.flush()
    del fp

    elapsed_total = time.time() - download_t0
    print(f"\n[DL] ✓ Done in {elapsed_total / 60:.0f} min")

    std = np.sqrt(running_m2 / max(total_n, 1))
    std = np.maximum(std, 1e-6)
    return {
        "channels": dl_channels,
        "mean": running_mean.astype(np.float32),
        "std": std.astype(np.float32),
        "n": total_n,
    }


# ─── coords/scalers ───────────────────────────────────────────────────

def write_coords(out_dir: Path, base_dataset: BaseDataset | None,
                 ds: xr.Dataset | None):
    """Пишем coords.npz. Если есть base — копируем оттуда; иначе из ds."""
    if base_dataset is not None and (base_dataset.base_dir / "coords.npz").exists():
        src = np.load(base_dataset.base_dir / "coords.npz")
        np.savez(out_dir / "coords.npz",
                 longitude=src["longitude"], latitude=src["latitude"])
        return
    assert ds is not None
    first_var = SURF_DYNAMIC[0]
    if first_var not in ds.data_vars:
        first_var = next(v for v in SURF_DYNAMIC if v in ds.data_vars)
    da = ds[first_var]
    np.savez(out_dir / "coords.npz",
             longitude=da.longitude.values.astype(np.float32),
             latitude=da.latitude.values.astype(np.float32))


def assemble_scalers(var_names: list[str],
                     base_ds: BaseDataset | None,
                     dl_stats: dict | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Собирает mean/std для всех каналов из base + dl_stats."""
    n = len(var_names)
    mean = np.zeros(n, dtype=np.float32)
    std = np.zeros(n, dtype=np.float32)
    welford_n = 0

    if dl_stats is not None:
        dl_map = {c: i for i, c in enumerate(dl_stats["channels"])}
        welford_n = dl_stats["n"]

    for i, ch in enumerate(var_names):
        if dl_stats is not None and ch in dl_map:
            j = dl_map[ch]
            mean[i] = dl_stats["mean"][j]
            std[i] = dl_stats["std"][j]
        elif base_ds is not None and base_ds.has(ch):
            m, s, n_b = base_ds.channel_stats(ch)
            mean[i] = m
            std[i] = s
            if welford_n == 0:
                welford_n = n_b
        else:
            raise RuntimeError(f"No stats source for channel {ch}")

    return mean, std, welford_n


# ─── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build 512×256 dataset (30 vars) from WeatherBench2 ERA5. "
                    "Optionally copy overlapping channels from existing _19f base."
    )
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output dir (e.g. /Volumes/Ext/.../global_512x256_30f_2010-2021_07deg)")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Existing _19f dataset; overlapping channels copied from disk (NO download)")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2022,
                        help="End year (exclusive)")
    parser.add_argument("--time-chunk", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_dir) if args.base_dir else None

    time_start = f"{args.start_year}-01-01"
    time_end = f"{args.end_year - 1}-12-31"

    print("=" * 70)
    print("BUILD DATASET — 512×256 — ERA5 — WeatherBench2  [30f]")
    print(f"  Period:   {args.start_year}–{args.end_year - 1}")
    print(f"  Grid:     512 × 256")
    print(f"  Features: {len(VAR_ORDER)} channels")
    print(f"  Output:   {out_dir}")
    print(f"  Base:     {base_dir or '(none, full download)'}")
    print("=" * 70)

    # --- Открываем base, если есть ---
    base_ds: BaseDataset | None = None
    if base_dir is not None:
        if not (base_dir / "dataset_info.json").exists():
            raise SystemExit(f"--base-dir {base_dir} не содержит dataset_info.json")
        base_ds = BaseDataset(base_dir)

    # --- Определяем размеры ---
    if base_ds is not None:
        n_time, n_lon, n_lat = base_ds.n_time, base_ds.n_lon, base_ds.n_lat
        # sanity: проверяем, что период совпадает
        info = json.loads((base_dir / "dataset_info.json").read_text())
        if info.get("time_start") != time_start or info.get("time_end") != time_end:
            print(f"[WARN] base dataset period {info.get('time_start')}–{info.get('time_end')}"
                  f" != requested {time_start}–{time_end}.")
            print(f"[WARN] Будем использовать base период (нельзя смешивать).")
            time_start = info["time_start"]
            time_end = info["time_end"]
    else:
        # без base — открываем zarr чтобы узнать n_time
        ds_probe = open_wb2(time_start, time_end)
        ref = ds_probe["2m_temperature"]
        n_time = ref.sizes["time"]
        n_lon = ref.sizes["longitude"]
        n_lat = ref.sizes["latitude"]
        del ds_probe

    n_feat = len(VAR_ORDER)
    est_gb = n_time * n_lon * n_lat * n_feat * 2 / (1024 ** 3)
    print(f"\n[INFO] Target shape: ({n_time}, {n_lon}, {n_lat}, {n_feat}) "
          f"float16 ≈ {est_gb:.1f} GB")

    # --- Какие каналы можно скопировать, какие надо качать ---
    if base_ds is not None:
        copy_channels = [c for c in VAR_ORDER if base_ds.has(c)]
        dl_channels   = [c for c in VAR_ORDER if not base_ds.has(c)]
    else:
        copy_channels = []
        dl_channels = list(VAR_ORDER)

    print(f"[PLAN] Copy from base: {len(copy_channels)} → {copy_channels}")
    print(f"[PLAN] Download:       {len(dl_channels)} → {dl_channels}")

    # --- Создаём/открываем выходной memmap ---
    out_path = out_dir / "data.npy"
    fresh = not out_path.exists()
    fp = np.memmap(str(out_path), dtype=np.float16,
                   mode="w+" if fresh else "r+",
                   shape=(n_time, n_lon, n_lat, n_feat))

    # --- Resume логика для COPY-фазы ---
    progress = load_progress(out_dir) if args.resume else None
    copy_done = set()
    if progress and "copy_done" in progress:
        copy_done = set(progress["copy_done"])
        print(f"[RESUME] Уже скопированы: {sorted(copy_done)}")

    # --- COPY: каналы из base ---
    if copy_channels and base_ds is not None:
        print(f"\n[COPY] Копируем {len(copy_channels)} каналов из base "
              f"({base_dir}) → {out_path}")
        for i, ch in enumerate(copy_channels):
            dst_idx = VAR_ORDER.index(ch)
            if ch in copy_done:
                print(f"  [{i+1}/{len(copy_channels)}] {ch:>8s} → idx {dst_idx:2d}  (skip, already copied)")
                continue
            t0 = time.time()
            base_ds.copy_channel(ch, fp, dst_idx, time_chunk=1000)
            dt = time.time() - t0
            ch_gb = n_time * n_lon * n_lat * 2 / (1024 ** 3)
            print(f"  [{i+1}/{len(copy_channels)}] {ch:>8s} → idx {dst_idx:2d}  "
                  f"{dt:.1f}s  ({ch_gb:.2f} GB, {ch_gb*1024/max(dt,0.1):.0f} MB/s)")
            copy_done.add(ch)
            # сохраняем прогресс копирования
            (out_dir / "progress.json").write_text(json.dumps({
                "copy_done": sorted(copy_done),
                "downloaded_channels": dl_channels,
                "last_completed_timestep": 0,
                "chunk_size": args.time_chunk,
            }))
        print(f"[COPY] ✓ Done")

    fp.flush()
    del fp

    # --- DOWNLOAD: новые каналы ---
    resume_from = None
    if args.resume and progress:
        saved_t = progress.get("last_completed_timestep", 0)
        if saved_t > 0:
            resume_from = max(0, saved_t - args.time_chunk)
            print(f"\n[RESUME] DL last_completed={saved_t}, restart from {resume_from}")

    dl_stats = download_delta(
        ds_open_args=(time_start, time_end),
        out_path=out_path,
        n_time=n_time, n_lon=n_lon, n_lat=n_lat, n_feat=n_feat,
        var_names=VAR_ORDER,
        dl_channels=dl_channels,
        time_chunk=args.time_chunk,
        resume_from=resume_from,
    )

    # --- coords ---
    if base_ds is not None:
        write_coords(out_dir, base_ds, None)
    else:
        ds = open_wb2(time_start, time_end)
        write_coords(out_dir, None, ds)

    # --- scalers ---
    mean, std, welford_n = assemble_scalers(VAR_ORDER, base_ds, dl_stats)
    np.savez(out_dir / "scalers.npz", mean=mean, std=std, n=welford_n)
    print(f"\n[STAT] ✓ Saved scalers.npz (n={welford_n})")
    for i, name in enumerate(VAR_ORDER):
        src = "BASE" if (base_ds and base_ds.has(name) and dl_stats and name not in dl_stats["channels"]) else \
              "BASE" if (base_ds and base_ds.has(name) and not dl_stats) else "DL  "
        print(f"  [{src}] {name:>8s}:  mean={mean[i]:+10.3f}  std={std[i]:10.3f}")

    (out_dir / "variables.json").write_text(
        json.dumps(VAR_ORDER, indent=2, ensure_ascii=False)
    )
    dataset_info = {
        "time_start": time_start, "time_end": time_end,
        "n_time": int(n_time), "n_lon": int(n_lon), "n_lat": int(n_lat),
        "n_feat": len(VAR_ORDER), "variables": VAR_ORDER,
        "dtype": "float16", "file": "data.npy",
        "size_gb": round(est_gb, 1),
        "base_dir": str(base_dir) if base_dir else None,
        "copied_channels": copy_channels,
        "downloaded_channels": dl_channels,
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))

    progress_path = out_dir / "progress.json"
    if progress_path.exists():
        progress_path.unlink()

    print()
    print("=" * 70)
    print("✓ Dataset 30f complete!")
    print(f"  {out_path} — {est_gb:.1f} GB")
    print(f"  {len(VAR_ORDER)} variables × {n_time} timesteps × {n_lon}×{n_lat}")
    print(f"  Copied from base: {len(copy_channels)}, Downloaded: {len(dl_channels)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

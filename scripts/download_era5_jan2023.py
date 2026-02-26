#!/usr/bin/env python3
"""
scripts/download_era5_jan2023.py

Скачивает ERA5 за январь 2023 из ARCO ERA5 (Google Cloud Storage),
интерполирует на сетку 512×256 и сохраняет в формате, совместимом с predict.py.

Зачем: WB2 512×256 Zarr заканчивается на 2022, а WRF-данные — январь 2023.
Поэтому качаем из ARCO ERA5 (полная ERA5 на GCS, есть 2023+).

Источник: gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
  - 0.25° (1440×721), почасово, все уровни давления

Выход:
  data/datasets/wb2_512x256_19f_jan2023/
    data.npy          — float16 memmap (T, 512, 256, 19)
    scalers.npz       — скопировано из тренировочного датасета
    coords.npz        — longitude/latitude
    variables.json    — названия каналов
    dataset_info.json — метаданные для dataloader

Использование:
    python scripts/download_era5_jan2023.py
    python scripts/download_era5_jan2023.py --start 2023-01-18 --end 2023-01-25
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import xarray as xr
import gcsfs

# ─── Конфигурация ─────────────────────────────────────────────────────

ARCO_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Также попробуем WB2 (вдруг обновили до 2023)
WB2_ZARR_512 = "gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr"

# Целевая сетка (как в тренировочном датасете)
TARGET_NLON = 512
TARGET_NLAT = 256

# 19 каналов в том же порядке, что и при обучении
VAR_ORDER = [
    "t2m", "10u", "10v", "msl", "tp",
    "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
]

# Маппинг: наше короткое имя → имя в ARCO ERA5 zarr
ARCO_SURFACE_MAP = {
    "t2m":  "2m_temperature",
    "10u":  "10m_u_component_of_wind",
    "10v":  "10m_v_component_of_wind",
    "msl":  "mean_sea_level_pressure",
    "tp":   "total_precipitation",
    "sp":   "surface_pressure",
    "tcwv": "total_column_water_vapour",
}

ARCO_PLEVEL_MAP = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "q": "specific_humidity",
}

LEVELS = [850, 500]

# Масштабирование (как при сборке тренировочного датасета)
# Нужно, чтобы значения помещались в float16 (max=65504)
SCALE_FACTORS = {
    "msl":    0.01,        # Pa  → hPa
    "sp":     0.01,        # Pa  → hPa
    "z_surf": 1/9.80665,   # м²/с² → м
    "z@850":  1/9.80665,
    "z@500":  1/9.80665,
}

# Пути по умолчанию
TRAIN_DATA_DIR = Path("data/datasets/wb2_512x256_19f_ar")
DEFAULT_OUT_DIR = Path("data/datasets/wb2_512x256_19f_jan2023")


# ─── Скачивание из ARCO ERA5 ──────────────────────────────────────────

def try_wb2_direct(time_start, time_end):
    """Попробовать взять данные напрямую из WB2 512×256 (если вдруг обновили)."""
    print(f"[1/4] Пробуем WB2 512×256 Zarr...")
    try:
        fs = gcsfs.GCSFileSystem(token="anon")
        store = fs.get_mapper(WB2_ZARR_512)
        ds = xr.open_zarr(store, consolidated=True)
        ds_sel = ds.sel(time=slice(time_start, time_end))
        
        n_time = ds_sel.sizes.get("time", 0)
        if n_time > 0:
            print(f"  ✓ WB2 имеет данные за {time_start}–{time_end} ({n_time} timesteps)")
            return ds_sel
        else:
            print(f"  ✗ WB2 не содержит данные за {time_start} (заканчивается на 2022)")
            return None
    except Exception as e:
        print(f"  ✗ WB2 недоступен: {e}")
        return None


def download_from_arco(time_start, time_end):
    """Скачать данные из ARCO ERA5 zarr-v3 (0.25°, hourly)."""
    print(f"[2/4] Открываем ARCO ERA5 Zarr...")
    print(f"  Источник: {ARCO_ZARR}")
    
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(ARCO_ZARR)
    
    t0 = time.time()
    ds = xr.open_zarr(store, consolidated=True)
    print(f"  Zarr открыт за {time.time() - t0:.1f}s")
    
    # Print available variables for debugging
    print(f"  Координаты: {list(ds.coords)}")
    print(f"  Переменные: {len(ds.data_vars)} vars")
    
    # Select time range
    ds = ds.sel(time=slice(time_start, time_end))
    n_time_raw = ds.sizes["time"]
    print(f"  Временной диапазон: {n_time_raw} часовых шагов ({time_start} → {time_end})")
    
    # Отбираем только 6-часовые шаги (00, 06, 12, 18) вместо resample
    # (resample ломается на новых pandas из-за deprecated 'base' argument)
    times = ds.time.values
    hours = [t.astype('datetime64[h]').astype(int) % 24 for t in times]
    mask_6h = np.array([h in (0, 6, 12, 18) for h in hours])
    ds_6h = ds.isel(time=mask_6h)
    n_time_6h = ds_6h.sizes["time"]
    print(f"  После фильтра 6h: {n_time_6h} шагов")
    
    # Целевые координаты: совпадают с тренировочной сеткой
    target_lon = np.linspace(0, 360, TARGET_NLON, endpoint=False).astype(np.float32)
    target_lat = np.linspace(-90, 90, TARGET_NLAT, endpoint=True).astype(np.float32)
    
    # Определяем имя координаты широты/долготы
    lat_coord = "latitude" if "latitude" in ds.coords else "lat"
    lon_coord = "longitude" if "longitude" in ds.coords else "lon"
    
    n_feat = len(VAR_ORDER)
    
    # ─── Скачиваем surface переменные ───
    print(f"\n[3/4] Скачиваем и обрабатываем {n_feat} переменных...")
    
    channels_6h = {}  # short_name -> np.ndarray (T_6h, lon, lat)
    
    for short_name, arco_name in ARCO_SURFACE_MAP.items():
        t1 = time.time()
        print(f"  {short_name} ({arco_name})...", end=" ", flush=True)
        
        if arco_name not in ds.data_vars:
            print(f"НЕ НАЙДЕНА! Доступные: {[v for v in ds.data_vars if short_name[:2] in v.lower()]}")
            raise RuntimeError(f"Variable '{arco_name}' not found in ARCO zarr")
        
        da = ds_6h[arco_name]
        
        # Интерполяция на целевую сетку (уже отфильтровано до 6h)
        da_interp = da.interp({lon_coord: target_lon, lat_coord: target_lat}, method="linear")
        
        # Загружаем в память
        arr = da_interp.values.astype(np.float32)
        
        # Нормализуем порядок осей: (T, lon, lat) — как в тренировочном датасете
        # ARCO может быть (time, latitude, longitude)
        da_dims = list(da_interp.dims)
        if da_dims.index(lat_coord) < da_dims.index(lon_coord):
            # (T, lat, lon) → (T, lon, lat)
            arr = np.swapaxes(arr, 1, 2)
        
        # Применяем масштабирование
        if short_name in SCALE_FACTORS:
            arr *= SCALE_FACTORS[short_name]
        
        channels_6h[short_name] = arr
        dt = time.time() - t1
        print(f"shape={arr.shape} [{dt:.1f}s]")
    
    # ─── Скачиваем pressure-level переменные ───
    for plev_short, arco_name in ARCO_PLEVEL_MAP.items():
        t1 = time.time()
        
        if arco_name not in ds.data_vars:
            print(f"  {plev_short} ({arco_name})... НЕ НАЙДЕНА!")
            raise RuntimeError(f"Variable '{arco_name}' not found in ARCO zarr")
        
        da = ds_6h[arco_name]
        
        # Нужно определить имя координаты уровня
        level_coord = "level" if "level" in da.coords else "pressure_level"
        
        for lev in LEVELS:
            ch_name = f"{plev_short}@{lev}"
            print(f"  {ch_name} ({arco_name}@{lev}hPa)...", end=" ", flush=True)
            
            da_lev = da.sel({level_coord: lev})
            da_interp = da_lev.interp({lon_coord: target_lon, lat_coord: target_lat}, method="linear")
            
            arr = da_interp.values.astype(np.float32)
            
            # Нормализуем порядок осей
            da_dims = list(da_interp.dims)
            if da_dims.index(lat_coord) < da_dims.index(lon_coord):
                arr = np.swapaxes(arr, 1, 2)
            
            if ch_name in SCALE_FACTORS:
                arr *= SCALE_FACTORS[ch_name]
            
            channels_6h[ch_name] = arr
            dt = time.time() - t1
            print(f"shape={arr.shape} [{dt:.1f}s]")
            t1 = time.time()
    
    return channels_6h, target_lon, target_lat


def get_static_from_training(n_time, train_data_dir=None):
    """
    Статические переменные (z_surf, lsm) не меняются во времени.
    Берём их из тренировочного датасета.
    """
    if train_data_dir is None:
        train_data_dir = TRAIN_DATA_DIR
    print(f"\n  Статические переменные (z_surf, lsm) из {train_data_dir}...")
    
    info_path = train_data_dir / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)
    
    shape = (info["n_time"], info["n_lon"], info["n_lat"], info["n_feat"])
    mm = np.memmap(str(train_data_dir / "data.npy"), dtype=np.float16, mode="r", shape=shape)
    
    var_names = info["variables"]
    
    static_channels = {}
    for var in ["z_surf", "lsm"]:
        idx = var_names.index(var)
        # Берём первый таймстеп (статика одинакова для всех)
        static_val = mm[0, :, :, idx].astype(np.float32)
        # Расширяем на n_time
        static_channels[var] = np.tile(static_val[np.newaxis, :, :], (n_time, 1, 1))
        print(f"    {var}: shape={static_channels[var].shape}, "
              f"range=[{static_val.min():.2f}, {static_val.max():.2f}]")
    
    del mm
    return static_channels


def assemble_and_save(channels, target_lon, target_lat, out_dir, train_data_dir=None):
    """Собираем все каналы в data.npy и сохраняем метаданные."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем размеры
    first_key = [k for k in VAR_ORDER if k in channels][0]
    n_time = channels[first_key].shape[0]
    n_lon = channels[first_key].shape[1]
    n_lat = channels[first_key].shape[2]
    n_feat = len(VAR_ORDER)
    
    print(f"\n[4/4] Собираем data.npy: ({n_time}, {n_lon}, {n_lat}, {n_feat})")
    
    if train_data_dir is None:
        train_data_dir = TRAIN_DATA_DIR
    
    # Добавляем статические переменные
    static_needed = [v for v in ["z_surf", "lsm"] if v not in channels]
    if static_needed:
        static_ch = get_static_from_training(n_time, train_data_dir)
        channels.update(static_ch)
    
    # Проверяем, что все каналы есть
    missing = [v for v in VAR_ORDER if v not in channels]
    if missing:
        raise RuntimeError(f"Missing channels: {missing}")
    
    # Собираем memmap
    data_path = out_dir / "data.npy"
    fp = np.memmap(str(data_path), dtype=np.float16, mode="w+",
                   shape=(n_time, n_lon, n_lat, n_feat))
    
    for i, var in enumerate(VAR_ORDER):
        arr = channels[var]
        if arr.shape != (n_time, n_lon, n_lat):
            raise RuntimeError(f"Shape mismatch for {var}: expected ({n_time},{n_lon},{n_lat}), got {arr.shape}")
        fp[:, :, :, i] = arr.astype(np.float16)
        print(f"  [{i+1:2d}/19] {var:>8s}: range=[{arr.min():+10.3f}, {arr.max():+10.3f}]")
    
    fp.flush()
    del fp
    
    size_mb = n_time * n_lon * n_lat * n_feat * 2 / (1024**2)
    print(f"  ✓ data.npy: {size_mb:.1f} MB")
    
    # Копируем scalers.npz из тренировочного датасета
    src_scalers = train_data_dir / "scalers.npz"
    dst_scalers = out_dir / "scalers.npz"
    shutil.copy2(src_scalers, dst_scalers)
    print(f"  ✓ scalers.npz (скопировано из {src_scalers})")
    
    # coords.npz
    np.savez(out_dir / "coords.npz",
             longitude=target_lon.astype(np.float32),
             latitude=target_lat.astype(np.float32))
    print(f"  ✓ coords.npz")
    
    # variables.json
    (out_dir / "variables.json").write_text(json.dumps(VAR_ORDER, indent=2))
    print(f"  ✓ variables.json")
    
    # dataset_info.json
    info = {
        "time_start": "2023-01-18",
        "time_end": "2023-01-25",
        "n_time": int(n_time),
        "n_lon": int(n_lon),
        "n_lat": int(n_lat),
        "n_feat": int(n_feat),
        "variables": VAR_ORDER,
        "dtype": "float16",
        "file": "data.npy",
        "size_gb": round(size_mb / 1024, 2),
        "source": "ARCO ERA5 (interpolated to 512x256)",
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"  ✓ dataset_info.json")


def process_wb2_direct(ds_wb2, out_dir, train_data_dir=None):
    """Если WB2 512×256 имеет данные за 2023, обрабатываем напрямую."""
    # Аналогично build_dataset_512x256.py
    from scripts.build_dataset_512x256 import (
        build_channel_parts, SCALE_FACTORS as WB2_SCALES, VAR_ORDER as WB2_ORDER
    )
    
    surf_parts, plev_groups, var_names = build_channel_parts(ds_wb2)
    
    channels = {}
    for name in var_names:
        if name in surf_parts:
            arr = surf_parts[name].values.astype(np.float32)
            if name in WB2_SCALES:
                arr *= WB2_SCALES[name]
            channels[name] = arr
    
    for src_var, lev_map, da in plev_groups:
        block = da.values.astype(np.float32)
        levels_arr = da.level.values
        for li, lev_val in enumerate(levels_arr):
            ch_name = lev_map[int(lev_val)]
            arr = block[:, li, :, :]
            if ch_name in WB2_SCALES:
                arr = arr * WB2_SCALES[ch_name]
            channels[ch_name] = arr
    
    if train_data_dir is None:
        train_data_dir = TRAIN_DATA_DIR
    coords = np.load(train_data_dir / "coords.npz")
    target_lon = coords["longitude"]
    target_lat = coords["latitude"]
    
    assemble_and_save(channels, target_lon, target_lat, out_dir, train_data_dir)


# ─── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 Jan 2023 for WRF comparison inference"
    )
    parser.add_argument("--start", type=str, default="2023-01-18",
                        help="Start date (default: 2023-01-18)")
    parser.add_argument("--end", type=str, default="2023-01-25",
                        help="End date, exclusive (default: 2023-01-25)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Output directory")
    parser.add_argument("--train-dir", type=str, default=str(TRAIN_DATA_DIR),
                        help="Training dataset dir (for scalers & static vars)")
    args = parser.parse_args()
    
    train_data_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    
    print("=" * 70)
    print("DOWNLOAD ERA5 FOR INFERENCE — Jan 2023 — WRF comparison")
    print(f"  Period:     {args.start} → {args.end}")
    print(f"  Grid:       {TARGET_NLON}×{TARGET_NLAT}")
    print(f"  Channels:   {len(VAR_ORDER)}")
    print(f"  Output:     {out_dir}")
    print(f"  Train data: {train_data_dir}")
    print("=" * 70)
    
    # Проверяем наличие тренировочных данных
    if not (train_data_dir / "scalers.npz").exists():
        raise FileNotFoundError(
            f"scalers.npz not found in {train_data_dir}! "
            "Need training scalers for normalization."
        )
    
    t_total = time.time()
    
    # Шаг 1: Пробуем WB2 напрямую
    ds_wb2 = try_wb2_direct(args.start, args.end)
    
    if ds_wb2 is not None:
        print("\n→ Используем WB2 512×256 (уже на правильной сетке)")
        process_wb2_direct(ds_wb2, out_dir, train_data_dir)
    else:
        print("\n→ Используем ARCO ERA5 (0.25° → интерполяция на 512×256)")
        channels, target_lon, target_lat = download_from_arco(args.start, args.end)
        assemble_and_save(channels, target_lon, target_lat, out_dir, train_data_dir)
    
    dt_total = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"✓ Готово за {dt_total / 60:.1f} мин!")
    print(f"  Датасет: {out_dir}")
    print(f"\n  Для инференса:")
    print(f"  python scripts/predict.py experiments/wb2_512x256_19f_ar_v2 \\")
    print(f"    --data-dir {out_dir} --ar-steps 4 --max-samples 20")
    print("=" * 70)


if __name__ == "__main__":
    main()

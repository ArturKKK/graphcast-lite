#!/usr/bin/env python3
"""
scripts/build_region_arco.py

Собирает РЕГИОНАЛЬНЫЙ датасет из ARCO ERA5 (0.25°, hourly) для дат после 2022,
когда WB2 Zarr уже не имеет данных.

Поддерживает ДВА формата датасета:
  A) 19-var chunked (для V2 модели wb2_512x256_19f_ar):
     scalers.npz содержит {mean, std, n}, нормализация: (x - mean) / std.
     Выход: data.npy + dataset_info.json (chunked timeseries формат).
  B) 15-var windowed (для старой модели wb2_64x32_15f_4pred):
     scalers.npz содержит {x_mean, x_scale, y_mean, y_scale},
     нормализация: (X - x_mean) / x_scale.
     Выход: X_train.pt, y_train.pt, X_test.pt, y_test.pt.

Формат определяется АВТОМАТИЧЕСКИ по ключам в scalers.npz.

Пример (19 var, V2 модель):
    python scripts/build_region_arco.py \
      --out-dir data/datasets/region_krsk_arco_19f \
      --start-date 2023-01-10 --end-date 2023-01-27 \
      --lon-min 83 --lon-max 98 --lat-min 50 --lat-max 60 \
      --obs-window 2 --pred-window 1 \
      --train-scalers data/datasets/wb2_512x256_19f_ar/scalers.npz

Пример (15 var, старая модель):
    python scripts/build_region_arco.py \
      --out-dir data/datasets/region_krsk_arco_15f_4obs_4pred \
      --start-date 2023-01-10 --end-date 2023-01-27 \
      --lon-min 83 --lon-max 98 --lat-min 50 --lat-max 60 \
      --obs-window 4 --pred-window 4 \
      --train-scalers data/datasets/region_nsko_1440x721_15f_4obs_4pred_week/scalers.npz
"""

import argparse
import json
import shutil
import time as _time
from pathlib import Path

import numpy as np
import xarray as xr
import gcsfs
import torch

# ─── Источник ──────────────────────────────────────────────────────────

ARCO_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ─── Маппинг переменных: имя в ARCO → наше короткое имя ───────────────

SURF_MAP = {
    "2m_temperature":               "t2m",
    "10m_u_component_of_wind":      "10u",
    "10m_v_component_of_wind":      "10v",
    "mean_sea_level_pressure":      "msl",
    "total_precipitation":          "tp",
    "surface_pressure":             "sp",
    "total_column_water_vapour":    "tcwv",
}

PLEV_MAP = {
    "temperature":           "t",
    "u_component_of_wind":   "u",
    "v_component_of_wind":   "v",
    "geopotential":          "z",
    "specific_humidity":     "q",
}

LEVELS = [850, 500]

# Порядок каналов для 15-var модели (как в experiments/wb2_64x32_15f_4pred/variables.json)
VAR_ORDER_15 = [
    "t2m", "10u", "10v", "msl", "tp",
    "t@850", "u@850", "v@850",
    "t@500", "u@500", "v@500",
    "z@850", "z@500",
    "q@850", "q@500",
]

# Порядок каналов для 19-var модели (как в experiments/wb2_512x256_19f_ar/variables.json)
VAR_ORDER_19 = [
    "t2m", "10u", "10v", "msl", "tp",
    "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
]

# Масштабирование для 19-var модели (как при сборке тренировочного датасета):
# Нужно, чтобы значения помещались в float16 (max=65504) и совпадали с тренировочными
SCALE_FACTORS_19 = {
    "msl":    0.01,        # Pa → hPa
    "sp":     0.01,        # Pa → hPa
    "z_surf": 1/9.80665,   # м²/с² → м
    "z@850":  1/9.80665,
    "z@500":  1/9.80665,
}

# Для 15-var модели скейлеры уже учитывают «сырые» единицы — масштабирование не нужно
SCALE_FACTORS_15 = {}


# ─── Скачивание региона из ARCO ERA5 ──────────────────────────────────

def _month_ranges(start_date: str, end_date: str):
    """Generate (year, month) tuples covering the date range."""
    from datetime import datetime
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    result = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        result.append((y, m))
        m += 1
        if m > 12:
            m = 1; y += 1
    return result


def _load_var_monthly(ds, var_name, lat_c, lon_c,
                      lon_min, lon_max, lat_min, lat_max,
                      months, scale_factor=1.0):
    """Load one variable month by month, only 6h steps, region-clipped.
    
    Returns: np.ndarray (T_6h, lon, lat) float32
    """
    chunks = []
    for i, (year, month) in enumerate(months):
        # Temporal slice for this month
        t_start = f"{year}-{month:02d}-01"
        if month == 12:
            t_end = f"{year+1}-01-01"
        else:
            t_end = f"{year}-{month+1:02d}-01"

        ds_m = ds.sel(time=slice(t_start, t_end))
        # Exclude first day of next month (slice is inclusive on Zarr)
        times_m = ds_m.time.values
        if len(times_m) == 0:
            continue
        # Filter to last day of this month max
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        t_end_exact = np.datetime64(f"{year}-{month:02d}-{last_day}T23:59:59")
        mask_month = times_m <= t_end_exact
        ds_m = ds_m.isel(time=mask_month)
        
        # 6h filter BEFORE .values  (critical: reduces HTTP requests 4x)
        times_m = ds_m.time.values
        if len(times_m) == 0:
            continue
        hours = np.array([t.astype("datetime64[h]").astype(int) % 24 for t in times_m])
        mask_6h = np.isin(hours, [0, 6, 12, 18])
        ds_m = ds_m.isel(time=mask_6h)

        if ds_m.sizes["time"] == 0:
            continue

        # Spatial clip + load
        da = ds_m[var_name].sel({
            lon_c: slice(lon_min, lon_max),
            lat_c: slice(lat_max, lat_min),
        })
        arr = da.values.astype(np.float32)  # NOW loads — small chunk

        # (T, lat, lon) → (T, lon, lat)
        dims = list(da.dims)
        if dims.index(lat_c) < dims.index(lon_c):
            arr = np.swapaxes(arr, dims.index(lat_c), dims.index(lon_c))

        if scale_factor != 1.0:
            arr *= scale_factor

        chunks.append(arr)

        if (i + 1) % 12 == 0 or i == len(months) - 1:
            done_steps = sum(c.shape[0] for c in chunks)
            print(f"  [{i+1}/{len(months)} months, {done_steps} steps]", end="", flush=True)

    return np.concatenate(chunks, axis=0)


def download_region(start_date: str, end_date: str,
                    lon_min: float, lon_max: float,
                    lat_min: float, lat_max: float,
                    scale_factors: dict):
    """
    Скачивает ERA5 для прямоугольного региона из ARCO:
    - грузит помесячно (избегаем 96k HTTP-запросов)
    - фильтрует до 6-часовых шагов ПЕРЕД загрузкой
    - применяет scale_factors если заданы
    - возвращает dict {var_name: np.ndarray (T, LON, LAT)}, координаты, кол-во шагов
    """
    print("=" * 60)
    print("Открываем ARCO ERA5 Zarr...")
    print(f"  Источник: {ARCO_ZARR}")

    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(ARCO_ZARR)

    t0 = _time.time()
    ds = xr.open_zarr(store, consolidated=True)
    print(f"  Zarr открыт за {_time.time() - t0:.1f} с")

    # Координатные имена
    lat_c = "latitude"  if "latitude"  in ds.coords else "lat"
    lon_c = "longitude" if "longitude" in ds.coords else "lon"

    months = _month_ranges(start_date, end_date)
    print(f"  Месяцев: {len(months)} ({months[0][0]}-{months[0][1]:02d} → {months[-1][0]}-{months[-1][1]:02d})")

    # Determine grid from metadata (no download needed)
    ds_peek = ds.sel({
        lon_c: slice(lon_min, lon_max),
        lat_c: slice(lat_max, lat_min),
    })
    lons = ds_peek[lon_c].values.astype(np.float32)
    lats = ds_peek[lat_c].values.astype(np.float32)
    n_6h_est = len(months) * 30 * 4  # rough estimate
    print(f"  Регион: lon [{lons.min():.2f}, {lons.max():.2f}]  ({len(lons)} точек)")
    print(f"          lat [{lats.min():.2f}, {lats.max():.2f}]  ({len(lats)} точек)")
    print(f"  Сетка: {len(lons)}×{len(lats)} = {len(lons)*len(lats)} нод")

    # ── Surface переменные ────────────────────────────────────────────
    channels = {}

    print(f"\n--- Surface переменные ---")
    for arco_name, short_name in SURF_MAP.items():
        t1 = _time.time()
        print(f"  {short_name:>6s} ({arco_name})...", flush=True)

        if arco_name not in ds.data_vars:
            raise RuntimeError(f"Переменная '{arco_name}' не найдена в ARCO")

        sf = scale_factors.get(short_name, 1.0)
        arr = _load_var_monthly(ds, arco_name, lat_c, lon_c,
                                lon_min, lon_max, lat_min, lat_max,
                                months, scale_factor=sf)

        channels[short_name] = arr
        dt = _time.time() - t1
        print(f"  → shape={arr.shape}  range=[{arr.min():.3f}, {arr.max():.3f}]  [{dt:.1f}s]")

    # ── Pressure-level переменные ─────────────────────────────────────
    n_6h = channels[list(channels.keys())[0]].shape[0]

    print(f"\n--- Pressure-level переменные ---")
    for arco_name, short_name in PLEV_MAP.items():
        if arco_name not in ds.data_vars:
            raise RuntimeError(f"Переменная '{arco_name}' не найдена в ARCO")

        level_c = "level" if "level" in ds[arco_name].coords else "pressure_level"

        for lev in LEVELS:
            ch = f"{short_name}@{lev}"
            t1 = _time.time()
            print(f"  {ch:>6s} ({arco_name}@{lev} hPa)...", flush=True)

            # Select level first, then monthly load
            ds_lev = ds.sel({level_c: lev})
            sf = scale_factors.get(ch, 1.0)
            arr = _load_var_monthly(ds_lev, arco_name, lat_c, lon_c,
                                    lon_min, lon_max, lat_min, lat_max,
                                    months, scale_factor=sf)

            channels[ch] = arr
            dt = _time.time() - t1
            print(f"  → shape={arr.shape}  range=[{arr.min():.3f}, {arr.max():.3f}]  [{dt:.1f}s]")

    return channels, lons, lats, n_6h


# ─── Построение сэмплов ──────────────────────────────────────────────

def build_samples(arr: np.ndarray, obs: int, pred: int):
    """
    Скользящее окно: из (T, LON, LAT, F) → (N, obs/pred, LON, LAT, F).
    """
    T = arr.shape[0]
    total = T - obs - pred + 1
    if total <= 0:
        raise RuntimeError(
            f"Мало данных: {T} шагов, нужно минимум obs+pred = {obs+pred}")

    X_list, Y_list = [], []
    for i in range(total):
        X_list.append(arr[i : i + obs])
        Y_list.append(arr[i + obs : i + obs + pred])
    return np.stack(X_list), np.stack(Y_list)


# ─── main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Строим РЕГИОНАЛЬНЫЙ ERA5 датасет из ARCO (0.25°) для 15- или 19-var модели.")

    p.add_argument("--out-dir",      type=str, required=True)
    p.add_argument("--start-date",   type=str, required=True,
                   help="Начало выгрузки, напр. 2023-01-10")
    p.add_argument("--end-date",     type=str, required=True,
                   help="Конец выгрузки, напр. 2023-01-27")
    p.add_argument("--lon-min",      type=float, default=83.0,
                   help="Западная граница региона (°E)")
    p.add_argument("--lon-max",      type=float, default=98.0,
                   help="Восточная граница региона (°E)")
    p.add_argument("--lat-min",      type=float, default=50.0,
                   help="Южная граница региона (°N)")
    p.add_argument("--lat-max",      type=float, default=60.0,
                   help="Северная граница региона (°N)")
    p.add_argument("--obs-window",   type=int, default=2)
    p.add_argument("--pred-window",  type=int, default=1)
    p.add_argument("--train-scalers", type=str, required=True,
                   help="Путь к scalers.npz из ТРЕНИРОВОЧНОГО набора")
    p.add_argument("--n-train",      type=int, default=4,
                   help="Сколько первых сэмплов отложить как «train» "
                        "(только для windowed формата, не для chunked)")
    p.add_argument("--static-from",  type=str, default=None,
                   help="Путь к тренировочному data.npy для z_surf/lsm "
                        "(только для 19-var; если не указан — z_surf/lsm считаются из geopotential/lsm в ARCO)")

    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Определяем формат по scalers ──────────────────────────────────
    sc_path = Path(args.train_scalers)
    sc = np.load(sc_path)
    sc_keys = set(sc.keys())

    if "mean" in sc_keys and "std" in sc_keys:
        # 19-var chunked формат (V2 модель)
        n_feat = len(sc["mean"])
        if n_feat == 19:
            mode = "19var"
            var_order = VAR_ORDER_19
            scale_factors = SCALE_FACTORS_19
            print(f"[MODE] 19-var chunked (V2 модель)")
        else:
            raise ValueError(f"scalers.npz имеет mean длины {n_feat}, ожидалось 19 или x_mean/x_scale формат")
    elif "x_mean" in sc_keys and "x_scale" in sc_keys:
        # 15-var windowed формат (старая модель)
        mode = "15var"
        var_order = VAR_ORDER_15
        scale_factors = SCALE_FACTORS_15
        print(f"[MODE] 15-var windowed (старая модель)")
    else:
        raise ValueError(f"Неизвестный формат scalers.npz: keys={sc_keys}")

    n_feat = len(var_order)
    print(f"  Переменных: {n_feat}")
    print(f"  Порядок: {var_order}")

    # ── 1. Скачиваем данные ───────────────────────────────────────────
    channels, lons, lats, n_time = download_region(
        args.start_date, args.end_date,
        args.lon_min, args.lon_max, args.lat_min, args.lat_max,
        scale_factors=scale_factors,
    )

    # ── 1b. Статические переменные (z_surf, lsm) для 19-var ──────────
    if mode == "19var":
        n_lon, n_lat = len(lons), len(lats)
        for static_var in ["z_surf", "lsm"]:
            if static_var in channels:
                continue  # уже скачали (не должно быть, но на всякий)

            if args.static_from:
                # Берём из тренировочного датасета и интерполируем
                print(f"\n  {static_var}: интерполируем из {args.static_from}...")
                import json as _json
                train_dir = Path(args.static_from)
                info = _json.loads((train_dir / "dataset_info.json").read_text())
                shape = (info["n_time"], info["n_lon"], info["n_lat"], info["n_feat"])
                mm = np.memmap(str(train_dir / "data.npy"), dtype=np.float16, mode="r", shape=shape)
                var_idx = info["variables"].index(static_var)
                global_field = mm[0, :, :, var_idx].astype(np.float32)  # (train_lon, train_lat)
                # Простая интерполяция: берём ближайшую точку из глобальной сетки
                train_coords = np.load(train_dir / "coords.npz")
                train_lons = train_coords["longitude"]
                train_lats = train_coords["latitude"]
                from scipy.interpolate import RegularGridInterpolator
                interp = RegularGridInterpolator(
                    (train_lons, train_lats), global_field,
                    method="nearest", bounds_error=False, fill_value=None)
                mesh_lon, mesh_lat = np.meshgrid(lons, lats, indexing="ij")
                static_2d = interp((mesh_lon, mesh_lat)).astype(np.float32)
                del mm
            else:
                # Скачиваем из ARCO напрямую
                print(f"\n  {static_var}: скачиваем из ARCO...")
                fs = gcsfs.GCSFileSystem(token="anon")
                store = fs.get_mapper(ARCO_ZARR)
                ds = xr.open_zarr(store, consolidated=True)
                lat_c = "latitude"  if "latitude"  in ds.coords else "lat"
                lon_c = "longitude" if "longitude" in ds.coords else "lon"
                ds = ds.sel({
                    lon_c: slice(args.lon_min, args.lon_max),
                    lat_c: slice(args.lat_max, args.lat_min),
                })
                if static_var == "z_surf":
                    # geopotential at surface — берём первый таймстеп
                    da = ds["geopotential"].isel(time=0)
                    if "level" in da.dims:
                        # Берём самый нижний уровень (1000 hPa)
                        da = da.sel(level=1000)
                    static_2d = da.values.astype(np.float32)
                    if static_var in scale_factors:
                        static_2d *= scale_factors[static_var]
                elif static_var == "lsm":
                    da = ds["land_sea_mask"].isel(time=0) if "land_sea_mask" in ds else None
                    if da is None:
                        print(f"    land_sea_mask не найден в ARCO, заполняем нулями")
                        static_2d = np.zeros((len(lons), len(lats)), dtype=np.float32)
                    else:
                        static_2d = da.values.astype(np.float32)

                # Переставляем оси если нужно (lat, lon) → (lon, lat)
                if static_2d.shape == (len(lats), len(lons)):
                    static_2d = static_2d.T

            channels[static_var] = np.tile(static_2d[np.newaxis, :, :], (n_time, 1, 1))
            print(f"    {static_var}: shape={channels[static_var].shape}  "
                  f"range=[{static_2d.min():.3f}, {static_2d.max():.3f}]")

    # ── 2. Собираем (T, LON, LAT, F) массив ──────────────────────────
    n_lon, n_lat = len(lons), len(lats)

    print(f"\nСборка массива: ({n_time}, {n_lon}, {n_lat}, {n_feat})")
    arr = np.zeros((n_time, n_lon, n_lat, n_feat), dtype=np.float32)
    for i, var in enumerate(var_order):
        if var not in channels:
            raise RuntimeError(f"Отсутствует канал: {var}")
        arr[:, :, :, i] = channels[var]
        print(f"  {i:2d}: {var:>8s}  "
              f"mean={channels[var].mean():.4f}  std={channels[var].std():.4f}")

    del channels  # освобождаем память

    # ── 3. Сохраняем в зависимости от формата ─────────────────────────
    if mode == "19var":
        _save_chunked(arr, lons, lats, var_order, sc_path, out_dir,
                      args.start_date, args.end_date)
    else:
        _save_windowed(arr, lons, lats, var_order, sc_path, sc, out_dir,
                       args.obs_window, args.pred_window, args.n_train)

    # Размер файлов
    total_mb = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file()) / (1024**2)
    print(f"\n  Общий размер: {total_mb:.1f} MB")
    print(f"\n{'='*60}")
    print(f"✓ Готово!")

    if mode == "19var":
        print(f"\nДля инференса:")
        print(f"  python scripts/predict.py experiments/wb2_512x256_19f_ar \\")
        print(f"    --data-dir {out_dir} --ar-steps 4 --max-samples 50 \\")
        print(f"    --region {args.lat_min} {args.lat_max} {args.lon_min} {args.lon_max} --per-channel")
    else:
        print(f"\nДля инференса:")
        print(f"  python scripts/predict.py experiments/wb2_64x32_15f_4pred \\")
        print(f"    --data-dir {out_dir} \\")
        print(f"    --region {args.lat_min} {args.lat_max} {args.lon_min} {args.lon_max} --per-channel")
    print(f"{'='*60}")


def _save_chunked(arr, lons, lats, var_order, sc_path, out_dir,
                  start_date, end_date):
    """Сохраняем в chunked timeseries формате (для V2 модели / dataloader_chunked)."""
    n_time, n_lon, n_lat, n_feat = arr.shape

    print(f"\n--- Сохраняем в CHUNKED формате (data.npy + dataset_info.json) ---")

    # data.npy — raw memmap float16
    data_path = out_dir / "data.npy"
    fp = np.memmap(str(data_path), dtype=np.float16, mode="w+",
                   shape=(n_time, n_lon, n_lat, n_feat))
    fp[:] = arr.astype(np.float16)
    fp.flush()
    del fp

    size_mb = n_time * n_lon * n_lat * n_feat * 2 / (1024**2)
    print(f"  data.npy: {size_mb:.1f} MB")

    # scalers.npz — копируем из тренировочного
    shutil.copy2(sc_path, out_dir / "scalers.npz")
    print(f"  scalers.npz (скопировано)")

    # coords.npz
    np.savez(out_dir / "coords.npz",
             longitude=lons.astype(np.float32),
             latitude=lats.astype(np.float32))
    print(f"  coords.npz")

    # variables.json
    (out_dir / "variables.json").write_text(json.dumps(var_order, indent=2))
    print(f"  variables.json")

    # dataset_info.json
    info = {
        "time_start": start_date,
        "time_end": end_date,
        "n_time": int(n_time),
        "n_lon": int(n_lon),
        "n_lat": int(n_lat),
        "n_feat": int(n_feat),
        "variables": var_order,
        "dtype": "float16",
        "file": "data.npy",
        "size_gb": round(size_mb / 1024, 2),
        "source": "ARCO ERA5 (regional, 0.25°)",
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"  dataset_info.json")


def _save_windowed(arr, lons, lats, var_order, sc_path, sc, out_dir,
                   obs_window, pred_window, n_train_min):
    """Сохраняем в windowed формате (X_train.pt / y_train.pt / ... для старой модели)."""
    n_time, n_lon, n_lat, n_feat = arr.shape

    print(f"\n--- Сохраняем в WINDOWED формате (X/y_train/test.pt) ---")

    # Строим сэмплы
    X, Y = build_samples(arr, obs_window, pred_window)
    n_samples = X.shape[0]
    print(f"  Сэмплов: {n_samples}  (obs={obs_window}, pred={pred_window})")

    # Flatten time→channel
    _, obs, LON, LAT, F = X.shape
    X = X.transpose(0, 2, 3, 1, 4).reshape(n_samples, LON, LAT, obs * F)
    Y = Y.transpose(0, 2, 3, 1, 4).reshape(n_samples, LON, LAT, pred_window * F)

    # Нормализуем тренировочными скейлерами
    x_mean, x_scale = sc["x_mean"], sc["x_scale"]
    y_mean, y_scale = sc["y_mean"], sc["y_scale"]

    Xn = ((X - x_mean) / x_scale).astype(np.float32)
    Yn = ((Y - y_mean) / y_scale).astype(np.float32)

    print(f"  Xn: mean={Xn.mean():.3f}, std={Xn.std():.3f}")
    print(f"  Yn: mean={Yn.mean():.3f}, std={Yn.std():.3f}")

    n_train = min(n_train_min, n_samples // 2)

    torch.save(torch.tensor(Xn[:n_train]), out_dir / "X_train.pt")
    torch.save(torch.tensor(Yn[:n_train]), out_dir / "y_train.pt")
    torch.save(torch.tensor(Xn),          out_dir / "X_test.pt")
    torch.save(torch.tensor(Yn),          out_dir / "y_test.pt")

    shutil.copy2(sc_path, out_dir / "scalers.npz")

    np.savez(out_dir / "coords.npz",
             longitude=lons.astype(np.float32),
             latitude=lats.astype(np.float32))

    (out_dir / "variables.json").write_text(
        json.dumps(var_order, ensure_ascii=False, indent=2))

    print(f"  Train: {n_train} samples (dummy)")
    print(f"  Test:  {n_samples} samples → {n_samples//2} val + {n_samples - n_samples//2} test")


if __name__ == "__main__":
    main()

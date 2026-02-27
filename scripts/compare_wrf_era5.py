#!/usr/bin/env python3
"""
scripts/compare_wrf_era5.py

Сравнение WRF vs ERA5 (ground truth) на домене d03 Красноярска.
WRF данные берутся из JSON-экспорта (wrf_d03_jan2023.json),
ERA5 — из датасета wb2_512x256_19f_jan2023.

Что делает:
  1. Загружает WRF поля T2/U10/V10/PSFC из JSON (25 часовых шагов, 96×84)
  2. Загружает ERA5 jan2023 (32 × 512 × 256 × 19), денормализует
  3. Интерполирует ERA5 на WRF-сетку для 5 совпадающих 6h шагов
  4. Считает RMSE / MAE / bias по каждой переменной

Запуск:
    python scripts/compare_wrf_era5.py \
      --wrf-json data/wrf_d03_jan2023.json \
      --era5-dir data/datasets/wb2_512x256_19f_jan2023 \
      --scalers-dir data/datasets/wb2_512x256_19f_ar
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ─── Маппинг переменных ──────────────────────────────────────────────

# WRF JSON key → наше имя в ERA5 → индекс в variables.json
# WRF PSFC в Pa, наш sp после денорм в hPa
VAR_MAP = {
    "t2_K":    {"era5": "t2m",  "unit": "K",    "era5_to_wrf": 1.0},
    "u10_ms":  {"era5": "10u",  "unit": "m/s",  "era5_to_wrf": 1.0},
    "v10_ms":  {"era5": "10v",  "unit": "m/s",  "era5_to_wrf": 1.0},
    "psfc_Pa": {"era5": "sp",   "unit": "hPa",  "era5_to_wrf": 0.01},
    # WRF PSFC в Pa → делим на 100 чтобы получить hPa (как ERA5 после денорм)
}

# Масштабные коэффициенты, которые применялись при сборке ERA5 датасета
# (x_raw * scale → x_stored, потом нормализовано scalers)
# После денорм: x_denorm = x_norm * std + mean  (в scaled units)
DATASET_SCALE = {
    "msl":    0.01,       # Pa → hPa
    "sp":     0.01,       # Pa → hPa
    "z_surf": 1/9.80665,  # m²/s² → m
    "z@850":  1/9.80665,
    "z@500":  1/9.80665,
}


def load_wrf_json(path):
    """Загрузка WRF из JSON."""
    with open(path) as f:
        d = json.load(f)

    domain = d["domain"]
    times = d["times"]
    lat = np.array(d["lat"], dtype=np.float32)  # (96, 84)
    lon = np.array(d["lon"], dtype=np.float32)  # (96, 84)

    fields = {}
    for key in ["t2_K", "u10_ms", "v10_ms", "psfc_Pa"]:
        fields[key] = np.array(d["domain_fields"][key], dtype=np.float32)  # (25, 96, 84)

    print(f"WRF: domain [{domain['lat_min']:.2f},{domain['lat_max']:.2f}]N "
          f"× [{domain['lon_min']:.2f},{domain['lon_max']:.2f}]E")
    print(f"  Grid: {lat.shape}, Times: {len(times)} hourly steps")
    print(f"  Period: {times[0]} → {times[-1]}")

    return fields, lat, lon, times, domain


def load_era5(era5_dir, scalers_dir=None):
    """
    Загрузка ERA5 jan2023.

    ВАЖНО: data.npy хранит данные в ФИЗИЧЕСКИХ единицах (K, m/s, hPa, м),
    а НЕ нормализованные. Нормализация (x-mean)/std делается в dataloader
    при обучении, но в файле значения «как есть».
    Поэтому денормализация НЕ нужна.
    """
    era5_dir = Path(era5_dir)

    with open(era5_dir / "dataset_info.json") as f:
        info = json.load(f)
    with open(era5_dir / "variables.json") as f:
        var_names = json.load(f)

    coords = np.load(era5_dir / "coords.npz")
    lons = coords["longitude"].astype(np.float64)   # (512,)
    lats = coords["latitude"].astype(np.float64)     # (256,)

    n_time = info["n_time"]
    n_lon  = info["n_lon"]
    n_lat  = info["n_lat"]
    n_feat = info["n_feat"]

    data_path = era5_dir / "data.npy"
    try:
        data = np.memmap(str(data_path), dtype=np.float16, mode="r",
                         shape=(n_time, n_lon, n_lat, n_feat))
    except Exception:
        data = np.load(data_path, allow_pickle=True, mmap_mode="r")
        if data.ndim == 1:
            data = data.reshape(n_time, n_lon, n_lat, n_feat)

    print(f"\nERA5 jan2023: shape={data.shape}, dtype={data.dtype}")
    print(f"  Period: {info.get('time_start','?')} → {info.get('time_end','?')}")
    print(f"  Grid: lon [{lons.min():.2f},{lons.max():.2f}] ({len(lons)}), "
          f"lat [{lats.min():.2f},{lats.max():.2f}] ({len(lats)})")
    print(f"  Variables: {var_names}")

    # Проверка: t2m должна быть ~200-320 K, не нормализованная ~[-3,3]
    sample_t2m = float(np.nanmean(data[0, :, :, var_names.index("t2m")]))
    if abs(sample_t2m) < 10:
        print(f"  ⚠ WARNING: t2m mean={sample_t2m:.2f} — данные НОРМАЛИЗОВАНЫ!")
        print(f"    Нужно указать --scalers-dir для денормализации")
    else:
        print(f"  ✓ t2m sample mean={sample_t2m:.1f} K — данные в физ. единицах")

    return data, lons, lats, var_names, info


def interpolate_era5_to_wrf(era5_field_2d, era5_lons, era5_lats, wrf_lat, wrf_lon):
    """
    Билинейная интерполяция ERA5 (регулярная сетка) на WRF точки (курвилинейная).

    era5_field_2d: (n_lon, n_lat) — поле ERA5
    era5_lons: (n_lon,)
    era5_lats: (n_lat,)
    wrf_lat, wrf_lon: (96, 84) — координаты WRF точек

    Returns: (96, 84) — интерполированное поле на WRF сетке
    """
    # RegularGridInterpolator принимает (n_lon, n_lat) данные
    # и точки как (lon, lat)
    interp = RegularGridInterpolator(
        (era5_lons, era5_lats),
        era5_field_2d,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # WRF точки
    points = np.stack([wrf_lon.ravel(), wrf_lat.ravel()], axis=-1)  # (96*84, 2)
    result = interp(points).reshape(wrf_lat.shape)  # (96, 84)

    return result


def compute_metrics(pred, truth):
    """RMSE, MAE, bias для массивов."""
    mask = ~(np.isnan(pred) | np.isnan(truth))
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    diff = pred[mask] - truth[mask]
    rmse = np.sqrt(np.mean(diff ** 2))
    mae  = np.mean(np.abs(diff))
    bias = np.mean(diff)
    return rmse, mae, bias


def main():
    p = argparse.ArgumentParser(description="WRF vs ERA5 comparison on Krasnoyarsk d03")
    p.add_argument("--wrf-json", required=True, help="Path to wrf_d03_jan2023.json")
    p.add_argument("--era5-dir", required=True, help="Path to wb2_512x256_19f_jan2023/")
    p.add_argument("--scalers-dir", default=None,
                   help="Dir with scalers.npz (default: era5-dir)")
    args = p.parse_args()

    # ─── 1. Загрузка данных ───────────────────────────────────────────
    wrf_fields, wrf_lat, wrf_lon, wrf_times, wrf_domain = load_wrf_json(args.wrf_json)
    era5_data, era5_lons, era5_lats, var_names, era5_info = load_era5(
        args.era5_dir, args.scalers_dir)

    # ─── 2. Совмещение временных шагов ────────────────────────────────
    # ERA5: start 2023-01-18 00:00, 6h steps, 32 шагов
    # WRF:  start 2023-01-20 00:00, hourly, 25 шагов (до 2023-01-21 00:00)
    #
    # ERA5 index 8  = 2023-01-20 00:00 (offset = 2 days × 4 steps/day)
    # ERA5 index 9  = 2023-01-20 06:00
    # ERA5 index 10 = 2023-01-20 12:00
    # ERA5 index 11 = 2023-01-20 18:00
    # ERA5 index 12 = 2023-01-21 00:00
    #
    # WRF hourly index: 0, 6, 12, 18, 24

    era5_start = era5_info.get("time_start", "2023-01-18")
    # Вычисляем offset от начала ERA5 до начала WRF
    from datetime import datetime
    era5_t0 = datetime.strptime(era5_start, "%Y-%m-%d")
    wrf_t0  = datetime.strptime(wrf_times[0][:10], "%Y-%m-%d")
    days_offset = (wrf_t0 - era5_t0).days
    era5_base_idx = days_offset * 4  # 6h шаги в сутке = 4

    # 5 совпадающих шагов: инициализация + 4 горизонта по 6h
    match_steps = []
    for h in range(5):  # 0h, 6h, 12h, 18h, 24h
        era5_idx = era5_base_idx + h
        wrf_hour_idx = h * 6  # 0, 6, 12, 18, 24
        if era5_idx < era5_data.shape[0] and wrf_hour_idx < len(wrf_times):
            match_steps.append((era5_idx, wrf_hour_idx, f"+{h*6:02d}h"))

    print(f"\nMatched 6h steps: {len(match_steps)}")
    for ei, wi, label in match_steps:
        print(f"  {label}: ERA5 idx={ei}, WRF time={wrf_times[wi]}")

    # ─── 3. Интерполяция и RMSE ───────────────────────────────────────
    print("\n" + "=" * 75)
    print("WRF vs ERA5  —  per-variable, per-horizon RMSE on d03 domain")
    print("=" * 75)

    # Результаты для итоговой таблицы
    results = {}  # var → {horizon → (rmse, mae, bias)}

    for wrf_key, vinfo in VAR_MAP.items():
        era5_name = vinfo["era5"]
        unit = vinfo["unit"]
        wrf_to_era5_scale = vinfo["era5_to_wrf"]

        if era5_name not in var_names:
            print(f"  SKIP {era5_name}: not in ERA5 dataset")
            continue

        vi = var_names.index(era5_name)
        wrf_field_all = wrf_fields[wrf_key]  # (25, 96, 84)

        results[era5_name] = {}
        all_rmse = []

        print(f"\n  {era5_name} ({unit}):")

        for era5_idx, wrf_hour_idx, label in match_steps:
            # ERA5: данные уже в физических единицах (K, m/s, hPa)
            era5_phys = era5_data[era5_idx, :, :, vi].astype(np.float32)  # (512, 256)

            # Интерполяция ERA5 на WRF-сетку
            era5_on_wrf = interpolate_era5_to_wrf(
                era5_phys, era5_lons, era5_lats, wrf_lat, wrf_lon)  # (96, 84)

            # WRF в тех же единицах что ERA5
            wrf_slice = wrf_field_all[wrf_hour_idx]  # (96, 84)
            wrf_converted = wrf_slice * wrf_to_era5_scale  # Pa→hPa для PSFC

            rmse, mae, bias = compute_metrics(wrf_converted, era5_on_wrf)
            results[era5_name][label] = (rmse, mae, bias)
            all_rmse.append(rmse)

            # Для отладки: средние значения
            era5_mean = np.nanmean(era5_on_wrf)
            wrf_mean  = np.nanmean(wrf_converted)

            print(f"    {label}: RMSE={rmse:.3f} {unit} | MAE={mae:.3f} | "
                  f"bias={bias:+.3f} | WRF_mean={wrf_mean:.1f} ERA5_mean={era5_mean:.1f}")

        avg_rmse = np.nanmean(all_rmse[1:])  # без 0h (инициализация = маленькая ошибка WRF)
        print(f"    AVG (без 0h): RMSE={avg_rmse:.3f} {unit}")

    # ─── 4. Итоговая таблица ──────────────────────────────────────────
    print("\n" + "=" * 75)
    print("ИТОГОВАЯ ТАБЛИЦА  —  WRF RMSE vs ERA5 на домене Красноярска")
    print("(для сравнения с GraphCast-lite)")
    print("=" * 75)

    header = f"{'Переменная':<12} | {'Единица':<8}"
    for _, _, label in match_steps:
        header += f" | {label:>8}"
    header += f" | {'AVG':>8}"
    print(header)
    print("-" * len(header))

    for wrf_key, vinfo in VAR_MAP.items():
        era5_name = vinfo["era5"]
        unit = vinfo["unit"]
        if era5_name not in results:
            continue

        row = f"{era5_name:<12} | {unit:<8}"
        vals = []
        for _, _, label in match_steps:
            r, _, _ = results[era5_name].get(label, (np.nan, 0, 0))
            row += f" | {r:>8.3f}"
            vals.append(r)
        avg = np.nanmean(vals[1:]) if len(vals) > 1 else np.nan
        row += f" | {avg:>8.3f}"
        print(row)

    print("=" * 75)
    print("\nПримечания:")
    print("  - +00h = инициализация WRF (ожидается малая ошибка vs ERA5)")
    print("  - +06h..+24h = прогноз WRF")
    print("  - AVG = среднее RMSE по горизонтам +06h..+24h (без инициализации)")
    print("  - PSFC сравнивается в hPa (WRF PSFC ÷ 100)")
    print("  - ERA5 интерполирована билинейно на WRF-сетку d03 (96×84)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/compare_wrf.py

Сравнение GraphCast-lite vs WRF vs ERA5 (ground truth) на WRF-домене.

Что делает:
  1. Загружает predictions.pt из нашего инференса
  2. Денормализует (pred * std + mean) обратно в физические единицы
  3. Извлекает точки в WRF d03 домене (55.5-56.4N, 92.3-93.6E)
  4. Загружает WRF netcdf, извлекает T2/U10/V10/PSFC, усредняет по домену
  5. Загружает ERA5 ground truth (из того же датасета), денормализует
  6. Сравнивает RMSE/MAE/bias для каждой переменной в физических единицах

Формат вывода: таблица для диплома.

Запуск на кластере:
    python scripts/compare_wrf.py \
      --predictions experiments/wb2_512x256_19f_ar/predictions.pt \
      --data-dir data/datasets/region_krsk_arco_19f \
      --wrf-path /path/to/wrfout_d03_2023-01-20_00:00:00_....nc \
      --experiment-dir experiments/wb2_512x256_19f_ar
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ─── Конфигурация ─────────────────────────────────────────────────────

# WRF d03 домен (Красноярск)
WRF_LAT_MIN, WRF_LAT_MAX = 55.5, 56.5
WRF_LON_MIN, WRF_LON_MAX = 92.0, 94.0

# Маппинг: наше имя → WRF имя → единица измерения → преобразование WRF→наше
# WRF T2 в Кельвинах, PSFC в Паскалях — совпадает с ERA5
VAR_MAPPING = {
    "t2m":  {"wrf_name": "T2",   "unit": "K",    "wrf_scale": 1.0,    "our_scale": 1.0},
    "10u":  {"wrf_name": "U10",  "unit": "m/s",  "wrf_scale": 1.0,    "our_scale": 1.0},
    "10v":  {"wrf_name": "V10",  "unit": "m/s",  "wrf_scale": 1.0,    "our_scale": 1.0},
    "sp":   {"wrf_name": "PSFC", "unit": "Pa",   "wrf_scale": 1.0,    "our_scale": 100.0},
    # sp в нашем датасете хранится в hPa (×0.01 при сборке), так что обратно ×100
}

# Для красивого вывода
UNIT_DISPLAY = {
    "t2m": "K (or degC delta)",
    "10u": "m/s",
    "10v": "m/s",
    "sp":  "hPa",
}


def load_predictions_and_truth(data_dir, pred_path, exp_dir, ar_steps):
    """
    Загружает предсказания и ground truth, денормализует.
    Поддерживает:
      - Новый формат predictions.pt (dict с "predictions" + "ground_truth")
      - Старый формат (голый тензор, truth из data.npy)
      - Flat grid (multires) и регулярную сетку
    Returns: pred_region, truth_region — np.ndarray (n_samples, n_region_nodes, P, C) в физ.единицах
             var_names, P, n_samples
    """
    data_dir = Path(data_dir)
    exp_dir = Path(exp_dir)

    # Загружаем scalers
    sc = np.load(data_dir / "scalers.npz")
    sc_keys = set(sc.keys())

    if "mean" in sc_keys:
        mean = sc["mean"].astype(np.float32)
        std = sc["std"].astype(np.float32)
    elif "y_mean" in sc_keys:
        mean = sc["y_mean"].astype(np.float32)
        std = sc["y_scale"].astype(np.float32)
    else:
        raise ValueError("Expected scalers with 'mean'/'std' or 'y_mean'/'y_scale'")

    # Загружаем variables
    with open(data_dir / "variables.json") as f:
        var_names = json.load(f)

    # Загружаем coords + определяем flat grid
    coords = np.load(data_dir / "coords.npz")
    lons = coords["longitude"].astype(np.float32)
    lats = coords["latitude"].astype(np.float32)

    # Flat grid: coords имеют shape (N_nodes,) и len(lats) == len(lons) == N
    # Regular grid: coords имеют 1D оси, len(lats) == n_lat, len(lons) == n_lon
    info_path = data_dir / "dataset_info.json"
    flat_grid = False
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        flat_grid = info.get("flat_grid", info.get("flat", False))
    # Auto-detect: if lats and lons have same length AND it doesn't equal n_lat*n_lon
    # of a plausible regular grid, it's flat
    if not flat_grid and len(lats) == len(lons) and len(lats) > max(512, 256):
        flat_grid = True
        print("  (auto-detected flat grid: %d nodes)" % len(lats))

    if flat_grid:
        # Flat grid: lats/lons are per-node (N,) — filter directly by bbox
        mask = ((lats >= WRF_LAT_MIN) & (lats <= WRF_LAT_MAX) &
                (lons >= WRF_LON_MIN) & (lons <= WRF_LON_MAX))
        region_indices = np.where(mask)[0]
        print("Region (flat grid): lat [%.1f, %.1f], lon [%.1f, %.1f]" % (
            WRF_LAT_MIN, WRF_LAT_MAX, WRF_LON_MIN, WRF_LON_MAX))
        print("  %d nodes in region" % len(region_indices))
    else:
        # Regular grid
        n_lon, n_lat = len(lons), len(lats)
        lat_mask = (lats >= WRF_LAT_MIN) & (lats <= WRF_LAT_MAX)
        lon_mask = (lons >= WRF_LON_MIN) & (lons <= WRF_LON_MAX)
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]

        region_indices = []
        for j in lon_idx:
            for i in lat_idx:
                region_indices.append(j * n_lat + i)
        region_indices = np.array(region_indices)

        print("Region (regular grid): lat [%.1f, %.1f], lon [%.1f, %.1f]" % (
            WRF_LAT_MIN, WRF_LAT_MAX, WRF_LON_MIN, WRF_LON_MAX))
        print("  lat points: %d, lon points: %d, total nodes: %d" % (
            len(lat_idx), len(lon_idx), len(region_indices)))

    # Загружаем predictions — поддержка нового dict формата
    bundle = torch.load(pred_path, map_location="cpu")

    if isinstance(bundle, dict) and "predictions" in bundle:
        # Новый формат: dict с predictions + ground_truth (оба нормализованы)
        preds_raw = bundle["predictions"]
        truth_raw = bundle["ground_truth"]
        C = bundle.get("n_features", len(var_names))
        print("Predictions (dict): pred=%s, truth=%s" % (
            str(preds_raw.shape), str(truth_raw.shape)))
    else:
        # Старый формат: голый тензор, truth реконструируем из data.npy
        preds_raw = bundle
        truth_raw = None
        C = len(var_names)
        print("Predictions (legacy tensor): %s" % str(preds_raw.shape))

    if preds_raw.dim() == 2:
        preds_raw = preds_raw.unsqueeze(0)

    n_samples = preds_raw.shape[0]
    G = preds_raw.shape[1]
    P = preds_raw.shape[2] // C

    print("Samples: %d, Grid: %d, Channels: %d, Horizons: %d" % (n_samples, G, C, P))

    # Денормализуем predictions
    preds_4d = preds_raw.numpy().reshape(n_samples, G, P, C)
    preds_phys = preds_4d * std[np.newaxis, np.newaxis, np.newaxis, :C] + mean[np.newaxis, np.newaxis, np.newaxis, :C]

    # Денормализуем ground truth
    if truth_raw is not None:
        # Из predictions.pt dict
        if truth_raw.dim() == 2:
            truth_raw = truth_raw.unsqueeze(0)
        truth_4d = truth_raw.numpy().reshape(n_samples, G, P, C)
        truth_phys = truth_4d * std[np.newaxis, np.newaxis, np.newaxis, :C] + mean[np.newaxis, np.newaxis, np.newaxis, :C]
    elif info_path.exists() and not flat_grid:
        # Старый путь: реконструкция из data.npy (только regular grid)
        shape = (info["n_time"], info["n_lon"], info["n_lat"], info["n_feat"])
        data = np.memmap(str(data_dir / "data.npy"), dtype=np.float16, mode="r", shape=shape)
        n_lon_d, n_lat_d = info["n_lon"], info["n_lat"]
        data_flat = data.reshape(shape[0], n_lon_d * n_lat_d, shape[3]).astype(np.float32)

        obs_w = 2
        total_samples = shape[0] - obs_w - ar_steps + 1
        split_idx = int(total_samples * 0.8)
        test_all = total_samples - split_idx
        val_size = test_all // 2
        test_only_start = split_idx + val_size

        n_test_samples = min(n_samples, test_all - val_size)
        truth_phys = np.zeros((n_test_samples, G, P, C), dtype=np.float32)
        for i in range(n_test_samples):
            global_t = test_only_start + i
            for p in range(P):
                t_idx = global_t + obs_w + p
                if t_idx < shape[0]:
                    truth_phys[i, :, p, :] = data_flat[t_idx]

        del data, data_flat
        n_samples = n_test_samples
        preds_phys = preds_phys[:n_test_samples]
    else:
        raise RuntimeError("No ground truth available: predictions.pt has no 'ground_truth' key "
                           "and dataset_info.json not found or flat_grid=True")

    # Извлекаем region
    pred_region = preds_phys[:, region_indices, :, :]
    truth_region = truth_phys[:, region_indices, :, :]

    return pred_region, truth_region, var_names, P, n_samples


def _load_wrf_json(wrf_path, n_horizons):
    """Загрузка WRF из JSON-экспорта (wrf_d03_jan2023.json)."""
    with open(wrf_path) as f:
        wrf_json = json.load(f)

    domain_mean = wrf_json.get("domain_mean", {})
    times = wrf_json.get("times", [])
    print("\nWRF dataset (JSON):")
    print("  Times: %d, domain: %s" % (len(times), wrf_json.get("domain", "")))

    # Маппинг JSON ключей → наше имя
    JSON_KEY_MAP = {
        "t2_K":    "t2m",
        "u10_ms":  "10u",
        "v10_ms":  "10v",
        "psfc_Pa": "sp",
    }

    wrf_data = {}
    for json_key, our_name in JSON_KEY_MAP.items():
        if json_key not in domain_mean:
            print("  WARNING: %s not found in WRF JSON" % json_key)
            continue

        hourly = np.array(domain_mean[json_key], dtype=np.float32)
        n_wrf_steps = len(hourly)

        # WRF почасовой → нужны 6h шаги: 0, 6, 12, 18, 24
        if n_wrf_steps >= 25:
            idx_6h = [0, 6, 12, 18, 24]
        elif n_wrf_steps >= 5:
            idx_6h = list(range(min(5, n_wrf_steps)))
        else:
            idx_6h = list(range(n_wrf_steps))

        wrf_6h = hourly[idx_6h]
        wrf_data[our_name] = wrf_6h

        print("  %s (%s): %d hourly -> %d 6h-steps, mean=%.2f" % (
            our_name, json_key, n_wrf_steps, len(wrf_6h), wrf_6h.mean()))

    return wrf_data


def load_wrf(wrf_path, n_horizons):
    """
    Загружает WRF данные. Поддерживает:
      - JSON экспорт (wrf_d03_jan2023.json) — domain_mean с почасовыми значениями
      - netCDF (xarray / netCDF4) — полные 3D поля
    Returns: dict{our_var_name: np.ndarray (n_6h_steps,)} усреднённые по домену значения
    """
    # JSON формат?
    if str(wrf_path).endswith(".json"):
        return _load_wrf_json(wrf_path, n_horizons)

    try:
        import xarray as xr
    except ImportError:
        try:
            from netCDF4 import Dataset as NC4Dataset
            return _load_wrf_netcdf4(wrf_path, n_horizons)
        except ImportError:
            print("WARNING: neither xarray nor netCDF4 available, skipping WRF")
            return None

    ds = xr.open_dataset(wrf_path)
    print("\nWRF dataset:")
    print("  Time steps: %d" % ds.sizes.get("Time", ds.sizes.get("time", 0)))

    wrf_data = {}
    for our_name, info in VAR_MAPPING.items():
        wrf_name = info["wrf_name"]
        if wrf_name not in ds.data_vars:
            print("  WARNING: %s not found in WRF" % wrf_name)
            continue

        da = ds[wrf_name]
        # WRF output: (Time, south_north, west_east) or similar
        # Усредняем по пространству
        spatial_dims = [d for d in da.dims if d != "Time" and d != "time"]
        domain_mean = da.mean(dim=spatial_dims).values.astype(np.float32)

        # WRF обычно почасовой, нам нужны 6h шаги: 0, 6, 12, 18, 24
        # (если 25 шагов: index 0, 6, 12, 18, 24)
        n_wrf_steps = len(domain_mean)
        if n_wrf_steps >= 25:
            idx_6h = [0, 6, 12, 18, 24]
        elif n_wrf_steps >= 5:
            # Может уже 6h
            idx_6h = list(range(min(5, n_wrf_steps)))
        else:
            idx_6h = list(range(n_wrf_steps))

        wrf_6h = domain_mean[idx_6h]

        # Преобразование единиц
        wrf_data[our_name] = wrf_6h * info["wrf_scale"]

        print("  %s (%s): %d hourly -> %d 6h-steps, mean=%.2f %s" % (
            our_name, wrf_name, n_wrf_steps, len(wrf_6h),
            wrf_6h.mean(), info["unit"]))

    ds.close()
    return wrf_data


def _load_wrf_netcdf4(wrf_path, n_horizons):
    """Fallback: загрузка через netCDF4."""
    from netCDF4 import Dataset as NC4Dataset

    ds = NC4Dataset(wrf_path, "r")
    print("\nWRF dataset (netCDF4):")

    wrf_data = {}
    for our_name, info in VAR_MAPPING.items():
        wrf_name = info["wrf_name"]
        if wrf_name not in ds.variables:
            print("  WARNING: %s not found in WRF" % wrf_name)
            continue

        arr = ds.variables[wrf_name][:].astype(np.float32)
        # (Time, south_north, west_east) → mean over space
        domain_mean = arr.mean(axis=tuple(range(1, arr.ndim)))

        n_wrf_steps = len(domain_mean)
        if n_wrf_steps >= 25:
            idx_6h = [0, 6, 12, 18, 24]
        else:
            idx_6h = list(range(min(5, n_wrf_steps)))

        wrf_6h = domain_mean[idx_6h] * info["wrf_scale"]
        wrf_data[our_name] = wrf_6h

        print("  %s (%s): %d -> %d 6h, mean=%.2f" % (
            our_name, wrf_name, n_wrf_steps, len(wrf_6h), wrf_6h.mean()))

    ds.close()
    return wrf_data


def compute_metrics(pred, truth):
    """RMSE, MAE, bias для 1D массивов."""
    diff = pred - truth
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    return rmse, mae, bias


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="predictions.pt from predict.py")
    p.add_argument("--data-dir", required=True, help="regional dataset dir")
    p.add_argument("--wrf-path", default=None, help="WRF netcdf or JSON file (optional)")
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--ar-steps", type=int, default=4, help="number of AR horizons")
    p.add_argument("--wrf-sample", type=int, default=None,
                   help="Force specific sample index for WRF comparison (0-based). "
                        "Default: auto-detect from WRF dates + dataset time_start.")
    args = p.parse_args()

    print("=" * 70)
    print("GraphCast-lite vs WRF comparison")
    print("=" * 70)

    # 1. Load our predictions + ground truth
    pred_region, truth_region, var_names, P, n_samples = load_predictions_and_truth(
        args.data_dir, args.predictions, args.experiment_dir, args.ar_steps)

    C = len(var_names)

    # Load predictions bundle for sample_offsets
    bundle = torch.load(args.predictions, map_location="cpu")
    sample_offsets = bundle.get("sample_offsets", list(range(n_samples))) if isinstance(bundle, dict) else list(range(n_samples))
    obs_window = bundle.get("obs_window", 2) if isinstance(bundle, dict) else 2

    # Усредняем наши предсказания и truth по пространству (region → скаляр)
    pred_mean = pred_region.mean(axis=1)   # (n_samples, P, C)
    truth_mean = truth_region.mean(axis=1)  # (n_samples, P, C)

    # 2. Наши метрики (по всем сэмплам и горизонтам, в физ. единицах)
    print("\n" + "=" * 70)
    print("OUR MODEL metrics (domain-averaged, physical units)")
    print("=" * 70)

    compare_vars = [v for v in VAR_MAPPING.keys() if v in var_names]

    for var in compare_vars:
        vi = var_names.index(var)
        info = VAR_MAPPING[var]

        p_vals = pred_mean[:, :, vi].flatten()
        t_vals = truth_mean[:, :, vi].flatten()

        unit = "hPa" if var == "sp" else info["unit"]
        rmse, mae, bias = compute_metrics(p_vals, t_vals)

        print("  %-5s: RMSE=%.3f %s | MAE=%.3f | bias=%+.3f | mean_truth=%.1f" % (
            var, rmse, unit, mae, bias, t_vals.mean()))

        for h in range(min(P, 4)):
            p_h = pred_mean[:, h, vi]
            t_h = truth_mean[:, h, vi]
            r, m, b = compute_metrics(p_h, t_h)
            print("    +%02dh: RMSE=%.3f | MAE=%.3f | bias=%+.3f" % ((h+1)*6, r, m, b))

    # 3. WRF comparison (if available)
    wrf_data = None
    if args.wrf_path:
        wrf_data = load_wrf(args.wrf_path, P)

        if wrf_data:
            # --- Определяем какой сэмпл соответствует WRF периоду ---
            wrf_sample_idx = args.wrf_sample

            if wrf_sample_idx is None:
                # Авто-определение: читаем time_start из dataset_info.json
                # и WRF start time, находим подходящий сэмпл
                wrf_sample_idx = _find_wrf_matching_sample(
                    args.data_dir, args.wrf_path, sample_offsets, obs_window, P)

            if wrf_sample_idx is not None and wrf_sample_idx < n_samples:
                print("\n" + "=" * 70)
                print("WRF vs ERA5 vs Ours — per-horizon (sample #%d, offset=%s)" % (
                    wrf_sample_idx, sample_offsets[wrf_sample_idx] if wrf_sample_idx < len(sample_offsets) else "?"))
                print("=" * 70)

                # Наши данные для этого сэмпла (domain-averaged)
                our_pred_h = pred_mean[wrf_sample_idx]   # (P, C)
                our_truth_h = truth_mean[wrf_sample_idx]  # (P, C)

                # WRF: 5 значений (init + 4 forecast), берём forecast (skip init)
                # WRF idx: 0=init(+00h), 1=+06h, 2=+12h, 3=+18h, 4=+24h
                # Наши горизонты: 0=+06h, 1=+12h, 2=+18h, 3=+24h

                for var in compare_vars:
                    if var not in wrf_data:
                        continue

                    vi = var_names.index(var)
                    info = VAR_MAPPING[var]
                    wrf_vals = wrf_data[var]  # (5,) — init,+6,+12,+18,+24h

                    if var == "sp":
                        wrf_vals_compare = wrf_vals / 100.0  # Pa -> hPa
                        unit = "hPa"
                    else:
                        wrf_vals_compare = wrf_vals
                        unit = info["unit"]

                    print("\n  %s (%s):" % (var, unit))
                    print("    Horizon | ERA5 truth | Our pred | WRF pred | Our err | WRF err")
                    print("    " + "-" * 65)

                    n_wrf = len(wrf_vals_compare)
                    n_horizons = min(P, n_wrf - 1)  # skip WRF init

                    our_errs = []
                    wrf_errs = []
                    for h in range(n_horizons):
                        era5_val = our_truth_h[h, vi]
                        our_val = our_pred_h[h, vi]
                        wrf_val = wrf_vals_compare[h + 1]  # +1 to skip WRF init (+00h)

                        our_err = abs(our_val - era5_val)
                        wrf_err = abs(wrf_val - era5_val)
                        our_errs.append(our_err)
                        wrf_errs.append(wrf_err)

                        winner = "<- us" if our_err < wrf_err else "<- WRF" if wrf_err < our_err else "   tie"
                        print("    +%02dh    | %8.2f   | %8.2f | %8.2f  | %6.2f  | %6.2f  %s" % (
                            (h+1)*6, era5_val, our_val, wrf_val, our_err, wrf_err, winner))

                    if our_errs:
                        our_rmse = np.sqrt(np.mean(np.array(our_errs)**2))
                        wrf_rmse = np.sqrt(np.mean(np.array(wrf_errs)**2))
                        print("    AVG     |            |          |          | %6.2f  | %6.2f  %s" % (
                            our_rmse, wrf_rmse,
                            "<- us" if our_rmse < wrf_rmse else "<- WRF"))
            else:
                print("\nWARNING: Could not find WRF-matching sample in predictions.")
                print("  Available sample offsets: %s" % sample_offsets)
                print("  Use --split all when running predict.py, or --wrf-sample N")

    # 4. Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for thesis)")
    print("=" * 70)

    # Header
    header = "%-8s | %-6s" % ("Var", "Unit")
    sep = "-" * 10
    for h in range(min(P, 4)):
        header += " | Our +%02dh | WRF +%02dh" % ((h+1)*6, (h+1)*6)
        sep += "-" * 22
    header += " | Our AVG | WRF AVG"
    sep += "-" * 20
    print(header)
    print(sep)

    # Определяем wrf_sample для таблицы
    wrf_si = args.wrf_sample
    if wrf_si is None and wrf_data:
        wrf_si = _find_wrf_matching_sample(
            args.data_dir, args.wrf_path, sample_offsets, obs_window, P)

    for var in compare_vars:
        vi = var_names.index(var)
        info = VAR_MAPPING[var]
        unit = "hPa" if var == "sp" else info["unit"]

        row = "%-8s | %-6s" % (var, unit)

        our_rmses = []
        wrf_rmses = []
        for h in range(min(P, 4)):
            # Our RMSE для этого горизонта (по всем сэмплам)
            p_h = pred_mean[:, h, vi]
            t_h = truth_mean[:, h, vi]
            our_r, _, _ = compute_metrics(p_h, t_h)
            our_rmses.append(our_r)

            # WRF для конкретного сэмпла
            if wrf_data and var in wrf_data and wrf_si is not None and wrf_si < n_samples:
                wrf_vals = wrf_data[var]
                if var == "sp":
                    wrf_vals = wrf_vals / 100.0
                era5_val = truth_mean[wrf_si, h, vi]
                if h + 1 < len(wrf_vals):
                    wrf_val = wrf_vals[h + 1]
                    wrf_r = abs(wrf_val - era5_val)
                    wrf_rmses.append(wrf_r)
                    row += " | %7.3f  | %7.3f " % (our_r, wrf_r)
                else:
                    row += " | %7.3f  |    N/A " % our_r
            else:
                row += " | %7.3f  |    N/A " % our_r

        # AVG
        our_avg = np.mean(our_rmses) if our_rmses else 0
        wrf_avg = np.mean(wrf_rmses) if wrf_rmses else 0
        if wrf_rmses:
            row += " | %7.3f | %7.3f" % (our_avg, wrf_avg)
        else:
            row += " | %7.3f |    N/A" % our_avg

        print(row)

    print("=" * 70)
    if wrf_si is not None:
        print("NOTE: WRF values are for sample #%d (matching WRF period)." % wrf_si)
        print("      Our RMSE is across all %d test samples." % n_samples)
    print()


def _find_wrf_matching_sample(data_dir, wrf_path, sample_offsets, obs_window, P):
    """
    Auto-detect which sample in predictions.pt corresponds to the WRF forecast period.
    
    Logic:
      - dataset_info.json has time_start (e.g. '2023-01-18' or '2023-01-18T00:00')
      - WRF JSON has 'times' list with first entry (e.g. '2023-01-20_00:00:00')
      - Each 6h timestep in dataset = 6 hours
      - Sample at offset t has predictions at [t+obs_window, ..., t+obs_window+P-1]
      - We need sample whose predictions[0] = WRF +06h (i.e. t+obs_window maps to WRF init + 6h)
      - Or more precisely: pred[h] should match WRF[h+1] (WRF[0] = init)
    """
    from datetime import datetime, timedelta

    data_dir = Path(data_dir)

    # Parse dataset time_start
    info_path = data_dir / "dataset_info.json"
    if not info_path.exists():
        print("  WARNING: dataset_info.json not found, cannot auto-detect WRF sample")
        return None

    with open(info_path) as f:
        info = json.load(f)

    time_start_str = info.get("time_start", "")
    if not time_start_str:
        print("  WARNING: time_start not found in dataset_info.json")
        return None

    # Parse time_start (various formats)
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:
        try:
            ds_start = datetime.strptime(time_start_str, fmt)
            break
        except ValueError:
            continue
    else:
        print("  WARNING: cannot parse time_start='%s'" % time_start_str)
        return None

    # Parse WRF start time
    wrf_init_time = None
    wrf_path = Path(wrf_path)
    if wrf_path.suffix == ".json":
        with open(wrf_path) as f:
            wrf_json = json.load(f)
        times = wrf_json.get("times", [])
        if times:
            t_str = times[0].replace("_", "T")[:19]
            for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"]:
                try:
                    wrf_init_time = datetime.strptime(t_str, fmt)
                    break
                except ValueError:
                    continue

    if wrf_init_time is None:
        print("  WARNING: cannot determine WRF init time, cannot auto-detect sample")
        return None

    dt_6h = timedelta(hours=6)

    # WRF forecast targets (skip init):
    # WRF +06h = wrf_init + 6h, +12h = wrf_init + 12h, etc.
    wrf_first_forecast = wrf_init_time + dt_6h  # first forecast horizon

    # For sample at offset t: prediction[0] = ds_start + (t + obs_window) * 6h
    # We want prediction[0] = wrf_first_forecast
    # → t + obs_window = (wrf_first_forecast - ds_start) / 6h
    # → t = (wrf_first_forecast - ds_start) / 6h - obs_window

    delta = wrf_first_forecast - ds_start
    delta_steps = delta.total_seconds() / (6 * 3600)

    target_offset = int(round(delta_steps)) - obs_window

    print("\n  [WRF date matching]")
    print("    Dataset start: %s" % ds_start.isoformat())
    print("    WRF init:      %s" % wrf_init_time.isoformat())
    print("    WRF +06h:      %s" % wrf_first_forecast.isoformat())
    print("    Target offset: %d (need pred[0] at global time %d)" % (
        target_offset, target_offset + obs_window))

    # Find matching sample in predictions
    for i, off in enumerate(sample_offsets):
        if off == target_offset:
            print("    ✓ Found: sample #%d (offset=%d)" % (i, off))
            return i

    # Try ±1 if exact match not found
    for delta_off in [-1, 1, -2, 2]:
        for i, off in enumerate(sample_offsets):
            if off == target_offset + delta_off:
                print("    ~ Approximate match: sample #%d (offset=%d, wanted %d)" % (
                    i, off, target_offset))
                return i

    print("    ✗ No matching sample found! Offsets in predictions: %s" % sample_offsets[:10])
    print("      Need offset=%d. Try: predict.py --split all" % target_offset)
    return None


if __name__ == "__main__":
    main()

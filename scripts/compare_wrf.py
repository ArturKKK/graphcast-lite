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
    Returns: pred_phys, truth_phys — dict{var_name: np.ndarray (n_samples, n_region_nodes, n_horizons)}
    """
    data_dir = Path(data_dir)
    exp_dir = Path(exp_dir)

    # Загружаем scalers
    sc = np.load(data_dir / "scalers.npz")
    sc_keys = set(sc.keys())

    if "mean" in sc_keys:
        mean = sc["mean"].astype(np.float32)  # (19,)
        std = sc["std"].astype(np.float32)
        scaler_mode = "chunked"
    else:
        raise ValueError("Expected chunked scalers (mean/std) for 19-var model")

    # Загружаем variables
    with open(data_dir / "variables.json") as f:
        var_names = json.load(f)

    # Загружаем coords
    coords = np.load(data_dir / "coords.npz")
    lons = coords["longitude"]
    lats = coords["latitude"]
    n_lon, n_lat = len(lons), len(lats)

    # Индексы точек в WRF домене
    lat_mask = (lats >= WRF_LAT_MIN) & (lats <= WRF_LAT_MAX)
    lon_mask = (lons >= WRF_LON_MIN) & (lons <= WRF_LON_MAX)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    region_indices = []
    for j in lon_idx:
        for i in lat_idx:
            region_indices.append(j * n_lat + i)
    region_indices = np.array(region_indices)

    print("Region: lat [%.1f, %.1f], lon [%.1f, %.1f]" % (
        WRF_LAT_MIN, WRF_LAT_MAX, WRF_LON_MIN, WRF_LON_MAX))
    print("  lat points: %d, lon points: %d, total nodes: %d" % (
        len(lat_idx), len(lon_idx), len(region_indices)))

    # Загружаем predictions
    preds_raw = torch.load(pred_path, map_location="cpu")
    print("Predictions shape: %s" % str(preds_raw.shape))

    # preds_raw: (n_samples, G, C*P) or (n_samples, G, C)
    if preds_raw.dim() == 2:
        preds_raw = preds_raw.unsqueeze(0)

    n_samples = preds_raw.shape[0]
    G = preds_raw.shape[1]
    CP = preds_raw.shape[2]
    C = len(var_names)
    P = CP // C

    print("Samples: %d, Grid: %d, Channels: %d, Horizons: %d" % (n_samples, G, C, P))

    # Денормализуем
    # predictions нормализованы как (x - mean) / std
    # Обратно: x_phys = pred * std + mean
    preds_np = preds_raw.numpy()  # (n_samples, G, C*P)

    # Reshape to (n_samples, G, P, C)
    preds_4d = preds_np.reshape(n_samples, G, P, C)

    # Денормализация
    preds_phys = preds_4d * std[np.newaxis, np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, np.newaxis, :]

    # Загружаем ground truth из датасета
    # Для chunked: нужно прочитать data.npy и восстановить truth
    info_path = data_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        shape = (info["n_time"], info["n_lon"], info["n_lat"], info["n_feat"])
        data = np.memmap(str(data_dir / "data.npy"), dtype=np.float16, mode="r", shape=shape)

        # data shape: (T, LON, LAT, F)
        # Flatten spatial: (T, LON*LAT, F) = (T, G, F)
        data_flat = data.reshape(shape[0], n_lon * n_lat, shape[3]).astype(np.float32)

        # Нужно знать какие timesteps соответствуют каким samples
        # predict.py c chunked dataset: test_only split = последние 20%, вторая половина
        obs_w = 2  # obs_window для v2 модели
        total_samples = shape[0] - obs_w - ar_steps + 1
        split_idx = int(total_samples * 0.8)
        test_start = split_idx
        test_all = total_samples - split_idx
        val_size = test_all // 2
        test_only_start = test_start + val_size

        print("Dataset: T=%d, total_samples=%d, test_only starts at sample %d" % (
            shape[0], total_samples, test_only_start))

        # Восстанавливаем ground truth для каждого test sample
        n_test_samples = min(n_samples, test_all - val_size)
        truth_phys = np.zeros((n_test_samples, G, P, C), dtype=np.float32)

        for i in range(n_test_samples):
            global_t = test_only_start + i  # индекс оконного семпла
            for p in range(P):
                t_idx = global_t + obs_w + p  # абсолютный timestep
                if t_idx < shape[0]:
                    frame = data_flat[t_idx]  # (G, F)
                    truth_phys[i, :, p, :] = frame

        del data, data_flat
    else:
        raise RuntimeError("dataset_info.json not found, can't load ground truth")

    # Извлекаем region
    pred_region = preds_phys[:, region_indices, :, :]   # (n_samples, n_reg, P, C)
    truth_region = truth_phys[:, region_indices, :, :]

    return pred_region, truth_region, var_names, P, n_samples


def load_wrf(wrf_path, n_horizons):
    """
    Загружает WRF netcdf, извлекает T2/U10/V10/PSFC, усредняет по домену.
    Returns: dict{our_var_name: np.ndarray (n_6h_steps,)} усреднённые по домену значения
    """
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
    p.add_argument("--wrf-path", default=None, help="WRF netcdf file (optional)")
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--ar-steps", type=int, default=4, help="number of AR horizons")
    args = p.parse_args()

    print("=" * 70)
    print("GraphCast-lite vs WRF comparison")
    print("=" * 70)

    # 1. Load our predictions + ground truth
    pred_region, truth_region, var_names, P, n_samples = load_predictions_and_truth(
        args.data_dir, args.predictions, args.experiment_dir, args.ar_steps)

    C = len(var_names)

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

        # Наши значения: уже в единицах scalers (для sp -> hPa, для t2m -> K)
        p_vals = pred_mean[:, :, vi].flatten()
        t_vals = truth_mean[:, :, vi].flatten()

        # sp в нашем датасете хранится в hPa (scale 0.01 при сборке)
        if var == "sp":
            # переводим в Pa для единообразия с WRF, или в hPa для красоты
            unit = "hPa"
        else:
            unit = info["unit"]

        rmse, mae, bias = compute_metrics(p_vals, t_vals)

        print("  %-5s: RMSE=%.3f %s | MAE=%.3f | bias=%+.3f | mean_truth=%.1f" % (
            var, rmse, unit, mae, bias, t_vals.mean()))

        # Per-horizon
        for h in range(min(P, 4)):
            p_h = pred_mean[:, h, vi]
            t_h = truth_mean[:, h, vi]
            r, m, b = compute_metrics(p_h, t_h)
            print("    +%02dh: RMSE=%.3f | MAE=%.3f | bias=%+.3f" % ((h+1)*6, r, m, b))

    # 3. WRF comparison (if available)
    if args.wrf_path:
        wrf_data = load_wrf(args.wrf_path, P)

        if wrf_data:
            print("\n" + "=" * 70)
            print("WRF vs ERA5 metrics (domain-averaged)")
            print("=" * 70)

            # WRF имеет фиксированный период (Jan 20-21 2023)
            # Нужно найти наш сэмпл, который покрывает этот период
            # Это зависит от start_date датасета и номера сэмпла
            # Пока: сравниваем средние по всем доступным шагам

            for var in compare_vars:
                if var not in wrf_data:
                    continue

                vi = var_names.index(var)
                info = VAR_MAPPING[var]

                wrf_vals = wrf_data[var]  # (n_6h_steps,)

                # Truth для тех же горизонтов (берём первые n_6h из наших)
                n_wrf = len(wrf_vals)

                # Наш truth (среднее по сэмплам для каждого горизонта)
                our_truth_h = truth_mean[:, :min(P, n_wrf), vi].mean(axis=0)  # (P,)
                our_pred_h = pred_mean[:, :min(P, n_wrf), vi].mean(axis=0)

                # Для sp: наши данные в hPa, WRF в Pa
                if var == "sp":
                    wrf_compare = wrf_vals[:min(P, n_wrf)] / 100.0  # Pa -> hPa
                    unit = "hPa"
                else:
                    wrf_compare = wrf_vals[:min(P, n_wrf)]
                    unit = info["unit"]

                n_compare = min(len(our_truth_h), len(wrf_compare))
                wrf_c = wrf_compare[:n_compare]
                era5_c = our_truth_h[:n_compare]
                ours_c = our_pred_h[:n_compare]

                rmse_wrf, mae_wrf, bias_wrf = compute_metrics(wrf_c, era5_c)
                rmse_ours, mae_ours, bias_ours = compute_metrics(ours_c, era5_c)

                print("")
                print("  %s (%s):" % (var, unit))
                print("    WRF vs ERA5:  RMSE=%.3f | MAE=%.3f | bias=%+.3f" % (
                    rmse_wrf, mae_wrf, bias_wrf))
                print("    Ours vs ERA5: RMSE=%.3f | MAE=%.3f | bias=%+.3f" % (
                    rmse_ours, mae_ours, bias_ours))
                if rmse_wrf > 0:
                    print("    Our/WRF RMSE ratio: %.2f (< 1 = we win)" % (rmse_ours / rmse_wrf))

    # 4. Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for thesis)")
    print("=" * 70)
    print("%-8s | %-12s | %-12s | %-12s" % ("Var", "Unit", "Our RMSE", "WRF RMSE"))
    print("-" * 50)
    for var in compare_vars:
        vi = var_names.index(var)
        info = VAR_MAPPING[var]
        p_vals = pred_mean[:, :, vi].flatten()
        t_vals = truth_mean[:, :, vi].flatten()
        rmse_ours, _, _ = compute_metrics(p_vals, t_vals)

        unit = "hPa" if var == "sp" else info["unit"]
        wrf_rmse_str = "N/A"

        if args.wrf_path and wrf_data and var in wrf_data:
            wrf_vals = wrf_data[var]
            our_truth_h = truth_mean[:, :min(P, len(wrf_vals)), vi].mean(axis=0)
            if var == "sp":
                wrf_c = wrf_vals[:min(P, len(wrf_vals))] / 100.0
            else:
                wrf_c = wrf_vals[:min(P, len(wrf_vals))]
            n_c = min(len(our_truth_h), len(wrf_c))
            rmse_wrf, _, _ = compute_metrics(wrf_c[:n_c], our_truth_h[:n_c])
            wrf_rmse_str = "%.3f" % rmse_wrf

        print("%-8s | %-12s | %-12.3f | %-12s" % (var, unit, rmse_ours, wrf_rmse_str))

    print("=" * 70)


if __name__ == "__main__":
    main()

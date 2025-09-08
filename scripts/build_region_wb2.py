#!/usr/bin/env python3
# scripts/build_region_wb2.py
# Собираем небольшой тайл из WB2 ERA5 1440x721 (6h) под инференс существующей модели.

import argparse, json
from pathlib import Path
import numpy as np
import xarray as xr
import gcsfs
import torch
import shutil

WB2_ERA5_1440x721 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

SURF = ["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind",
        "mean_sea_level_pressure"]
PLEV = ["temperature","u_component_of_wind","v_component_of_wind","geopotential","specific_humidity"]
LEVELS = [850, 500]

RENAME_SURF = {"2m_temperature":"t2m","10m_u_component_of_wind":"10u",
               "10m_v_component_of_wind":"10v","mean_sea_level_pressure":"msl"}
RENAME_PLEV = {"temperature":"t","u_component_of_wind":"u","v_component_of_wind":"v",
               "geopotential":"z","specific_humidity":"q"}

def _resolve_tp_name(ds: xr.Dataset) -> str:
    for cand in ["total_precipitation","total_precipitation_6hr","tp"]:
        if cand in ds.data_vars: return cand
    fuzzy = [v for v in ds.data_vars if "precip" in v]
    if fuzzy:
        print(f"[WARN] using precipitation var: {fuzzy[0]}")
        return fuzzy[0]
    raise RuntimeError("Total precipitation variable not found")

def _drop_nonindex_coords(da: xr.DataArray) -> xr.DataArray:
    non_index = [c for c in da.coords if c not in da.dims]
    if non_index: da = da.reset_coords(non_index, drop=True)
    return da

def _stack_channels(ds: xr.Dataset):
    parts = {}
    for v in SURF:
        parts[RENAME_SURF[v]] = ds[v]
    parts["tp"] = ds[_resolve_tp_name(ds)]
    for v in PLEV:
        if v not in ds.data_vars: continue
        for lev in LEVELS:
            if "level" not in ds[v].dims: continue
            if lev not in ds.level.values: continue
            key = f"{RENAME_PLEV[v]}@{lev}"
            arr = ds[v].sel(level=lev).reset_coords("level", drop=True)
            parts[key] = _drop_nonindex_coords(arr)

    base = ["t2m","10u","10v","msl","tp"]
    for lev in LEVELS: base += [f"t@{lev}", f"u@{lev}", f"v@{lev}"]
    extras = []
    for lev in LEVELS:
        if f"z@{lev}" in parts: extras.append(f"z@{lev}")
    for lev in LEVELS:
        if f"q@{lev}" in parts: extras.append(f"q@{lev}")
    ordered = base + extras

    arrs = [parts[k].transpose("time","longitude","latitude") for k in ordered]
    da = xr.concat(arrs, dim="feature", coords="minimal",
                   compat="override", combine_attrs="drop_conflicts"
    ).transpose("time","longitude","latitude","feature")
    return da, ordered

def _build_samples(arr: np.ndarray, obs: int, pred: int):
    X_list, Y_list = [], []
    total = arr.shape[0] - obs - pred + 1
    for i in range(total):
        X_list.append(arr[i:i+obs])
        Y_list.append(arr[i+obs:i+obs+pred])
    return np.stack(X_list), np.stack(Y_list)

def main():
    p = argparse.ArgumentParser("Build REGION dataset from WB2 1440x721 for inference.")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--start-date", type=str, required=True)  # например: 2019-07-01
    p.add_argument("--end-date", type=str, required=True)    # например: 2019-07-05
    p.add_argument("--lon-min", type=float, default=75.0)
    p.add_argument("--lon-max", type=float, default=90.0)
    p.add_argument("--lat-min", type=float, default=50.0)    # южнее
    p.add_argument("--lat-max", type=float, default=60.0)    # севернее
    p.add_argument("--obs-window", type=int, default=4)
    p.add_argument("--pred-window", type=int, default=4)
    p.add_argument("--train-scalers", type=str, required=True,
                   help="Путь к scalers.npz из ТРЕНИРОВОЧНОГО набора (64x32)")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Open WB2 (1440x721) Zarr ===")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB2_ERA5_1440x721)
    ds = xr.open_zarr(store, consolidated=True)
    print(ds["2m_temperature"].encoding.get("chunks"))  # или ds["2m_temperature"].chunks // temp

    # Важно: в WB2 широта обычно убывает; slice(max,min) корректен.
    ds = ds.sel(
        time=slice(args.start_date, args.end_date),
        longitude=slice(args.lon_min, args.lon_max),
        latitude=slice(args.lat_max, args.lat_min),
    )

    print("dims:", dict(ds.dims))
    da, var_order = _stack_channels(ds)                # (time, lon, lat, feat)
    arr = da.values.astype(np.float32)

    # Нужно как минимум obs+pred шагов по времени
    if arr.shape[0] < (args.obs_window + args.pred_window):
        raise RuntimeError("Недостаточно времени: нужно минимум obs+pred шагов (6ч шаг).")

    X, Y = _build_samples(arr, args.obs_window, args.pred_window)
    n, obs, LON, LAT, F = X.shape
    # Свернём время в канал (как на трейне, want_feats_flattened=True)
    X = X.transpose(0,2,3,1,4).reshape(n, LON, LAT, obs*F)
    Y = Y.transpose(0,2,3,1,4).reshape(n, LON, LAT, args.pred_window*F)

    # Загрузим ТРЕНИРОВОЧНЫЕ скейлеры и нормируем так же, как на обучении
    sc_path = Path(args.train_scalers)
    sc = np.load(sc_path)
    x_mean, x_scale = sc["x_mean"], sc["x_scale"]
    y_mean, y_scale = sc["y_mean"], sc["y_scale"]

    Xn = ((X - x_mean) / x_scale).astype(np.float32)
    Yn = ((Y - y_mean) / y_scale).astype(np.float32)

    # Сохраняем файлы в том же формате, который ждёт твой predict.py
    torch.save(torch.tensor(Xn), out_dir/"X_test.pt")
    torch.save(torch.tensor(Yn), out_dir/"y_test.pt")
    (out_dir/"variables.json").write_text(json.dumps(var_order, ensure_ascii=False, indent=2))
    # Подложим ТЕ ЖЕ скейлеры в новый датасет (predict.py возьмёт их отсюда)
    shutil.copy2(sc_path, out_dir/"scalers.npz")

    # Чтобы модель знала точные координаты — сохраним сетку (может пригодиться)
    np.savez(out_dir/"coords.npz",
             longitude=da.longitude.values.astype(np.float32),
             latitude=da.latitude.values.astype(np.float32))

    print(f"Saved to {out_dir}")
    print(f"X_test: {tuple(torch.load(out_dir/'X_test.pt').shape)}  (lon={LON}, lat={LAT}, obs*feat={obs*F})")
    print(f"y_test: {tuple(torch.load(out_dir/'y_test.pt').shape)}  (pred*feat={args.pred_window*F})")
    print(f"vars: {len(var_order)} → {var_order}")

if __name__ == "__main__":
    main()

# python scripts/build_region_wb2.py \
#   --out-dir data/datasets/region_nsko_1440x721_15f_4obs_4pred \
#   --start-date 2019-07-01 --end-date 2019-07-05 \
#   --lon-min 75 --lon-max 90 --lat-min 50 --lat-max 60 \
#   --obs-window 4 --pred-window 4 \
#   --train-scalers data/datasets/wb2_64x32_zq_15f_4obs_4pred/scalers.npz

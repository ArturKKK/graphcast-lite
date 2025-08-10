#!/usr/bin/env python3
# scripts/build_dataset_wb2.py
# 2.5D датасет из WeatherBench2 ERA5 Zarr (6h, 64x32, 13 уровней), минимум трафика.

import argparse, json
from pathlib import Path
import numpy as np
import xarray as xr
import gcsfs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

WB2_ERA5_64x32 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

SURF = ["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind",
        "mean_sea_level_pressure"]  # осадки детектируем отдельно (имя может отличаться)
PLEV = ["temperature","u_component_of_wind","v_component_of_wind", "geopotential", "specific_humidity"]
LEVELS = [850, 500]

RENAME_SURF = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "msl",
}
# ДОБАВИЛ z и q
RENAME_PLEV = {"temperature":"t", "u_component_of_wind":"u", "v_component_of_wind":"v",
               "geopotential":"z", "specific_humidity":"q"}

def _open_wb2(time_start: str, time_end: str):
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB2_ERA5_64x32)
    ds = xr.open_zarr(store, consolidated=True)
    ds = ds.sel(time=slice(time_start, time_end))
    # sanity
    print("dims:", dict(ds.dims))
    if "level" in ds.dims:
        print("levels:", ds.level.values.tolist())
    return ds

def _resolve_tp_name(ds: xr.Dataset) -> str:
    for cand in ["total_precipitation", "total_precipitation_6hr", "tp"]:
        if cand in ds.data_vars:
            return cand
    fuzzy = [v for v in ds.data_vars if "precip" in v]
    if fuzzy:
        print(f"[WARN] using precipitation var: {fuzzy[0]}")
        return fuzzy[0]
    raise RuntimeError(
        "Total precipitation variable not found. Tried: "
        "'total_precipitation', 'total_precipitation_6hr', 'tp'."
    )

def _drop_nonindex_coords(da: xr.DataArray) -> xr.DataArray:
    # удалить любые неиндексные координаты (оставим только time/lat/lon индексы)
    non_index = [c for c in da.coords if c not in da.dims]
    if non_index:
        da = da.reset_coords(non_index, drop=True)
    return da

def _stack_channels(ds: xr.Dataset) -> (xr.DataArray, list):
    parts = {}
    # surface
    for v in SURF:
        if v not in ds.data_vars:
            raise RuntimeError(f"Missing surface var in WB2: {v}")
        parts[RENAME_SURF[v]] = ds[v]
    # precip
    tp_name = _resolve_tp_name(ds)
    parts["tp"] = ds[tp_name]

    # upper air @ levels
    for v in PLEV:
        if v not in ds.data_vars:
            # geopotential/specific_humidity могут отсутствовать в некоторых сторах — просто пропустим
            print(f"[INFO] skip (no WB2 var): {v}")
            continue
        for lev in LEVELS:
            if "level" not in ds[v].dims:
                print(f"[INFO] skip {v} (no 'level' dim)")
                continue
            if lev not in ds.level.values:
                print(f"[INFO] skip {v}@{lev} (level not in ds)")
                continue
            key = f"{RENAME_PLEV[v]}@{lev}"
            arr = ds[v].sel(level=lev).reset_coords("level", drop=True)
            arr = _drop_nonindex_coords(arr)
            parts[key] = arr

    base = ["t2m","10u","10v","msl","tp"]
    for lev in LEVELS:
        base += [f"t@{lev}", f"u@{lev}", f"v@{lev}"]
    extras = []
    for lev in LEVELS:
        if f"z@{lev}" in parts: extras.append(f"z@{lev}")
    for lev in LEVELS:
        if f"q@{lev}" in parts: extras.append(f"q@{lev}")
    ordered = base + extras

    arrs = [parts[k].transpose("time","longitude","latitude") for k in ordered]
    da = xr.concat(
        arrs, dim="feature",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    ).transpose("time","longitude","latitude","feature")

    print(f"[VARS] using {len(ordered)} channels: {ordered}")
    return da, ordered

def _build_samples(arr: np.ndarray, obs: int, pred: int):
    X_list, Y_list = [], []
    total = arr.shape[0] - obs - pred + 1
    for i in range(total):
        X_list.append(arr[i:i+obs])
        Y_list.append(arr[i+obs:i+obs+pred])
    X = np.stack(X_list)  # (samples, obs, lon, lat, feat)
    Y = np.stack(Y_list)
    return X, Y

def build_dataset(out_dir: Path, start_date: str, end_date: str,
                  obs_window: int, pred_window: int, test_size: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== Build dataset (WB2 ERA5 Zarr 6h 64x32) ===")
    print(f"Dates: {start_date} .. {end_date} | obs={obs_window} | pred={pred_window}")

    ds = _open_wb2(start_date, end_date)

    da, var_order = _stack_channels(ds)                # (time, lon, lat, feat)
    arr = da.values.astype(np.float32)

    X, Y = _build_samples(arr, obs_window, pred_window)
    n, obs, lon, lat, feat = X.shape
    X = X.transpose(0,2,3,1,4).reshape(n, lon, lat, obs*feat)
    Y = Y.transpose(0,2,3,1,4).reshape(n, lon, lat, pred_window*feat)

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=test_size, random_state=42, shuffle=False)

    sx, sy = StandardScaler(), StandardScaler()
    X_tr = sx.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    Y_tr = sy.fit_transform(Y_tr.reshape(-1, Y_tr.shape[-1])).reshape(Y_tr.shape)
    X_te = sx.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)
    Y_te = sy.transform(Y_te.reshape(-1, Y_te.shape[-1])).reshape(Y_te.shape)

    torch.save(torch.tensor(X_tr, dtype=torch.float32), out_dir/"X_train.pt")
    torch.save(torch.tensor(Y_tr, dtype=torch.float32), out_dir/"y_train.pt")
    torch.save(torch.tensor(X_te, dtype=torch.float32), out_dir/"X_test.pt")
    torch.save(torch.tensor(Y_te, dtype=torch.float32), out_dir/"y_test.pt")
    (out_dir/"variables.json").write_text(json.dumps(var_order, ensure_ascii=False, indent=2))

    print(f"Saved tensors to {out_dir}")
    print(f"X_train: {tuple(torch.load(out_dir/'X_train.pt').shape)}  (lon={lon}, lat={lat}, obs*feat={obs*feat})")
    print(f"y_train: {tuple(torch.load(out_dir/'y_train.pt').shape)}  (pred*feat={pred_window*feat})")

def parse_args():
    p = argparse.ArgumentParser(description="Build 2.5D dataset from WeatherBench2 ERA5 Zarr (6h, 64x32).")
    p.add_argument("--out-dir", type=str, default="data/datasets/wb2_64x32_zq_15f_4obs_1pred")
    p.add_argument("--start-date", type=str, default="2010-01-01")
    p.add_argument("--end-date", type=str, default="2020-01-01")
    p.add_argument("--obs-window", type=int, default=4)
    p.add_argument("--pred-window", type=int, default=1)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        out_dir=Path(args.out_dir),
        start_date=args.start_date, end_date=args.end_date,
        obs_window=args.obs_window, pred_window=args.pred_window,
        test_size=args.test_size,
    )

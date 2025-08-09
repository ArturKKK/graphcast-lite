#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Собираем 2.5D датасет из ARCO ERA5 (GCS) под графовую модель:
- single_level: <var>/surface.nc
- pressure_level: <var>/<level>.nc (или <var>/<level>/*.nc)
Ресэмплим на 6h, интерполируем на сетку 128x64, нормируем per-channel, делаем лаги.
Сохраняем: X_train.pt, y_train.pt, X_test.pt, y_test.pt, variables.json

ЛОГИ: перед скачиванием печатаем сколько файлов и общий вес (оценка), далее прогресс скачивания
и отметки по основным этапам обработки.
"""

from __future__ import annotations
import argparse
import os
import json
import tempfile
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple
import time

import numpy as np
import xarray as xr
import gcsfs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# -------------------- Настройки по умолчанию --------------------

GCS_SINGLE = "gcs://gcp-public-data-arco-era5/raw/date-variable-single_level"
GCS_PRESS  = "gcs://gcp-public-data-arco-era5/raw/date-variable-pressure_level"

TARGET_NLON = 128
TARGET_NLAT = 64

SURFACE_VARS = [
    "2m_temperature",            # -> t2m
    "10m_u_component_of_wind",   # -> 10u
    "10m_v_component_of_wind",   # -> 10v
    "mean_sea_level_pressure",   # -> msl
    "total_precipitation",       # -> tp (сумма за 6h)
]
PLEVEL_VARS  = ["temperature", "u_component_of_wind", "v_component_of_wind"]
PLEVELS_HPA  = [850, 500]  # 2 уровня → всего 11 каналов вместе с поверхностью

RENAME_SURF = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "msl",
    "total_precipitation": "tp",
}
RENAME_PLEV = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}

def variables_order(levels: List[int]) -> List[str]:
    order = ["t2m","10u","10v","msl","tp"]
    for lev in levels:
        order += [f"t@{lev}", f"u@{lev}", f"v@{lev}"]
    return order

# -------------------- Утилиты GCS и логирования --------------------

def _fs() -> gcsfs.GCSFileSystem:
    return gcsfs.GCSFileSystem(token="anon")

def _date_range(start: str, end: str) -> List[dt.date]:
    a = dt.date.fromisoformat(start); b = dt.date.fromisoformat(end)
    out = []
    d = a
    while d <= b:
        out.append(d)
        d += dt.timedelta(days=1)
    return out

def _listdir(path: str) -> List[str]:
    fs = _fs()
    return fs.ls(path)

def _exists(path: str) -> bool:
    fs = _fs()
    try:
        return fs.exists(path)
    except Exception:
        return False

def _sizeof_fmt(num: float) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if abs(num) < 1024.0:
            return f"{num:,.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"

def _safe_local_name(url: str) -> str:
    parts = url.strip("/").split("/")
    tail = "_".join(parts[-5:])  # год, месяц, день, var, файл/уровень
    tail = tail.replace("/", "_")
    return tail

def _estimate_total_bytes(urls: List[str]) -> Tuple[int, List[Tuple[str,int]]]:
    fs = _fs()
    total = 0
    detailed = []
    for u in urls:
        try:
            info = fs.info(u)
            sz = int(info.get("size", 0))
        except Exception:
            sz = 0
        total += sz
        detailed.append((u, sz))
    return total, detailed

# -------------------- Разрешение путей --------------------

def _resolve_single_paths(dates: List[dt.date], vars_list: List[str]) -> Dict[str, List[str]]:
    """
    Возвращает {var: [gs://.../surface.nc, ...]} для single_level.
    """
    out = {v: [] for v in vars_list}
    for d in dates:
        day_dir = f"{GCS_SINGLE}/{d.year:04d}/{d.month:02d}/{d.day:02d}/"
        for v in vars_list:
            var_dir = f"{day_dir}{v}/"
            try:
                lst = _listdir(var_dir)
            except Exception as e:
                print(f"[WARN] single miss dir {var_dir}: {e}")
                lst = []
            # предпочтительно surface.nc
            cand = [p for p in lst if p.endswith("/surface.nc")]
            if not cand:
                cand = [p for p in lst if p.endswith(".nc")]
            if cand:
                out[v].append(cand[0])
            else:
                print(f"[WARN] single miss nc under {var_dir}")
    return out

def _resolve_plevel_paths(dates: List[dt.date], vars_list: List[str], levels_hpa: List[int]) -> Dict[str, Dict[int, List[str]]]:
    """
    Возвращает {var: {lev: [gs://.../*.nc, ...]}} для pressure_level.
    """
    out = {v: {lev: [] for lev in levels_hpa} for v in vars_list}
    for d in dates:
        day_dir = f"{GCS_PRESS}/{d.year:04d}/{d.month:02d}/{d.day:02d}/"
        for v in vars_list:
            var_dir = f"{day_dir}{v}/"
            try:
                lst = _listdir(var_dir)
            except Exception as e:
                print(f"[WARN] plevel miss dir {var_dir}: {e}")
                lst = []
            for lev in levels_hpa:
                picked = None
                # 1) Путь вида .../<var>/<lev>.nc
                exact = [p for p in lst if p.endswith(f"/{lev}.nc")]
                if exact:
                    picked = exact[0]
                else:
                    # 2) Подкаталог .../<var>/<lev>/ *.nc
                    subdirs = [p for p in lst if p.rstrip("/").endswith(f"/{lev}")]
                    for sd in subdirs:
                        try:
                            lst2 = _listdir(sd if sd.endswith("/") else sd + "/")
                            nc2 = [p for p in lst2 if p.endswith(".nc")]
                            if nc2:
                                picked = nc2[0]
                                break
                        except Exception:
                            pass
                if picked:
                    out[v][lev].append(picked)
                else:
                    print(f"[WARN] plevel miss {v}@{lev} for {d}")
    return out

# -------------------- Скачивание и открытие --------------------

def _download_urls(urls: List[str], desc: str) -> List[str]:
    """
    Скачиваем список gs:// url в temp-каталог. Печатаем оценку общего размера и прогресс.
    Возвращаем список локальных путей.
    """
    if not urls:
        return []
    fs = _fs()
    total_bytes, detailed = _estimate_total_bytes(urls)
    print(f"[DOWNLOAD PLAN] {desc}: files={len(urls)} | estimated size={_sizeof_fmt(total_bytes)}")

    tmpdir = tempfile.mkdtemp(prefix="arco_nc_")
    downloaded = 0
    locals_ = []
    t0 = time.perf_counter()

    for i, (u, sz) in enumerate(detailed, 1):
        local = os.path.join(tmpdir, _safe_local_name(u))
        # обычный fs.get без пофайлового прогресса — печатаем по завершении файла
        fs.get(u, local)
        locals_.append(local)
        downloaded += sz
        eta = (time.perf_counter() - t0)
        speed = downloaded / max(eta, 1e-6)
        print(f"  [{i}/{len(detailed)}] {os.path.basename(local)} "
              f"{_sizeof_fmt(sz)}  | done={_sizeof_fmt(downloaded)}/{_sizeof_fmt(total_bytes)} "
              f"| avg={_sizeof_fmt(speed)}/s")

    print(f"[DOWNLOAD DONE] {desc} in {time.perf_counter()-t0:.1f}s")
    return locals_

def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    # координаты как latitude/longitude по возрастанию широты
    if "lon" in ds.coords: ds = ds.rename({"lon": "longitude"})
    if "lat" in ds.coords: ds = ds.rename({"lat": "latitude"})
    if ds.latitude[0] > ds.latitude[-1]:
        ds = ds.sortby("latitude")
    return ds

def _open_and_merge_surface(files_dict: Dict[str, List[str]]) -> xr.Dataset:
    # flatten список url-ов и скачать
    all_urls = [u for urls in files_dict.values() for u in urls]
    locals_all = _download_urls(all_urls, "single_level")
    # разложим назад по переменным
    by_var = {v: [] for v in files_dict}
    idx = 0
    for v, urls in files_dict.items():
        by_var[v] = locals_all[idx: idx+len(urls)]
        idx += len(urls)

    dsets = []
    for v, locals_ in by_var.items():
        if not locals_:
            print(f"[WARN] no files for surface var {v}")
            continue
        t0 = time.perf_counter()
        ds_v = xr.open_mfdataset(
            locals_,
            engine="scipy",          # CDF classic
            combine="by_coords",
            chunks={"time": 24},
            decode_times=True,
        )
        print(f"[OPEN] surface {v}: files={len(locals_)} dims={dict(ds_v.dims)} "
              f"in {time.perf_counter()-t0:.1f}s")
        dsets.append(ds_v.rename({v: RENAME_SURF.get(v, v)}))
    if not dsets:
        raise RuntimeError("No surface datasets opened.")
    ds = xr.merge(dsets, compat="override")
    return _normalize_coords(ds)

def _open_and_merge_plevel(files_nested: Dict[str, Dict[int, List[str]]]) -> xr.Dataset:
    # flatten
    flat = []
    for v, mp in files_nested.items():
        for lev, urls in mp.items():
            for u in urls: flat.append((v, lev, u))
    locals_all = _download_urls([u for _,_,u in flat], "pressure_level")

    # разложим назад
    by_vl: Dict[Tuple[str,int], List[str]] = {}
    j = 0
    for v, lev, _ in flat:
        by_vl.setdefault((v,lev), []).append(locals_all[j]); j += 1

    dsets = []
    for (v, lev), locals_ in by_vl.items():
        if not locals_:
            print(f"[WARN] no files for plevel {v}@{lev}")
            continue
        t0 = time.perf_counter()
        ds_vl = xr.open_mfdataset(
            locals_,
            engine="scipy",
            combine="by_coords",
            chunks={"time": 24},
            decode_times=True,
        )
        if "level" not in ds_vl.dims:
            ds_vl = ds_vl.expand_dims({"level": [lev]})
        print(f"[OPEN] plevel {v}@{lev}: files={len(locals_)} dims={dict(ds_vl.dims)} "
              f"in {time.perf_counter()-t0:.1f}s")
        dsets.append(ds_vl)
    if not dsets:
        raise RuntimeError("No pressure-level datasets opened.")
    ds = xr.merge(dsets, compat="override")
    ds = _normalize_coords(ds)
    ds = ds.rename(RENAME_PLEV)
    if "level" in ds.dims:
        ds = ds.sortby("level")
    return ds

# -------------------- Преобразования --------------------

def _resample_6h_surface(ds: xr.Dataset) -> xr.Dataset:
    print("[STEP] resample surface to 6H ...", end="", flush=True)
    t0 = time.perf_counter()
    parts = []
    keep = {k: v for k, v in ds.data_vars.items() if k != "tp"}
    if keep:
        parts.append(xr.Dataset(keep).resample(time="6H").nearest())
    if "tp" in ds:
        parts.append(ds[["tp"]].resample(time="6H").sum())
    out = xr.merge(parts)
    print(f" done in {time.perf_counter()-t0:.1f}s; time_len={out.sizes.get('time')}")
    return out

def _resample_6h_plevel(ds: xr.Dataset) -> xr.Dataset:
    print("[STEP] resample pressure to 6H ...", end="", flush=True)
    t0 = time.perf_counter()
    out = ds.resample(time="6H").nearest()
    print(f" done in {time.perf_counter()-t0:.1f}s; time_len={out.sizes.get('time')}")
    return out

def _interp_to_target_grid(ds: xr.Dataset) -> xr.Dataset:
    print(f"[STEP] interp to grid {TARGET_NLON}x{TARGET_NLAT} ...", end="", flush=True)
    t0 = time.perf_counter()
    tlon = np.linspace(0.0, 360.0, TARGET_NLON, endpoint=False)
    tlat = np.linspace(-90.0, 90.0, TARGET_NLAT, endpoint=True)
    out = ds.interp(longitude=tlon, latitude=tlat, method="linear")
    print(f" done in {time.perf_counter()-t0:.1f}s")
    return out

def _stack_channels(ds_surf: xr.Dataset, ds_pl: xr.Dataset, levels: List[int]) -> xr.DataArray:
    print("[STEP] stack channels ...", end="", flush=True)
    t0 = time.perf_counter()
    parts = {
        "t2m":  ds_surf["t2m"],
        "10u":  ds_surf["10u"],
        "10v":  ds_surf["10v"],
        "msl":  ds_surf["msl"],
        "tp":   ds_surf["tp"] if "tp" in ds_surf else xr.zeros_like(ds_surf["t2m"]),
    }
    for lev in levels:
        parts[f"t@{lev}"] = ds_pl["t"].sel(level=lev)
        parts[f"u@{lev}"] = ds_pl["u"].sel(level=lev)
        parts[f"v@{lev}"] = ds_pl["v"].sel(level=lev)
    ordered = variables_order(levels)
    arrs = [parts[k].transpose("time","longitude","latitude") for k in ordered]
    da = xr.concat(arrs, dim="feature").transpose("time","longitude","latitude","feature")
    print(f" done in {time.perf_counter()-t0:.1f}s; shape={tuple(da.shape)}")
    return da

# -------------------- Лаги и сохранение --------------------

def _build_samples(arr: np.ndarray, obs: int, pred: int):
    print(f"[STEP] build lagged samples obs={obs} pred={pred} ...", end="", flush=True)
    t0 = time.perf_counter()
    X_list, Y_list = [], []
    total = arr.shape[0] - obs - pred + 1
    for i in range(total):
        X_list.append(arr[i:i+obs])
        Y_list.append(arr[i+obs:i+obs+pred])
    X = np.stack(X_list)  # (samples, obs, lon, lat, feat)
    Y = np.stack(Y_list)
    print(f" done in {time.perf_counter()-t0:.1f}s; samples={len(X)}")
    return X, Y

def build_dataset(out_dir: Path,
                  start_date: str,
                  end_date: str,
                  obs_window: int,
                  pred_window: int,
                  test_size: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = _date_range(start_date, end_date)
    var_order = variables_order(PLEVELS_HPA)

    print("=== Build dataset (ARCO ERA5) ===")
    print(f"Dates: {start_date} .. {end_date}  (days={len(dates)})")
    print(f"Grid: {TARGET_NLON}x{TARGET_NLAT} | obs={obs_window} | pred={pred_window}")
    print(f"Surface vars: {', '.join(SURFACE_VARS)}")
    print(f"Pressure vars: {', '.join(PLEVEL_VARS)} @ levels: {', '.join(map(str, PLEVELS_HPA))}")
    print(f"Out: {out_dir}")
    t_all = time.perf_counter()

    try:
        # surface
        files_surf = _resolve_single_paths(dates, SURFACE_VARS)
        ds_surf = _open_and_merge_surface(files_surf)
        ds_surf = _resample_6h_surface(ds_surf)
        ds_surf = _interp_to_target_grid(ds_surf)

        # pressure levels
        files_pl = _resolve_plevel_paths(dates, PLEVEL_VARS, PLEVELS_HPA)
        ds_pl = _open_and_merge_plevel(files_pl)
        ds_pl = _resample_6h_plevel(ds_pl)
        ds_pl = _interp_to_target_grid(ds_pl)

        # стек каналов
        da = _stack_channels(ds_surf, ds_pl, PLEVELS_HPA)
        arr = da.values.astype(np.float32)  # (time, NLON, NLAT, F)

    except Exception as e:
        print(f"[WARN] Remote read failed: {e}\nUsing random data instead.")
        T = 40
        arr = np.random.randn(T, TARGET_NLON, TARGET_NLAT, len(var_order)).astype(np.float32)

    # лаги
    X, Y = _build_samples(arr, obs_window, pred_window)
    n, obs, nl, nt, f = X.shape
    X = X.transpose(0,2,3,1,4).reshape(n, nl, nt, obs*f)
    Y = Y.transpose(0,2,3,1,4).reshape(n, nl, nt, pred_window*f)

    # split + нормализация per-channel
    print("[STEP] split + normalize ...", end="", flush=True)
    t0 = time.perf_counter()
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=test_size, random_state=42, shuffle=False)
    sx, sy = StandardScaler(), StandardScaler()
    X_tr = sx.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    Y_tr = sy.fit_transform(Y_tr.reshape(-1, Y_tr.shape[-1])).reshape(Y_tr.shape)
    X_te = sx.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)
    Y_te = sy.transform(Y_te.reshape(-1, Y_te.shape[-1])).reshape(Y_te.shape)
    print(f" done in {time.perf_counter()-t0:.1f}s")

    # save
    torch.save(torch.tensor(X_tr, dtype=torch.float32), out_dir/"X_train.pt")
    torch.save(torch.tensor(Y_tr, dtype=torch.float32), out_dir/"y_train.pt")
    torch.save(torch.tensor(X_te, dtype=torch.float32), out_dir/"X_test.pt")
    torch.save(torch.tensor(Y_te, dtype=torch.float32), out_dir/"y_test.pt")
    (out_dir/"variables.json").write_text(json.dumps(var_order, ensure_ascii=False, indent=2))

    print(f"\nSaved tensors to {out_dir}")
    print(f"X_train: {tuple(torch.load(out_dir/'X_train.pt').shape)}  (lon={TARGET_NLON}, lat={TARGET_NLAT}, obs*feat={obs*f})")
    print(f"y_train: {tuple(torch.load(out_dir/'y_train.pt').shape)}  (pred*feat={pred_window*f})")
    print(f"=== Done in {time.perf_counter()-t_all:.1f}s ===")

# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build 2.5D ERA5 dataset from ARCO (GCS).")
    p.add_argument("--out-dir", type=str, default="data/datasets/128x64_11f_2obs_1pred")
    p.add_argument("--start-date", type=str, default="2012-02-01")
    p.add_argument("--end-date", type=str, default="2012-02-02")
    p.add_argument("--obs-window", type=int, default=2)
    p.add_argument("--pred-window", type=int, default=1)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        out_dir=Path(args.out_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        obs_window=args.obs_window,
        pred_window=args.pred_window,
        test_size=args.test_size,
    )

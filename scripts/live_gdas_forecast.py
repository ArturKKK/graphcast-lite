#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig
from src.constants import FileNames
from src.data.data_configs import DatasetMetadata
from src.main import load_model_from_experiment_config
from src.postprocessing.mos_correction import (
    apply_mos_t2m, load_mos_table,
    apply_learned_mos_t2m, apply_learned_mos_wind, load_learned_mos,
    load_wind_scaling, apply_wind_scaling,
)
from src.utils import load_from_json_file
from src.unet.model import WeatherUNet

CITY_BBOX = (55.5, 56.5, 92.0, 94.0)
GDAS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

# ── Fields needed for selective GDAS download (idx shortName : level) ──
SELECTIVE_FIELDS = [
    ("PRMSL", "mean sea level"),
    ("TMP",   "2 m above ground"),
    ("UGRD",  "10 m above ground"),
    ("VGRD",  "10 m above ground"),
    ("PRES",  "surface"),
    ("PWAT",  "entire atmosphere"),
    ("PRATE", "surface"),
    ("TMP",   "850 mb"), ("UGRD", "850 mb"), ("VGRD", "850 mb"),
    ("HGT",   "850 mb"), ("SPFH", "850 mb"),
    ("TMP",   "500 mb"), ("UGRD", "500 mb"), ("VGRD", "500 mb"),
    ("HGT",   "500 mb"), ("SPFH", "500 mb"),
    ("HGT",   "surface"),
]

DEFAULT_VAR_ORDER = [
    "t2m", "10u", "10v", "msl", "tp",
    "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
]

GROUP_FILTERS = {
    "t2m": [{"typeOfLevel": "heightAboveGround", "shortName": "2t"}],
    "10u": [{"typeOfLevel": "heightAboveGround", "shortName": "10u"}],
    "10v": [{"typeOfLevel": "heightAboveGround", "shortName": "10v"}],
    "msl": [
        {"typeOfLevel": "meanSea", "shortName": "prmsl"},
        {"typeOfLevel": "meanSea", "shortName": "mslma"},
    ],
    "sp": [
        {"typeOfLevel": "surface", "shortName": "sp"},
        {"typeOfLevel": "surface", "shortName": "pres"},
    ],
    "tcwv": [
        {"typeOfLevel": "atmosphereSingleLayer", "shortName": "pwat"},
        {"typeOfLevel": "atmosphereSingleLayer", "shortName": "tcwv"},
    ],
    "isobaric_t": [{"typeOfLevel": "isobaricInhPa", "shortName": "t"}],
    "isobaric_u": [{"typeOfLevel": "isobaricInhPa", "shortName": "u"}],
    "isobaric_v": [{"typeOfLevel": "isobaricInhPa", "shortName": "v"}],
    "isobaric_z": [
        {"typeOfLevel": "isobaricInhPa", "shortName": "gh"},
        {"typeOfLevel": "isobaricInhPa", "shortName": "z"},
    ],
    "isobaric_q": [{"typeOfLevel": "isobaricInhPa", "shortName": "q"}],
    "tp": [
        {"typeOfLevel": "surface", "shortName": "tp"},
        {"typeOfLevel": "surface", "shortName": "acpcp"},
        {"typeOfLevel": "surface", "shortName": "prate"},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download recent GDAS analyses and run live GraphCast-lite inference")
    parser.add_argument("--experiment-dir", default="experiments/multires_nores_freeze6")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Full dataset dir. Optional when --runtime-bundle is provided",
    )
    parser.add_argument("--runtime-bundle", default=None, help="Optional lightweight bundle dir exported from cluster for live inference")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--cache-dir", default="results/live_gdas/cache")
    parser.add_argument("--out-dir", default="results/live_gdas/latest")
    parser.add_argument("--ar-steps", type=int, default=4)
    parser.add_argument("--obs-cycles", type=int, default=2)
    parser.add_argument("--lag-hours", type=int, default=6)
    parser.add_argument("--max-lookback-cycles", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--keep-grib", action="store_true")
    parser.add_argument("--mos-table", default=None,
                        help="Path to MOS bias correction JSON (from build_mos_table.py). "
                             "If provided, t2m predictions are corrected for station bias.")
    parser.add_argument("--learned-mos", default=None,
                        help="Path to learned MOS model (.joblib from build_learned_mos.py). "
                             "ML-based t2m correction (overrides --mos-table if both given).")
    parser.add_argument("--spatial-idw", action="store_true",
                        help="Interpolate MOS bias correction to ALL grid nodes via IDW "
                             "(instead of correcting only station grid points).")
    parser.add_argument("--cycle", default=None, help="Optional cycle anchor in ISO form, e.g. 2026-03-23T12:00:00+00:00")
    parser.add_argument("--lapse-target-elevation", type=float, default=None,
                        help="Target station elevation in metres for lapse-rate t2m correction. "
                             "E.g. 277 for Opytnoe Pole (Krasnoyarsk). Corrects for grid cell "
                             "topography averaging at coarse resolution.")
    parser.add_argument("--input-bias-t2m", type=float, default=None,
                        help="Additive bias correction (°C) to apply to GDAS t2m input "
                             "*before* normalization. Positive value warms the input. "
                             "Use to compensate systematic GDAS cold/warm bias for the region.")
    parser.add_argument("--wind-scale", default=None,
                        help="Path to wind monthly scaling JSON (from build_wind_scale.py). "
                             "Applies seasonal multiplicative wind speed correction.")
    parser.add_argument("--selective", action="store_true",
                        help="Download only the ~18 GRIB messages needed (via byte-range). "
                             "Reduces ~440 MB download to ~14 MB per cycle (96.8%% savings).")
    parser.add_argument("--cascade-experiment", default=None,
                        help="UNet downscaler experiment dir for cascade inference (e.g. experiments/downscaler_krsk)")
    parser.add_argument("--cascade-data", default=None,
                        help="Downscaler dataset dir with scalers/coords/static_fine.npy. "
                             "Falls back to live_runtime_bundle if not set.")
    args = parser.parse_args()
    if args.data_dir is None and args.runtime_bundle is None:
        parser.error("either --data-dir or --runtime-bundle must be provided")
    return args


# ── MOS station registry ────────────────────────────────────────────
# Mirrors STATIONS from build_learned_mos.py; used at inference time
# to apply corrections at all trained station grid points.
_MOS_STATIONS = [
    {"lat": 56.173, "lon": 92.493, "elev": 287, "name": "Yemelyanovo"},
    {"lat": 56.283, "lon": 90.517, "elev": 257, "name": "Achinsk"},
    {"lat": 56.200, "lon": 95.633, "elev": 207, "name": "Kansk"},
    {"lat": 53.740, "lon": 91.385, "elev": 253, "name": "Abakan"},
    {"lat": 57.683, "lon": 93.267, "elev": 93,  "name": "Kazachinskoe"},
    {"lat": 56.900, "lon": 93.133, "elev": 180, "name": "Bolshaya Murta"},
    {"lat": 56.967, "lon": 90.683, "elev": 181, "name": "Novobirilyussy"},
    {"lat": 56.500, "lon": 93.283, "elev": 164, "name": "Sukhobuzimskoe"},
    {"lat": 56.217, "lon": 89.550, "elev": 290, "name": "Bogotol"},
    {"lat": 56.067, "lon": 92.733, "elev": 235, "name": "Minino"},
    {"lat": 56.117, "lon": 92.200, "elev": 479, "name": "Kaca"},
    {"lat": 55.933, "lon": 92.283, "elev": 275, "name": "Shumiha"},
    {"lat": 57.633, "lon": 92.267, "elev": 179, "name": "Pirovskoe"},
    {"lat": 57.200, "lon": 94.550, "elev": 168, "name": "Taseevo"},
    {"lat": 56.650, "lon": 90.550, "elev": 231, "name": "Bolshoj Uluj"},
    {"lat": 56.850, "lon": 95.217, "elev": 188, "name": "Dzerzhinskoe"},
    {"lat": 56.033, "lon": 90.317, "elev": 256, "name": "Nazarovo"},
    {"lat": 56.100, "lon": 91.667, "elev": 332, "name": "Kemchug"},
    {"lat": 56.167, "lon": 95.267, "elev": 357, "name": "Solyanka"},
]


def _get_mos_stations(mos_bundle: dict) -> list[dict]:
    """Return station list for MOS correction.

    Uses the hardcoded registry (kept in sync with build_learned_mos.py).
    """
    return _MOS_STATIONS


def cycle_floor_6h(dt_utc: datetime) -> datetime:
    floored_hour = (dt_utc.hour // 6) * 6
    return dt_utc.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def cycle_url(cycle_dt: datetime) -> str:
    day = cycle_dt.strftime("%Y%m%d")
    hour = cycle_dt.strftime("%H")
    file_name = f"gdas.t{hour}z.pgrb2.0p25.f000"
    return f"{GDAS_BASE}/gdas.{day}/{hour}/atmos/{file_name}"


def cycle_label(cycle_dt: datetime) -> str:
    return cycle_dt.strftime("%Y%m%d_%H")


def find_recent_cycles(obs_cycles: int, lag_hours: int, max_lookback_cycles: int) -> list[datetime]:
    now_utc = datetime.now(timezone.utc)
    anchor = cycle_floor_6h(now_utc - timedelta(hours=lag_hours))
    found: list[datetime] = []
    checked = 0
    session = requests.Session()
    while len(found) < obs_cycles and checked < max_lookback_cycles:
        candidate = anchor - timedelta(hours=6 * checked)
        url = cycle_url(candidate)
        try:
            response = session.head(url, allow_redirects=True, timeout=60)
            if response.status_code == 200:
                found.append(candidate)
        except requests.RequestException:
            pass
        checked += 1
    if len(found) < obs_cycles:
        raise RuntimeError(f"Could not find {obs_cycles} available GDAS cycles within {max_lookback_cycles} lookback steps")
    return sorted(found)


def download_file(url: str, dest: Path, timeout: int) -> Path:
    expected_bytes = 0
    try:
        head_response = requests.head(url, allow_redirects=True, timeout=min(timeout, 60))
        if head_response.ok:
            expected_bytes = int(head_response.headers.get("Content-Length", 0))
    except requests.RequestException:
        expected_bytes = 0

    if dest.exists() and dest.stat().st_size > 0:
        existing_size = dest.stat().st_size
        if expected_bytes and existing_size == expected_bytes:
            print(f"[cache] Using existing {dest}")
            return dest
        print(f"[cache] Removing incomplete cached file {dest} ({existing_size} bytes, expected {expected_bytes or 'unknown'})")
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("Content-Length", 0)) or expected_bytes
        written = 0
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total_bytes:
                    pct = 100.0 * written / total_bytes
                    print(f"  {written / (1024**2):7.1f} MB / {total_bytes / (1024**2):7.1f} MB  ({pct:5.1f}%)", end="\r", flush=True)
        if total_bytes:
            print()
        if total_bytes and written != total_bytes:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded truncated file {dest}: wrote {written} bytes, expected {total_bytes}")
    return dest


def download_selective(url: str, dest: Path, timeout: int) -> Path:
    """Download only the needed GRIB messages via HTTP byte-range requests.

    Fetches the NOMADS .idx file to get byte offsets, then downloads only
    the 18 fields required by our model (~14 MB instead of ~440 MB).
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache] Using existing {dest}")
        return dest

    idx_url = url + ".idx"
    try:
        idx_resp = requests.get(idx_url, timeout=min(timeout, 60))
        idx_resp.raise_for_status()
    except requests.RequestException:
        print(f"[selective] .idx not available, falling back to full download")
        return download_file(url, dest, timeout)

    # Parse idx: "NUM:OFFSET:d=DATE:SHORTNAME:LEVEL:anl:"
    entries = []
    for line in idx_resp.text.strip().split("\n"):
        parts = line.split(":")
        if len(parts) >= 6:
            entries.append({
                "num": int(parts[0]),
                "offset": int(parts[1]),
                "shortName": parts[3],
                "level": parts[4],
            })

    if not entries:
        print(f"[selective] Empty idx, falling back to full download")
        return download_file(url, dest, timeout)

    # Find byte ranges for each needed field
    ranges = []
    for shortName, level_desc in SELECTIVE_FIELDS:
        for i, e in enumerate(entries):
            if e["shortName"] == shortName and level_desc in e["level"]:
                start = e["offset"]
                end = entries[i + 1]["offset"] - 1 if i + 1 < len(entries) else None
                ranges.append((shortName, level_desc, start, end))
                break

    if len(ranges) < len(SELECTIVE_FIELDS) - 2:  # tolerate 2 missing
        missing = len(SELECTIVE_FIELDS) - len(ranges)
        print(f"[selective] Only {len(ranges)}/{len(SELECTIVE_FIELDS)} fields found in idx "
              f"({missing} missing), falling back to full download")
        return download_file(url, dest, timeout)

    ranges.sort(key=lambda r: r[2])
    dest.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    session = requests.Session()
    with dest.open("wb") as f:
        for shortName, level_desc, start, end in ranges:
            range_header = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
            resp = session.get(url, headers={"Range": range_header}, timeout=timeout)
            resp.raise_for_status()
            f.write(resp.content)
            total_bytes += len(resp.content)

    print(f"[selective] Downloaded {len(ranges)} fields: "
          f"{total_bytes / 1024 / 1024:.1f} MB (vs ~440 MB full)")
    return dest


def open_cfgrib_dataset(grib_path: Path, payload_name: str) -> xr.Dataset | None:
    for filter_by_keys in GROUP_FILTERS[payload_name]:
        try:
            ds = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": filter_by_keys, "indexpath": ""},
            )
            if len(ds.data_vars) == 0:
                continue
            return ds
        except Exception:
            continue
    return None


def open_gdas_payload(grib_path: Path) -> dict[str, xr.Dataset]:
    payload: dict[str, xr.Dataset] = {}
    for group_name in GROUP_FILTERS:
        ds = open_cfgrib_dataset(grib_path, group_name)
        if ds is not None:
            payload[group_name] = ds
    if not payload:
        raise RuntimeError(f"Could not read any GDAS groups from {grib_path}")
    return payload


def get_var_order(data_dir: Path) -> list[str]:
    variables_path = data_dir / "variables.json"
    if variables_path.exists():
        with variables_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return DEFAULT_VAR_ORDER.copy()


def get_var_order_from_bundle(bundle_dir: Path) -> list[str]:
    variables_path = bundle_dir / "variables.json"
    if variables_path.exists():
        with variables_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return DEFAULT_VAR_ORDER.copy()


def load_scalers(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scalers = np.load(data_dir / "scalers.npz")
    if "mean" in scalers:
        mean = scalers["mean"].astype(np.float32)
        std = scalers["std"].astype(np.float32)
        return mean, std, mean, std
    return (
        scalers["x_mean"].astype(np.float32),
        scalers["x_scale"].astype(np.float32),
        scalers["y_mean"].astype(np.float32),
        scalers["y_scale"].astype(np.float32),
    )


def load_coords(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = np.load(data_dir / "coords.npz")
    return coords["latitude"].astype(np.float32), coords["longitude"].astype(np.float32)


def load_coords_from_bundle(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    coords = np.load(bundle_dir / "coords.npz")
    is_regional = coords["is_regional"] if "is_regional" in coords else None
    return (
        coords["latitude"].astype(np.float32),
        coords["longitude"].astype(np.float32),
        is_regional,
    )


def load_template_static(data_dir: Path, var_order: list[str]) -> dict[str, np.ndarray]:
    static_values: dict[str, np.ndarray] = {}
    info_path = data_dir / "dataset_info.json"
    data_path = data_dir / "data.npy"
    if not info_path.exists() or not data_path.exists():
        return static_values

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info_vars = info.get("variables", var_order)
    dtype = np.float16
    if info.get("flat", False):
        shape = (info["n_time"], info["n_nodes"], info["n_feat"])
    else:
        shape = (info["n_time"], info["n_lon"], info["n_lat"], info["n_feat"])
    mmap_arr = np.memmap(str(data_path), dtype=dtype, mode="r", shape=shape)
    for name in ["z_surf", "lsm"]:
        if name not in info_vars:
            continue
        idx = info_vars.index(name)
        if info.get("flat", False):
            values = mmap_arr[0, :, idx].astype(np.float32)
        else:
            values = mmap_arr[0, :, :, idx].astype(np.float32).reshape(-1)
        static_values[name] = values
    return static_values


def load_template_static_from_bundle(bundle_dir: Path) -> dict[str, np.ndarray]:
    static_values: dict[str, np.ndarray] = {}
    static_path = bundle_dir / "static_fields.npz"
    if not static_path.exists():
        return static_values
    payload = np.load(static_path)
    for name in payload.files:
        static_values[name] = payload[name].astype(np.float32)
    return static_values


def build_metadata_from_arrays(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    n_features: int,
    obs_window: int,
    pred_window: int,
    is_regional: np.ndarray | None = None,
) -> DatasetMetadata:
    metadata = DatasetMetadata(
        flattened=True,
        num_latitudes=0,
        num_longitudes=0,
        num_features=n_features,
        obs_window=obs_window,
        pred_window=pred_window,
    )
    metadata.flat_grid = True
    metadata.num_grid_nodes = len(latitudes)
    metadata.cordinates = (latitudes.astype(np.float32), longitudes.astype(np.float32))
    metadata.is_regional = is_regional
    return metadata


def load_runtime_assets(
    data_dir: Path | None,
    runtime_bundle_dir: Path | None,
    obs_window: int,
    pred_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, dict[str, np.ndarray], DatasetMetadata]:
    if runtime_bundle_dir is not None:
        required = ["coords.npz", "scalers.npz", "variables.json"]
        for required_name in required:
            required_path = runtime_bundle_dir / required_name
            if not required_path.exists():
                raise FileNotFoundError(f"Missing runtime bundle file: {required_path}")

        x_mean, x_std, y_mean, y_std = load_scalers(runtime_bundle_dir)
        var_order = get_var_order_from_bundle(runtime_bundle_dir)
        latitudes, longitudes, is_regional = load_coords_from_bundle(runtime_bundle_dir)
        template_static = load_template_static_from_bundle(runtime_bundle_dir)
        missing_static = [name for name in ["z_surf", "lsm"] if name in var_order and name not in template_static]
        if missing_static:
            raise FileNotFoundError(
                "Runtime bundle is incomplete: missing static fields "
                f"{missing_static} in {runtime_bundle_dir / 'static_fields.npz'}"
            )
        metadata = build_metadata_from_arrays(
            latitudes=latitudes,
            longitudes=longitudes,
            n_features=len(var_order),
            obs_window=obs_window,
            pred_window=pred_window,
            is_regional=is_regional,
        )
        return x_mean, x_std, y_mean, y_std, var_order, latitudes, longitudes, template_static, metadata

    if data_dir is None:
        raise ValueError("data_dir must be provided when runtime_bundle_dir is not set")

    for required_name in ["coords.npz", "scalers.npz", "variables.json"]:
        required_path = data_dir / required_name
        if not required_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {required_path}")

    x_mean, x_std, y_mean, y_std = load_scalers(data_dir)
    var_order = get_var_order(data_dir)
    latitudes, longitudes = load_coords(data_dir)
    template_static = load_template_static(data_dir, var_order)
    is_regional = None
    coords = np.load(data_dir / "coords.npz")
    if "is_regional" in coords:
        is_regional = coords["is_regional"]
    metadata = build_metadata_from_arrays(
        latitudes=latitudes,
        longitudes=longitudes,
        n_features=len(var_order),
        obs_window=obs_window,
        pred_window=pred_window,
        is_regional=is_regional,
    )
    return x_mean, x_std, y_mean, y_std, var_order, latitudes, longitudes, template_static, metadata


def build_interpolator(data_array: xr.DataArray) -> RegularGridInterpolator:
    lat_name = "latitude" if "latitude" in data_array.coords else "lat"
    lon_name = "longitude" if "longitude" in data_array.coords else "lon"
    lats = np.asarray(data_array[lat_name].values, dtype=np.float32)
    lons = np.asarray(data_array[lon_name].values, dtype=np.float32) % 360.0
    values = np.asarray(data_array.values, dtype=np.float32)

    lat_order = np.argsort(lats)
    lon_order = np.argsort(lons)
    lats_sorted = lats[lat_order]
    lons_sorted = lons[lon_order]
    values_sorted = values[np.ix_(lat_order, lon_order)]

    extended_lons = np.concatenate([lons_sorted, [lons_sorted[0] + 360.0]])
    extended_values = np.concatenate([values_sorted, values_sorted[:, :1]], axis=1)

    return RegularGridInterpolator(
        (lats_sorted, extended_lons),
        extended_values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def interp_to_nodes(data_array: xr.DataArray, node_lats: np.ndarray, node_lons: np.ndarray) -> np.ndarray:
    interpolator = build_interpolator(data_array)
    points = np.column_stack([node_lats.astype(np.float32), np.mod(node_lons, 360.0).astype(np.float32)])
    values = interpolator(points)
    return np.asarray(values, dtype=np.float32)


def select_field(payload: dict[str, xr.Dataset], group_name: str, candidates: list[str], level: int | None = None) -> xr.DataArray | None:
    ds = payload.get(group_name)
    if ds is None:
        return None
    for candidate in candidates:
        if candidate in ds.data_vars:
            da = ds[candidate]
            if level is not None:
                level_name = None
                for dim_name in ["isobaricInhPa", "level"]:
                    if dim_name in da.dims or dim_name in da.coords:
                        level_name = dim_name
                        break
                if level_name is None:
                    continue
                da = da.sel({level_name: level})
            return da
    return None


def extract_live_channels(
    payload: dict[str, xr.Dataset],
    node_lats: np.ndarray,
    node_lons: np.ndarray,
    var_order: list[str],
    template_static: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str]]:
    extracted: dict[str, np.ndarray] = {}
    warnings: list[str] = []

    var_specs = {
        "t2m": ("t2m", ["2t", "t2m", "t"], None),
        "10u": ("10u", ["10u", "u10", "u"], None),
        "10v": ("10v", ["10v", "v10", "v"], None),
        "msl": ("msl", ["prmsl", "mslma"], None),
        "sp": ("sp", ["sp", "pres"], None),
        "tcwv": ("tcwv", ["pwat", "tcwv"], None),
        "t@850": ("isobaric_t", ["t"], 850),
        "u@850": ("isobaric_u", ["u"], 850),
        "v@850": ("isobaric_v", ["v"], 850),
        "z@850": ("isobaric_z", ["gh", "z"], 850),
        "q@850": ("isobaric_q", ["q"], 850),
        "t@500": ("isobaric_t", ["t"], 500),
        "u@500": ("isobaric_u", ["u"], 500),
        "v@500": ("isobaric_v", ["v"], 500),
        "z@500": ("isobaric_z", ["gh", "z"], 500),
        "q@500": ("isobaric_q", ["q"], 500),
        "tp": ("tp", ["tp", "acpcp", "prate"], None),
    }

    for name in var_order:
        if name in template_static:
            extracted[name] = template_static[name].astype(np.float32)
            continue
        spec = var_specs.get(name)
        if spec is None:
            warnings.append(f"Unsupported variable {name}; filling zeros")
            extracted[name] = np.zeros_like(node_lats, dtype=np.float32)
            continue
        group_name, candidates, level = spec
        da = select_field(payload, group_name, candidates, level=level)
        if da is None:
            if name == "tp":
                warnings.append("GDAS analysis does not expose tp in this path; filling zeros")
            else:
                warnings.append(f"Missing {name} in GDAS payload; filling zeros")
            extracted[name] = np.zeros_like(node_lats, dtype=np.float32)
            continue
        values = interp_to_nodes(da, node_lats, node_lons)
        # Training scalers for this experiment expect pressure in hPa, while GDAS provides Pa.
        if name in {"msl", "sp"}:
            values = values / 100.0
        extracted[name] = values
    return extracted, warnings


def normalize_frame(frame: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return (frame - x_mean[None, :]) / x_std[None, :]


def denormalize_prediction(prediction: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return prediction * y_std[None, None, :] + y_mean[None, None, :]


def build_city_mask(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = CITY_BBOX
    return (
        (latitudes >= lat_min)
        & (latitudes <= lat_max)
        & (longitudes >= lon_min)
        & (longitudes <= lon_max)
    )


def render_t2m_plot(out_path: Path, latitudes: np.ndarray, longitudes: np.ndarray, prediction_phys: np.ndarray, var_order: list[str]) -> None:
    if "t2m" not in var_order:
        return
    city_mask = build_city_mask(latitudes, longitudes)
    if city_mask.sum() == 0:
        return
    t2m_idx = var_order.index("t2m")
    steps = prediction_phys.shape[1]
    fig, axes = plt.subplots(1, steps, figsize=(4.5 * steps, 4.5))
    if steps == 1:
        axes = [axes]
    for step_idx, ax in enumerate(axes):
        values_c = prediction_phys[:, step_idx, t2m_idx][city_mask] - 273.15
        scatter = ax.scatter(longitudes[city_mask], latitudes[city_mask], c=values_c, s=18, cmap="RdYlBu_r", edgecolors="none")
        ax.set_title(f"t2m +{(step_idx + 1) * 6}h")
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_city(out_path: Path, prediction_phys: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray, var_order: list[str], cycles: list[datetime], warnings: list[str]) -> None:
    city_mask = build_city_mask(latitudes, longitudes)
    lines = ["# GDAS Live Forecast", ""]
    lines.append("Input cycles:")
    for cycle_dt in cycles:
        lines.append(f"- {cycle_dt.isoformat()}")
    lines.append("")
    lines.append("Warnings:")
    if warnings:
        lines.extend(f"- {line}" for line in warnings)
    else:
        lines.append("- none")
    lines.append("")
    if city_mask.sum() > 0:
        lines.append("City-area means:")
        for step_idx in range(prediction_phys.shape[1]):
            lines.append(f"- Horizon +{(step_idx + 1) * 6}h")
            for name in ["t2m", "10u", "10v", "msl"]:
                if name not in var_order:
                    continue
                idx = var_order.index(name)
                values = prediction_phys[:, step_idx, idx][city_mask]
                if name == "t2m":
                    values = values - 273.15
                    unit = "C"
                elif name == "msl":
                    unit = "hPa"
                else:
                    unit = "m/s"
                lines.append(f"  {name}: mean={values.mean():.2f} {unit} min={values.min():.2f} max={values.max():.2f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None
    runtime_bundle_dir = Path(args.runtime_bundle) if args.runtime_bundle else None
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = experiment_dir / FileNames.EXPERIMENT_CONFIG
    ckpt_path = Path(args.ckpt) if args.ckpt else experiment_dir / FileNames.SAVED_MODEL
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    exp_cfg = ExperimentConfig(**load_from_json_file(str(cfg_path)))
    obs_window = int(exp_cfg.data.obs_window_used)
    if obs_window != args.obs_cycles:
        print(f"[live] obs_window={obs_window}; overriding requested obs_cycles={args.obs_cycles}")
    obs_cycles = obs_window
    no_residual = not bool(getattr(exp_cfg, "use_residual", False))

    x_mean, x_std, y_mean, y_std, var_order, latitudes, longitudes, template_static, metadata = load_runtime_assets(
        data_dir=data_dir,
        runtime_bundle_dir=runtime_bundle_dir,
        obs_window=obs_window,
        pred_window=1,
    )

    if args.cycle:
        anchor = datetime.fromisoformat(args.cycle)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        cycles = [anchor - timedelta(hours=6 * offset) for offset in reversed(range(obs_cycles))]
    else:
        cycles = find_recent_cycles(obs_cycles, args.lag_hours, args.max_lookback_cycles)
    print("[live] using cycles:")
    for cycle_dt in cycles:
        print(f"  - {cycle_dt.isoformat()}")

    cycle_payloads = []
    _download = download_selective if args.selective else download_file
    for cycle_dt in cycles:
        label = cycle_label(cycle_dt)
        grib_path = cache_dir / f"gdas_{label}.grib2"
        _download(cycle_url(cycle_dt), grib_path, args.timeout)
        payload = open_gdas_payload(grib_path)
        cycle_payloads.append((cycle_dt, grib_path, payload))

    warnings: list[str] = []
    obs_frames = []
    for cycle_dt, _, payload in cycle_payloads:
        extracted, cycle_warnings = extract_live_channels(payload, latitudes, longitudes, var_order, template_static)
        warnings.extend([f"{cycle_dt.isoformat()}: {line}" for line in cycle_warnings])
        frame = np.stack([extracted[name] for name in var_order], axis=-1).astype(np.float32)

        # --- GDAS input bias correction for t2m ---
        if args.input_bias_t2m is not None and "t2m" in var_order:
            t_idx = var_order.index("t2m")
            frame[:, t_idx] += args.input_bias_t2m  # t2m is in K here, bias in °C = K
            if cycle_dt == cycle_payloads[0][0]:  # print once
                print(f"[Input bias] t2m += {args.input_bias_t2m:+.2f}°C applied to GDAS input")

        obs_frames.append(normalize_frame(frame, x_mean[:len(var_order)], x_std[:len(var_order)]))

    input_tensor = np.stack(obs_frames, axis=1).reshape(len(latitudes), obs_window * len(var_order)).astype(np.float32)
    input_path = out_dir / "input_normalized.npy"
    np.save(input_path, input_tensor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_experiment_config(
        exp_cfg,
        device,
        metadata,
        coordinates=(latitudes, longitudes),
        flat_grid=getattr(metadata, "flat_grid", False),
    )
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    G = len(latitudes)
    C = len(var_order)
    curr_state = torch.from_numpy(input_tensor).unsqueeze(0).to(device).view(1, G, obs_window, C)
    ar_outs = []
    with torch.no_grad():
        for _ in range(args.ar_steps):
            inp = curr_state.view(1, G, -1)
            pred = model(inp, attention_threshold=0.0)
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)
            if no_residual:
                step_out = pred
            else:
                step_out = curr_state[:, :, -1, :] + pred
            ar_outs.append(step_out.cpu())
            curr_state = torch.cat([curr_state[:, :, 1:, :], step_out.unsqueeze(2).to(device)], dim=2)

    prediction_norm = torch.cat(ar_outs, dim=2).squeeze(0).numpy().reshape(G, args.ar_steps, C)
    prediction_phys = denormalize_prediction(prediction_norm, y_mean[:C], y_std[:C])

    # --- Cascade: UNet downscaler refinement for ROI ---
    cascade_phys = None  # (H, W, AR, C) in physical units, or None
    cascade_lats = None
    cascade_lons = None
    if args.cascade_experiment:
        from scripts.predict_cascade import crop_and_upsample_roi

        cas_dir = Path(args.cascade_experiment)
        cas_cfg = json.loads((cas_dir / "config.json").read_text()) if (cas_dir / "config.json").exists() else {}
        cas_obs = cas_cfg.get("obs_window", 2)
        cas_bf = cas_cfg.get("base_filters", 64)
        cas_residual = cas_cfg.get("residual", False)
        cas_static_ch = cas_cfg.get("static_channels", [7, 8])

        # Load cascade scalers & coords
        cas_data_dir = Path(args.cascade_data) if args.cascade_data else runtime_bundle_dir
        if cas_data_dir and (cas_data_dir / "coords.npz").exists():
            cas_coords = np.load(cas_data_dir / "coords.npz")
            cascade_lats = cas_coords["latitude"].astype(np.float64)
            cascade_lons = cas_coords["longitude"].astype(np.float64)
        else:
            # Default Krasnoyarsk ROI 0.25°
            cascade_lats = np.linspace(50, 60, 41)
            cascade_lons = np.linspace(83, 98, 61)

        # Load UNet scalers (may differ from GNN scalers)
        cas_sc_path = cas_data_dir / "scalers.npz" if cas_data_dir else None
        if cas_sc_path and cas_sc_path.exists():
            cas_sc = np.load(cas_sc_path)
            cas_mean = cas_sc["mean"].astype(np.float32)
            cas_std = cas_sc["std"].astype(np.float32)
        else:
            cas_mean, cas_std = y_mean[:C], y_std[:C]

        # Static fields for UNet
        n_static = 0
        static_t = None
        if cas_data_dir and (cas_data_dir / "static_fine.npy").exists():
            sf = np.load(cas_data_dir / "static_fine.npy")  # (lon, lat, N_static)
            sf = sf.transpose(1, 0, 2)  # (H, W, N_static)
            n_static = sf.shape[2]
            static_t = torch.from_numpy(sf).permute(2, 0, 1).unsqueeze(0).float().to(device)

        in_ch = cas_obs * C + n_static
        unet = WeatherUNet(in_channels=in_ch, out_channels=C, base_filters=cas_bf).to(device)
        unet_ckpt = cas_dir / "best_model.pth"
        unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
        unet.eval()
        print(f"[Cascade] UNet loaded from {unet_ckpt}, params={unet.num_params:,}")

        roi = (50.0, 60.0, 83.0, 98.0)
        H_fine, W_fine = len(cascade_lats), len(cascade_lons)
        cascade_phys = np.zeros((H_fine, W_fine, args.ar_steps, C), dtype=np.float32)

        flat_grid = getattr(metadata, "flat_grid", False)
        for step in range(args.ar_steps):
            gnn_step_norm = prediction_norm[:, step, :]  # (G, C)
            coarse_gnn = crop_and_upsample_roi(
                gnn_step_norm, latitudes, longitudes, roi,
                cascade_lats, cascade_lons, flat_grid=flat_grid,
            )  # (H, W, C) in GNN-normalized space
            coarse_phys_hw = coarse_gnn * y_std[:C] + y_mean[:C]
            coarse_ds_norm = (coarse_phys_hw - cas_mean) / cas_std

            frames = [coarse_ds_norm] * cas_obs
            x_unet = torch.from_numpy(np.concatenate(frames, axis=-1)).permute(2, 0, 1).unsqueeze(0).float().to(device)
            if static_t is not None:
                x_unet = torch.cat([x_unet, static_t], dim=1)
            with torch.no_grad():
                unet_out = unet(x_unet)
            if cas_residual:
                x_last = x_unet[:, (cas_obs - 1) * C : cas_obs * C, :, :]
                unet_out = x_last + unet_out
            cas_norm = unet_out[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            cascade_phys[:, :, step, :] = cas_norm * cas_std + cas_mean

        print(f"[Cascade] Refined {args.ar_steps} steps to {H_fine}×{W_fine} grid")

    # --- Lapse-rate elevation correction for t2m ---
    # Grid cells average topography at 0.25° resolution, which may differ from
    # actual station elevation. Apply a standard lapse rate of 6.5 °C/km.
    if args.lapse_target_elevation is not None and "t2m" in var_order and "z_surf" in var_order:
        z_idx = var_order.index("z_surf")
        t_idx = var_order.index("t2m")
        z_grid = prediction_phys[:, 0, z_idx]  # gpm (static, same all steps)
        z_target = float(args.lapse_target_elevation)
        LAPSE_RATE = 6.5e-3  # K per metre
        delta_z = z_grid - z_target  # positive means grid is higher → grid is colder
        delta_t = delta_z * LAPSE_RATE  # positive correction = warm up
        for step in range(prediction_phys.shape[1]):
            prediction_phys[:, step, t_idx] += delta_t
        city_mask = build_city_mask(latitudes, longitudes)
        mean_correction = delta_t[city_mask].mean() if city_mask.any() else delta_t.mean()
        print(f"[Lapse] Elevation correction: grid→{z_target:.0f}m, "
              f"city-average ΔT={mean_correction:+.2f}°C")

    # --- MOS bias correction ---
    mos_table = None
    learned_mos_bundle = None
    last_cycle = cycles[-1]
    forecast_valid_times = [
        last_cycle + timedelta(hours=6 * (step + 1))
        for step in range(args.ar_steps)
    ]

    # Learned MOS takes priority over static MOS
    if args.learned_mos:
        lmos_path = Path(args.learned_mos)
        if lmos_path.exists():
            learned_mos_bundle = load_learned_mos(lmos_path)
            # Build station list from trained stations metadata
            mos_stations = _get_mos_stations(learned_mos_bundle)
            result = apply_learned_mos_t2m(
                prediction_phys, var_order, learned_mos_bundle,
                latitudes, longitudes, forecast_valid_times,
                stations=mos_stations,
                spatial_idw=args.spatial_idw,
            )
            if isinstance(result, tuple):
                prediction_phys, n_pts = result
            else:
                prediction_phys = result
                n_pts = 1
            print(f"[Learned MOS] Applied ML t2m correction from {lmos_path.name}")
            print(f"  Stations: {len(mos_stations)}, corrected grid points: {n_pts}")
            print(f"  Test MAE: {learned_mos_bundle.get('test_mae', '?')}°C")
            # Wind MOS (if wind_model present in bundle)
            if learned_mos_bundle.get("wind_model") is not None:
                prediction_phys, n_wind_pts = apply_learned_mos_wind(
                    prediction_phys, var_order, learned_mos_bundle,
                    latitudes, longitudes, forecast_valid_times,
                    stations=mos_stations,
                    spatial_idw=args.spatial_idw,
                )
                print(f"[Learned MOS] Applied ML wind correction, grid points: {n_wind_pts}")
                print(f"  Wind test MAE: {learned_mos_bundle.get('wind_test_mae', '?')} m/s")
        else:
            print(f"[Learned MOS] WARNING: model not found at {lmos_path}, skipping")

    # ── Seasonal wind scaling ────────────────────────────────────────
    if args.wind_scale:
        ws_path = Path(args.wind_scale)
        if ws_path.exists():
            wind_scale_table = load_wind_scaling(ws_path)
            prediction_phys, n_ws_pts = apply_wind_scaling(
                prediction_phys, var_order, wind_scale_table,
                latitudes, longitudes, forecast_valid_times,
                spatial_idw=args.spatial_idw,
            )
            meta = wind_scale_table.get('_meta', {})
            print(f"[Wind Scale] Applied seasonal wind correction from {ws_path.name}")
            print(f"  Period: {meta.get('period', '?')}, corrected grid points: {n_ws_pts}")
        else:
            print(f"[Wind Scale] WARNING: table not found at {ws_path}, skipping")
    elif args.mos_table:
        mos_path = Path(args.mos_table)
        if mos_path.exists():
            mos_table = load_mos_table(mos_path)
            prediction_phys = apply_mos_t2m(
                prediction_phys, var_order, mos_table, forecast_valid_times,
            )
            print(f"[MOS] Applied t2m bias correction from {mos_path.name}")
            for vt in forecast_valid_times:
                from src.postprocessing.mos_correction import get_t2m_bias
                bias = get_t2m_bias(mos_table, vt)
                print(f"  +{(forecast_valid_times.index(vt)+1)*6:02d}h ({vt.strftime('%H:%MZ')}): bias={bias:+.2f}°C")
        else:
            print(f"[MOS] WARNING: table not found at {mos_path}, skipping correction")

    payload = {
        "cycles": [cycle.isoformat() for cycle in cycles],
        "var_names": var_order,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "input_normalized": input_tensor,
        "prediction_normalized": prediction_norm,
        "prediction_physical": prediction_phys,
        "warnings": warnings,
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir),
        "runtime_bundle": str(runtime_bundle_dir) if runtime_bundle_dir else None,
        "mos_applied": mos_table is not None,
        "learned_mos_applied": learned_mos_bundle is not None,
        "input_bias_t2m": args.input_bias_t2m,
        "cascade_experiment": args.cascade_experiment,
    }
    if cascade_phys is not None:
        payload["cascade_physical"] = cascade_phys  # (H, W, AR, C)
        payload["cascade_lats"] = cascade_lats
        payload["cascade_lons"] = cascade_lons
    torch.save(payload, out_dir / "forecast.pt")

    render_t2m_plot(out_dir / "t2m_city_forecast.png", latitudes, longitudes, prediction_phys, var_order)
    summarize_city(out_dir / "summary.txt", prediction_phys, latitudes, longitudes, var_order, cycles, warnings)

    if not args.keep_grib:
        for _, grib_path, _ in cycle_payloads:
            try:
                grib_path.unlink()
            except OSError:
                pass

    print(f"[live] Saved forecast to {out_dir}")


if __name__ == "__main__":
    main()
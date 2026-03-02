#!/usr/bin/env python3
"""
scripts/build_region_cds.py

Собирает РЕГИОНАЛЬНЫЙ датасет из CDS ERA5 API (Copernicus) на сетке 0.25deg.
Это альтернатива build_region_arco.py для случаев когда ARCO GCS тормозит.

CDS серверы в Европе, обычно быстрее из РФ.

Требования:
  pip install cdsapi netcdf4
  Файл ~/.cdsapirc с ключом API:
    url: https://cds.climate.copernicus.eu/api
    key: YOUR_UID:YOUR_API_KEY

Пример:
    python scripts/build_region_cds.py \
      --out-dir data/datasets/region_krsk_cds_19f \
      --start-date 2023-01-10 --end-date 2023-01-27 \
      --lon-min 83 --lon-max 98 --lat-min 50 --lat-max 60 \
      --train-scalers data/datasets/wb2_512x256_19f_ar/scalers.npz
"""

import argparse
import json
import os
import shutil
import ssl
import tempfile
import time as _time
import zipfile
from pathlib import Path

import numpy as np

# Disable SSL verification globally (proxy/VPN often break SSL with self-signed certs)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Маппинг переменных ───────────────────────────────────────────────

# CDS "reanalysis-era5-single-levels" short names
SURF_VARS_CDS = {
    "2m_temperature":            "t2m",
    "10m_u_component_of_wind":   "10u",
    "10m_v_component_of_wind":   "10v",
    "mean_sea_level_pressure":   "msl",
    "total_precipitation":       "tp",
    "surface_pressure":          "sp",
    "total_column_water_vapour": "tcwv",
    "geopotential":              "z_surf",   # surface geopotential
    "land_sea_mask":             "lsm",
}

# CDS "reanalysis-era5-pressure-levels" short names
PLEV_VARS_CDS = {
    "temperature":           "t",
    "u_component_of_wind":   "u",
    "v_component_of_wind":   "v",
    "geopotential":          "z",
    "specific_humidity":     "q",
}

LEVELS = [850, 500]

VAR_ORDER_19 = [
    "t2m", "10u", "10v", "msl", "tp",
    "sp", "tcwv",
    "z_surf", "lsm",
    "t@850", "u@850", "v@850", "z@850", "q@850",
    "t@500", "u@500", "v@500", "z@500", "q@500",
]

SCALE_FACTORS = {
    "msl":    0.01,        # Pa -> hPa
    "sp":     0.01,        # Pa -> hPa
    "z_surf": 1/9.80665,   # m2/s2 -> m
    "z@850":  1/9.80665,
    "z@500":  1/9.80665,
}


def check_cdsapi():
    """Проверяем наличие cdsapi и ключа."""
    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed. Run: pip install cdsapi")
        raise SystemExit(1)

    rc_path = Path.home() / ".cdsapirc"
    if not rc_path.exists():
        print("=" * 60)
        print("ERROR: ~/.cdsapirc not found!")
        print()
        print("1. Register at https://cds.climate.copernicus.eu/")
        print("2. Go to your profile page, copy your API key")
        print("3. Create ~/.cdsapirc with:")
        print()
        print("   url: https://cds.climate.copernicus.eu/api")
        print("   key: YOUR_UID:YOUR_API_KEY")
        print()
        print("Registration is free and instant.")
        print("=" * 60)
        raise SystemExit(1)

    return cdsapi.Client()


def _unzip_if_needed(filepath, tmpdir, prefix):
    """If CDS returned a .zip, extract and return path(s) to .nc files.
    Returns a list of paths (even if just one file)."""
    if zipfile.is_zipfile(filepath):
        print("  (Downloaded as .zip, extracting...)")
        extract_dir = os.path.join(tmpdir, prefix + "_unzipped")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.extractall(extract_dir)
        nc_files = sorted(
            [os.path.join(extract_dir, f)
             for f in os.listdir(extract_dir) if f.endswith(".nc")]
        )
        if nc_files:
            print("  Extracted %d .nc files" % len(nc_files))
            return nc_files
        all_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir)]
        if all_files:
            return all_files
    return [filepath]


def make_date_list(start_date, end_date):
    """Generate list of dates between start and end (inclusive)."""
    from datetime import datetime, timedelta
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def _half_year_ranges(start_date, end_date):
    """Return list of (year, month_start, month_end) tuples covering [start_date, end_date].
    Each range covers ~6 months: Jan-Jun or Jul-Dec."""
    from datetime import datetime
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    result = []
    y = s.year
    while y <= e.year:
        for m_start, m_end in [(1, 6), (7, 12)]:
            # Clip to actual date range
            if (y, m_end) < (s.year, s.month):
                continue
            if (y, m_start) > (e.year, e.month):
                continue
            result.append((y, m_start, m_end))
        y += 1
    return result


def _days_for_range(year, m_start, m_end, start_date, end_date):
    """Return list of day strings for months [m_start..m_end] of year, clipped to date range."""
    import calendar
    from datetime import datetime
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    days = []
    for m in range(m_start, m_end + 1):
        last_day = calendar.monthrange(year, m)[1]
        for d in range(1, last_day + 1):
            dt = datetime(year, m, d)
            if s <= dt <= e:
                days.append("%02d" % d)
    return days


def _months_for_range(m_start, m_end):
    return ["%02d" % m for m in range(m_start, m_end + 1)]


def download_surface(client, dates, area, tmpdir):
    """
    Download surface variables from CDS in half-year batches.
    area = [lat_max, lon_min, lat_min, lon_max] (North/West/South/East)
    Returns: list of paths to netcdf files.
    """
    print("\nDownloading SURFACE variables from CDS (half-year batches)...")
    print("  Variables: %s" % list(SURF_VARS_CDS.keys()))
    print("  Dates: %s to %s (%d days)" % (dates[0], dates[-1], len(dates)))
    print("  Area: N=%.1f W=%.1f S=%.1f E=%.1f" % tuple(area))

    ranges = _half_year_ranges(dates[0], dates[-1])
    print("  Batches: %d" % len(ranges))
    all_files = []
    t0 = _time.time()

    for idx, (year, m_start, m_end) in enumerate(ranges):
        label = "%04d-H%d" % (year, 1 if m_start == 1 else 2)
        outfile = os.path.join(tmpdir, "surface_%s.nc" % label)
        if os.path.exists(outfile) and os.path.getsize(outfile) > 1000:
            print("  [%d/%d] %s — cached" % (idx+1, len(ranges), label))
            all_files.append(outfile)
            continue

        days = _days_for_range(year, m_start, m_end, dates[0], dates[-1])
        months = _months_for_range(m_start, m_end)
        if not days:
            continue

        print("  [%d/%d] %s (%d days)..." % (idx+1, len(ranges), label, len(days)),
              end=" ", flush=True)
        t1 = _time.time()

        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": list(SURF_VARS_CDS.keys()),
                    "year": [str(year)],
                    "month": months,
                    "day": sorted(set(days)),
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "area": area,
                    "grid": [0.25, 0.25],
                    "download_format": "unarchived",
                    "format": "netcdf",
                },
                outfile,
            )
            extracted = _unzip_if_needed(outfile, tmpdir, "surface_%s" % label)
            all_files.extend(extracted)
            sz = sum(os.path.getsize(f) for f in extracted) / 1024**2
            print("%.1f MB [%.0fs]" % (sz, _time.time() - t1))
        except Exception as e:
            print("FAILED: %s" % e)
            continue

    total_mb = sum(os.path.getsize(f) for f in all_files if os.path.exists(f)) / 1024**2
    dt = _time.time() - t0
    print("  Surface total: %.1f MB in %.0fs" % (total_mb, dt))

    return all_files


def download_pressure(client, dates, area, tmpdir):
    """
    Download pressure level variables from CDS.
    Split by LEVEL to keep request size within CDS limits:
      half-year × 5 vars × 1 level ≈ 3600 fields (OK)
    vs half-year × 5 vars × 2 levels ≈ 7200 fields (403 Forbidden).
    Returns: list of paths to netcdf files.
    """
    print("\nDownloading PRESSURE LEVEL variables from CDS (per-level, half-year batches)...")
    print("  Variables: %s" % list(PLEV_VARS_CDS.keys()))
    print("  Levels: %s hPa (each downloaded separately)" % LEVELS)

    ranges = _half_year_ranges(dates[0], dates[-1])
    n_batches = len(ranges) * len(LEVELS)
    print("  Batches: %d (%d half-years × %d levels)" % (n_batches, len(ranges), len(LEVELS)))
    all_files = []
    t0 = _time.time()
    batch_idx = 0

    for year, m_start, m_end in ranges:
        half = 1 if m_start == 1 else 2
        days = _days_for_range(year, m_start, m_end, dates[0], dates[-1])
        months = _months_for_range(m_start, m_end)
        if not days:
            batch_idx += len(LEVELS)
            continue

        for level in LEVELS:
            batch_idx += 1
            label = "%04d-H%d-L%d" % (year, half, level)
            outfile = os.path.join(tmpdir, "pressure_%s.nc" % label)
            if os.path.exists(outfile) and os.path.getsize(outfile) > 1000:
                print("  [%d/%d] %s — cached" % (batch_idx, n_batches, label))
                all_files.append(outfile)
                continue

            print("  [%d/%d] %s (%d days)..." % (batch_idx, n_batches, label, len(days)),
                  end=" ", flush=True)
            t1 = _time.time()

            try:
                client.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": list(PLEV_VARS_CDS.keys()),
                        "pressure_level": [str(level)],
                        "year": [str(year)],
                        "month": months,
                        "day": sorted(set(days)),
                        "time": ["00:00", "06:00", "12:00", "18:00"],
                        "area": area,
                        "grid": [0.25, 0.25],
                        "download_format": "unarchived",
                        "format": "netcdf",
                    },
                    outfile,
                )
                extracted = _unzip_if_needed(outfile, tmpdir, "pressure_%s" % label)
                all_files.extend(extracted)
                sz = sum(os.path.getsize(f) for f in extracted) / 1024**2
                print("%.1f MB [%.0fs]" % (sz, _time.time() - t1))
            except Exception as e:
                print("FAILED: %s" % e)
                continue

    total_mb = sum(os.path.getsize(f) for f in all_files if os.path.exists(f)) / 1024**2
    dt = _time.time() - t0
    print("  Pressure total: %.1f MB in %.0fs" % (total_mb, dt))

    return all_files


def parse_netcdf(surf_paths, pres_paths, scale_factors):
    """
    Parse downloaded netcdf files into channels dict.
    surf_paths / pres_paths: list of .nc file paths (CDS may split into multiple files).
    Returns: channels dict, lons, lats, n_time.
    """
    import xarray as xr

    channels = {}

    # ── Surface ────────────────────────────────────────
    print("\nParsing surface netcdf (%d files)..." % len(surf_paths))
    if len(surf_paths) == 1:
        ds_s = xr.open_dataset(surf_paths[0])
    else:
        # Merge all surface nc files into one dataset
        datasets = []
        for p in surf_paths:
            try:
                datasets.append(xr.open_dataset(p))
            except Exception as e:
                print("  WARNING: could not open %s: %s" % (p, e))
        ds_s = xr.merge(datasets)
        for d in datasets:
            d.close()
    print("  Available vars: %s" % list(ds_s.data_vars))

    # CDS netcdf variable short names
    # t2m -> 't2m', u10 -> 'u10', v10 -> 'v10', msl -> 'msl',
    # tp -> 'tp', sp -> 'sp', tcwv -> 'tcwv', z -> 'z', lsm -> 'lsm'
    CDS_NC_MAP = {
        "t2m":    "t2m",
        "u10":    "10u",
        "v10":    "10v",
        "msl":    "msl",
        "tp":     "tp",
        "sp":     "sp",
        "tcwv":   "tcwv",
        "z":      "z_surf",
        "lsm":    "lsm",
    }

    # Determine coord names
    lat_c = "latitude" if "latitude" in ds_s.coords else "lat"
    lon_c = "longitude" if "longitude" in ds_s.coords else "lon"

    lats = ds_s[lat_c].values.astype(np.float32)
    lons = ds_s[lon_c].values.astype(np.float32)

    # CDS returns lat descending (90 -> -90), sort ascending
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        flip_lat = True
    else:
        flip_lat = False

    for nc_name, our_name in CDS_NC_MAP.items():
        if nc_name not in ds_s.data_vars:
            # Try alternative names
            alt_names = {"u10": "10u", "v10": "10v"}
            nc_name_try = alt_names.get(nc_name, None)
            if nc_name_try and nc_name_try in ds_s.data_vars:
                nc_name = nc_name_try
            else:
                print("  WARNING: %s not found in surface file, available: %s" % (
                    nc_name, list(ds_s.data_vars)))
                continue

        da = ds_s[nc_name]
        arr = da.values.astype(np.float32)  # (time, lat, lon) or (lat, lon) for static

        if arr.ndim == 2:
            # Static variable (lsm, z at surface) - no time dim
            if flip_lat:
                arr = arr[::-1, :]
            # Duplicate across time
            n_time_s = ds_s.sizes.get("time", ds_s.sizes.get("valid_time", 1))
            arr = np.tile(arr[np.newaxis, :, :], (n_time_s, 1, 1))
        elif arr.ndim == 3:
            if flip_lat:
                arr = arr[:, ::-1, :]
        else:
            print("  WARNING: unexpected ndim=%d for %s" % (arr.ndim, nc_name))
            continue

        # (time, lat, lon) -> (time, lon, lat)
        arr = np.swapaxes(arr, 1, 2)

        if our_name in scale_factors:
            arr *= scale_factors[our_name]

        channels[our_name] = arr
        print("  %-8s <- %-6s shape=%s range=[%.3f, %.3f]" % (
            our_name, nc_name, arr.shape, arr.min(), arr.max()))

    n_time = arr.shape[0]
    ds_s.close()

    # ── Pressure levels ────────────────────────────────
    print("\nParsing pressure level netcdf (%d files)..." % len(pres_paths))
    if len(pres_paths) == 1:
        ds_p = xr.open_dataset(pres_paths[0])
    else:
        datasets = []
        for p in pres_paths:
            try:
                datasets.append(xr.open_dataset(p))
            except Exception as e:
                print("  WARNING: could not open %s: %s" % (p, e))
        ds_p = xr.merge(datasets)
        for d in datasets:
            d.close()
    print("  Available vars: %s" % list(ds_p.data_vars))

    # CDS netcdf: pressure level var short names
    CDS_PLEV_NC_MAP = {
        "t":   "t",
        "u":   "u",
        "v":   "v",
        "z":   "z",
        "q":   "q",
    }

    level_c = None
    for candidate in ["level", "pressure_level", "isobaricInhPa"]:
        if candidate in ds_p.coords:
            level_c = candidate
            break
    if level_c is None:
        print("  WARNING: no level coordinate found")
        level_c = "level"

    for nc_name, short_name in CDS_PLEV_NC_MAP.items():
        if nc_name not in ds_p.data_vars:
            print("  WARNING: %s not found" % nc_name)
            continue

        da = ds_p[nc_name]

        for lev in LEVELS:
            ch = "%s@%d" % (short_name, lev)
            da_lev = da.sel({level_c: lev})
            arr = da_lev.values.astype(np.float32)

            if flip_lat:
                arr = arr[:, ::-1, :]

            # (time, lat, lon) -> (time, lon, lat)
            arr = np.swapaxes(arr, 1, 2)

            if ch in scale_factors:
                arr *= scale_factors[ch]

            channels[ch] = arr
            print("  %-8s <- %s@%d shape=%s range=[%.3f, %.3f]" % (
                ch, nc_name, lev, arr.shape, arr.min(), arr.max()))

    ds_p.close()

    return channels, lons, lats, n_time


def main():
    p = argparse.ArgumentParser(
        description="Build regional ERA5 dataset from CDS API (0.25deg, 19-var)")

    p.add_argument("--out-dir",       required=True)
    p.add_argument("--start-date",    required=True, help="e.g. 2023-01-10")
    p.add_argument("--end-date",      required=True, help="e.g. 2023-01-27")
    p.add_argument("--lon-min",       type=float, default=83.0)
    p.add_argument("--lon-max",       type=float, default=98.0)
    p.add_argument("--lat-min",       type=float, default=50.0)
    p.add_argument("--lat-max",       type=float, default=60.0)
    p.add_argument("--train-scalers", required=True,
                   help="Path to scalers.npz from training dataset")
    p.add_argument("--keep-nc",       action="store_true",
                   help="Keep downloaded .nc files (for debugging)")
    p.add_argument("--pressure-only", action="store_true",
                   help="Re-download only pressure levels (reuse cached surface)")

    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify scalers
    sc_path = Path(args.train_scalers)
    sc = np.load(sc_path)
    if "mean" not in sc or "std" not in sc:
        print("ERROR: scalers must have 'mean' and 'std' keys (19-var format)")
        raise SystemExit(1)
    n_feat = len(sc["mean"])
    if n_feat != 19:
        print("ERROR: expected 19 features, got %d" % n_feat)
        raise SystemExit(1)

    print("=" * 60)
    print("Building regional ERA5 dataset from CDS API")
    print("  Region: lon [%.1f, %.1f], lat [%.1f, %.1f]" % (
        args.lon_min, args.lon_max, args.lat_min, args.lat_max))
    print("  Period: %s to %s" % (args.start_date, args.end_date))
    print("  Grid: 0.25 deg (native ERA5)")
    print("  Variables: 19 (surface + pressure levels)")
    print("=" * 60)

    # Check CDS API
    client = check_cdsapi()

    # Date list
    dates = make_date_list(args.start_date, args.end_date)
    print("  Days: %d" % len(dates))

    # Estimate grid size
    n_lon_est = int((args.lon_max - args.lon_min) / 0.25) + 1
    n_lat_est = int((args.lat_max - args.lat_min) / 0.25) + 1
    n_time_est = len(dates) * 4  # 4 per day (6h)
    print("  Expected grid: %d x %d = %d nodes" % (n_lon_est, n_lat_est, n_lon_est * n_lat_est))
    print("  Expected timesteps: %d" % n_time_est)
    est_mb = n_time_est * n_lon_est * n_lat_est * 19 * 4 / 1024**2
    print("  Estimated download: ~%.0f MB (before compression)" % est_mb)

    # CDS area format: [North, West, South, East]
    area = [args.lat_max, args.lon_min, args.lat_min, args.lon_max]

    # Download — use out_dir/_tmp as persistent tmpdir (resumable!)
    tmpdir = str(out_dir / "_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    print("\n  Temp dir: %s (persistent, for resume)" % tmpdir)

    t_start = _time.time()

    if args.pressure_only:
        # Reuse cached surface files from tmpdir
        import glob
        surf_paths = sorted(glob.glob(os.path.join(tmpdir, "surface_*.nc")))
        if not surf_paths:
            print("ERROR: --pressure-only but no cached surface files in %s" % tmpdir)
            raise SystemExit(1)
        print("\n  Reusing %d cached surface files" % len(surf_paths))
    else:
        surf_paths = download_surface(client, dates, area, tmpdir)
    pres_paths = download_pressure(client, dates, area, tmpdir)

    t_download = _time.time() - t_start
    print("\nTotal download time: %.0fs (%.1f min)" % (t_download, t_download / 60))

    # Parse
    channels, lons, lats, n_time = parse_netcdf(surf_paths, pres_paths, SCALE_FACTORS)

    # Optionally keep nc files
    if args.keep_nc:
        for i, sp in enumerate(surf_paths):
            shutil.copy2(sp, out_dir / ("surface_%d.nc" % i))
        for i, pp in enumerate(pres_paths):
            shutil.copy2(pp, out_dir / ("pressure_%d.nc" % i))
        print("\nKept .nc files in %s" % out_dir)

    # Cleanup temp — only if we actually got pressure data
    has_pressure = any(v in channels for v in ["t@850", "u@850", "t@500"])
    if has_pressure and os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
        print("  Cleaned up tmp dir")
    elif not has_pressure:
        print("  WARNING: pressure data missing! Keeping tmp dir for --pressure-only resume")

    # Assemble (T, LON, LAT, F) array
    n_lon, n_lat = len(lons), len(lats)
    print("\nAssembling: (%d, %d, %d, %d)" % (n_time, n_lon, n_lat, n_feat))

    arr = np.zeros((n_time, n_lon, n_lat, n_feat), dtype=np.float32)
    for i, var in enumerate(VAR_ORDER_19):
        if var not in channels:
            print("  MISSING: %s — filling with zeros!" % var)
            continue
        ch = channels[var]
        # Handle static vars broadcast
        if ch.shape[0] == 1 and n_time > 1:
            ch = np.tile(ch, (n_time, 1, 1))
        arr[:, :, :, i] = ch[:n_time]
        print("  %2d: %-8s mean=%.4f  std=%.4f" % (i, var, ch.mean(), ch.std()))

    del channels

    # Save chunked format
    print("\nSaving chunked format...")

    # data.npy
    fp = np.memmap(str(out_dir / "data.npy"), dtype=np.float16, mode="w+",
                   shape=(n_time, n_lon, n_lat, n_feat))
    fp[:] = arr.astype(np.float16)
    fp.flush()
    del fp
    size_mb = n_time * n_lon * n_lat * n_feat * 2 / 1024**2
    print("  data.npy: %.1f MB" % size_mb)

    # scalers.npz
    shutil.copy2(sc_path, out_dir / "scalers.npz")
    print("  scalers.npz (copied)")

    # coords.npz
    np.savez(out_dir / "coords.npz",
             longitude=lons, latitude=lats)
    print("  coords.npz")

    # variables.json
    (out_dir / "variables.json").write_text(json.dumps(VAR_ORDER_19, indent=2))
    print("  variables.json")

    # dataset_info.json
    info = {
        "time_start": args.start_date,
        "time_end": args.end_date,
        "n_time": int(n_time),
        "n_lon": int(n_lon),
        "n_lat": int(n_lat),
        "n_feat": int(n_feat),
        "variables": VAR_ORDER_19,
        "dtype": "float16",
        "file": "data.npy",
        "size_gb": round(size_mb / 1024, 3),
        "source": "CDS ERA5 API (regional, 0.25 deg)",
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False))
    print("  dataset_info.json")

    total_mb = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file()) / 1024**2
    print("\nTotal size: %.1f MB" % total_mb)
    print("\n" + "=" * 60)
    print("DONE!")
    print("\nFor inference:")
    print("  python scripts/predict.py experiments/wb2_512x256_19f_ar \\")
    print("    --data-dir %s --ar-steps 4 --max-samples 50 \\" % out_dir)
    print("    --region %.1f %.1f %.1f %.1f --per-channel" % (
        args.lat_min, args.lat_max, args.lon_min, args.lon_max))
    print("=" * 60)


if __name__ == "__main__":
    main()

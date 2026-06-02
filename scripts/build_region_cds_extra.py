#!/usr/bin/env python3
"""
scripts/build_region_cds_extra.py

Качает РЕГИОНАЛЬНЫЕ extra-каналы (10 plev: z/t/u/v/q @ 250 и @ 1000)
из CDS ERA5 API на 0.25° для дальнейшего МЕРДЖА (без интерполяции)
в multires_russia_33f.

Это сиблинг build_region_cds.py, но качает только pressure-level @ 250 и @ 1000.
Формат выхода аналогичен global_512x256_extra_2010-2021_07deg.

Требования:
  pip install cdsapi netcdf4 xarray
  ~/.cdsapirc с ключом API.

Пример (full 2010-2021 Russia):
    python scripts/build_region_cds_extra.py \\
        --out-dir /data/datasets/region_russia_645x165_extra_2010-2021_025deg \\
        --start-date 2010-01-01 --end-date 2021-12-31 \\
        --lon-min 19 --lon-max 180 --lat-min 41 --lat-max 82
"""

import argparse
import json
import os
import shutil
import time as _time
import zipfile
from pathlib import Path

import numpy as np

# Disable SSL verification (Russian VPN/proxy may break self-signed certs)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Variables ─────────────────────────────────────────────────────────
PLEV_VARS_CDS = {
    "geopotential":         "z",
    "temperature":          "t",
    "u_component_of_wind":  "u",
    "v_component_of_wind":  "v",
    "specific_humidity":    "q",
}

LEVELS = [250, 1000]

# Order MUST match global_512x256_extra channel layout for scaler merge
VAR_ORDER_EXTRA = [
    "z@250", "t@250", "u@250", "v@250", "q@250",
    "z@1000", "t@1000", "u@1000", "v@1000", "q@1000",
]

SCALE_FACTORS = {
    "z@250":  1 / 9.80665,    # m^2/s^2 -> m (geopotential height)
    "z@1000": 1 / 9.80665,
}


def check_cdsapi():
    try:
        import cdsapi
    except ImportError:
        raise SystemExit("ERROR: cdsapi not installed. pip install cdsapi")
    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        raise SystemExit("ERROR: ~/.cdsapirc not found. Register at https://cds.climate.copernicus.eu/")
    return cdsapi.Client()


def _unzip_if_needed(filepath, tmpdir, prefix):
    if zipfile.is_zipfile(filepath):
        extract_dir = os.path.join(tmpdir, prefix + "_unzipped")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.extractall(extract_dir)
        nc_files = sorted(
            os.path.join(extract_dir, f)
            for f in os.listdir(extract_dir) if f.endswith(".nc")
        )
        if nc_files:
            return nc_files
    return [filepath]


def _half_year_ranges(start_date, end_date):
    from datetime import datetime
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    result = []
    y = s.year
    while y <= e.year:
        for m_start, m_end in [(1, 6), (7, 12)]:
            if (y, m_end) < (s.year, s.month):
                continue
            if (y, m_start) > (e.year, e.month):
                continue
            result.append((y, m_start, m_end))
        y += 1
    return result


def _days_for_range(year, m_start, m_end, start_date, end_date):
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


def download_pressure(client, start_date, end_date, area, tmpdir):
    """Half-year × per-level batches. Resumable through cache files in tmpdir."""
    print("\nDownloading PRESSURE LEVELS from CDS (per-level, half-year batches)...")
    print("  Variables: %s" % list(PLEV_VARS_CDS))
    print("  Levels: %s hPa" % LEVELS)

    ranges = _half_year_ranges(start_date, end_date)
    n_batches = len(ranges) * len(LEVELS)
    print("  Batches: %d (%d half-years × %d levels)" % (n_batches, len(ranges), len(LEVELS)))
    all_files = []
    t0 = _time.time()
    batch_idx = 0

    for year, m_start, m_end in ranges:
        half = 1 if m_start == 1 else 2
        days = _days_for_range(year, m_start, m_end, start_date, end_date)
        months = _months_for_range(m_start, m_end)
        if not days:
            batch_idx += len(LEVELS)
            continue

        for level in LEVELS:
            batch_idx += 1
            label = "%04d-H%d-L%d" % (year, half, level)
            outfile = os.path.join(tmpdir, "extra_%s.nc" % label)
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
                        "variable": list(PLEV_VARS_CDS),
                        "pressure_level": [str(level)],
                        "year": [str(year)],
                        "month": months,
                        "day": sorted(set(days)),
                        "time": ["00:00", "06:00", "12:00", "18:00"],
                        "area": area,  # [N, W, S, E]
                        "grid": [0.25, 0.25],
                        "download_format": "unarchived",
                        "format": "netcdf",
                    },
                    outfile,
                )
                extracted = _unzip_if_needed(outfile, tmpdir, "extra_%s" % label)
                all_files.extend(extracted)
                sz = sum(os.path.getsize(f) for f in extracted) / 1024**2
                print("%.1f MB [%.0fs]" % (sz, _time.time() - t1))
            except Exception as e:
                print("FAILED: %s" % e)
                continue

    total_mb = sum(os.path.getsize(f) for f in all_files if os.path.exists(f)) / 1024**2
    print("  Total: %.1f MB in %.0f min" % (total_mb, (_time.time() - t0) / 60))
    return all_files


def _detect_coords(ds):
    lat_c = "latitude" if "latitude" in ds.coords else "lat"
    lon_c = "longitude" if "longitude" in ds.coords else "lon"
    level_c = None
    for cand in ["level", "pressure_level", "isobaricInhPa"]:
        if cand in ds.coords:
            level_c = cand
            break
    time_c = "valid_time" if "valid_time" in ds.coords else "time"
    return lat_c, lon_c, level_c, time_c


def assemble_streaming(paths, out_dir, ref_coords=None):
    """Stream per-batch into a memmap to avoid OOM.

    Returns: (n_time, n_lon, n_lat, lons, lats, mean, std).
    """
    import xarray as xr

    print("\nStreaming parse of %d netcdf files..." % len(paths))

    # Pass 1: collect time axis and verify spatial coords from L250 files (per half-year).
    # We rely on filename schema extra_YYYY-Hx-Lyyy.nc.
    times_per_batch = {}  # (year, half) -> np.array(datetime64)
    lats_ref = None
    lons_ref = None
    flip_lat = False
    level_c_name = None

    sample_paths = sorted(p for p in paths if "-L250." in os.path.basename(p))
    if not sample_paths:
        sample_paths = sorted(paths)

    for p in sample_paths:
        ds = xr.open_dataset(p)
        lat_c, lon_c, level_c, time_c = _detect_coords(ds)
        if level_c_name is None:
            level_c_name = level_c
        lats = ds[lat_c].values.astype(np.float32)
        lons = ds[lon_c].values.astype(np.float32)
        if lats_ref is None:
            if ref_coords is not None:
                ref = np.load(ref_coords)
                ref_lats = ref["latitude"].astype(np.float32)
                ref_lons = ref["longitude"].astype(np.float32)
                if len(ref_lats) != len(lats) or len(ref_lons) != len(lons):
                    raise SystemExit("ref coords shape mismatch: ref=(%d,%d) cds=(%d,%d)" % (
                        len(ref_lats), len(ref_lons), len(lats), len(lons)))
                flip_lat = (lats[0] > lats[-1]) != (ref_lats[0] > ref_lats[-1])
                lats_eff = lats[::-1] if flip_lat else lats
                if not np.allclose(lats_eff, ref_lats, atol=1e-3):
                    raise SystemExit("lat values differ from ref (max diff %.4f)" % np.abs(lats_eff - ref_lats).max())
                if not np.allclose(lons, ref_lons, atol=1e-3):
                    raise SystemExit("lon values differ from ref (max diff %.4f)" % np.abs(lons - ref_lons).max())
                lats_ref = ref_lats
                lons_ref = ref_lons
                print("  Aligned to ref coords (flip_lat=%s)" % flip_lat)
            else:
                flip_lat = lats[0] > lats[-1]
                lats_ref = lats[::-1] if flip_lat else lats
                lons_ref = lons
        base = os.path.basename(p)
        # extra_YYYY-Hx-Lyyy.nc
        try:
            tag = base.replace("extra_", "").replace(".nc", "")
            yh = tag.split("-L")[0]  # YYYY-Hx
            year_s, half_s = yh.split("-H")
            key = (int(year_s), int(half_s))
        except Exception:
            key = base
        times_per_batch[key] = ds[time_c].values
        ds.close()

    # Build global sorted unique time axis
    all_times = np.concatenate(list(times_per_batch.values()))
    all_times = np.unique(all_times)
    all_times.sort()
    n_time = len(all_times)
    n_lat = len(lats_ref)
    n_lon = len(lons_ref)
    n_feat = len(VAR_ORDER_EXTRA)
    print("  n_time=%d n_lon=%d n_lat=%d n_feat=%d" % (n_time, n_lon, n_lat, n_feat))

    # Time -> index lookup
    time_to_idx = {t: i for i, t in enumerate(all_times)}

    # Allocate memmap (float16) on disk for the output
    out_path = Path(out_dir) / "data_extra.npy"
    print("  Allocating memmap: %s (~%.1f GB)" % (
        out_path, n_time * n_lon * n_lat * n_feat * 2 / 1024**3))
    fp = np.memmap(out_path, dtype=np.float16, mode="w+",
                   shape=(n_time, n_lon, n_lat, n_feat))

    # Welford running stats per channel (float64)
    count = np.zeros(n_feat, dtype=np.float64)
    mean = np.zeros(n_feat, dtype=np.float64)
    m2 = np.zeros(n_feat, dtype=np.float64)

    var_indices = {v: i for i, v in enumerate(VAR_ORDER_EXTRA)}

    # Pass 2: process each batch file
    for bi, p in enumerate(sorted(paths)):
        base = os.path.basename(p)
        # parse level from name
        try:
            tag = base.replace("extra_", "").replace(".nc", "")
            lvl = int(tag.split("-L")[1])
        except Exception:
            ds_tmp = xr.open_dataset(p)
            lvl = int(ds_tmp[level_c_name].values.ravel()[0])
            ds_tmp.close()

        ds = xr.open_dataset(p)
        lat_c, lon_c, level_c, time_c = _detect_coords(ds)
        times_b = ds[time_c].values
        # indices in global axis
        try:
            idx = np.array([time_to_idx[t] for t in times_b], dtype=np.int64)
        except KeyError as e:
            raise SystemExit("Time %s not in global axis" % e)

        print("  [%d/%d] %s lvl=%d Tb=%d..." % (bi + 1, len(paths), base, lvl, len(times_b)),
              flush=True)

        for short in ["z", "t", "u", "v", "q"]:
            if short not in ds.data_vars:
                continue
            ch_name = "%s@%d" % (short, lvl)
            if ch_name not in var_indices:
                continue
            ci = var_indices[ch_name]
            da = ds[short]
            if level_c is not None and level_c in da.dims:
                da = da.sel({level_c: lvl})
            arr = da.values.astype(np.float32)  # (time, lat, lon)
            if flip_lat:
                arr = arr[:, ::-1, :]
            arr = np.swapaxes(arr, 1, 2)  # (time, lon, lat)
            if ch_name in SCALE_FACTORS:
                arr *= SCALE_FACTORS[ch_name]
            # write into memmap at correct time indices
            fp[idx, :, :, ci] = arr.astype(np.float16)
            # update Welford
            flat = arr.reshape(-1).astype(np.float64)
            n_b = flat.size
            mean_b = flat.mean()
            m2_b = ((flat - mean_b) ** 2).sum()
            n_a = count[ci]
            delta = mean_b - mean[ci]
            tot = n_a + n_b
            mean[ci] = mean[ci] + delta * (n_b / tot)
            m2[ci] = m2[ci] + m2_b + (delta ** 2) * (n_a * n_b / tot)
            count[ci] = tot
            del arr, flat
        ds.close()

    fp.flush()
    del fp

    std = np.sqrt(m2 / np.maximum(count, 1)).astype(np.float32)
    mean_f32 = mean.astype(np.float32)
    for i, v in enumerate(VAR_ORDER_EXTRA):
        print("  %2d %-8s mean=%.4f std=%.4f (n=%g)" % (i, v, mean_f32[i], std[i], count[i]))

    return n_time, n_lon, n_lat, lons_ref, lats_ref, mean_f32, std


def main():
    p = argparse.ArgumentParser(description="Build regional EXTRA dataset (plev@250 + plev@1000) from CDS")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date",   required=True, help="YYYY-MM-DD")
    p.add_argument("--lon-min",    type=float, required=True)
    p.add_argument("--lon-max",    type=float, required=True)
    p.add_argument("--lat-min",    type=float, required=True)
    p.add_argument("--lat-max",    type=float, required=True)
    p.add_argument("--keep-nc",    action="store_true")
    p.add_argument("--ref-coords", default=None,
                   help="coords.npz of existing region 19f to align lat orientation with")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Regional ERA5 EXTRA (plev@250 + plev@1000) from CDS")
    print("  Region: lon[%.2f, %.2f] lat[%.2f, %.2f]" % (
        args.lon_min, args.lon_max, args.lat_min, args.lat_max))
    print("  Period: %s .. %s" % (args.start_date, args.end_date))
    print("  Grid: 0.25 deg, step 6h")
    print("  Channels: %s" % VAR_ORDER_EXTRA)
    print("=" * 60)

    client = check_cdsapi()

    # area = [North, West, South, East]
    area = [args.lat_max, args.lon_min, args.lat_min, args.lon_max]
    tmpdir = str(out_dir / "_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    print("\nTmp (resumable cache): %s" % tmpdir)

    t_start = _time.time()
    paths = download_pressure(client, args.start_date, args.end_date, area, tmpdir)
    print("\nDownload phase: %.1f min" % ((_time.time() - t_start) / 60))

    if not paths:
        raise SystemExit("ERROR: no files downloaded")

    channels_info = assemble_streaming(paths, out_dir, ref_coords=args.ref_coords)
    n_time, n_lon, n_lat, lons, lats, mean, std = channels_info
    n_feat = len(VAR_ORDER_EXTRA)

    print("\nSaving scalers / coords / metadata...")
    np.savez(out_dir / "scalers_extra.npz", mean=mean, std=std)
    np.savez(out_dir / "coords.npz", longitude=lons.astype(np.float32), latitude=lats.astype(np.float32))
    (out_dir / "variables_extra.json").write_text(json.dumps(VAR_ORDER_EXTRA, indent=2))

    info = {
        "time_start": args.start_date,
        "time_end": args.end_date,
        "n_time": int(n_time),
        "n_lon": int(n_lon),
        "n_lat": int(n_lat),
        "n_feat_extra": int(n_feat),
        "variables_extra": VAR_ORDER_EXTRA,
        "dtype": "float16",
        "file": "data_extra.npy",
        "size_gb": round(n_time * n_lon * n_lat * n_feat * 2 / 1024**3, 3),
        "source": "CDS ERA5 reanalysis-era5-pressure-levels (regional, 0.25°)",
        "region": {
            "lon_min": args.lon_min, "lon_max": args.lon_max,
            "lat_min": args.lat_min, "lat_max": args.lat_max,
        },
    }
    (out_dir / "dataset_info_extra.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    if not args.keep_nc:
        shutil.rmtree(tmpdir, ignore_errors=True)

    total_gb = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file()) / 1024**3
    print("\n" + "=" * 60)
    print("DONE in %.1f min. Total: %.2f GB" % ((_time.time() - t_start) / 60, total_gb))
    print("Output: %s" % out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

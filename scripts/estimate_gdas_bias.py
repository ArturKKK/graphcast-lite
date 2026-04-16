#!/usr/bin/env python3
"""
Estimate recent GDAS t2m bias vs METAR observations for a station.

Downloads recent METAR reports (via Iowa State ASOS) and compares with
GDAS analysis t2m at the nearest grid point to compute rolling bias.

Usage:
    python scripts/estimate_gdas_bias.py
    python scripts/estimate_gdas_bias.py --station UNKL --days 14
    python scripts/estimate_gdas_bias.py --station UNKL --days 7 --output results/gdas_bias.json

Output: prints recommended --input-bias-t2m value for live_gdas_forecast.py
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


# Station registry: ICAO → (lat, lon, elev_m)
STATIONS = {
    "UNKL": ("Krasnoyarsk (Yemelyanovo)", 56.173, 92.493, 277),
}


def fetch_metar_hourly(icao: str, start: datetime, end: datetime) -> list[dict]:
    """Fetch hourly METAR temps from Iowa State ASOS network."""
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={icao}"
        f"&data=tmpf"
        f"&tz=Etc/UTC"
        f"&format=comma"
        f"&latlon=no"
        f"&elev=no"
        f"&missing=M"
        f"&trace=T"
        f"&direct=no"
        f"&report_type=3"
        f"&year1={start.year}&month1={start.month}&day1={start.day}"
        f"&year2={end.year}&month2={end.month}&day2={end.day}"
    )
    result = subprocess.run(
        ["curl", "-s", "--max-time", "30", url],
        capture_output=True, text=True, check=True,
    )

    records = []
    for line in result.stdout.strip().split("\n"):
        if line.startswith("station") or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            time_str = parts[1].strip()
            tmpf_str = parts[2].strip()
            if tmpf_str == "M":
                continue
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
            dt = dt.replace(tzinfo=timezone.utc)
            tmpf = float(tmpf_str)
            tmpc = (tmpf - 32) * 5 / 9
            records.append({"time": dt, "t2m_c": tmpc})
        except (ValueError, IndexError):
            continue

    return records


def fetch_gdas_t2m_at_point(cycle_dt: datetime, lat: float, lon: float,
                            cache_dir: Path) -> float | None:
    """Get GDAS analysis t2m at a point for a given cycle using cfgrib.

    Returns temperature in °C, or None if unavailable.
    """
    try:
        import xarray as xr
    except ImportError:
        print("ERROR: xarray required. pip install xarray cfgrib", file=sys.stderr)
        return None

    ymd = cycle_dt.strftime("%Y%m%d")
    hh = cycle_dt.strftime("%H")
    url = (
        f"https://www.ncei.noaa.gov/data/global-forecast-system/"
        f"access/grid-003-1.0-degree/analysis/{ymd}/"
        f"gfs_3_{ymd}_{hh}00_000.grb2"
    )
    # Use NOMADS for recent data (faster)
    url_nomads = (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
        f"gdas.{ymd}/{hh}/atmos/gdas.t{hh}z.pgrb2.0p25.f000"
    )
    grib_path = cache_dir / f"gdas_{ymd}_{hh}.grib2"

    if not grib_path.exists():
        for attempt_url in [url_nomads]:
            try:
                subprocess.run(
                    ["curl", "-s", "--max-time", "120", "-o", str(grib_path), attempt_url],
                    check=True, capture_output=True,
                )
                if grib_path.stat().st_size > 1000:
                    break
                grib_path.unlink(missing_ok=True)
            except subprocess.CalledProcessError:
                grib_path.unlink(missing_ok=True)
                continue

    if not grib_path.exists() or grib_path.stat().st_size < 1000:
        return None

    try:
        ds = xr.open_dataset(
            str(grib_path),
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t"}},
        )
        t2m = ds["t2m"]
        # Nearest-neighbor interpolation
        lon_360 = lon % 360
        val = float(t2m.sel(latitude=lat, longitude=lon_360, method="nearest").values)
        return val - 273.15  # K → °C
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Estimate GDAS t2m bias vs METAR")
    parser.add_argument("--station", default="UNKL", help="ICAO station code")
    parser.add_argument("--days", type=int, default=7, help="Number of recent days to analyze")
    parser.add_argument("--output", default=None, help="Save results as JSON")
    parser.add_argument("--cache-dir", default="data/temp_train/gdas_bias_cache")
    parser.add_argument("--hours", nargs="*", type=int, default=[0, 6, 12, 18],
                        help="UTC hours to compare (synoptic hours)")
    args = parser.parse_args()

    info = STATIONS.get(args.station)
    if info is None:
        print(f"Unknown station {args.station}. Known: {list(STATIONS.keys())}")
        sys.exit(1)
    name, lat, lon, elev = info
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=args.days)

    print(f"Station: {args.station} ({name})")
    print(f"Period: {start.date()} → {end.date()} ({args.days} days)")
    print(f"Synoptic hours: {args.hours}")
    print()

    # Fetch METAR
    print("[1/2] Fetching METAR observations...")
    metars = fetch_metar_hourly(args.station, start, end)
    print(f"  Got {len(metars)} records")

    # Round METAR to nearest hour and average
    metar_by_hour: dict[str, list[float]] = {}
    for rec in metars:
        key = rec["time"].strftime("%Y%m%d_%H")
        metar_by_hour.setdefault(key, []).append(rec["t2m_c"])

    metar_hourly = {k: np.mean(v) for k, v in metar_by_hour.items()}

    # Compare at synoptic hours
    print("[2/2] Comparing with GDAS at synoptic hours...")
    pairs = []
    dt = start
    while dt < end:
        for hh in args.hours:
            t = dt.replace(hour=hh)
            key = t.strftime("%Y%m%d_%H")
            if key not in metar_hourly:
                continue
            metar_c = metar_hourly[key]
            gdas_c = fetch_gdas_t2m_at_point(t, lat, lon, cache_dir)
            if gdas_c is None:
                continue
            bias = metar_c - gdas_c
            pairs.append({
                "time": t.isoformat(),
                "hour": hh,
                "metar_c": round(metar_c, 2),
                "gdas_c": round(gdas_c, 2),
                "bias": round(bias, 2),
            })
            print(f"  {t.strftime('%Y-%m-%d %HUTC')}: METAR={metar_c:+.1f}°C  "
                  f"GDAS={gdas_c:+.1f}°C  bias={bias:+.1f}°C")
        dt += timedelta(days=1)

    if not pairs:
        print("\nERROR: No matching pairs found.")
        sys.exit(1)

    biases = [p["bias"] for p in pairs]
    mean_bias = np.mean(biases)
    std_bias = np.std(biases)

    # Per-hour breakdown
    print(f"\n{'='*50}")
    print(f"GDAS t2m bias summary ({len(pairs)} matchups)")
    print(f"{'='*50}")
    print(f"Overall mean: {mean_bias:+.2f}°C ± {std_bias:.2f}°C")
    for hh in sorted(set(p["hour"] for p in pairs)):
        hh_biases = [p["bias"] for p in pairs if p["hour"] == hh]
        print(f"  {hh:02d} UTC: {np.mean(hh_biases):+.2f}°C (n={len(hh_biases)})")

    print(f"\n→ Recommended: --input-bias-t2m {mean_bias:+.1f}")
    print(f"  (Apply this to live_gdas_forecast.py to warm GDAS input by {mean_bias:+.1f}°C)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "station": args.station,
            "period": f"{start.date()} to {end.date()}",
            "n_pairs": len(pairs),
            "mean_bias": round(float(mean_bias), 3),
            "std_bias": round(float(std_bias), 3),
            "pairs": pairs,
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

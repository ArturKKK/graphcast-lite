#!/usr/bin/env python3
"""
Build MOS (Model Output Statistics) bias correction table for t2m.

Downloads ERA5 hourly data from Open-Meteo archive API and station
observations from Meteostat, then computes:
    bias[month][hour_utc] = mean(station_t2m - era5_t2m)

Usage:
    python scripts/build_mos_table.py
    python scripts/build_mos_table.py --start-date 2020-01-01 --end-date 2024-12-31
    python scripts/build_mos_table.py --output live_runtime_bundle/mos_bias_t2m.json
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# WMO 29570 — Krasnoyarsk (Opytnoe Pole), main synoptic station
STATION_WMO = "29570"
STATION_NAME = "Krasnoyarsk (Opytnoe Pole)"
STATION_LAT = 56.017
STATION_LON = 92.750


def fetch_era5_openmeteo(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch ERA5 hourly t2m from Open-Meteo archive API, month by month."""
    all_rows: list[dict] = []
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    total_records = 0

    for year in range(start_year, end_year + 1):
        year_count = 0
        for month in range(1, 13):
            import calendar
            last_day = calendar.monthrange(year, month)[1]
            m_start = f"{year}-{month:02d}-01"
            m_end = f"{year}-{month:02d}-{last_day:02d}"
            if m_end < start_date or m_start > end_date:
                continue
            if m_start < start_date:
                m_start = start_date
            if m_end > end_date:
                m_end = end_date

            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m"
                f"&start_date={m_start}&end_date={m_end}"
                f"&timezone=GMT"
            )
            for attempt in range(3):
                try:
                    result = subprocess.run(
                        ["curl", "-s", "--max-time", "30", url],
                        capture_output=True, text=True, check=True,
                    )
                    data = json.loads(result.stdout)
                    break
                except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                    if attempt == 2:
                        print(f"    WARNING: failed month {m_start}: {exc}")
                        data = None
                        break
                    time.sleep(2)

            if data is None:
                continue

            times = data["hourly"]["time"]
            temps = data["hourly"]["temperature_2m"]
            for t, temp in zip(times, temps):
                if temp is not None:
                    all_rows.append({"time": pd.Timestamp(t), "era5_t2m_C": temp})
            year_count += len(times)

        total_records += year_count
        print(f"  ERA5 {year}: {year_count} records")

    df = pd.DataFrame(all_rows)
    return df


def fetch_station_isd_lite(station_usaf: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch station hourly observations from NOAA ISD-Lite (fast, no deps)."""
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    all_rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        url = (
            f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/"
            f"{year}/{station_usaf}-99999-{year}.gz"
        )
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", "30", url],
                capture_output=True, check=True,
            )
            import gzip
            text = gzip.decompress(result.stdout).decode("utf-8")
        except (subprocess.CalledProcessError, Exception) as exc:
            print(f"    WARNING: ISD-Lite {year} failed: {exc}")
            continue

        for line in text.strip().split("\n"):
            parts = line.split()
            if len(parts) < 5:
                continue
            y, m, d, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            temp_raw = int(parts[4])
            if temp_raw == -9999:
                continue
            temp_c = temp_raw / 10.0
            ts = pd.Timestamp(year=y, month=m, day=d, hour=h)
            if ts < pd.Timestamp(start_date) or ts > pd.Timestamp(end_date + " 23:59"):
                continue
            all_rows.append({"time": ts, "station_t2m_C": temp_c})

        print(f"    ISD-Lite {year}: {sum(1 for r in all_rows if r['time'].year == year)} records")

    if not all_rows:
        raise RuntimeError(f"No ISD-Lite data for station {station_usaf}")

    return pd.DataFrame(all_rows)


def compute_bias_table(merged: pd.DataFrame) -> tuple[dict, dict]:
    """Compute bias[month][hour_utc] = mean(station - ERA5) in degC."""
    merged = merged.copy()
    merged["month"] = merged["time"].dt.month
    merged["hour"] = merged["time"].dt.hour
    merged["bias"] = merged["station_t2m_C"] - merged["era5_t2m_C"]

    # Remove extreme outliers (instrument errors, data glitches)
    merged = merged[merged["bias"].abs() < 15.0]

    bias_table: dict[str, dict[str, float]] = {}
    count_table: dict[str, dict[str, int]] = {}

    for month in range(1, 13):
        bias_table[str(month)] = {}
        count_table[str(month)] = {}
        for hour in range(24):
            mask = (merged["month"] == month) & (merged["hour"] == hour)
            subset = merged.loc[mask, "bias"]
            n = len(subset)
            if n >= 10:
                bias_table[str(month)][str(hour)] = round(float(subset.mean()), 3)
            else:
                bias_table[str(month)][str(hour)] = 0.0
            count_table[str(month)][str(hour)] = n

    return bias_table, count_table


def print_bias_heatmap(bias_table: dict) -> None:
    """Print a compact ASCII heatmap of the bias table."""
    print("\n  Bias table (station - ERA5, °C):")
    header = "       " + "".join(f"{h:>5d}h" for h in range(0, 24, 3))
    print(header)
    for month in range(1, 13):
        row = f"  M{month:02d}  "
        for hour in range(0, 24, 3):
            val = bias_table[str(month)][str(hour)]
            row += f"{val:+5.1f} "
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MOS bias table for t2m")
    parser.add_argument("--output", default="live_runtime_bundle/mos_bias_t2m.json")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()

    print(f"[MOS] Building bias table for WMO {STATION_WMO} ({STATION_NAME})")
    print(f"[MOS] Period: {args.start_date} to {args.end_date}")

    # 1. ERA5 from Open-Meteo
    print(f"\n[1/3] Fetching ERA5 hourly t2m from Open-Meteo...")
    era5_df = fetch_era5_openmeteo(STATION_LAT, STATION_LON, args.start_date, args.end_date)
    print(f"  Total ERA5 records: {len(era5_df)}")

    # 2. Station observations from NOAA ISD-Lite
    print(f"\n[2/3] Fetching station observations from NOAA ISD-Lite...")
    station_df = fetch_station_isd_lite("029570", args.start_date, args.end_date)
    print(f"  Total station records: {len(station_df)}")

    # 3. Merge and compute bias
    print(f"\n[3/3] Computing bias table...")
    merged = pd.merge(era5_df, station_df, on="time", how="inner")
    print(f"  Matched hourly records: {len(merged)}")

    if len(merged) < 100:
        raise RuntimeError(f"Too few matched records ({len(merged)}) for reliable MOS table")

    bias_table, count_table = compute_bias_table(merged)

    # Stats
    all_biases = [bias_table[m][h] for m in bias_table for h in bias_table[m]]
    print(f"  Overall mean bias: {np.mean(all_biases):+.3f} °C")
    print(f"  Range: [{np.min(all_biases):+.3f}, {np.max(all_biases):+.3f}] °C")

    print_bias_heatmap(bias_table)

    # Save
    result = {
        "station_wmo": STATION_WMO,
        "station_name": STATION_NAME,
        "station_lat": STATION_LAT,
        "station_lon": STATION_LON,
        "variable": "t2m",
        "unit_bias": "degC",
        "period": f"{args.start_date} to {args.end_date}",
        "description": (
            "MOS bias correction for t2m. Apply as: corrected_t2m = model_t2m + bias. "
            "Bias = mean(station_obs - ERA5) per (month, hour_utc). "
            "Positive bias means station is warmer than ERA5 (e.g. urban heat island at night)."
        ),
        "n_matched_records": len(merged),
        "bias_table": bias_table,
        "n_samples": count_table,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[MOS] Saved bias table to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download ISD-Lite data for all Russia MOS stations.

Reads station list from data/russia_isd_stations.json,
downloads ISD-Lite files for 2016-2024 from NOAA,
saves to data/isd_lite_russia/.

Usage:
    python scripts/download_russia_isd_lite.py [--years 2016-2024] [--out-dir data/isd_lite_russia]
"""
import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ISD_LITE_BASE = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"


def download_one(usaf: str, wban: str, year: int, out_dir: Path) -> tuple[str, int, bool]:
    """Download one station-year file. Returns (key, year, success)."""
    fname = f"{usaf}-{wban}-{year}.gz"
    url = f"{ISD_LITE_BASE}/{year}/{fname}"
    out_path = out_dir / fname

    if out_path.exists() and out_path.stat().st_size > 100:
        return (usaf, year, True)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "graphcast-lite/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 50:
            return (usaf, year, False)
        out_path.write_bytes(data)
        return (usaf, year, True)
    except Exception:
        return (usaf, year, False)


def main():
    parser = argparse.ArgumentParser(description="Download ISD-Lite for Russia MOS stations")
    parser.add_argument("--stations", default="data/russia_isd_stations.json")
    parser.add_argument("--years", default="2016-2024", help="Year range, e.g. 2016-2024")
    parser.add_argument("--out-dir", default="data/isd_lite_russia")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    stations_path = repo_root / args.stations
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(stations_path) as f:
        stations = json.load(f)

    y_start, y_end = map(int, args.years.split("-"))
    years = list(range(y_start, y_end + 1))

    print(f"Stations: {len(stations)}")
    print(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"Total files: {len(stations) * len(years)}")
    print(f"Output: {out_dir}")
    print(f"Workers: {args.workers}")
    print()

    tasks = []
    for usaf, meta in stations.items():
        wban = meta.get("wban", "99999")
        for year in years:
            tasks.append((usaf, wban, year))

    done = 0
    ok = 0
    fail = 0
    failed_list = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, usaf, wban, yr, out_dir): (usaf, yr)
            for usaf, wban, yr in tasks
        }
        for fut in as_completed(futures):
            usaf, yr, success = fut.result()
            done += 1
            if success:
                ok += 1
            else:
                fail += 1
                failed_list.append(f"{usaf}/{yr}")

            if done % 200 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{len(tasks)}] ok={ok} fail={fail} "
                    f"rate={rate:.1f}/s ETA={eta:.0f}s"
                )

    print(f"\nDone: {ok} downloaded, {fail} failed")
    if failed_list:
        print(f"Failed ({len(failed_list)}):")
        for f_ in failed_list[:20]:
            print(f"  {f_}")
        if len(failed_list) > 20:
            print(f"  ... and {len(failed_list) - 20} more")

    # Summary
    total_size = sum(f.stat().st_size for f in out_dir.iterdir() if f.suffix == ".gz")
    print(f"\nTotal size: {total_size / 1e6:.1f} MB ({len(list(out_dir.glob('*.gz')))} files)")


if __name__ == "__main__":
    main()

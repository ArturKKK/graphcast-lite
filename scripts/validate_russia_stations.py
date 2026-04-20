#!/usr/bin/env python3
"""
Validate downloaded ISD-Lite data for Russia MOS stations.

Reads station list + downloaded .gz files, checks actual observation counts
per station-year, and produces a final filtered station list.

Usage:
    python scripts/validate_russia_stations.py \
        [--stations data/russia_isd_stations.json] \
        [--isd-dir data/isd_lite_russia] \
        [--min-obs-year 1200] \
        [--min-good-years 7] \
        [--years 2016-2024] \
        [--output data/russia_mos_stations.json]
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def count_obs_in_file(path: Path) -> dict:
    """Count observations and check t2m availability in an ISD-Lite .gz file.

    ISD-Lite format: fixed-width, 24 chars/line min, field 1 = year, field 5 = air temp.
    Missing = -9999.
    """
    obs_total = 0
    obs_t2m = 0
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                obs_total += 1
                try:
                    t = int(parts[4])
                    if t != -9999:
                        obs_t2m += 1
                except ValueError:
                    pass
    except Exception:
        return {"total": 0, "t2m": 0}
    return {"total": obs_total, "t2m": obs_t2m}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", default="data/russia_isd_stations.json")
    parser.add_argument("--isd-dir", default="data/isd_lite_russia")
    parser.add_argument("--min-obs-year", type=int, default=1200,
                        help="Min t2m observations per year (1200 ≈ 3.3/day)")
    parser.add_argument("--min-good-years", type=int, default=7,
                        help="Min years meeting obs threshold (out of total)")
    parser.add_argument("--years", default="2016-2024")
    parser.add_argument("--output", default="data/russia_mos_stations.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    stations_path = repo_root / args.stations
    isd_dir = repo_root / args.isd_dir
    out_path = repo_root / args.output

    with open(stations_path) as f:
        stations = json.load(f)

    y_start, y_end = map(int, args.years.split("-"))
    years = list(range(y_start, y_end + 1))

    print(f"Stations: {len(stations)}")
    print(f"Years: {years[0]}-{years[-1]}")
    print(f"Thresholds: >={args.min_obs_year} t2m obs/yr, >={args.min_good_years}/{len(years)} years")
    print()

    results = {}
    for i, (usaf, meta) in enumerate(stations.items()):
        wban = meta.get("wban", "99999")
        year_obs = {}
        for yr in years:
            fpath = isd_dir / f"{usaf}-{wban}-{yr}.gz"
            if fpath.exists():
                counts = count_obs_in_file(fpath)
                year_obs[yr] = counts
            else:
                year_obs[yr] = {"total": 0, "t2m": 0}

        good_years = sum(
            1 for yr in years if year_obs[yr]["t2m"] >= args.min_obs_year
        )
        avg_t2m = sum(year_obs[yr]["t2m"] for yr in years) / len(years)
        avg_total = sum(year_obs[yr]["total"] for yr in years) / len(years)

        results[usaf] = {
            **meta,
            "year_obs": {str(yr): year_obs[yr] for yr in years},
            "good_years": good_years,
            "avg_t2m_obs": round(avg_t2m),
            "avg_total_obs": round(avg_total),
            "obs_per_day": round(avg_t2m / 365, 1),
        }

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(stations)}] processed")

    # Filter
    good_stations = {
        usaf: info for usaf, info in results.items()
        if info["good_years"] >= args.min_good_years
    }

    # Remove year_obs from output (too bulky)
    output = {}
    for usaf, info in sorted(good_stations.items()):
        output[usaf] = {
            "name": info["name"],
            "lat": info["lat"],
            "lon": info["lon"],
            "elev": info["elev"],
            "icao": info.get("icao"),
            "wban": info.get("wban", "99999"),
            "avg_t2m_obs_yr": info["avg_t2m_obs"],
            "obs_per_day": info["obs_per_day"],
            "good_years": info["good_years"],
        }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Input: {len(stations)} stations")
    print(f"Passed: {len(output)} stations")
    print(f"Rejected: {len(stations) - len(output)} stations")
    print(f"Output: {out_path}")

    def region(lon):
        if lon < 40: return "NW Russia"
        elif lon < 60: return "Central/Volga"
        elif lon < 68: return "Ural"
        elif lon < 85: return "W.Siberia"
        elif lon < 105: return "C.Siberia"
        elif lon < 120: return "E.Siberia"
        elif lon < 140: return "Yakutia/Amur"
        else: return "Far East Coast"

    rcounts = Counter(region(v["lon"]) for v in output.values())
    print(f"\nBy region:")
    for r in ["NW Russia", "Central/Volga", "Ural", "W.Siberia",
              "C.Siberia", "E.Siberia", "Yakutia/Amur", "Far East Coast"]:
        print(f"  {r:20s}: {rcounts.get(r, 0):3d}")

    freq = Counter()
    for v in output.values():
        opd = v["obs_per_day"]
        if opd >= 20:
            freq["hourly+"] += 1
        elif opd >= 6:
            freq["8x/day"] += 1
        else:
            freq["4x/day"] += 1
    print(f"\nBy frequency:")
    for k in ["hourly+", "8x/day", "4x/day"]:
        print(f"  {k:12s}: {freq.get(k, 0)}")

    # Rejected stations detail
    rejected = {
        usaf: info for usaf, info in results.items()
        if info["good_years"] < args.min_good_years
    }
    if rejected:
        rej_reasons = Counter()
        for usaf, info in rejected.items():
            if info["avg_t2m_obs"] == 0:
                rej_reasons["no_data"] += 1
            elif info["good_years"] == 0:
                rej_reasons["very_sparse"] += 1
            else:
                rej_reasons["partially_sparse"] += 1
        print(f"\nRejected breakdown:")
        for k, v in rej_reasons.most_common():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

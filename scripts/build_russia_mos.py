#!/usr/bin/env python3
"""
Learned MOS for Russia: ML post-processor for t2m + wind (ERA5 → station).

Builds a single HistGradientBoostingRegressor over all Russian ISD stations.
ERA5 features downloaded from Open-Meteo Archive API (with caching).
Station observations loaded from local ISD-Lite .gz files.

Stations loaded from JSON (e.g. data/russia_mos_stations.json).
ISD-Lite data from local dir (e.g. data/isd_lite_russia/).

Usage:
    # Full Russia MOS (689 stations)
    python scripts/build_russia_mos.py

    # Quick test on 10 stations
    python scripts/build_russia_mos.py --max-stations 10

    # Custom paths
    python scripts/build_russia_mos.py \
        --stations-file data/russia_mos_stations.json \
        --isd-dir data/isd_lite_russia \
        --output live_runtime_bundle/learned_mos_russia.joblib
"""

import argparse
import calendar
import gzip
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Open-Meteo Archive API hourly ERA5 variables
ERA5_HOURLY_VARS = [
    "temperature_2m",
    "dewpoint_2m",
    "windspeed_10m",
    "winddirection_10m",
    "surface_pressure",
    "cloudcover",
    "shortwave_radiation",
    "precipitation",
]

FEATURE_COLUMNS = [
    "era5_temperature_2m",
    "era5_dewpoint_2m",
    "era5_windspeed_10m",
    "wind_dir_sin",
    "wind_dir_cos",
    "era5_surface_pressure",
    "era5_cloudcover",
    "era5_shortwave_radiation",
    "era5_precipitation",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "solar_elevation",
    "dewpoint_depression",
    "era5_t2m_lag6h",
    "delta_t2m_6h",
    "station_lat",
    "station_lon",
    "station_elev",
]


# ── Solar elevation ─────────────────────────────────────────────────
def solar_elevation(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    """Approximate solar elevation angle (degrees). Spencer 1971."""
    doy = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0
    gamma = 2 * math.pi * (doy - 1) / 365.0
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                     - 0.032077 * math.sin(gamma)
                     - 0.014615 * math.cos(2 * gamma)
                     - 0.04089 * math.sin(2 * gamma))
    solar_time = hour * 60 + eqt + 4 * lon_deg
    ha = math.radians(solar_time / 4.0 - 180.0)
    lat_rad = math.radians(lat_deg)
    sin_elev = (math.sin(lat_rad) * math.sin(decl)
                + math.cos(lat_rad) * math.cos(decl) * math.cos(ha))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))


# ── Data loading ────────────────────────────────────────────────────
def load_station_isd_lite(usaf: str, wban: str, isd_dir: Path,
                          start_year: int, end_year: int) -> pd.DataFrame:
    """Load station observations from local ISD-Lite .gz files."""
    rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        fpath = isd_dir / f"{usaf}-{wban}-{year}.gz"
        if not fpath.exists():
            continue
        try:
            with gzip.open(fpath, "rt") as f:
                text = f.read()
        except Exception:
            continue
        for line in text.strip().split("\n"):
            parts = line.split()
            if len(parts) < 5:
                continue
            y, m, d, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            temp_raw = int(parts[4])
            if temp_raw == -9999:
                continue
            rec: dict = {
                "time": pd.Timestamp(year=y, month=m, day=d, hour=h),
                "station_t2m_C": temp_raw / 10.0,
            }
            if len(parts) >= 9:
                ws_raw = int(parts[8])
                wd_raw = int(parts[7])
                if ws_raw != -9999:
                    rec["station_wind_speed_ms"] = ws_raw / 10.0
                if wd_raw != -9999:
                    rec["station_wind_dir"] = float(wd_raw)
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_era5_openmeteo(lat: float, lon: float,
                         start_year: int, end_year: int,
                         cache_dir: Path) -> pd.DataFrame:
    """Fetch ERA5 hourly from Open-Meteo Archive, with per-year caching."""
    vars_str = ",".join(ERA5_HOURLY_VARS)
    all_dfs: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        # Check per-year cache
        cache_key = f"era5_{lat:.3f}_{lon:.3f}_{year}.csv"
        cache_file = cache_dir / cache_key
        if cache_file.exists():
            ydf = pd.read_csv(cache_file, parse_dates=["time"])
            all_dfs.append(ydf)
            continue

        year_dfs: list[pd.DataFrame] = []
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            m_start = f"{year}-{month:02d}-01"
            m_end = f"{year}-{month:02d}-{last_day:02d}"

            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&hourly={vars_str}"
                f"&start_date={m_start}&end_date={m_end}"
                f"&timezone=GMT"
            )
            data = None
            for attempt in range(3):
                try:
                    result = subprocess.run(
                        ["curl", "-s", "--compressed", "--max-time", "60", url],
                        capture_output=True, text=True, check=True,
                    )
                    data = json.loads(result.stdout)
                    if "hourly" in data:
                        break
                    data = None
                except (subprocess.CalledProcessError, json.JSONDecodeError):
                    pass
                time.sleep(2 * (attempt + 1))

            if data is None:
                continue

            hourly = data["hourly"]
            mdf = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
            for var in ERA5_HOURLY_VARS:
                mdf[f"era5_{var}"] = hourly.get(var)
            year_dfs.append(mdf)
            time.sleep(0.3)

        if year_dfs:
            ydf = pd.concat(year_dfs, ignore_index=True)
            ydf.to_csv(cache_file, index=False)
            all_dfs.append(ydf)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def process_one_station(
    usaf: str, info: dict, isd_dir: Path, cache_dir: Path,
    start_year: int, end_year: int,
) -> pd.DataFrame | None:
    """Process one station: load ISD-Lite + fetch ERA5 + merge + features."""
    wban = info.get("wban", "99999")
    lat, lon, elev = info["lat"], info["lon"], info["elev"]

    # Check merged cache first
    merged_cache = cache_dir / f"merged_{usaf}_{start_year}_{end_year}.csv"
    if merged_cache.exists():
        merged = pd.read_csv(merged_cache, parse_dates=["time"])
        if "bias" not in merged.columns:
            merged["bias"] = merged["station_t2m_C"] - merged["era5_temperature_2m"]
        merged = build_features(merged, lat, lon, elev)
        return merged

    # Load ISD-Lite (local)
    station = load_station_isd_lite(usaf, wban, isd_dir, start_year, end_year)
    if station.empty or len(station) < 100:
        return None

    # Fetch ERA5 (with per-year caching)
    era5 = fetch_era5_openmeteo(lat, lon, start_year, end_year, cache_dir)
    if era5.empty:
        return None

    # Merge
    merged = pd.merge(era5, station, on="time", how="inner")
    merged = merged.dropna(subset=["era5_temperature_2m", "station_t2m_C"])
    merged["bias"] = merged["station_t2m_C"] - merged["era5_temperature_2m"]
    merged = merged[merged["bias"].abs() < 20.0]

    if "station_wind_speed_ms" in merged.columns:
        merged["wind_bias"] = (
            merged["station_wind_speed_ms"] - merged["era5_windspeed_10m"] / 3.6
        )

    # Cache merged
    merged.to_csv(merged_cache, index=False)

    merged = build_features(merged, lat, lon, elev)
    return merged


# ── Feature engineering ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, lat: float, lon: float,
                   elev: float) -> pd.DataFrame:
    """Add derived features to merged ERA5+station DataFrame."""
    out = df.copy()

    hour = out["time"].dt.hour
    doy = out["time"].dt.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    out["solar_elevation"] = [
        solar_elevation(lat, lon, t.to_pydatetime()) for t in out["time"]
    ]
    out["dewpoint_depression"] = (
        out["era5_temperature_2m"] - out["era5_dewpoint_2m"]
    )

    wd_rad = np.deg2rad(out["era5_winddirection_10m"])
    out["wind_dir_sin"] = np.sin(wd_rad)
    out["wind_dir_cos"] = np.cos(wd_rad)

    out["era5_t2m_lag6h"] = out["era5_temperature_2m"].shift(6)
    out["delta_t2m_6h"] = out["era5_temperature_2m"] - out["era5_t2m_lag6h"]

    out["station_lat"] = lat
    out["station_lon"] = lon
    out["station_elev"] = elev

    return out


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Build Russia-wide learned MOS")
    parser.add_argument("--stations-file", default="data/russia_mos_stations.json",
                        help="JSON file with station registry")
    parser.add_argument("--isd-dir", default="data/isd_lite_russia",
                        help="Directory with downloaded ISD-Lite .gz files")
    parser.add_argument("--output", default="live_runtime_bundle/learned_mos_russia.joblib")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--cache-dir", default="data/temp_train/mos_cache_russia",
                        help="Cache directory for ERA5 downloads & merged CSVs")
    parser.add_argument("--max-stations", type=int, default=0,
                        help="Limit number of stations (0 = all, for quick testing)")
    parser.add_argument("--era5-workers", type=int, default=4,
                        help="Parallel workers for ERA5 download (be gentle with API)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    stations_path = repo_root / args.stations_file
    isd_dir = repo_root / args.isd_dir
    cache_dir = repo_root / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(stations_path) as f:
        stations = json.load(f)

    if args.max_stations > 0:
        keys = list(stations.keys())[:args.max_stations]
        stations = {k: stations[k] for k in keys}

    print(f"Stations: {len(stations)}")
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"ISD-Lite dir: {isd_dir}")
    print(f"Cache: {cache_dir}")
    print(f"ERA5 workers: {args.era5_workers}")
    print()

    # ── Step 1: Process all stations ────────────────────────────────
    all_merged: list[pd.DataFrame] = []
    ok_count = 0
    fail_count = 0
    t0 = time.time()

    station_items = list(stations.items())

    # Sequential processing (ERA5 API is rate-limited anyway)
    for i, (usaf, info) in enumerate(station_items):
        try:
            merged = process_one_station(
                usaf, info, isd_dir, cache_dir,
                args.start_year, args.end_year,
            )
            if merged is not None and len(merged) > 100:
                all_merged.append(merged)
                ok_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
            if i < 10:
                print(f"  ERROR {usaf}: {e}")

        if (i + 1) % 25 == 0 or (i + 1) == len(station_items):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(station_items) - i - 1) / rate if rate > 0 else 0
            total_rows = sum(len(m) for m in all_merged)
            print(
                f"  [{i+1}/{len(station_items)}] "
                f"ok={ok_count} fail={fail_count} "
                f"rows={total_rows:,} "
                f"rate={rate:.2f} st/s "
                f"ETA={eta/60:.0f}m"
            )

    if not all_merged:
        print("ERROR: No data. Exiting.")
        sys.exit(1)

    df = pd.concat(all_merged, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["bias"])

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DATA COLLECTION DONE in {elapsed/60:.1f} min")
    print(f"  Stations: {ok_count} ok, {fail_count} failed")
    print(f"  Total samples: {len(df):,}")
    print(f"  Period: {df['time'].min()} → {df['time'].max()}")
    print(f"  Mean bias: {df['bias'].mean():.2f}°C  Std: {df['bias'].std():.2f}°C")
    print(f"{'='*60}")

    # ── Step 2: Chronological split ─────────────────────────────────
    train_mask = df["time"].dt.year <= 2022
    val_mask = df["time"].dt.year == 2023
    test_mask = df["time"].dt.year == 2024

    X_train = df.loc[train_mask, FEATURE_COLUMNS].values
    y_train = df.loc[train_mask, "bias"].values
    X_val = df.loc[val_mask, FEATURE_COLUMNS].values
    y_val = df.loc[val_mask, "bias"].values
    X_test = df.loc[test_mask, FEATURE_COLUMNS].values
    y_test = df.loc[test_mask, "bias"].values

    print(f"Split: train={len(X_train):,} (≤2022) | val={len(X_val):,} (2023) | test={len(X_test):,} (2024)")
    df_test = df.loc[test_mask].copy()

    # ── Step 3: Train ───────────────────────────────────────────────
    print("\n[Training] HistGradientBoostingRegressor (t2m bias)...")
    model = HistGradientBoostingRegressor(
        max_iter=2000,
        max_depth=10,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=20,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s | iterations: {model.n_iter_}")

    # ── Step 4: Evaluate ────────────────────────────────────────────
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    era5_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
    era5_rmse = np.sqrt(mean_squared_error(y_test, np.zeros_like(y_test)))

    # Static MOS baseline
    df_train = df.loc[train_mask].copy()
    df_train["month"] = df_train["time"].dt.month
    df_train["hour"] = df_train["time"].dt.hour
    static_bias = df_train.groupby(["month", "hour"])["bias"].mean().to_dict()

    df_test["month"] = df_test["time"].dt.month
    df_test["hour"] = df_test["time"].dt.hour
    y_static = df_test.apply(
        lambda r: static_bias.get((r["month"], r["hour"]), 0.0), axis=1
    ).values
    static_mae = mean_absolute_error(y_test, y_static)
    static_rmse = np.sqrt(mean_squared_error(y_test, y_static))

    learned_mae = mean_absolute_error(y_test, y_pred_test)
    learned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    val_mae = mean_absolute_error(y_val, y_pred_val)

    print(f"\n{'='*60}")
    print(f"  RESULTS — Test 2024 ({len(y_test):,} samples, {ok_count} stations)")
    print(f"{'='*60}")
    print(f"{'Method':<25s} {'MAE':>8s} {'RMSE':>8s}")
    print(f"{'-'*41}")
    print(f"{'Raw ERA5 (no corr.)':<25s} {era5_mae:8.3f} {era5_rmse:8.3f}")
    print(f"{'Static MOS (month×hour)':<25s} {static_mae:8.3f} {static_rmse:8.3f}")
    print(f"{'Learned MOS (HistGBM)':<25s} {learned_mae:8.3f} {learned_rmse:8.3f}")
    print(f"{'='*60}")
    print(f"  vs ERA5:       {(1 - learned_mae/era5_mae)*100:+.1f}%")
    print(f"  vs static MOS: {(1 - learned_mae/static_mae)*100:+.1f}%")
    print(f"  Val 2023 MAE:  {val_mae:.3f}°C")

    # Per-region
    print(f"\nPer-region MAE (test 2024):")
    regions = {
        "NW Russia (<40E)": lambda r: r["station_lon"] < 40,
        "Central (40-60E)": lambda r: 40 <= r["station_lon"] < 60,
        "Ural (60-68E)": lambda r: 60 <= r["station_lon"] < 68,
        "W.Siberia (68-85E)": lambda r: 68 <= r["station_lon"] < 85,
        "C.Siberia (85-105E)": lambda r: 85 <= r["station_lon"] < 105,
        "E.Siberia (105-120E)": lambda r: 105 <= r["station_lon"] < 120,
        "Far East (>120E)": lambda r: r["station_lon"] >= 120,
    }
    for rname, mask_fn in regions.items():
        rmask = df_test.apply(mask_fn, axis=1).values
        if rmask.sum() == 0:
            continue
        r_era5 = np.mean(np.abs(y_test[rmask]))
        r_learn = np.mean(np.abs(y_test[rmask] - y_pred_test[rmask]))
        print(f"  {rname:25s}: ERA5={r_era5:.2f}  Learned={r_learn:.2f}°C  (n={rmask.sum():,})")

    # Per-season
    print(f"\nPer-season MAE (test 2024):")
    seasons = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
    for name, months in seasons.items():
        mask = df_test["month"].isin(months).values
        if mask.sum() == 0:
            continue
        print(
            f"  {name}: ERA5={np.mean(np.abs(y_test[mask])):.2f}  "
            f"Learned={np.mean(np.abs(y_test[mask] - y_pred_test[mask])):.2f}°C  "
            f"(n={mask.sum():,})"
        )

    # Feature importances
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        print(f"\nFeature importances (top 10):")
        for rank, i in enumerate(np.argsort(imp)[::-1][:10]):
            print(f"  {rank+1:2d}. {FEATURE_COLUMNS[i]:<28s} {imp[i]:.4f}")

    # ── Step 5: Wind MOS ────────────────────────────────────────────
    wind_model = None
    wind_test_mae = None
    if "wind_bias" in df.columns:
        df_wind = df.dropna(subset=FEATURE_COLUMNS + ["wind_bias"]).copy()
        df_wind = df_wind[df_wind["wind_bias"].abs() < 15.0]

        if len(df_wind) > 1000:
            print(f"\n{'='*60}")
            print(f"Wind MOS: {len(df_wind):,} samples with wind obs")
            print(f"Mean wind bias: {df_wind['wind_bias'].mean():.2f} m/s")

            w_train = df_wind[df_wind["time"].dt.year <= 2022]
            w_test = df_wind[df_wind["time"].dt.year == 2024]
            print(f"Split: train={len(w_train):,} | test={len(w_test):,}")

            if len(w_train) > 100 and len(w_test) > 0:
                print("[Training] Wind HistGBR...")
                wind_model = HistGradientBoostingRegressor(
                    max_iter=2000,
                    max_depth=10,
                    learning_rate=0.05,
                    min_samples_leaf=50,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.05,
                    n_iter_no_change=20,
                    random_state=42,
                )
                t0 = time.time()
                wind_model.fit(w_train[FEATURE_COLUMNS].values, w_train["wind_bias"].values)
                dt_w = time.time() - t0
                print(f"  Done in {dt_w:.1f}s | iterations: {wind_model.n_iter_}")

                yw_pred = wind_model.predict(w_test[FEATURE_COLUMNS].values)
                yw_test = w_test["wind_bias"].values
                wind_test_mae = float(mean_absolute_error(yw_test, yw_pred))
                wind_era5_mae = float(mean_absolute_error(yw_test, np.zeros_like(yw_test)))
                print(f"  Wind test MAE: {wind_test_mae:.3f} m/s  (raw: {wind_era5_mae:.3f})")
                print(f"  Improvement: {(1 - wind_test_mae / wind_era5_mae) * 100:+.1f}%")

    # ── Step 6: Save ────────────────────────────────────────────────
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    station_usafs = [usaf for usaf in stations.keys()]
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "stations_trained": station_usafs,
        "n_stations": ok_count,
        "period": f"{args.start_year}-{args.end_year}",
        "split": "train≤2022 / val=2023 / test=2024",
        "test_mae": round(float(learned_mae), 4),
        "test_rmse": round(float(learned_rmse), 4),
        "era5_mae": round(float(era5_mae), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    if wind_model is not None:
        bundle["wind_model"] = wind_model
        bundle["wind_test_mae"] = round(wind_test_mae, 4)
    joblib.dump(bundle, out_path)
    print(f"\n[Saved] {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    print("Done!")


if __name__ == "__main__":
    main()

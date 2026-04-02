#!/usr/bin/env python3
"""
Learned MOS v2: ML post-processor for t2m (ERA5 → station).

Downloads ERA5 hourly multi-variable data from Open-Meteo Archive API and
station observations from NOAA ISD-Lite, then trains a
HistGradientBoostingRegressor to predict: bias = station_t2m - ERA5_t2m.

Key features vs v1:
  - More ERA5 variables (dewpoint, cloudcover, radiation)
  - Solar elevation as feature (Spencer 1971)
  - Dewpoint depression (stability proxy)
  - Proper chronological train/val/test split (≤2022 / 2023 / 2024)
  - Multi-station support
  - Predicts BIAS (not absolute t2m) → applied as correction
  - Comparison with static MOS baseline
  - joblib serialization with metadata

Usage:
    python scripts/build_learned_mos.py
    python scripts/build_learned_mos.py --start-year 2016 --end-year 2024
    python scripts/build_learned_mos.py --stations 029570 029574
"""

import argparse
import gzip
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Station registry ────────────────────────────────────────────────
STATIONS = {
    "029570": {
        "name": "Krasnoyarsk (Opytnoe Pole)",
        "lat": 56.017, "lon": 92.750, "elev": 277,
    },
    "029574": {
        "name": "Emelyanovo (airport)",
        "lat": 56.167, "lon": 92.617, "elev": 262,
    },
    "029612": {
        "name": "Achinsk",
        "lat": 56.267, "lon": 90.500, "elev": 284,
    },
    "029698": {
        "name": "Kansk",
        "lat": 56.200, "lon": 95.717, "elev": 232,
    },
    "029866": {
        "name": "Abakan",
        "lat": 53.750, "lon": 91.400, "elev": 247,
    },
}

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


# ── Data fetching ───────────────────────────────────────────────────
def fetch_era5_openmeteo(lat: float, lon: float,
                         start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch ERA5 hourly from Open-Meteo Archive, monthly requests with compression."""
    import calendar
    vars_str = ",".join(ERA5_HOURLY_VARS)
    all_dfs: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
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
                time.sleep(2)

            if data is None:
                continue

            hourly = data["hourly"]
            mdf = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
            for var in ERA5_HOURLY_VARS:
                mdf[f"era5_{var}"] = hourly.get(var)
            year_dfs.append(mdf)
            time.sleep(0.5)

        if year_dfs:
            ydf = pd.concat(year_dfs, ignore_index=True)
            all_dfs.append(ydf)
            n = ydf["era5_temperature_2m"].notna().sum()
            print(f"    ERA5 {year}: {n} hours")
        else:
            print(f"    ERA5 {year}: FAILED")

    if not all_dfs:
        raise RuntimeError("No ERA5 data fetched")
    return pd.concat(all_dfs, ignore_index=True)


def fetch_station_isd_lite(usaf: str,
                           start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch hourly temperature from NOAA ISD-Lite."""
    rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        url = (
            f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/"
            f"{year}/{usaf}-99999-{year}.gz"
        )
        try:
            result = subprocess.run(
                ["curl", "-s", "--compressed", "--max-time", "30", url],
                capture_output=True, check=True,
            )
            text = gzip.decompress(result.stdout).decode("utf-8")
        except Exception as exc:
            print(f"    WARNING: ISD-Lite {year}: {exc}")
            continue

        count = 0
        for line in text.strip().split("\n"):
            parts = line.split()
            if len(parts) < 5:
                continue
            y, m, d, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            temp_raw = int(parts[4])
            if temp_raw == -9999:
                continue
            rows.append({
                "time": pd.Timestamp(year=y, month=m, day=d, hour=h),
                "station_t2m_C": temp_raw / 10.0,
            })
            count += 1
        print(f"    ISD-Lite {year}: {count} records")

    if not rows:
        raise RuntimeError(f"No ISD-Lite data for {usaf}")
    return pd.DataFrame(rows)


# ── Feature engineering ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, lat: float, lon: float,
                   elev: float) -> pd.DataFrame:
    """Add derived features to merged ERA5+station DataFrame."""
    out = df.copy()

    hour = out["time"].dt.hour
    doy = out["time"].dt.dayofyear

    # Cyclic time encoding
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Solar elevation
    out["solar_elevation"] = [
        solar_elevation(lat, lon, t.to_pydatetime())
        for t in out["time"]
    ]

    # Dewpoint depression (stability / cloud base proxy)
    out["dewpoint_depression"] = (
        out["era5_temperature_2m"] - out["era5_dewpoint_2m"]
    )

    # Wind direction → sin/cos
    wd_rad = np.deg2rad(out["era5_winddirection_10m"])
    out["wind_dir_sin"] = np.sin(wd_rad)
    out["wind_dir_cos"] = np.cos(wd_rad)

    # Temperature trend (6h lag)
    out["era5_t2m_lag6h"] = out["era5_temperature_2m"].shift(6)
    out["delta_t2m_6h"] = out["era5_temperature_2m"] - out["era5_t2m_lag6h"]

    # Static geographic features
    out["station_lat"] = lat
    out["station_lon"] = lon
    out["station_elev"] = elev

    return out


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


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Build learned MOS for t2m")
    parser.add_argument("--output", default="live_runtime_bundle/learned_mos_t2m.joblib")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--stations", nargs="*", default=["029570"],
                        help="USAF station IDs")
    parser.add_argument("--cache-dir", default="data/temp_train/mos_cache",
                        help="Cache directory for downloaded data")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download & merge ────────────────────────────────────
    all_merged: list[pd.DataFrame] = []

    for usaf in args.stations:
        info = STATIONS.get(usaf)
        if info is None:
            print(f"Unknown station {usaf}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Station {usaf}: {info['name']} ({info['lat']}°N, {info['lon']}°E)")
        print(f"{'='*60}")

        cache_file = cache_dir / f"merged_{usaf}_{args.start_year}_{args.end_year}.csv"
        if cache_file.exists():
            print(f"  Loading cached: {cache_file}")
            merged = pd.read_csv(cache_file, parse_dates=["time"])
        else:
            print(f"\n  [1/2] Fetching ERA5 hourly (Open-Meteo Archive)...")
            era5 = fetch_era5_openmeteo(
                info["lat"], info["lon"], args.start_year, args.end_year
            )
            print(f"  Total ERA5: {len(era5)}")

            print(f"\n  [2/2] Fetching station obs (NOAA ISD-Lite)...")
            station = fetch_station_isd_lite(usaf, args.start_year, args.end_year)
            print(f"  Total station: {len(station)}")

            merged = pd.merge(era5, station, on="time", how="inner")
            merged = merged.dropna(subset=["era5_temperature_2m", "station_t2m_C"])
            merged["bias"] = merged["station_t2m_C"] - merged["era5_temperature_2m"]
            merged = merged[merged["bias"].abs() < 20.0]
            print(f"  Matched: {len(merged)}")

            merged.to_csv(cache_file, index=False)
            print(f"  Cached → {cache_file}")

        if "bias" not in merged.columns:
            merged["bias"] = merged["station_t2m_C"] - merged["era5_temperature_2m"]
        merged = build_features(merged, info["lat"], info["lon"], info["elev"])
        all_merged.append(merged)

    if not all_merged:
        print("ERROR: No data. Exiting.")
        sys.exit(1)

    df = pd.concat(all_merged, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["bias"])

    print(f"\n{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"Period: {df['time'].min()} → {df['time'].max()}")
    print(f"Mean bias: {df['bias'].mean():.2f}°C  Std: {df['bias'].std():.2f}°C")
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

    print(f"Split: train={len(X_train)} (≤2022) | val={len(X_val)} (2023) | test={len(X_test)} (2024)")
    df_test = df.loc[test_mask].copy()

    # ── Step 3: Train ───────────────────────────────────────────────
    print("\n[Training] HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s | iterations: {model.n_iter_}")

    # ── Step 4: Evaluate ────────────────────────────────────────────
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Baseline: raw ERA5 (zero correction)
    era5_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
    era5_rmse = np.sqrt(mean_squared_error(y_test, np.zeros_like(y_test)))

    # Baseline: static MOS (mean bias per month×hour from train)
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

    # Learned MOS
    learned_mae = mean_absolute_error(y_test, y_pred_test)
    learned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    val_mae = mean_absolute_error(y_val, y_pred_val)

    print(f"\n{'='*60}")
    print(f"  RESULTS — Test 2024 ({len(y_test)} samples)")
    print(f"{'='*60}")
    print(f"{'Method':<25s} {'MAE':>8s} {'RMSE':>8s}")
    print(f"{'-'*41}")
    print(f"{'Raw ERA5 (no corr.)':<25s} {era5_mae:8.3f} {era5_rmse:8.3f}")
    print(f"{'Static MOS (month×hour)':<25s} {static_mae:8.3f} {static_rmse:8.3f}")
    print(f"{'Learned MOS (HistGBM)':<25s} {learned_mae:8.3f} {learned_rmse:8.3f}")
    print(f"{'='*60}")
    print(f"  vs ERA5:      {(1 - learned_mae/era5_mae)*100:+.1f}%")
    print(f"  vs static MOS: {(1 - learned_mae/static_mae)*100:+.1f}%")
    print(f"  Val 2023 MAE: {val_mae:.3f}°C")

    # Per-season
    print(f"\nPer-season MAE on test 2024:")
    seasons = {"DJF": [12,1,2], "MAM": [3,4,5], "JJA": [6,7,8], "SON": [9,10,11]}
    for name, months in seasons.items():
        mask = df_test["month"].isin(months)
        if mask.sum() == 0:
            continue
        s = mask.values
        print(f"  {name}: ERA5={np.mean(np.abs(y_test[s])):.2f}  "
              f"Static={np.mean(np.abs(y_test[s]-y_static[s])):.2f}  "
              f"Learned={np.mean(np.abs(y_test[s]-y_pred_test[s])):.2f}°C  (n={mask.sum()})")

    # Per hour (every 3h)
    print(f"\nPer-hour MAE on test 2024:")
    for h in range(0, 24, 3):
        mask = (df_test["hour"] == h).values
        if mask.sum() == 0:
            continue
        e = np.mean(np.abs(y_test[mask]))
        l = np.mean(np.abs(y_test[mask] - y_pred_test[mask]))
        print(f"  {h:02d} UTC: ERA5={e:.2f}  Learned={l:.2f}  (Δ={e-l:+.2f})")

    # Feature importances
    try:
        imp = model.feature_importances_
        print(f"\nFeature importances (top 10):")
        for rank, i in enumerate(np.argsort(imp)[::-1][:10]):
            print(f"  {rank+1:2d}. {FEATURE_COLUMNS[i]:<28s} {imp[i]:.4f}")
    except AttributeError:
        from sklearn.inspection import permutation_importance
        print(f"\nPermutation importances (top 10, test set):")
        r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        imp = r.importances_mean
        for rank, i in enumerate(np.argsort(imp)[::-1][:10]):
            print(f"  {rank+1:2d}. {FEATURE_COLUMNS[i]:<28s} {imp[i]:.4f}")

    # ── Step 5: Save ────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "stations_trained": args.stations,
        "period": f"{args.start_year}-{args.end_year}",
        "split": "train≤2022 / val=2023 / test=2024",
        "test_mae": round(float(learned_mae), 4),
        "test_rmse": round(float(learned_rmse), 4),
        "era5_mae": round(float(era5_mae), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    joblib.dump(bundle, out_path)
    print(f"\n[Saved] {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    print("Done!")


if __name__ == "__main__":
    main()
